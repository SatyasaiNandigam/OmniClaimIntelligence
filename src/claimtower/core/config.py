"""Config and pack loader (T0.5). Business rules live in YAML, never in Python.

`load_settings()` reads configs/*.yaml and the active pack, validates every file against a
Pydantic model, cross-checks references, and raises ONE ConfigError listing all problems.
Call it at startup. Env vars USE_MOCK_PAS / USE_MOCK_CLAIMS / USE_MOCK_RECON /
LLM_REPLAY_MODE override the flags in app.yaml; CLAIMTOWER_ROOT overrides the repo root.
"""
from __future__ import annotations

import os
import re
import string
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, get_args

import yaml
from pydantic import ConfigDict, Field, ValidationError, model_validator

from claimtower.core.errors import ConfigError
from claimtower.core.schemas import SettlementInputs
from claimtower.core.schemas.common import Contract, GapType, IssueStatus, QuestionCategory, QuestionStatus
from claimtower.core.schemas.policy import ClauseType, RelationType


# ----------------------------------------------------------------- models
class LLMConfig(Contract):
    provider: str
    text_model: str
    vision_model: str
    embedding_model: str | None = None                                        # for clause retrieval (T3.4); null = embeddings off
    embedding_dimensions: int | None = Field(default=None, ge=8)              # shorter vectors: cheaper, smaller cassettes
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)       # null = not sent (reasoning models only accept their default)
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None   # reasoning models: trade accuracy for cost/latency


class Flags(Contract):
    use_mock_pas: bool = True
    use_mock_claims: bool = True
    use_mock_recon: bool = True
    llm_replay_mode: bool = False


class AppConfig(Contract):
    llm: LLMConfig
    abstention: dict[str, int | float] = {}
    flags: Flags = Flags()
    active_pack: str


class AutonomyOp(Contract):
    mode: Literal["autonomous", "draft_only", "controlled_api", "human_reviewed", "authorised_human_only"]
    roles: list[str] = []

    @model_validator(mode="after")
    def _roles_for_human_only(self) -> "AutonomyOp":
        if self.mode == "authorised_human_only" and not self.roles:
            raise ValueError("authorised_human_only operations must list at least one role")
        return self


class AutonomyMatrix(Contract):
    operations: dict[str, AutonomyOp]


class RBAC(Contract):
    roles: list[str] = Field(min_length=1)
    approval_order: list[str] = []                  # SAMPLE: who may decide what a lower role may; later = more authority

    @model_validator(mode="after")
    def _order_in_roles(self) -> "RBAC":
        unknown = sorted(set(self.approval_order) - set(self.roles))
        if unknown:
            raise ValueError(f"approval_order names roles that are not in roles: {unknown}")
        return self


# Guardrail 4: extraction confidence must never drive routing.
_FORBIDDEN_ROUTING_FEATURES = {"extraction_confidence", "confidence"}
_NUMERIC_OPS = ("gt", "gte", "lt", "lte")
Tier3 = Literal["simple", "medium", "complex"]


class Condition(Contract):
    """One routing test on a feature, e.g. {feature: exposure_inr, gt: 100000, lte: 500000, code: EXPOSURE_MEDIUM_BAND}.
    All operators given on one condition must hold; `code` is emitted as a reason code when it matches."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    feature: str
    is_: str | bool | None = Field(default=None, alias="is")
    in_: list[str] | None = Field(default=None, alias="in")
    gt: float | None = None
    gte: float | None = None
    lt: float | None = None
    lte: float | None = None
    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")

    @model_validator(mode="after")
    def _has_operator(self) -> "Condition":
        if self.is_ is None and self.in_ is None and all(getattr(self, o) is None for o in _NUMERIC_OPS):
            raise ValueError(f"condition {self.code}: needs one of is / in / gt / gte / lt / lte")
        return self


class MandatoryReferral(Contract):
    any_of: list[Condition] = Field(min_length=1)       # any match -> complex, regardless of other rules


class TierRule(Contract):
    all_of: list[Condition] = []
    any_of: list[Condition] = []
    default: bool = False


class RoutingTiers(Contract):
    simple: TierRule
    medium: TierRule
    complex: TierRule


class RoutingRules(Contract):
    """Evaluation order (first matching stage decides the tier; its matching codes are the reason codes):
    1 mandatory_referral.any_of -> complex | 2 tiers.complex.any_of -> complex | 3 tiers.simple.all_of -> simple
    | 4 tiers.medium.any_of -> medium | 5 otherwise -> complex (default)."""

    rules_version: str
    features: dict[str, list[str] | Literal["number", "boolean"]]
    approver_roles: dict[Tier3, str]
    mandatory_referral: MandatoryReferral
    tiers: RoutingTiers

    @model_validator(mode="after")
    def _checks(self) -> "RoutingRules":
        if not self.tiers.complex.default:
            raise ValueError("tiers.complex must be the default tier (default: true)")
        if not self.tiers.simple.all_of or self.tiers.simple.any_of:
            raise ValueError("tiers.simple must use all_of only")
        if not self.tiers.medium.any_of or self.tiers.medium.all_of:
            raise ValueError("tiers.medium must use any_of only")
        if set(self.approver_roles) != {"simple", "medium", "complex"}:
            raise ValueError("approver_roles must define simple, medium and complex")
        banned = {f for f in self.features if f in _FORBIDDEN_ROUTING_FEATURES}
        conds = [*self.mandatory_referral.any_of, *self.tiers.complex.any_of, *self.tiers.simple.all_of, *self.tiers.medium.any_of]
        banned |= {c.feature for c in conds} & _FORBIDDEN_ROUTING_FEATURES
        if banned:
            raise ValueError(f"extraction confidence is not an allowed routing feature: {sorted(banned)}")
        for c in conds:
            kind = self.features.get(c.feature)
            if kind is None:
                raise ValueError(f"condition {c.code}: unknown feature '{c.feature}' (declare it under features)")
            if kind == "number":
                if c.is_ is not None or c.in_ is not None:
                    raise ValueError(f"condition {c.code}: numeric feature '{c.feature}' takes gt/gte/lt/lte only")
            elif kind == "boolean":
                if not isinstance(c.is_, bool) or c.in_ is not None or any(getattr(c, o) is not None for o in _NUMERIC_OPS):
                    raise ValueError(f"condition {c.code}: boolean feature '{c.feature}' takes 'is: true|false' only")
            else:
                used = ([c.is_] if isinstance(c.is_, str) else []) + (c.in_ or [])
                bad = set(used) - set(kind)
                if bad or any(getattr(c, o) is not None for o in _NUMERIC_OPS) or isinstance(c.is_, bool):
                    raise ValueError(f"condition {c.code}: value(s) {sorted(map(str, bad)) or c.is_} not allowed for "
                                     f"'{c.feature}'; allowed: {kind}")
        return self


class PackMeta(Contract):
    name: str
    version: str
    claim_types: list[str] = Field(min_length=1)


FieldType = Literal["text", "narrative", "date", "date_range", "money", "id"]


class FieldSpec(Contract):
    """One field the extractor looks for in a document type (T2.3). The first label is the canonical one; the others are synonyms."""

    labels: list[str] = []                  # matched case-insensitively against the label of 'Label: value' lines
    type: FieldType = "text"
    required: bool = False                  # missing required fields are reported, never silently skipped
    pattern: str | None = None              # id fields: regex the value must contain (case-insensitive)
    from_body: bool = False                 # no label? take the sentence-like body lines instead (lower confidence)
    description: str = ""                   # what the field is, for the model reading the document (the rules ignore it)

    @model_validator(mode="after")
    def _checks(self) -> "FieldSpec":
        if not self.labels and not self.from_body:
            raise ValueError("a field needs at least one label or from_body: true")
        if self.pattern is not None:
            if self.type != "id":
                raise ValueError("'pattern' is only for type: id")
            try:
                re.compile(self.pattern)
            except re.error as e:
                raise ValueError(f"bad field pattern {self.pattern!r}: {e}") from None
        self.labels = [lab.strip().lower() for lab in self.labels]
        return self


class TableConfig(Contract):
    """How rows of a 'Label: amount' table are classified. Regexes are case-insensitive and match the whole label."""

    total_labels: list[str]
    subtotal_labels: list[str]
    tax_labels: list[str]
    discount_labels: list[str]
    skip_labels: list[str]                  # money rows that are not line items (e.g. 'Estimated loss')

    @model_validator(mode="after")
    def _compile(self) -> "TableConfig":
        for pat in [*self.total_labels, *self.subtotal_labels, *self.tax_labels, *self.discount_labels, *self.skip_labels]:
            try:
                re.compile(pat)
            except re.error as e:
                raise ValueError(f"bad table pattern {pat!r}: {e}") from None
        return self


class ExtractorConfig(Contract):
    """Confidence and layout parameters. SAMPLE values tuned on the labelled set (evals/baselines/extraction.json), not insurer policy."""

    low_confidence: float = Field(ge=0, le=1)             # below this a value is FLAGGED (never hidden)
    label_confidence: float = Field(ge=0, le=1)           # value found under the canonical label
    synonym_confidence: float = Field(ge=0, le=1)         # ...under a synonym label
    body_confidence: float = Field(ge=0, le=1)            # unlabelled body text
    unparseable_confidence: float = Field(ge=0, le=1)     # a label was found but the value would not parse
    ambiguous_factor: float = Field(ge=0, le=1)           # multiplier for ambiguous dates (e.g. 03/04/2026)
    conflict_factor: float = Field(ge=0, le=1)            # multiplier when the document states different values for one field
    table_mismatch_confidence: float = Field(ge=0, le=1)  # items/total that do not reconcile
    ocr_confidence_cap: float = Field(ge=0, le=1)
    llm_confidence_cap: float = Field(ge=0, le=1)
    wrap_gap_ratio: float = Field(gt=0, le=1)             # PDF lines closer than this x line height belong to the same paragraph
    wrap_x_tolerance: float = Field(ge=0)                 # ...and must start at about the same x
    body_min_words: int = Field(ge=1)
    table_tolerance: Decimal = Field(ge=0)                # allowed |stated total - computed total|
    boilerplate_patterns: list[str] = []                  # lines that are never body text (page footers, banners)
    table: TableConfig

    @model_validator(mode="after")
    def _compile(self) -> "ExtractorConfig":
        for pat in self.boilerplate_patterns:
            try:
                re.compile(pat)
            except re.error as e:
                raise ValueError(f"bad boilerplate pattern {pat!r}: {e}") from None
        return self


class DocTypeSpec(Contract):
    """What the rule classifier looks for. Regexes are case-insensitive; phrases are literal text (matched on word boundaries)."""

    title: str = ""
    title_patterns: list[str] = []          # regexes tried against heading-like lines near the top of page 1
    phrases: list[str] = []                 # supporting phrases anywhere in the first pages
    phrases_needed: int = Field(default=3, ge=1)          # distinct phrases that give full phrase score
    negative_phrases: list[str] = []        # phrases that argue AGAINST this type
    structure_patterns: list[str] = []      # regexes on ANY of the first lines (e.g. an e-mail header block); counts as heading evidence
    structure_needed: int = Field(default=2, ge=1)        # distinct structure patterns that must match
    fields: dict[str, FieldSpec] = {}       # what the extractor pulls out of this document type (T2.3)
    table: bool = False                     # extract and reconcile a 'Label: amount' table (estimates, invoices)
    table_expects_tax: bool = False         # flag a table with no tax row (informational)

    @model_validator(mode="after")
    def _regexes_compile(self) -> "DocTypeSpec":
        for pat in [*self.title_patterns, *self.structure_patterns]:
            try:
                re.compile(pat)
            except re.error as e:
                raise ValueError(f"bad pattern {pat!r}: {e}") from None
        return self


class ClassifierConfig(Contract):
    """Thresholds are SAMPLE values chosen on the labelled set (see evals/baselines/classification.json), not insurer policy."""

    title_weight: float = Field(ge=0, le=1)
    phrase_weight: float = Field(ge=0, le=1)
    negative_penalty: float = Field(ge=0, le=1)
    min_score: float = Field(ge=0, le=1)                  # below this the classifier abstains
    min_margin: float = Field(ge=0, le=1)                 # winner must beat the runner-up by this much
    answer_without_heading: bool                          # False: vocabulary alone never decides a type (it must have heading evidence)
    min_phrase_hits: int = Field(ge=0)                    # ...and heading evidence alone is not enough: this many supporting phrases too
    title_zone_lines: int = Field(ge=1)
    title_max_chars: int = Field(ge=10)
    pages_scanned: int = Field(ge=1)
    ocr_confidence_cap: float = Field(ge=0, le=1)         # text from OCR is less trustworthy
    llm_min_confidence: float = Field(ge=0, le=1)
    llm_confidence_cap: float = Field(ge=0, le=1)
    photo_confidence: float = Field(ge=0, le=1)
    container_confidence: float = Field(ge=0, le=1)       # e-mail recognised from its headers

    @model_validator(mode="after")
    def _weights(self) -> "ClassifierConfig":
        if abs(self.title_weight + self.phrase_weight - 1) > 1e-9:
            raise ValueError("title_weight + phrase_weight must equal 1")
        return self


class DocumentTypes(Contract):
    classifier: ClassifierConfig
    extractor: ExtractorConfig
    types: dict[str, DocTypeSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _has_fallback(self) -> "DocumentTypes":
        if "other" not in self.types:
            raise ValueError("types must include 'other' (classifier must be able to abstain)")
        if "unknown" in self.types:
            raise ValueError("'unknown' is the abstention value, not a document type")
        return self


class IntakeChecklistCfg(Contract):
    loss_type: str | None = None                      # the pack's line of business: assumed when a claim states none
    required_fields: list[dict[str, str]] = []       # header fields a registration needs: {name, description}
    required_documents: list[dict[str, str]] = []    # {type, description}: received when a stored document has that detected type
    required_photo_views: list[Any] = []


class ClauseLink(Contract):
    """Which policy clauses a question bears on: clause types (any of) AND, when given, a word pattern in the clause heading or text.
    A `fallback` link is used only when no earlier link of the question found a clause (e.g. the cover when no sublimit exists)."""

    types: list[ClauseType] = Field(min_length=1)
    match: list[str] = []
    fallback: bool = False

    @model_validator(mode="after")
    def _valid(self) -> "ClauseLink":
        for p in self.match:
            try:
                re.compile(p, re.IGNORECASE)
            except re.error as e:
                raise ValueError(f"bad clause-link pattern {p!r}: {e}") from e
        return self


class QuestionTemplate(Contract):
    """A canonical open question: what must be known to decide, and which clauses it bears on (T4.1)."""

    id: str
    text: str
    category: QuestionCategory
    establishing_criteria: list[str] = []         # what would establish the answer (used by T4.4 / T4.5)
    needs_policy: bool = True                     # False: asked even when there is no policy snapshot (e.g. was the policy in force?)
    criteria_mode: Literal["all", "any"] = "all"   # all: every criterion must be met; any: the criteria are alternatives (one is enough)
    clause_links: list[ClauseLink] = []


class ModelQuestionsConfig(Contract):
    """Claim-specific questions the model may ADD to the template questions (T4.1). Checked, never trusted: values are SAMPLE.
    A question is accepted only if it cites existing clauses / facts and quotes them verbatim; it never replaces a template question."""

    enabled: bool = False
    max_questions: int = Field(default=4, ge=0)
    min_confidence: float = Field(default=0.6, ge=0, le=1)
    confidence_cap: float = Field(default=0.85, ge=0, le=1)
    max_text_chars: int = Field(default=200, ge=40)
    max_criteria: int = Field(default=3, ge=1)
    max_clause_chars: int = Field(default=700, ge=100)
    max_fact_chars: int = Field(default=300, ge=50)
    duplicate_overlap: float = Field(default=0.6, ge=0, le=1)       # word overlap with an existing question above which it is a repeat
    duplicate_check: bool = True                  # a second model look: does a standard question already ask for the same evidence?
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    forbidden_patterns: list[str] = []            # a question must ask what is known, never pre-judge the decision

    @model_validator(mode="after")
    def _valid(self) -> "ModelQuestionsConfig":
        if self.min_confidence > self.confidence_cap:
            raise ValueError("min_confidence cannot be above confidence_cap: no proposal could ever be accepted")
        for p in self.forbidden_patterns:
            try:
                re.compile(p, re.IGNORECASE)
            except re.error as e:
                raise ValueError(f"bad forbidden pattern {p!r}: {e}") from e
        return self


class FactLinkConfig(Contract):
    """How facts are attached to open questions (T4.2). The fact templates' question tags are the floor; the model judges the
    DIRECTION of each link (supporting / contradicting / neutral), may add links the tags missed and may mark a tagged link unrelated.
    Checked, never trusted."""

    enabled: bool = False
    min_confidence: float = Field(default=0.6, ge=0, le=1)          # a judgement below this is ignored (the tagged link stays neutral)
    confidence_cap: float = Field(default=0.9, ge=0, le=1)
    max_rationale_chars: int = Field(default=300, ge=40)
    max_links_per_fact: int = Field(default=3, ge=1)
    batch_size: int = Field(default=20, ge=1)                       # facts per model call
    max_fact_chars: int = Field(default=400, ge=50)
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None

    @model_validator(mode="after")
    def _valid(self) -> "FactLinkConfig":
        if self.min_confidence > self.confidence_cap:
            raise ValueError("min_confidence cannot be above confidence_cap: no judgement could ever be accepted")
        return self


ContradictionSeverity = Literal["low", "medium", "high"]


def _value_patterns(patterns: list[str], what: str) -> list[str]:
    if not patterns:
        raise ValueError(f"{what} needs at least one pattern")
    for p in patterns:
        try:
            rx = re.compile(p, re.IGNORECASE)
        except re.error as e:
            raise ValueError(f"bad pattern {p!r} in {what}: {e}") from e
        if "value" not in rx.groupindex:
            raise ValueError(f"pattern {p!r} in {what} needs a (?P<value>...) group")
    return patterns


class DateRule(Contract):
    """Facts that state the SAME date (e.g. the date of loss) must agree; two different dates are a contradiction."""

    id: str
    question: str
    severity: ContradictionSeverity
    sources: list[str]

    @model_validator(mode="after")
    def _valid(self) -> "DateRule":
        _value_patterns(self.sources, f"date rule {self.id}")
        return self


class OrderRule(Contract):
    """A `later` date may not be before an `earlier` one (an invoice or inspection dated before the loss happened)."""

    id: str
    question: str
    severity: ContradictionSeverity
    earlier: list[str]
    later: list[str]

    @model_validator(mode="after")
    def _valid(self) -> "OrderRule":
        _value_patterns(self.earlier, f"order rule {self.id} (earlier)")
        _value_patterns(self.later, f"order rule {self.id} (later)")
        return self


class AmountBand(Contract):
    min_gap_pct: float = Field(gt=0)               # relative gap |a - b| / max(a, b), in percent
    severity: ContradictionSeverity


class AmountRule(Contract):
    """Two amounts that should agree (what is claimed and what is assessed): a gap above the lowest band is a contradiction."""

    id: str
    question: str
    a: list[str]
    b: list[str]
    bands: list[AmountBand] = Field(min_length=1)

    @model_validator(mode="after")
    def _valid(self) -> "AmountRule":
        _value_patterns(self.a, f"amount rule {self.id} (a)")
        _value_patterns(self.b, f"amount rule {self.id} (b)")
        gaps = [b.min_gap_pct for b in self.bands]
        if gaps != sorted(gaps) or len(set(gaps)) != len(gaps):
            raise ValueError(f"amount rule {self.id}: bands must have strictly increasing min_gap_pct")
        return self


class ContradictionConfig(Contract):
    """Contradiction detector (T4.3). Dates and amounts are compared by RULES; narrative conflicts (cause, extent, timing) are proposed by a
    model per open question and CHECKED: both sides cite real facts, quoted verbatim, with a rationale. Values are SAMPLE."""

    enabled: bool = False                          # the model part (the rules always run)
    min_confidence: float = Field(default=0.6, ge=0, le=1)
    confidence_cap: float = Field(default=0.9, ge=0, le=1)
    max_explanation_chars: int = Field(default=400, ge=40)
    max_per_question: int = Field(default=3, ge=1)
    max_fact_chars: int = Field(default=400, ge=50)
    model_categories: list[QuestionCategory] = ["cause", "timing", "extent"]       # questions the model looks at
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    severity_rubric: list[str] = []                # shown to the model; the severity of a rule contradiction comes from the rule
    date_rules: list[DateRule] = []
    order_rules: list[OrderRule] = []
    amount_rules: list[AmountRule] = []

    @model_validator(mode="after")
    def _valid(self) -> "ContradictionConfig":
        if self.min_confidence > self.confidence_cap:
            raise ValueError("min_confidence cannot be above confidence_cap: no contradiction could ever be accepted")
        ids = [r.id for r in (*self.date_rules, *self.order_rules, *self.amount_rules)]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate contradiction rule ids")
        return self


SUFFICIENCY_CONDITIONS = ("blocking_contradiction", "unclear_criteria", "all_met", "any_met", "any_partly", "supporting_link",
                          "linked_fact")


class CriterionRule(Contract):
    """A criterion decided from the POLICY, not from facts: no model is asked about it."""

    question: str
    criterion: str                                  # the exact text of one of the question's establishing criteria
    met_if: Literal["policy_answer_known", "linked_clause_type"]
    types: list[ClauseType] = []

    @model_validator(mode="after")
    def _valid(self) -> "CriterionRule":
        if self.met_if == "linked_clause_type" and not self.types:
            raise ValueError("met_if linked_clause_type needs clause types")
        return self


class SufficiencyRow(Contract):
    """One row of the decision table. All listed conditions must hold (an absent one is ignored); the first matching row decides."""

    id: str
    when: dict[str, bool] = {}
    status: QuestionStatus
    gap: GapType | None = None
    abstain: bool = False

    @model_validator(mode="after")
    def _valid(self) -> "SufficiencyRow":
        bad = sorted(set(self.when) - set(SUFFICIENCY_CONDITIONS))
        if bad:
            raise ValueError(f"row {self.id}: unknown condition(s) {bad}; known: {list(SUFFICIENCY_CONDITIONS)}")
        return self


class PurposeConfig(Contract):
    """Settling a question on the evidence as a whole. The listed criteria are examples of what would establish a question; for a question about HOW something happened (its cause, its timing)
    different evidence can answer it just as well. When a question is only partly evidenced by its criteria, a second model call asks whether the facts together answer what the question is FOR.
    It is accepted only if it cites the facts, quotes one verbatim, no linked fact points the other way and no contradiction blocks. Off, or a question category not listed: nothing changes."""

    enabled: bool = False
    categories: list[str] = []                      # the question categories it may apply to (a question about a required document or an amount is never settled this way)
    min_confidence: float = Field(default=0.75, ge=0, le=1)
    max_rationale_chars: int = Field(default=300, ge=40)


class SufficiencyConfig(Contract):
    """Sufficiency classifier (T4.4): for each question, which criteria are met by the evidence, and from that the status and the gap type
    (contradictory / missing / insufficient) through the decision table. A criterion about the policy is decided from the snapshot; one
    about facts is judged by a model, CHECKED (cited facts, verbatim quote), which may say 'unclear': the question then abstains."""

    enabled: bool = False                           # the model part: without it fact criteria are 'unclear' (unmet if no fact bears)
    min_confidence: float = Field(default=0.6, ge=0, le=1)
    confidence_cap: float = Field(default=0.9, ge=0, le=1)
    max_rationale_chars: int = Field(default=300, ge=40)
    max_fact_chars: int = Field(default=400, ge=50)
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    blocking_severities: list[Literal["low", "medium", "high"]] = ["medium", "high"]   # a contradiction at these severities blocks
    criteria_rules: list[CriterionRule] = []
    purpose: PurposeConfig = PurposeConfig()
    table: list[SufficiencyRow] = Field(min_length=1)

    @model_validator(mode="after")
    def _valid(self) -> "SufficiencyConfig":
        if self.min_confidence > self.confidence_cap:
            raise ValueError("min_confidence cannot be above confidence_cap: no judgement could ever be accepted")
        ids = [r.id for r in self.table]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate decision-table row ids")
        if self.table[-1].when:
            raise ValueError("the last row of the decision table must have no conditions, so that every case is decided")
        return self


class RequestRole(Contract):
    """Someone a request can be addressed to, and which kinds of document they can supply."""

    label: str
    internal: bool = False                          # an internal team (never the claimant or a third party)
    # How the request reaches them. direct: the system may send it (the claimant). via_claimant_or_adjuster: a third party (plumber,
    # surveyor, contractor): the system only DRAFTS it, and the claimant or an adjuster passes it on. internal: asked inside the insurer.
    delivery: Literal["direct", "via_claimant_or_adjuster", "internal"] = "direct"
    document_types: list[str] = []                  # document types (document_types.yaml) this role can produce

    @model_validator(mode="after")
    def _valid(self) -> "RequestRole":
        if self.internal != (self.delivery == "internal"):
            raise ValueError("an internal role must have delivery 'internal', and only an internal role may")
        return self


class CriterionRole(Contract):
    question: str
    criterion: str                                  # the exact text of one of the question's establishing criteria
    role: str


class RequestConfig(Contract):
    """Minimum-evidence requests (T4.5): the smallest, most specific request that would resolve each open question, one per addressee.
    WHO is asked is decided by this table; the model only writes the items, CHECKED: each item serves a listed need, asks for no document
    already on file, and never pre-judges the decision. A need the model misses gets a rule-written item: nothing unresolved is dropped."""

    enabled: bool = False                           # the model writes the items; without it the rule-written items are the request
    max_items_per_request: int = Field(default=4, ge=1)
    max_item_chars: int = Field(default=250, ge=40)
    duplicate_overlap: float = Field(default=0.6, ge=0, le=1)
    repair_rounds: int = Field(default=1, ge=0)    # times the writer is asked again, with the refusal reasons, while needs are uncovered
    max_known_facts: int = Field(default=6, ge=0)  # facts shown to the model per need, so the request does not ask for what is known
    max_fact_chars: int = Field(default=300, ge=50)
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    default_role: str
    roles: dict[str, RequestRole]
    criteria_roles: list[CriterionRole] = []
    conflict_roles: dict[str, str] = {}             # question id -> who is asked to resolve a contradiction on it
    forbidden_patterns: list[str] = []

    @model_validator(mode="after")
    def _valid(self) -> "RequestConfig":
        if self.default_role not in self.roles:
            raise ValueError(f"default_role {self.default_role!r} is not one of the roles")
        used = [self.default_role, *[c.role for c in self.criteria_roles], *self.conflict_roles.values()]
        unknown = sorted({r for r in used if r not in self.roles})
        if unknown:
            raise ValueError(f"unknown role(s) {unknown}; roles are {sorted(self.roles)}")
        for p in self.forbidden_patterns:
            try:
                re.compile(p, re.IGNORECASE)
            except re.error as e:
                raise ValueError(f"bad forbidden pattern {p!r}: {e}") from e
        return self


SCENARIOS = ("A", "B", "C")
ISSUE_KINDS = ("cover", "exclusion", "amount")


class CounterfactualFraming(Contract):
    """The three 'what if' findings for one material question."""

    question: str
    issue_kinds: list[Literal["cover", "exclusion", "amount"]] = ["cover", "exclusion", "amount"]   # issue kinds a finding re-assesses
    a: str                                          # scenario A: a finding that supports the claim as presented
    b: str                                          # scenario B: a finding that points away from it
    c: str                                          # scenario C: an inconclusive finding


class CounterfactualConfig(Contract):
    """Counterfactual scenarios (T4.6): for each unresolved MATERIAL question, what each possible finding would change, never what the
    decision will be. Effects per kind of coverage issue and next steps are pack content; a model may rewrite the wording for the claim,
    CHECKED by the linter (no verdict language, conditional, only known issues), falling back to the rule text. Values are SAMPLE."""

    enabled: bool = False                           # the model rewrite; without it the rule text is the scenario
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    max_implication_chars: int = Field(default=420, ge=80)
    max_next_step_chars: int = Field(default=260, ge=40)
    framings: list[CounterfactualFraming] = []
    effects: dict[str, dict[str, str]]              # issue kind (cover | exclusion | amount) -> scenario (A | B | C) -> effect on the issue
    review_suffix: dict[str, str] = {}              # scenario -> sentence appended to the implication (e.g. B: needs senior human review)
    next_steps: dict[str, str]                      # scenario -> next step; {outstanding} = who still has to supply evidence
    forbidden_patterns: list[str] = []              # wording that predicts or recommends a decision
    conditional_markers: list[str] = []             # an implication must be conditional: it contains at least one of these words

    @model_validator(mode="after")
    def _valid(self) -> "CounterfactualConfig":
        for kind, by in self.effects.items():
            if kind not in ISSUE_KINDS or set(by) != set(SCENARIOS):
                raise ValueError(f"effects[{kind!r}] must be one of {ISSUE_KINDS} with scenarios {SCENARIOS}")
        missing = [k for k in ISSUE_KINDS if k not in self.effects]
        if missing:
            raise ValueError(f"effects is missing issue kind(s) {missing}")
        if set(self.next_steps) != set(SCENARIOS) or not set(self.review_suffix) <= set(SCENARIOS):
            raise ValueError("next_steps needs scenarios A, B and C; review_suffix may name only A, B or C")
        for p in self.forbidden_patterns:
            try:
                re.compile(p, re.IGNORECASE)
            except re.error as e:
                raise ValueError(f"bad forbidden pattern {p!r}: {e}") from e
        return self


class QuestionTemplates(Contract):
    questions: list[QuestionTemplate] = []
    model_questions: ModelQuestionsConfig = ModelQuestionsConfig()
    fact_links: FactLinkConfig = FactLinkConfig()
    contradictions: ContradictionConfig = ContradictionConfig()
    sufficiency: SufficiencyConfig | None = None
    requests: RequestConfig | None = None
    counterfactuals: CounterfactualConfig | None = None

    @model_validator(mode="after")
    def _unique(self) -> "QuestionTemplates":
        ids = [q.id for q in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate question ids")
        by_id = {q.id: q for q in self.questions}
        if self.sufficiency is not None and self.questions:
            for cr in self.sufficiency.criteria_rules:
                if cr.question not in by_id or cr.criterion not in by_id[cr.question].establishing_criteria:
                    raise ValueError(f"sufficiency criteria_rules: {cr.question!r} has no criterion {cr.criterion!r}")
        if self.requests is not None and self.questions:
            for cr in self.requests.criteria_roles:
                if cr.question not in by_id or cr.criterion not in by_id[cr.question].establishing_criteria:
                    raise ValueError(f"requests criteria_roles: {cr.question!r} has no criterion {cr.criterion!r}")
            for q in self.requests.conflict_roles:
                if q not in by_id:
                    raise ValueError(f"requests conflict_roles names unknown question {q!r}")
        if self.counterfactuals is not None and self.questions:
            for fr in self.counterfactuals.framings:
                if fr.question not in by_id:
                    raise ValueError(f"counterfactuals framings name unknown question {fr.question!r}")
        c = self.contradictions
        for r in (*c.date_rules, *c.order_rules, *c.amount_rules):
            if self.questions and r.question not in ids:
                raise ValueError(f"contradiction rule {r.id} names unknown question {r.question!r}")
        return self


FACT_PLACEHOLDERS = frozenset({"value", "normalised", "sentence", "label", "description", "amount", "currency"})


class FactTemplate(Contract):
    """How one extracted field becomes a Fact: the proposition wording and which open questions it bears on.
    The DIRECTION (supports / contradicts) is not decided here - that is the fact-question linker (T4.2)."""

    proposition: str                              # uses {value} {normalised} {sentence} {label} {description} {amount} {currency}
    questions: list[str] = []
    split_sentences: bool = False                 # narrative fields: one Fact per sentence (each with its own verbatim evidence)

    @model_validator(mode="after")
    def _placeholders(self) -> "FactTemplate":
        names = {f for _, f, _, _ in string.Formatter().parse(self.proposition) if f}
        unknown = names - FACT_PLACEHOLDERS
        if unknown:
            raise ValueError(f"unknown placeholder(s) {sorted(unknown)} in '{self.proposition}'; allowed: {sorted(FACT_PLACEHOLDERS)}")
        if self.split_sentences and "sentence" not in names:
            raise ValueError("a split_sentences template must use {sentence}")
        if not self.split_sentences and "sentence" in names:
            raise ValueError("{sentence} is only for split_sentences templates")
        return self


class FactTemplates(Contract):
    """Keys are '<document type>.<field>', plus '<type>.line_item' and '<type>.total_amount' for table documents."""

    templates: dict[str, FactTemplate]


def check_fact_templates(dt: "DocumentTypes", qt: QuestionTemplates, ft: FactTemplates) -> list[str]:
    """Cross-checks that make a pack fail at startup instead of silently producing no facts for a field."""
    problems: list[str] = []
    expected: set[str] = set()
    for t, spec in dt.types.items():
        expected |= {f"{t}.{name}" for name in spec.fields}
        if spec.table:
            expected |= {f"{t}.line_item", f"{t}.total_amount"}
    for key in sorted(set(ft.templates) - expected):
        problems.append(f"fact_templates.yaml: '{key}' does not match any extractable field or table item")
    for key in sorted(expected - set(ft.templates)):
        problems.append(f"fact_templates.yaml: no template for '{key}' (every extractable field must say how it becomes a fact)")
    known = {q.id for q in qt.questions}
    for key, tpl in ft.templates.items():
        for q in tpl.questions:
            if q not in known:
                problems.append(f"fact_templates.yaml: '{key}' tags unknown question '{q}' (see question_templates.yaml)")
    return problems


class CalcStep(Contract):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    id: str
    op: Literal["set", "subtract", "cap", "floor"]
    from_: str | None = Field(default=None, alias="from")
    value: float | int | str | None = None
    optional: bool = False

    @model_validator(mode="after")
    def _shape(self) -> "CalcStep":
        if self.op == "floor" and self.value is None:
            raise ValueError(f"step '{self.id}': floor needs 'value'")
        if self.op != "floor" and not self.from_:
            raise ValueError(f"step '{self.id}': {self.op} needs 'from'")
        return self


# Names a calculation rule may read: the typed inputs of the deterministic engine (core.schemas.SettlementInputs).
SETTLEMENT_INPUTS: frozenset[str] = frozenset(SettlementInputs.model_fields) - {"unavailable_inputs", "input_citations"}


class CalculationRules(Contract):
    """Ordered settlement steps. Structural invariants enforced here (so a bad pack fails at startup, not at payout time):
    first step is the only `set`; each input is used by at most one step (no repeated deduction); the last step is a floor at 0
    (a payable is never negative); a non-optional cap on remaining_policy_limit exists (a payable never exceeds the remaining limit)."""

    version: str
    currency: str
    rounding: dict[str, Any]
    steps: list[CalcStep] = Field(min_length=1)
    forecast_upper_sources: list[str] = []      # patterns (a named group `value`) of facts that state an amount claimed or estimated: the upper end of the forecast range while the repair cost is not validated

    @model_validator(mode="after")
    def _checks(self) -> "CalculationRules":
        for p in self.forecast_upper_sources:
            try:
                re.compile(p)
            except re.error as e:
                raise ValueError(f"bad pattern in forecast_upper_sources: {p!r} ({e})") from None
        ids = [s.id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate step ids")
        places = self.rounding.get("places")
        if self.rounding.get("mode") != "half_up" or not isinstance(places, int) or isinstance(places, bool) or not 0 <= places <= 6:
            raise ValueError("rounding must be {mode: half_up, places: <int 0-6>}")
        if self.steps[0].op != "set" or any(s.op == "set" for s in self.steps[1:]):
            raise ValueError("the first step must be 'set' and no other step may be 'set'")
        if self.steps[0].optional:
            raise ValueError("the 'set' step cannot be optional")
        refs = [s.from_ for s in self.steps if s.from_]
        unknown = sorted(set(refs) - SETTLEMENT_INPUTS)
        if unknown:
            raise ValueError(f"unknown input name(s) {unknown}; allowed: {sorted(SETTLEMENT_INPUTS)}")
        repeated = sorted({r for r in refs if refs.count(r) > 1})
        if repeated:
            raise ValueError(f"input(s) {repeated} used by more than one step (would deduct or cap twice)")
        floors = [s for s in self.steps if s.op == "floor"]
        if self.steps[-1].op != "floor" or any(Decimal(str(f.value)) != 0 for f in floors):
            raise ValueError("the last step must be a floor at 0 so a payable can never be negative")
        if not any(s.op == "cap" and s.from_ == "remaining_policy_limit" and not s.optional for s in self.steps):
            raise ValueError("rules must cap at remaining_policy_limit (non-optional) so a payable never exceeds it")
        return self


class ClauseTypeRule(Contract):
    """Keywords (regular expressions, case-insensitive) that decide a clause's type, by where the evidence is found."""

    type: ClauseType
    heading_keywords: list[str] = []         # the clause's own heading, e.g. 'Standard deductible'
    section_keywords: list[str] = []         # the section it sits in, e.g. 'WHAT IS NOT COVERED'
    title_keywords: list[str] = []           # the document title, e.g. 'Unoccupied Property Condition Endorsement'
    text_cues: list[str] = []                # the clause text, e.g. '^\S.{0,60} means\b' - last resort


class RelationRule(Contract):
    """Which clause types a clause can act on, and how the target is found (T3.2)."""

    source_kind: Literal["wording", "endorsement"]
    source_types: list[ClauseType]
    relation: RelationType
    target_types: list[ClauseType]
    # overlap: shared words; term: the defined term appears; exception: 'However ...' after an exclusion
    method: Literal["overlap", "term", "exception"]


class RelationConfig(Contract):
    """How clauses are linked and how precedence is read. Words and cues are SAMPLE conventions, extensible per pack."""

    stopwords: list[str]
    heading_weight: int = Field(default=2, ge=1)
    text_weight: int = Field(default=1, ge=1)
    # a target needs at least this many shared words, else no relation (never a guess)
    min_score: int = Field(default=1, ge=1)
    explicit_reference_patterns: list[str]                   # need (?P<id>...): 'Endorsement END-01'
    exception_cues: list[str]                                # an exclusion that starts like this is an exception to the exclusion before it
    precedence_clause_cues: list[str]                        # a wording clause that states how conflicting endorsements are resolved
    later_prevails_cues: list[str]                           # ... and says the later endorsement wins
    prevails_over_patterns: list[str]                        # an endorsement stating it prevails over another; need (?P<id>...)
    effective_date_cues: list[str]                           # a date an endorsement states for itself; need (?P<date>D Month YYYY)
    rules: list[RelationRule]

    @model_validator(mode="after")
    def _valid(self) -> "RelationConfig":
        for where, needs in (("explicit_reference_patterns", "id"), ("prevails_over_patterns", "id"), ("effective_date_cues", "date")):
            for p in getattr(self, where):
                try:
                    rx = re.compile(p, re.IGNORECASE)
                except re.error as e:
                    raise ValueError(f"bad pattern in relations.{where}: '{p}' ({e})") from None
                if needs not in rx.groupindex:
                    raise ValueError(f"relations.{where}: '{p}' needs a (?P<{needs}>...) group")
        for where in ("exception_cues", "precedence_clause_cues", "later_prevails_cues"):
            for p in getattr(self, where):
                try:
                    re.compile(p, re.IGNORECASE)
                except re.error as e:
                    raise ValueError(f"bad pattern in relations.{where}: '{p}' ({e})") from None
        if not self.rules:
            raise ValueError("relations.rules must not be empty")
        for r in self.rules:
            if r.method == "term" and r.relation != "defines":
                raise ValueError("the 'term' method can only produce 'defines'")
            if r.method == "exception" and r.relation != "excepts":
                raise ValueError("the 'exception' method can only produce 'excepts'")
        return self


class RetrievalConfig(Contract):
    """Hybrid clause retrieval (T3.4): keyword (BM25) + embeddings, fused by reciprocal rank. All numbers are SAMPLE starting points."""

    stopwords: list[str]                                     # function words only: 'cover', 'damage', 'pay' matter when searching a policy
    k: int = Field(default=5, ge=1)                          # ranked results returned
    bm25_k1: float = Field(default=1.2, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    heading_weight: int = Field(default=2, ge=1)             # a clause heading counts this many times in its keyword document
    rrf_k: int = Field(default=60, ge=1)                     # reciprocal rank fusion constant
    keyword_weight: float = Field(default=1.0, ge=0)
    dense_weight: float = Field(default=1.0, ge=0)
    # below this cosine similarity an embedding match is ignored (no weak guesses)
    min_dense_similarity: float = Field(default=0.2, ge=0, le=1)

    @model_validator(mode="after")
    def _valid(self) -> "RetrievalConfig":
        if self.keyword_weight == 0 and self.dense_weight == 0:
            raise ValueError("keyword_weight and dense_weight cannot both be 0")
        return self


class LLMTypingConfig(Contract):
    """Optional model tie-breaker for clauses no rule could type (T3.1 -> 4a). Checked, never trusted: values are SAMPLE."""

    enabled: bool = False                            # off unless the pack turns it on (this pack does: see policy_structure.yaml)
    min_confidence: float = Field(ge=0, le=1)        # a verdict below this stays unclassified
    confidence_cap: float = Field(ge=0, le=1)        # a model answer is never reported as more certain than this
    max_clause_chars: int = Field(default=1500, ge=100)
    # None = the client's default; typing a clause may need more thought than transcribing a page
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    # A clause typed with one of these must state an amount or point to one (a payout rule says HOW MUCH):
    # a cheap check the model cannot talk its way past.
    types_needing_amount: list[ClauseType] = []
    amount_cues: list[str] = []

    @model_validator(mode="after")
    def _valid(self) -> "LLMTypingConfig":
        if self.min_confidence > self.confidence_cap:
            raise ValueError("min_confidence cannot be above confidence_cap: no answer could ever be accepted")
        if self.types_needing_amount and not self.amount_cues:
            raise ValueError("types_needing_amount needs at least one amount cue")
        for p in self.amount_cues:
            try:
                re.compile(p, re.IGNORECASE)
            except re.error as e:
                raise ValueError(f"bad pattern in llm_typing.amount_cues: '{p}' ({e})") from None
        return self


class PolicyStructure(Contract):
    """How a policy document is cut into clauses (T3.1). Patterns are conventions of the wording style, SAMPLE and extensible per pack.

    A clause heading pattern needs the named groups `number` and `title`; a section pattern needs `title`.
    Type rules are tried in order; the evidence sources in this order: heading, section, document title, text. No match = the clause
    is NOT guessed: it is returned as unclassified for a human."""

    noise_patterns: list[str]                # banner, footer and page-number lines that are not policy text
    section_patterns: list[str]
    clause_patterns: list[str]
    endorsement_title_patterns: list[str]
    max_heading_chars: int = Field(default=80, ge=10)
    types: list[ClauseTypeRule]
    relations: RelationConfig
    retrieval: RetrievalConfig
    llm_typing: LLMTypingConfig

    @model_validator(mode="after")
    def _valid(self) -> "PolicyStructure":
        def compiled(p: str, where: str) -> re.Pattern[str]:
            try:
                return re.compile(p, 0 if where in ("section_patterns", "clause_patterns") else re.IGNORECASE)
            except re.error as e:
                raise ValueError(f"bad pattern in {where}: '{p}' ({e})") from None

        for where, pats in (("noise_patterns", self.noise_patterns), ("endorsement_title_patterns", self.endorsement_title_patterns)):
            for p in pats:
                compiled(p, where)
        for p in self.section_patterns:
            if "title" not in compiled(p, "section_patterns").groupindex:
                raise ValueError(f"section pattern '{p}' needs a (?P<title>...) group")
        for p in self.clause_patterns:
            if not {"number", "title"} <= set(compiled(p, "clause_patterns").groupindex):
                raise ValueError(f"clause pattern '{p}' needs (?P<number>...) and (?P<title>...) groups")
        if not self.clause_patterns:
            raise ValueError("at least one clause pattern is required")
        seen: set[str] = set()
        for r in self.types:
            if r.type in seen:
                raise ValueError(f"clause type '{r.type}' has more than one rule")
            seen.add(r.type)
            for where in ("heading_keywords", "section_keywords", "title_keywords", "text_cues"):
                for p in getattr(r, where):
                    compiled(p, where)
        return self


IssueKind = Literal["condition", "cover", "exclusion", "amount"]
CONTROL_STATES = ("settled", "contested", "open", "partial", "missing")
READINGS = ("favourable", "adverse", "unclear", "ambiguous", "not_judged")
ROW_CONDITIONS = ("kind", "control", "reading", "not_in_force", "no_citation")


class IssueSpec(Contract):
    """One row of the coverage issue matrix: what the issue is, which open questions control it, and which policy clauses it cites."""

    id: str
    issue: str
    kind: IssueKind                                 # condition (policy in force) | cover (an insuring clause) | exclusion | amount (a limit, sublimit, deductible)
    questions: list[str] = Field(min_length=1)      # the open questions whose answers decide it; a row exists only if they are all in the register
    clauses: list[ClauseLink] = Field(min_length=1)


class IssueRow(Contract):
    """One row of the status decision table. Every listed condition must hold (a missing one is ignored); the first matching row decides."""

    id: str
    when: dict[str, Any] = {}
    status: IssueStatus
    note: str = ""
    reason: str = ""                                # T5.2: why the system does not conclude on this row (a code; empty when it does)

    @model_validator(mode="after")
    def _valid(self) -> "IssueRow":
        for k, v in self.when.items():
            if k not in ROW_CONDITIONS:
                raise ValueError(f"row {self.id}: unknown condition {k!r}; known: {list(ROW_CONDITIONS)}")
            if k in ("not_in_force", "no_citation"):
                if not isinstance(v, bool):
                    raise ValueError(f"row {self.id}: {k} must be true or false")
                continue
            allowed = {"kind": get_args(IssueKind), "control": CONTROL_STATES, "reading": READINGS}[k]
            vals = v if isinstance(v, list) else [v]
            if not vals or any(x not in allowed for x in vals):
                raise ValueError(f"row {self.id}: {k} must be one or more of {list(allowed)}")
        return self


class ReadingConfig(Contract):
    """The model's reading of whether the facts favour the claim as presented under a clause (cover condition met, exclusion not engaged). Checked,
    never trusted: it must cite offered facts and clauses and quote a fact verbatim; anything else is `unclear` and the row abstains. Values are SAMPLE."""

    enabled: bool = False                           # without it a settled row is only `identified`: the clauses are cited, nothing is concluded
    min_confidence: float = Field(default=0.6, ge=0, le=1)
    confidence_cap: float = Field(default=0.9, ge=0, le=1)
    max_rationale_chars: int = Field(default=300, ge=40)
    max_clause_chars: int = Field(default=800, ge=100)
    max_fact_chars: int = Field(default=300, ge=50)
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    check_ambiguity: bool = True                    # T5.2: first ask whether the clauses admit two reasonable readings that point opposite ways (then: abstain, show both)
    # T5.2 rule: the ambiguity question is asked only where a clause of one of these types has wording that makes it an exception (a rule, not a model call)
    ambiguity_trigger_types: list[str] = ["exclusion"]
    ambiguity_trigger_patterns: list[str] = [r"\bhowever\b", r"\bexcept\b", r"\bunless\b", r"\bprovided that\b"]
    verify_supports: bool = True                    # a second, independent model question: does the quote itself state how or when the water escaped?

    @model_validator(mode="after")
    def _valid(self) -> "ReadingConfig":
        if self.min_confidence > self.confidence_cap:
            raise ValueError("min_confidence cannot be above confidence_cap: no reading could ever be accepted")
        return self


class CoverageConfig(Contract):
    """The coverage issue matrix (T5.1): rows, how each row's status follows from its questions, and the wording rules. No row ever says a claim is covered,
    excluded, approved or denied: it says how well the issue is resolved on the evidence, and a human decides."""

    issues: list[IssueSpec] = Field(min_length=1)
    table: list[IssueRow] = Field(min_length=1)
    reading: ReadingConfig = ReadingConfig()
    forbidden_patterns: list[str] = []              # wording that states a coverage conclusion

    @model_validator(mode="after")
    def _valid(self) -> "CoverageConfig":
        ids = [i.id for i in self.issues]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate coverage issue ids")
        rows = [r.id for r in self.table]
        if len(rows) != len(set(rows)):
            raise ValueError("duplicate status-table row ids")
        if self.table[-1].when:
            raise ValueError("the last row of the status table must have no conditions, so that every case is decided")
        for p in self.forbidden_patterns:
            try:
                re.compile(p, re.IGNORECASE)
            except re.error as e:
                raise ValueError(f"bad forbidden pattern {p!r}: {e}") from e
        return self


class Pack(Contract):
    meta: PackMeta
    document_types: DocumentTypes
    intake_checklist: IntakeChecklistCfg
    question_templates: QuestionTemplates
    fact_templates: FactTemplates
    policy_structure: PolicyStructure
    coverage: CoverageConfig | None = None
    calculation: dict[str, dict[str, Any]]     # file stem -> raw rules dict; settlement.rules_loader turns it into engines
    root: Path
    prompts_dir: Path


class Settings(Contract):
    root: Path
    app: AppConfig
    autonomy: AutonomyMatrix
    routing: RoutingRules
    rbac: RBAC
    pack: Pack


# ----------------------------------------------------------------- loading
def find_root() -> Path:
    env = os.environ.get("CLAIMTOWER_ROOT")
    return Path(env) if env else Path(__file__).resolve().parents[3]


def _read(path: Path, model: type[Contract], problems: list[str]):
    rel = f"{path.parent.name}/{path.name}"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        problems.append(f"{rel}: file not found")
        return None
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        where = f" (line {mark.line + 1}, col {mark.column + 1})" if mark else ""
        problems.append(f"{rel}: invalid YAML{where}: {getattr(e, 'problem', e)}")
        return None
    try:
        return model.model_validate(data if data is not None else {})
    except ValidationError as e:
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"]) or "<root>"
            problems.append(f"{rel}: {loc}: {err['msg']}")
        return None


def _env_flag(name: str) -> bool | None:
    v = os.environ.get(name)
    return None if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


def load_pack(name: str, root: Path | None = None, problems: list[str] | None = None) -> Pack | None:
    root = root or find_root()
    own = problems is None
    problems = [] if problems is None else problems
    pdir = root / "packs" / name
    if not pdir.is_dir():
        problems.append(f"packs/{name}: pack folder not found")
    else:
        meta = _read(pdir / "pack.yaml", PackMeta, problems)
        if meta and meta.name != name:
            problems.append(f"packs/{name}/pack.yaml: name '{meta.name}' does not match folder")
        dt = _read(pdir / "document_types.yaml", DocumentTypes, problems)
        ic = _read(pdir / "intake_checklist.yaml", IntakeChecklistCfg, problems)
        qt = _read(pdir / "question_templates.yaml", QuestionTemplates, problems)
        ft = _read(pdir / "fact_templates.yaml", FactTemplates, problems)
        ps = _read(pdir / "policy_structure.yaml", PolicyStructure, problems)
        cov = _read(pdir / "coverage_issues.yaml", CoverageConfig, problems) if (pdir / "coverage_issues.yaml").is_file() else None
        if cov and qt:
            known = {q.id for q in qt.questions}
            for i in cov.issues:
                problems += [f"coverage_issues.yaml: issue {i.id} names unknown question '{q}' (see question_templates.yaml)" for q in i.questions if q not in known]
        if dt and qt and ft:
            problems.extend(check_fact_templates(dt, qt, ft))
        calc: dict[str, dict[str, Any]] = {}
        for f in sorted((pdir / "calculation").glob("*.yaml")):
            if _read(f, CalculationRules, problems) is not None:
                calc[f.stem] = yaml.safe_load(f.read_text(encoding="utf-8"))
        if not calc and not any(p.startswith(f"packs/{name}/calculation") for p in problems):
            problems.append(f"packs/{name}/calculation: no calculation rule files found")
        if not problems and meta and dt and ic and qt and ft and ps:
            return Pack(meta=meta, document_types=dt, intake_checklist=ic, question_templates=qt, fact_templates=ft,
                        policy_structure=ps, coverage=cov, calculation=calc, root=pdir, prompts_dir=pdir / "prompts")
    if own and problems:
        raise ConfigError("Invalid pack:\n  - " + "\n  - ".join(problems))
    return None


def available_packs(root: Path | None = None) -> list[str]:
    base = (root or find_root()) / "packs"
    return sorted(p.name for p in base.iterdir() if (p / "pack.yaml").is_file()) if base.is_dir() else []


def load_settings(root: Path | None = None) -> Settings:
    """Load + validate everything. Raises ConfigError listing ALL problems found."""
    root = Path(root) if root else find_root()
    problems: list[str] = []
    cdir = root / "configs"
    app = _read(cdir / "app.yaml", AppConfig, problems)
    autonomy = _read(cdir / "autonomy_matrix.yaml", AutonomyMatrix, problems)
    routing = _read(cdir / "routing_rules.yaml", RoutingRules, problems)
    rbac = _read(cdir / "rbac.yaml", RBAC, problems)

    pack = None
    if app:
        flags = app.flags.model_copy(update={k: v for k, v in {
            "use_mock_pas": _env_flag("USE_MOCK_PAS"), "use_mock_claims": _env_flag("USE_MOCK_CLAIMS"),
            "use_mock_recon": _env_flag("USE_MOCK_RECON"), "llm_replay_mode": _env_flag("LLM_REPLAY_MODE"),
        }.items() if v is not None})
        app = app.model_copy(update={"flags": flags})
        pack = load_pack(app.active_pack, root, problems)
    if routing and rbac:
        unknown = set(routing.approver_roles.values()) - set(rbac.roles)
        if unknown:
            problems.append(f"routing_rules.yaml: approver_roles use unknown roles {sorted(unknown)} (not in rbac.yaml)")
    if autonomy and rbac:
        known = set(rbac.roles)
        for op, cfg in autonomy.operations.items():
            unknown = set(cfg.roles) - known
            if unknown:
                problems.append(f"autonomy_matrix.yaml: {op}: unknown roles {sorted(unknown)} (not in rbac.yaml)")
    if problems:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(problems))
    return Settings(root=root, app=app, autonomy=autonomy, routing=routing, rbac=rbac, pack=pack)

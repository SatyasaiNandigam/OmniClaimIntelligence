import pytest
from claimtower.agents.base import StatePatch
from claimtower.core.schemas import ClaimState
from claimtower.core.state import OwnershipViolation, apply_patch


def test_foreign_field_rejected():
    with pytest.raises(OwnershipViolation):
        apply_patch(ClaimState(claim_id="c1"), StatePatch(agent_id="gap", updates={"settlement": None}),
                    owns=frozenset({"open_questions"}))

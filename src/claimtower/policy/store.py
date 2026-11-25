"""Relational store for decomposed policy clauses (T3.1: "stored in relational tables"). SQLite via the standard library.

Tables (snapshots are added by T3.3):
    policy_documents(doc_id PK, title, kind, page_count)
    policy_clauses(clause_id PK, doc_id FK, clause_type, heading, text, page, span_start, span_end, number, section, type_source,
                   type_confidence)      -- type_source 'llm' = typed by the optional model, with its (capped) confidence
    policy_unclassified(doc_id FK, page, number, heading, section, text, reason, clause_id, span_start, span_end)
    policy_relations(pool, source_clause_id, target_clause_id, relation, basis, evidence)   -- T3.2
        (pool = one policy: its wording plus the endorsements attached to it)
    policy_precedence(pool PK, kind, statement, clause_id)

Saving a document replaces everything stored for it, in one transaction: re-ingesting a document never leaves stale clauses behind.
`clauses(...)` returns contract objects (PolicyClause); `ClauseInfo` (number, section, how the type was decided) comes back with them.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

from claimtower.core.schemas.policy import ClauseRelation, PolicyClause
from claimtower.policy.decompose import ClauseInfo, Decomposition, Unclassified
from claimtower.policy.relations import PolicyGraph, PrecedenceRule, Relation

SCHEMA = """
CREATE TABLE IF NOT EXISTS policy_documents (
    doc_id TEXT PRIMARY KEY, title TEXT, kind TEXT NOT NULL, page_count INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS policy_clauses (
    clause_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES policy_documents(doc_id) ON DELETE CASCADE,
    clause_type TEXT NOT NULL, heading TEXT, text TEXT NOT NULL, page INTEGER NOT NULL, span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL, number TEXT NOT NULL, section TEXT, type_source TEXT NOT NULL, type_confidence REAL);
CREATE INDEX IF NOT EXISTS ix_policy_clauses_doc ON policy_clauses(doc_id, page, span_start);
CREATE TABLE IF NOT EXISTS policy_unclassified (
    doc_id TEXT NOT NULL REFERENCES policy_documents(doc_id) ON DELETE CASCADE, page INTEGER NOT NULL, number TEXT NOT NULL,
    heading TEXT NOT NULL, section TEXT, text TEXT NOT NULL, reason TEXT NOT NULL, clause_id TEXT NOT NULL DEFAULT '',
    span_start INTEGER NOT NULL DEFAULT 0, span_end INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS policy_relations (
    pool TEXT NOT NULL, source_clause_id TEXT NOT NULL, target_clause_id TEXT NOT NULL, relation TEXT NOT NULL,
    basis TEXT NOT NULL, evidence TEXT NOT NULL, PRIMARY KEY (pool, source_clause_id, target_clause_id, relation));
CREATE INDEX IF NOT EXISTS ix_policy_relations_target ON policy_relations(pool, target_clause_id);
CREATE TABLE IF NOT EXISTS policy_precedence (pool TEXT PRIMARY KEY, kind TEXT NOT NULL, statement TEXT NOT NULL, clause_id TEXT);
"""


class PolicyStore:
    def __init__(self, path: Path | str = ":memory:") -> None:
        self._db = sqlite3.connect(str(path))
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        self._db.close()

    def save(self, d: Decomposition, page_count: int) -> None:
        with self._db:                                                                  # one transaction
            self._db.execute("DELETE FROM policy_documents WHERE doc_id = ?", (d.doc_id,))        # cascades to clauses
            self._db.execute("INSERT INTO policy_documents VALUES (?, ?, ?, ?)", (d.doc_id, d.title, d.kind, page_count))
            for c in d.clauses:
                info = d.info[c.clause_id]
                assert c.page is not None and c.span is not None
                self._db.execute("INSERT INTO policy_clauses VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                                 (c.clause_id, d.doc_id, c.clause_type, c.heading, c.text, c.page, c.span[0], c.span[1],
                                  info.number, info.section, info.type_source, info.confidence))
            for u in d.unclassified:
                self._db.execute("INSERT INTO policy_unclassified VALUES (?,?,?,?,?,?,?,?,?,?)",
                                 (d.doc_id, u.page, u.number, u.heading, u.section, u.text, u.reason, u.clause_id, u.span[0], u.span[1]))

    def save_graph(self, graph: PolicyGraph) -> None:
        """Replace the stored relations and precedence rule of one policy (pool)."""
        with self._db:
            self._db.execute("DELETE FROM policy_relations WHERE pool = ?", (graph.pool,))
            self._db.execute("DELETE FROM policy_precedence WHERE pool = ?", (graph.pool,))
            for r in graph.relations:
                self._db.execute("INSERT INTO policy_relations VALUES (?,?,?,?,?,?)",
                                 (graph.pool, r.source, r.target, r.relation, r.basis, r.evidence))
            p = graph.precedence
            self._db.execute("INSERT INTO policy_precedence VALUES (?,?,?,?)", (graph.pool, p.kind, p.statement, p.clause_id))

    def relations(self, pool: str, clause_id: str | None = None) -> list[Relation]:
        sql, args = "SELECT * FROM policy_relations WHERE pool = ?", [pool]
        if clause_id:
            sql, args = sql + " AND (source_clause_id = ? OR target_clause_id = ?)", [*args, clause_id, clause_id]
        return [Relation(r["source_clause_id"], r["target_clause_id"], r["relation"], r["basis"], r["evidence"])
                for r in self._db.execute(sql + " ORDER BY source_clause_id, target_clause_id, relation", args)]

    def relation_contracts(self, pool: str) -> list[ClauseRelation]:
        return [r.as_contract() for r in self.relations(pool)]

    def precedence(self, pool: str) -> PrecedenceRule | None:
        r = self._db.execute("SELECT * FROM policy_precedence WHERE pool = ?", (pool,)).fetchone()
        return PrecedenceRule(r["kind"], r["statement"], r["clause_id"]) if r else None

    def documents(self) -> list[dict]:
        return [dict(r) for r in self._db.execute("SELECT * FROM policy_documents ORDER BY doc_id")]

    def clauses(self, doc_id: str | None = None, clause_type: str | None = None) -> list[PolicyClause]:
        sql, args = "SELECT * FROM policy_clauses WHERE 1=1", []
        if doc_id:
            sql, args = sql + " AND doc_id = ?", [*args, doc_id]
        if clause_type:
            sql, args = sql + " AND clause_type = ?", [*args, clause_type]
        return [self._clause(r) for r in self._db.execute(sql + " ORDER BY doc_id, page, span_start", args)]

    def clause(self, clause_id: str) -> PolicyClause | None:
        r = self._db.execute("SELECT * FROM policy_clauses WHERE clause_id = ?", (clause_id,)).fetchone()
        return self._clause(r) if r else None

    def info(self, clause_id: str) -> ClauseInfo | None:
        r = self._db.execute("SELECT number, section, type_source, type_confidence FROM policy_clauses WHERE clause_id = ?",
                             (clause_id,)).fetchone()
        return ClauseInfo(r["number"], r["section"], r["type_source"], r["type_confidence"]) if r else None

    def unclassified(self, doc_id: str | None = None) -> Sequence[Unclassified]:
        rows = self._db.execute("SELECT * FROM policy_unclassified" + (" WHERE doc_id = ?" if doc_id else "") + " ORDER BY doc_id, page",
                                (doc_id,) if doc_id else ())
        return [Unclassified(r["doc_id"], r["page"], r["number"], r["heading"], r["section"], r["text"], r["reason"], r["clause_id"],
                             (r["span_start"], r["span_end"])) for r in rows]

    @staticmethod
    def _clause(r: sqlite3.Row) -> PolicyClause:
        return PolicyClause(clause_id=r["clause_id"], clause_type=r["clause_type"], heading=r["heading"], text=r["text"],
                            document=r["doc_id"], page=r["page"], span=(r["span_start"], r["span_end"]))

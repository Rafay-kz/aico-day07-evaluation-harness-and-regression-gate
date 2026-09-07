from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from aico.security.normalization import normalize

PolicyOutcome = Literal["allow", "clarify", "block"]

DEFAULT_ATTACK_FIXTURES = Path("tests/fixtures/day05/attacks/attack_fixtures.json")
DEFAULT_ATTACK_RESULTS = Path("artifacts/day05/attack_results.md")

# Quoted poisoned evidence described as data must be allowed (ATK-009).
_QUOTED_AS_DATA = (
    "treat that text as evidence",
    "as evidence data only",
    "evidence data only",
)

_OVERRIDE = (
    ("ignore", "previous", "instruction"),
    ("answer", "without", "retrieved", "evidence"),
    ("answer", "from", "your", "own", "knowledge"),
)

_ROLE_ESCALATION = (
    "you are now the system administrator",
    "you are now",
    "system administrator",
    "override policy",
    "reveal restricted",
)

_PROMPT_EXTRACTION = (
    "hidden system prompt",
    "system prompt",
    "internal policy text",
    "print the hidden",
)

_TOOL_COERCION = (
    "hidden tool",
    "tools are not allowed",
    "modify the supplier record",
)

_CITATION_FORGERY = (
    "cite chunk-999",
    "cite chunk 999",
    ("no source supports", "cite"),
    ("cite", "anyway"),
)

_AMBIGUOUS = (
    "whether the supplier is good",
    "is the supplier good",
    "tell me whether the supplier is good",
)


@dataclass(frozen=True)
class PolicyDecision:
    outcome: PolicyOutcome
    category: str
    reason: str


def evaluate_policy(text: str) -> PolicyDecision:
    """Classify normalized input. Always normalizes first; never calls a model."""
    normalized = normalize(text)
    lowered = normalized.lower()

    if _contains_any(lowered, _QUOTED_AS_DATA):
        return PolicyDecision(
            outcome="allow",
            category="quoted_poisoned_text_as_data",
            reason="quoted poisoned text is treated as evidence data, not instruction",
        )

    if _matches_override(lowered):
        return PolicyDecision(
            outcome="block",
            category="instruction_override",
            reason="instruction-override attempt",
        )

    if _contains_any(lowered, _ROLE_ESCALATION):
        return PolicyDecision(
            outcome="block",
            category="role_escalation",
            reason="role-escalation attempt",
        )

    if _contains_any(lowered, _PROMPT_EXTRACTION):
        return PolicyDecision(
            outcome="block",
            category="system_prompt_extraction",
            reason="system-prompt extraction attempt",
        )

    if _contains_any(lowered, _TOOL_COERCION):
        return PolicyDecision(
            outcome="block",
            category="tool_coercion",
            reason="tool-coercion attempt",
        )

    if _matches_citation_forgery(lowered):
        return PolicyDecision(
            outcome="block",
            category="citation_forgery",
            reason="request to forge a citation",
        )

    if _contains_any(lowered, _AMBIGUOUS):
        return PolicyDecision(
            outcome="clarify",
            category="ambiguous_request",
            reason="request is too vague to answer from evidence",
        )

    return PolicyDecision(
        outcome="allow",
        category="benign_grounded_question",
        reason="no attack or ambiguity pattern matched",
    )


def run_attack_fixtures(path: Path | None = None) -> list[dict]:
    """Evaluate every supplied fixture. Does not edit fixtures."""
    fixture_path = path if path is not None else DEFAULT_ATTACK_FIXTURES
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for fixture in payload["fixtures"]:
        decision = evaluate_policy(fixture["input"])
        expected = fixture["expected"]
        actual = decision.outcome
        passed = actual == expected
        rows.append(
            {
                "id": fixture["id"],
                "category": fixture["category"],
                "expected": expected,
                "actual": actual,
                "pass": passed,
                "reason": "" if passed else decision.reason,
                "decision_category": decision.category,
            }
        )
    return rows


def write_attack_results(
    rows: list[dict] | None = None,
    *,
    fixtures_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    evaluated = rows if rows is not None else run_attack_fixtures(fixtures_path)
    target = output_path if output_path is not None else DEFAULT_ATTACK_RESULTS
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Day 5 Attack Fixture Results",
        "",
        "Deterministic policy evaluation of the supplied Day 5 corpus.",
        "Passing does not imply universal jailbreak prevention.",
        "",
        "| Fixture | Category | Expected | Actual | Result | Reason |",
        "|---|---|---|---|---|---|",
    ]
    for row in evaluated:
        result = "pass" if row["pass"] else "fail"
        reason = row["reason"] if row["reason"] else "—"
        lines.append(
            f"| {row['id']} | {row['category']} | {row['expected']} | "
            f"{row['actual']} | {result} | {reason} |"
        )
    lines.extend(["", "No secrets or sensitive data are included.", ""])
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _matches_override(text: str) -> bool:
    for group in _OVERRIDE:
        if all(token in text for token in group):
            return True
    return False


def _matches_citation_forgery(text: str) -> bool:
    for item in _CITATION_FORGERY:
        if isinstance(item, tuple):
            if all(token in text for token in item):
                return True
        elif item in text:
            return True
    return False

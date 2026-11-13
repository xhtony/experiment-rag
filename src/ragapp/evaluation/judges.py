"""LLM-as-judge primitives used by the unified evaluator.

Two-step faithfulness protocol (RAGAS-style):
1. Decompose the answer into atomic factual claims.
2. Give a verdict per claim against the reference context:
   supported / partially_supported / unsupported / contradicted.
"""

from typing import List, Literal

from pydantic import BaseModel, Field

from ragapp.shared.llm import get_judge_llm

# --- claim decomposition ------------------------------------------------------


class _Claims(BaseModel):
    claims: List[str] = Field(
        description="Atomic, self-contained factual claims made in the answer."
    )


_DECOMPOSE_SYSTEM = (
    "Decompose the given answer into a list of atomic factual claims. Each claim "
    "must be a single, self-contained statement that can be verified independently. "
    "Ignore hedging, meta-commentary, and refusal statements."
)


def decompose_claims(answer_text: str) -> list[str]:
    if not answer_text.strip():
        return []
    llm = get_judge_llm().with_structured_output(_Claims)
    return llm.invoke(
        [("system", _DECOMPOSE_SYSTEM), ("human", f"Answer:\n{answer_text}")]
    ).claims


# --- claim verification -------------------------------------------------------

Verdict = Literal["supported", "partially_supported", "unsupported", "contradicted"]


class ClaimVerdict(BaseModel):
    claim: str
    verdict: Verdict
    evidence_chunk_id: str = Field(
        default="", description="chunk_id of the passage that supports/contradicts the claim, if any."
    )


class _ClaimVerdicts(BaseModel):
    verdicts: List[ClaimVerdict]


_VERIFY_SYSTEM = (
    "You verify factual claims against reference passages from university policy "
    "documents. For each claim, decide:\n"
    "- supported: the claim follows directly from the passages\n"
    "- partially_supported: part of the claim is in the passages, part is not\n"
    "- unsupported: the passages say nothing that confirms the claim\n"
    "- contradicted: the passages state the opposite\n"
    "Judge ONLY against the passages, not your own knowledge."
)


def verify_claims(claims: list[str], context: str) -> list[ClaimVerdict]:
    if not claims:
        return []
    llm = get_judge_llm().with_structured_output(_ClaimVerdicts)
    claim_list = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    result = llm.invoke(
        [
            ("system", _VERIFY_SYSTEM),
            ("human", f"Reference passages:\n\n{context}\n\nClaims:\n{claim_list}"),
        ]
    )
    return result.verdicts


VERDICT_WEIGHT = {
    "supported": 1.0,
    "partially_supported": 0.5,
    "unsupported": 0.0,
    "contradicted": 0.0,
}


# --- sentence-level citation support -------------------------------------------


class _SupportVerdict(BaseModel):
    supports: bool = Field(
        description="True if the passage supports the sentence's factual content."
    )


_CITATION_SYSTEM = (
    "Decide whether the given passage supports the factual content of the given "
    "sentence. Answer strictly based on the passage."
)


def citation_supports(sentence: str, passage: str) -> bool:
    llm = get_judge_llm().with_structured_output(_SupportVerdict)
    return llm.invoke(
        [
            ("system", _CITATION_SYSTEM),
            ("human", f"Passage:\n{passage}\n\nSentence:\n{sentence}"),
        ]
    ).supports

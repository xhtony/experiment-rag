"""C0 - No-RAG LLM baseline.

Graph:  generate -> END

Purpose: measures how much the bare LLM hallucinates on institution-specific
policy questions when it has no access to the corpus.
"""

from langgraph.graph import END, START, StateGraph

from ragapp.schemas import AnswerPayload, Condition, PlainAnswer
from ragapp.shared.llm import get_generator_llm
from ragapp.shared.prompts import C0_SYSTEM
from ragapp.pipelines.state import RAGState


def _generate(state: RAGState) -> dict:
    llm = get_generator_llm().with_structured_output(PlainAnswer)
    result: PlainAnswer = llm.invoke(
        [("system", C0_SYSTEM), ("human", state["question"])]
    )
    payload = AnswerPayload(
        condition=Condition.C0,
        question_id=state.get("question_id", -1),
        answer_text=result.answer if not result.refused else "",
        refused=result.refused,
        refusal_reason=result.refusal_reason,
    )
    return {"payload": payload}


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("generate", _generate)
    g.add_edge(START, "generate")
    g.add_edge("generate", END)
    return g.compile()

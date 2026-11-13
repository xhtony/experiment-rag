"""C1 - Standard vector RAG baseline.

Graph:  retrieve (dense, top-k) -> generate -> END

Single dense retrieval pass against Qdrant, standard grounded prompt with
sentence-level citations. This is the reference RAG condition.
"""

from langgraph.graph import END, START, StateGraph

from ragapp.schemas import Condition
from ragapp.pipelines.nodes import make_dense_retrieve_node, make_grounded_generate_node
from ragapp.pipelines.state import RAGState


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", make_dense_retrieve_node())
    g.add_node("generate", make_grounded_generate_node(Condition.C1))
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()

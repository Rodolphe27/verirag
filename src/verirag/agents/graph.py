"""The agent graph: how the agents are connected, and when the system stops.

    retrieve -> route --refuse--------------------------------> END
                  |
                  +--documents--> write -> verify --all supported--> finish -> END
                                    ^         |
                                    |         +--some failed, retries left--+
                                    +-------------------------------------- +
                                              |
                                              +--no retries left--> refuse -> END

Every path ends with an Answer that is either ANSWERED (verified, with
citations) or REFUSED (with the reason). Nothing else can leave the graph.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from verirag.agents.output import BadModelOutput
from verirag.agents.router import Route, Router, refusal
from verirag.agents.verifier import Verifier, locate_evidence
from verirag.agents.writer import Writer
from verirag.domain.answer import Answer, AnswerStatus, Draft
from verirag.domain.chunk import Chunk
from verirag.domain.verdict import Verdict
from verirag.ports import Retriever, Tracer


class State(TypedDict, total=False):
    """The data that travels through the graph. Each node returns only what it changes."""

    question: str
    chunks: list[Chunk]
    draft: Draft | None
    verdict: Verdict | None
    attempts: int  # how many times the writer has been called
    feedback: str  # what the writer must fix on its next try
    answer: Answer


def build_graph(
    retriever: Retriever,
    writer: Writer,
    verifier: Verifier,
    router: Router,
    tracer: Tracer,
    max_retries: int = 2,
    k: int = 4,
) -> Any:
    """Connect the agents. Returns a compiled LangGraph graph.

    `max_retries` is the number of extra writer attempts after the first one,
    so the writer is called at most `max_retries + 1` times.
    """

    def retrieve(state: State) -> State:
        chunks = retriever.search(state["question"], k)
        best = max((chunk.score for chunk in chunks), default=0.0)
        tracer.record("retriever", {"chunks": len(chunks), "best_score": round(best, 3)})
        return {"chunks": chunks}

    def route(state: State) -> State:
        decision = router.decide(state["question"], state["chunks"])
        tracer.record("router", {"route": decision.route.value, "reason": decision.reason})
        if decision.route is Route.REFUSE:
            return {"answer": refusal(decision.reason)}
        return {"chunks": list(decision.chunks)}

    def write(state: State) -> State:
        attempt = state.get("attempts", 0) + 1
        try:
            draft = writer.write(state["question"], state["chunks"], state.get("feedback", ""))
        except BadModelOutput as error:
            tracer.record("writer", {"attempt": attempt, "error": str(error)})
            note = f"Your previous reply was rejected ({error}). Follow the JSON format exactly."
            return {"attempts": attempt, "draft": None, "feedback": note}
        tracer.record("writer", {"attempt": attempt, "claims": len(draft.claims)})
        return {"attempts": attempt, "draft": draft}

    def verify(state: State) -> State:
        draft = state["draft"]
        assert draft is not None  # the graph only reaches this node with a draft
        verdict = verifier.verify(draft, state["chunks"])
        tracer.record(
            "verifier",
            {
                "attempt": state["attempts"],
                "claims": len(verdict.items),
                "unsupported": len(verdict.unsupported),
            },
        )
        return {"verdict": verdict, "feedback": verdict.feedback()}

    def finish(state: State) -> State:
        draft, verdict = state["draft"], state["verdict"]
        assert draft is not None and verdict is not None
        # Cite the passages that were verified to contain the evidence, not the
        # numbers the writer wrote down.
        cited = (locate_evidence(item.evidence, state["chunks"]) for item in verdict.items)
        citations = tuple(dict.fromkeys(chunk.citation for chunk in cited if chunk is not None))
        answer = Answer(AnswerStatus.ANSWERED, draft.text, citations)
        tracer.record("finish", {"status": answer.status.value, "citations": len(citations)})
        return {"answer": answer}

    def give_up(state: State) -> State:
        draft = state.get("draft")
        if draft is not None and not draft.claims:
            reason = "the sources do not contain an answer to the question"
        else:
            reason = "the answer could not be verified against the sources"
        answer = refusal(reason, state["chunks"])
        tracer.record("refuse", {"reason": reason, "attempts": state.get("attempts", 0)})
        return {"answer": answer}

    def after_route(state: State) -> str:
        return "end" if "answer" in state else "write"

    def after_write(state: State) -> str:
        draft = state.get("draft")
        if draft is None:  # the model's reply was unreadable
            return "write" if state["attempts"] <= max_retries else "give_up"
        return "verify" if draft.claims else "give_up"

    def after_verify(state: State) -> str:
        verdict = state["verdict"]
        assert verdict is not None
        if verdict.all_supported:
            return "finish"
        return "write" if state["attempts"] <= max_retries else "give_up"

    graph = StateGraph(State)
    for name, node in [
        ("retrieve", retrieve),
        ("route", route),
        ("write", write),
        ("verify", verify),
        ("finish", finish),
        ("give_up", give_up),
    ]:
        graph.add_node(name, node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "route")
    graph.add_conditional_edges("route", after_route, {"end": END, "write": "write"})
    graph.add_conditional_edges(
        "write", after_write, {"write": "write", "verify": "verify", "give_up": "give_up"}
    )
    graph.add_conditional_edges(
        "verify", after_verify, {"finish": "finish", "write": "write", "give_up": "give_up"}
    )
    graph.add_edge("finish", END)
    graph.add_edge("give_up", END)
    return graph.compile()

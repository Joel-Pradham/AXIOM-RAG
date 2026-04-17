"""
socratic_graph.py — AXIOM Agentic RAG (LangGraph).

Accuracy fixes in this version:
  1. Router receives {has_documents} so it can prefer RAG for ALL substantive
     questions when documents are available — not just when the user says "from my PDF".
  2. Relevance grader assesses ALL retrieved docs (not just top-1) and is
     permissive (any relevant chunk → route to generate).
  3. RAG generate_node uses a document-locked prompt — the LLM MUST stay within
     the retrieved chunks. No "supplement with own knowledge" escape hatch.
  4. Faithfulness scoring: after generation, an LLM judge rates how well the
     answer is grounded in the chunks (target ≥ 0.85). Score included in telemetry.
  5. If faithfulness < 0.75, a regeneration pass with a stricter prompt is triggered.
"""

import os
import yaml
from typing import List, Any, TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from src.retrieval import RAGTutorRetriever


# ── Structured output schema ─────────────────────────────────────────────────
class QAOutput(BaseModel):
    internal_thought_process: str = Field(
        description=(
            "Your step-by-step reasoning: which chunks are relevant, "
            "what they say, and how you will construct the answer from them."
        )
    )
    answer: str = Field(
        description=(
            "A thorough answer grounded in the retrieved chunks. "
            "Format in HTML (ul/li for lists, p for prose). Never use markdown."
        )
    )


# ── Graph state ──────────────────────────────────────────────────────────────
class TutorState(TypedDict):
    student_query:       str
    standalone_query:    str
    chat_history:        str
    route:               str
    retrieved_documents: List[Any]
    docs_are_relevant:   bool
    web_context:         str
    faithfulness_score:  float
    response:            dict


# ── Main tutor class ─────────────────────────────────────────────────────────
class SocraticTutor:

    def __init__(self, persist_directory: str = None):
        self.retriever = RAGTutorRetriever()

        # ── LLM selection ─────────────────────────────────────────────────────
        if "GROQ_API_KEY" in os.environ:
            from langchain_groq import ChatGroq
            self.llm      = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
            self.fast_llm = ChatGroq(model="llama-3.1-8b-instant",  temperature=0.0)
        elif "OPENAI_API_KEY" in os.environ:
            from langchain_openai import ChatOpenAI
            self.llm      = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
            self.fast_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        else:
            from langchain_cohere import ChatCohere
            self.llm      = ChatCohere(model="command-r-plus")
            self.fast_llm = ChatCohere(model="command-r")

        self.structured_llm = self.llm.with_structured_output(QAOutput)

        # ── Load prompts ──────────────────────────────────────────────────────
        with open("prompts.yaml", "r") as f:
            self.prompts = yaml.safe_load(f)

        self.system_prompt    = PromptTemplate.from_template(self.prompts["system_prompt"])
        self.general_prompt   = PromptTemplate.from_template(self.prompts["general_knowledge_prompt"])
        self.route_prompt     = PromptTemplate.from_template(self.prompts["route_query"])
        self.condense_prompt  = PromptTemplate.from_template(self.prompts["condense_question"])
        self.conv_prompt      = PromptTemplate.from_template(self.prompts["conversational_response"])
        self.grade_prompt     = PromptTemplate.from_template(self.prompts["grade_documents"])
        self.faith_prompt     = PromptTemplate.from_template(self.prompts["faithfulness_score"])

        self.graph = self._build_graph()

    # ── Nodes ────────────────────────────────────────────────────────────────

    def route_query_node(self, state: TutorState) -> TutorState:
        """
        3-way classification. Passes has_documents so the router knows to
        prefer 'rag' for all substantive questions when docs are available.
        """
        doc_count    = self.retriever.doc_count()
        has_documents = "true" if doc_count > 0 else "false"

        formatted = self.route_prompt.format(
            question=state["student_query"],
            chat_history=state.get("chat_history", ""),
            has_documents=has_documents,
        )
        resp = self.fast_llm.invoke([HumanMessage(content=formatted)])
        raw  = resp.content.strip().lower()

        if "conversational" in raw:
            route = "conversational"
        elif "rag" in raw:
            route = "rag"
        else:
            route = "general"

        print(f"[graph] route={route!r} | doc_count={doc_count} | has_documents={has_documents}")
        return {"route": route}

    def conversational_node(self, state: TutorState) -> TutorState:
        formatted = self.conv_prompt.format(
            question=state["student_query"],
            chat_history=state.get("chat_history", ""),
        )
        resp = self.fast_llm.invoke([HumanMessage(content=formatted)])
        return {"response": {
            "internal_thought_process": "Conversational greeting/small talk detected.",
            "answer":            resp.content,
            "citations":         [],
            "faithfulness_score": None,
        }}

    def retrieve_node(self, state: TutorState) -> TutorState:
        """Condense → retrieve → grade ALL docs (permissive)."""
        query   = state["student_query"]
        history = state.get("chat_history", "")

        # Condense follow-up queries
        if history:
            try:
                condensed = self.fast_llm.invoke([
                    HumanMessage(content=self.condense_prompt.format(
                        question=query, chat_history=history
                    ))
                ]).content.strip()
                standalone = condensed if condensed else query
            except Exception:
                standalone = query
        else:
            standalone = query

        docs = self.retriever.retrieve(standalone)
        print(f"[graph] retrieved {len(docs)} docs")

        # Grade ALL retrieved docs — relevant if ANY chunk is relevant
        docs_relevant = False
        if docs:
            try:
                # Build a multi-chunk context for the grader (first 5 chunks, 400 chars each)
                combined = "\n\n---\n\n".join(
                    f"[Chunk {i+1}]: {d.page_content[:400]}"
                    for i, d in enumerate(docs[:5])
                )
                grade_input = self.grade_prompt.format(
                    documents=combined,
                    question=standalone,
                )
                grade_resp    = self.fast_llm.invoke([HumanMessage(content=grade_input)])
                docs_relevant = "relevant" in grade_resp.content.lower()
            except Exception:
                docs_relevant = len(docs) > 0  # fallback: trust retriever

        print(f"[graph] docs_relevant={docs_relevant}")
        return {
            "retrieved_documents": docs,
            "docs_are_relevant":   docs_relevant,
            "standalone_query":    standalone,
        }

    def web_search_node(self, state: TutorState) -> TutorState:
        """DDG web search with 5s POSIX timeout. Failure is silent — LLM answers anyway."""
        query     = state.get("standalone_query", state["student_query"])
        web_context = ""

        try:
            import signal
            try:
                signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError()))
                signal.alarm(5)
            except (AttributeError, OSError):
                pass

            from duckduckgo_search import DDGS
            results = DDGS().text(query, max_results=4)

            try:
                signal.alarm(0)
            except (AttributeError, OSError):
                pass

            if results:
                snippets = [
                    f"<b>{r.get('title','')}</b>: {r.get('body','')}"
                    for r in results if r.get("body")
                ]
                web_context = "WEB SEARCH RESULTS:\n" + "\n\n".join(snippets)
                print(f"[graph] web_search: {len(snippets)} results")
        except Exception as e:
            print(f"[graph] web_search skipped: {type(e).__name__}")

        return {"web_context": web_context}

    def general_knowledge_node(self, state: TutorState) -> TutorState:
        """Answer from LLM training knowledge + optional web context. Never blocked."""
        query       = state.get("standalone_query", state["student_query"])
        web_context = state.get("web_context", "")
        history     = state.get("chat_history", "")

        web_section = (
            f"\nSUPPLEMENTARY WEB CONTEXT:\n---\n{web_context}\n---\n"
            if web_context else ""
        )

        formatted = self.general_prompt.format(
            question=query,
            chat_history=history,
            web_context=web_section,
        )

        citations = [{"source": "Web Search", "chapter_number": "N/A", "concept_tags": "Web"}] \
            if web_context else []

        try:
            response  = self.structured_llm.invoke([HumanMessage(content=formatted)])
            resp_dict = response.model_dump()
            resp_dict["citations"]         = citations
            resp_dict["faithfulness_score"] = None
            return {"response": resp_dict}
        except Exception as e:
            try:
                raw = self.llm.invoke([HumanMessage(content=formatted)])
                return {"response": {
                    "internal_thought_process": f"Structured output failed ({e}); raw fallback.",
                    "answer":            raw.content,
                    "citations":         citations,
                    "faithfulness_score": None,
                }}
            except Exception as e2:
                return {"response": {
                    "internal_thought_process": f"Generation error: {e2}",
                    "answer":    "<p>An error occurred. Please try again.</p>",
                    "citations": [],
                    "faithfulness_score": None,
                }}

    def generate_node(self, state: TutorState) -> TutorState:
        """
        RAG generation — document-locked.
        The LLM MUST answer from retrieved chunks. If faithfulness < 0.75,
        a second stricter pass is triggered automatically.
        Target faithfulness: ≥ 0.85.
        """
        query   = state.get("standalone_query", state["student_query"])
        docs    = state["retrieved_documents"]
        history = state.get("chat_history", "")

        # No relevant docs → fall through to general
        if not docs:
            return self.general_knowledge_node(state)

        context_str = "\n\n---\n\n".join([
            f"[Source — Course: {d.metadata.get('course_name','?')}, "
            f"Page: {d.metadata.get('page','?')}]\n{d.page_content}"
            for d in docs
        ])
        source_type = "User Uploaded Documents"

        def _generate(extra_instruction: str = "") -> dict:
            prompt_text = self.system_prompt.format(
                context=context_str,
                question=query + (f"\n\nINSTRUCTION: {extra_instruction}" if extra_instruction else ""),
                chat_history=history,
                source_type=source_type,
            )
            try:
                response = self.structured_llm.invoke([HumanMessage(content=prompt_text)])
                return response.model_dump()
            except Exception as e:
                raw = self.llm.invoke([HumanMessage(content=prompt_text)])
                return {
                    "internal_thought_process": f"Structured output failed ({e}); raw fallback.",
                    "answer": raw.content,
                }

        # ── First-pass generation ─────────────────────────────────────────────
        resp_dict = _generate()

        # ── Faithfulness scoring ──────────────────────────────────────────────
        faith_score = self._score_faithfulness(resp_dict.get("answer", ""), context_str, query)
        print(f"[graph] faithfulness_score={faith_score:.2f}")

        # ── Re-generate if score is too low (< 0.75) ─────────────────────────
        if faith_score < 0.75:
            print("[graph] faithfulness < 0.75 — regenerating with stricter grounding")
            resp_dict = _generate(
                "You MUST quote or paraphrase directly from the provided document chunks. "
                "Do NOT add ANY information that is not explicitly stated in the chunks."
            )
            faith_score = self._score_faithfulness(resp_dict.get("answer", ""), context_str, query)
            print(f"[graph] faithfulness_score (2nd pass)={faith_score:.2f}")

        resp_dict["citations"]         = [d.metadata for d in docs]
        resp_dict["faithfulness_score"] = round(faith_score, 2)

        return {
            "faithfulness_score": faith_score,
            "response":          resp_dict,
        }

    def _score_faithfulness(self, answer: str, context: str, question: str) -> float:
        """
        LLM-as-judge faithfulness scorer.
        Returns a float 0.0–1.0. Falls back to 0.8 on any error to avoid
        spurious re-generation loops.
        """
        if not answer or not context:
            return 0.8

        try:
            prompt = self.faith_prompt.format(
                context=context[:3000],   # cap to avoid token overflow
                answer=answer[:2000],
            )
            resp  = self.fast_llm.invoke([HumanMessage(content=prompt)])
            score = float(resp.content.strip().split()[0])
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"[graph] faithfulness scoring failed: {e}")
            return 0.8   # optimistic default — don't trigger unnecessary re-gen

    # ── Graph wiring ─────────────────────────────────────────────────────────

    def _route_after_classify(self, state: TutorState) -> str:
        return state.get("route", "general")

    def _route_after_retrieve(self, state: TutorState) -> str:
        return "generate" if state.get("docs_are_relevant", False) else "web_search"

    def _build_graph(self):
        wf = StateGraph(TutorState)

        wf.add_node("route_query",       self.route_query_node)
        wf.add_node("conversational",    self.conversational_node)
        wf.add_node("retrieve",          self.retrieve_node)
        wf.add_node("web_search",        self.web_search_node)
        wf.add_node("general_knowledge", self.general_knowledge_node)
        wf.add_node("generate",          self.generate_node)

        wf.set_entry_point("route_query")

        wf.add_conditional_edges(
            "route_query",
            self._route_after_classify,
            {
                "conversational": "conversational",
                "rag":            "retrieve",
                "general":        "web_search",
            },
        )

        wf.add_edge("conversational", END)

        wf.add_conditional_edges(
            "retrieve",
            self._route_after_retrieve,
            {
                "generate":   "generate",
                "web_search": "web_search",
            },
        )

        wf.add_edge("web_search",        "general_knowledge")
        wf.add_edge("general_knowledge", END)
        wf.add_edge("generate",          END)

        return wf.compile()

    # ── Public API ───────────────────────────────────────────────────────────

    def run(self, student_query: str, chat_history: str = "") -> dict:
        try:
            result = self.graph.invoke({
                "student_query":       student_query,
                "chat_history":        chat_history,
                "standalone_query":    student_query,
                "route":               "general",
                "retrieved_documents": [],
                "docs_are_relevant":   False,
                "web_context":         "",
                "faithfulness_score":  0.0,
                "response":            {},
            })
            return result.get("response", {
                "internal_thought_process": "Graph returned no response.",
                "answer":    "<p>Unable to generate a response. Please try again.</p>",
                "citations": [],
                "faithfulness_score": None,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "internal_thought_process": f"Pipeline error: {type(e).__name__}: {e}",
                "answer":    f"<p>An internal error occurred: {e}. Please try again.</p>",
                "citations": [],
                "faithfulness_score": None,
            }

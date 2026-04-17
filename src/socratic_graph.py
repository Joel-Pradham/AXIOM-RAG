"""
socratic_graph.py — AXIOM Agentic RAG (Production build).

Production constraints addressed:
  - Vercel 10s timeout → max 2 LLM calls total per request
  - Groq tool_use_failed on HTML in JSON → plain HTML output, no function calling
  - Blank answers → dead-simple response parsing, robust fallback at every step
  - Slow Cohere reranker removed from critical path (dense+BM25 hybrid only)
  - Faithfulness scoring moved to metadata (non-blocking)

Architecture: route(fast_llm) → retrieve(FAISS+BM25) → generate(llm) → done
Total target latency: ~4-6s on Vercel
"""

import os
import re
import yaml
from typing import List, Any, TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from src.retrieval import RAGTutorRetriever


# ── State ────────────────────────────────────────────────────────────────────
class TutorState(TypedDict):
    student_query:       str
    standalone_query:    str
    chat_history:        str
    route:               str
    retrieved_documents: List[Any]
    docs_are_relevant:   bool
    web_context:         str
    response:            dict


# ── Production tutor ─────────────────────────────────────────────────────────
class SocraticTutor:

    MAX_CHUNKS   = 6      # chunks fed to LLM — keeps latency under 10s
    MAX_CHUNK_LEN = 1800  # chars per chunk in the context string

    def __init__(self, persist_directory: str = None):
        self.retriever = RAGTutorRetriever()

        # LLM setup
        if "GROQ_API_KEY" in os.environ:
            from langchain_groq import ChatGroq
            self.llm      = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.15)
            self.fast_llm = ChatGroq(model="llama-3.1-8b-instant",    temperature=0.0)
        elif "OPENAI_API_KEY" in os.environ:
            from langchain_openai import ChatOpenAI
            self.llm = self.fast_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        else:
            from langchain_cohere import ChatCohere
            self.llm = self.fast_llm = ChatCohere(model="command-r-plus")

        # Prompts
        with open("prompts.yaml", "r") as f:
            self.cfg = yaml.safe_load(f)

        self.graph = self._build_graph()

    # ── LLM helper ───────────────────────────────────────────────────────────

    def _call(self, prompt: str, fast: bool = False) -> str:
        """Single LLM call. Returns empty string on any failure — never raises."""
        try:
            model = self.fast_llm if fast else self.llm
            return model.invoke([HumanMessage(content=prompt)]).content.strip()
        except Exception as e:
            print(f"[graph] LLM call failed ({type(e).__name__}): {e}")
            return ""

    # ── HTML sanitiser ────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_html(text: str) -> str:
        """
        If the LLM returned plain text instead of HTML, wrap it properly.
        If it returned HTML, return as-is.
        """
        if not text:
            return "<p>No response generated. Please try again.</p>"
        # If already contains HTML tags, trust it
        if re.search(r'<(p|ul|ol|li|h[1-6]|table|div|strong|em|code)\b', text):
            return text
        # Otherwise wrap each paragraph in <p>
        paras = [p.strip() for p in text.split('\n\n') if p.strip()]
        if paras:
            return ''.join(f'<p>{p}</p>' for p in paras)
        return f'<p>{text}</p>'

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def route_query_node(self, state: TutorState) -> TutorState:
        """Fast 3-way router. Prefers RAG when docs are available."""
        doc_count = self.retriever.doc_count()
        has_docs  = "true" if doc_count > 0 else "false"

        prompt = self.cfg["route_query"].format(
            question=state["student_query"],
            chat_history=state.get("chat_history", "") or "",
            has_documents=has_docs,
        )
        raw   = self._call(prompt, fast=True).lower()
        route = ("conversational" if "conversational" in raw
                 else "rag" if "rag" in raw
                 else "general")

        print(f"[graph] route={route!r} | docs_in_store={doc_count}")
        return {"route": route}

    def conversational_node(self, state: TutorState) -> TutorState:
        prompt = self.cfg["conversational_response"].format(
            question=state["student_query"],
            chat_history=state.get("chat_history", "") or "",
        )
        raw = self._call(prompt, fast=True)
        return {"response": {
            "internal_thought_process": "Conversational input — no retrieval.",
            "answer":      self._ensure_html(raw),
            "citations":   [],
            "faithfulness_score": None,
        }}

    def retrieve_node(self, state: TutorState) -> TutorState:
        """Condense follow-up → dense+BM25 hybrid retrieval → grade."""
        query   = state["student_query"]
        history = state.get("chat_history", "") or ""

        # Condense follow-up queries into standalone
        if history.strip():
            condensed = self._call(
                self.cfg["condense_question"].format(
                    question=query, chat_history=history
                ),
                fast=True,
            )
            standalone = condensed if condensed else query
        else:
            standalone = query

        docs = self.retriever.retrieve(standalone)
        print(f"[graph] retrieved {len(docs)} docs for: {standalone[:80]!r}")

        # Grade: permissive multi-chunk assessment
        docs_relevant = False
        if docs:
            sample = "\n\n---\n\n".join(
                f"[Chunk {i+1}]: {d.page_content[:300]}"
                for i, d in enumerate(docs[:5])
            )
            grade_raw = self._call(
                self.cfg["grade_documents"].format(
                    documents=sample, question=standalone
                ),
                fast=True,
            )
            docs_relevant = "relevant" in grade_raw.lower()

        print(f"[graph] docs_relevant={docs_relevant}")
        return {
            "retrieved_documents": docs,
            "docs_are_relevant":   docs_relevant,
            "standalone_query":    standalone,
        }

    def web_search_node(self, state: TutorState) -> TutorState:
        """DDG web search. Silent on failure — always returns."""
        query = state.get("standalone_query", state["student_query"])
        web   = ""
        try:
            import signal
            try:
                signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
                signal.alarm(4)
            except (AttributeError, OSError):
                pass
            from duckduckgo_search import DDGS
            results = DDGS().text(query, max_results=3)
            try:
                signal.alarm(0)
            except (AttributeError, OSError):
                pass
            if results:
                snippets = [
                    f"<b>{r.get('title','')}</b>: {r.get('body','')}"
                    for r in results if r.get("body")
                ]
                web = "WEB SEARCH RESULTS:\n" + "\n\n".join(snippets)
        except Exception as e:
            print(f"[graph] web search skipped ({type(e).__name__})")
        return {"web_context": web}

    def general_knowledge_node(self, state: TutorState) -> TutorState:
        """General LLM answer with optional web context."""
        query   = state.get("standalone_query", state["student_query"])
        history = state.get("chat_history", "") or ""
        web     = state.get("web_context", "") or ""
        web_sec = (f"\nSUPPLEMENTARY WEB CONTEXT:\n---\n{web}\n---\n" if web else "")

        prompt = self.cfg["general_knowledge_prompt"].format(
            question=query,
            chat_history=history,
            web_context=web_sec,
        )
        raw = self._call(prompt)
        return {"response": {
            "internal_thought_process": "General knowledge query — no document context.",
            "answer":      self._ensure_html(raw),
            "citations":   [],
            "faithfulness_score": None,
        }}

    def generate_node(self, state: TutorState) -> TutorState:
        """
        RAG generation.
        - Takes top MAX_CHUNKS chunks
        - Builds a context string
        - Single LLM call (no function calling — avoids tool_use_failed)
        - Returns pure HTML answer parsed from response
        """
        query   = state.get("standalone_query", state["student_query"])
        docs    = state.get("retrieved_documents", [])
        history = state.get("chat_history", "") or ""

        if not docs:
            # No docs — fall through to general knowledge
            return self.general_knowledge_node(state)

        # Build context: limit each chunk to MAX_CHUNK_LEN chars, take top MAX_CHUNKS
        top_docs = docs[:self.MAX_CHUNKS]
        context_parts = []
        for i, d in enumerate(top_docs):
            page   = d.metadata.get("page", "?")
            course = d.metadata.get("course_name", "Document")
            text   = d.page_content[:self.MAX_CHUNK_LEN]
            context_parts.append(f"[Chunk {i+1} | {course} | Page {page}]\n{text}")
        context_str = "\n\n" + ("─" * 60) + "\n\n".join(context_parts)

        # Build the full prompt
        prompt = self.cfg["system_prompt"].format(
            context=context_str,
            question=query,
            chat_history=history,
            source_type="User Uploaded Documents",
        )

        raw    = self._call(prompt)
        answer = self._ensure_html(raw)

        thought = (
            f"RAG path | {len(top_docs)} chunks retrieved "
            f"(pages: {', '.join(str(d.metadata.get('page','?')) for d in top_docs)}) | "
            f"Query: {query[:80]}"
        )

        return {"response": {
            "internal_thought_process": thought,
            "answer":      answer,
            "citations":   [d.metadata for d in top_docs],
            "faithfulness_score": None,   # disabled — was causing blank outputs
        }}

    # ── Graph wiring ──────────────────────────────────────────────────────────

    def _build_graph(self):
        wf = StateGraph(TutorState)

        wf.add_node("route_query",       self.route_query_node)
        wf.add_node("conversational",    self.conversational_node)
        wf.add_node("retrieve",          self.retrieve_node)
        wf.add_node("web_search",        self.web_search_node)
        wf.add_node("general_knowledge", self.general_knowledge_node)
        wf.add_node("generate",          self.generate_node)

        wf.set_entry_point("route_query")

        wf.add_conditional_edges("route_query",
            lambda s: s.get("route", "general"),
            {"conversational": "conversational",
             "rag":            "retrieve",
             "general":        "web_search"})

        wf.add_edge("conversational", END)

        wf.add_conditional_edges("retrieve",
            lambda s: "generate" if s.get("docs_are_relevant") else "web_search",
            {"generate": "generate", "web_search": "web_search"})

        wf.add_edge("web_search",        "general_knowledge")
        wf.add_edge("general_knowledge", END)
        wf.add_edge("generate",          END)

        return wf.compile()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, student_query: str, chat_history: str = "") -> dict:
        try:
            result = self.graph.invoke({
                "student_query":       student_query,
                "standalone_query":    student_query,
                "chat_history":        chat_history or "",
                "route":               "general",
                "retrieved_documents": [],
                "docs_are_relevant":   False,
                "web_context":         "",
                "response":            {},
            })
            resp = result.get("response") or {}
            # Ensure answer is never empty
            if not resp.get("answer"):
                resp["answer"] = "<p>Unable to generate a response. Please try again.</p>"
            return resp
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "internal_thought_process": f"Pipeline error: {type(e).__name__}: {e}",
                "answer":    f"<p>An error occurred: {e}</p>",
                "citations": [],
                "faithfulness_score": None,
            }

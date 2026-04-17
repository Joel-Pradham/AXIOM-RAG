"""
socratic_graph.py — AXIOM Agentic RAG pipeline (LangGraph).

Routing logic (3-way):
  conversational  → conversational_node → END
  rag             → retrieve_node → (relevant docs?) → generate_node → END
                                  → (no relevant docs) → general_knowledge_node → END
  general         → general_knowledge_node → END
                    (web search attempted first; LLM answers regardless of result)

Key design decisions for Vercel:
  - DuckDuckGo is attempted in general_knowledge_node with a 5-second timeout.
  - If DDG fails (rate-limited on Vercel IPs), the Groq LLM answers from its
    own knowledge — the answer is NEVER blocked by a failed web search.
  - The RAG system_prompt no longer forbids LLM knowledge — it uses docs as
    primary and LLM knowledge as supplement.
  - Document relevance is checked via a quick LLM grader so irrelevant uploaded
    docs don't pollute general-knowledge answers.
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


# ── Structured output schema ────────────────────────────────────────────────────
class QAOutput(BaseModel):
    internal_thought_process: str = Field(
        description="Your detailed reasoning chain: identify the query type, relevant knowledge, and plan the response."
    )
    answer: str = Field(
        description=(
            "A thorough, exhaustive, expert-level answer. "
            "Cover ALL relevant details, subtypes, and examples. "
            "Format in HTML (ul/li for lists, p for prose). "
            "Never use markdown."
        )
    )


# ── Graph state ─────────────────────────────────────────────────────────────────
class TutorState(TypedDict):
    student_query:      str
    standalone_query:   str
    chat_history:       str
    route:              str          # "conversational" | "rag" | "general"
    retrieved_documents: List[Any]
    docs_are_relevant:  bool
    web_context:        str          # web search results (may be empty)
    response:           dict


# ── Main tutor class ─────────────────────────────────────────────────────────────
class SocraticTutor:

    def __init__(self, persist_directory: str = None):
        # persist_directory kept for backwards-compat; ignored (store.py manages it)
        self.retriever = RAGTutorRetriever()

        # ── LLM selection ──────────────────────────────────────────────────
        if "GROQ_API_KEY" in os.environ:
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
            # Use a smaller/faster model for classification tasks
            self.fast_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
        elif "OPENAI_API_KEY" in os.environ:
            from langchain_openai import ChatOpenAI
            self.llm      = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
            self.fast_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        else:
            from langchain_cohere import ChatCohere
            self.llm      = ChatCohere(model="command-r-plus")
            self.fast_llm = ChatCohere(model="command-r")

        self.structured_llm = self.llm.with_structured_output(QAOutput)

        # ── Load prompts ───────────────────────────────────────────────────
        with open("prompts.yaml", "r") as f:
            self.prompts = yaml.safe_load(f)

        self.system_prompt      = PromptTemplate.from_template(self.prompts["system_prompt"])
        self.general_prompt     = PromptTemplate.from_template(self.prompts["general_knowledge_prompt"])
        self.route_prompt       = PromptTemplate.from_template(self.prompts["route_query"])
        self.condense_prompt    = PromptTemplate.from_template(self.prompts["condense_question"])
        self.conv_prompt        = PromptTemplate.from_template(self.prompts["conversational_response"])
        self.grade_prompt       = PromptTemplate.from_template(self.prompts["grade_documents"])

        # ── Build graph ────────────────────────────────────────────────────
        self.graph = self._build_graph()

    # ── Nodes ────────────────────────────────────────────────────────────────────

    def route_query_node(self, state: TutorState) -> TutorState:
        """Classify into conversational / rag / general."""
        formatted = self.route_prompt.format(
            question=state["student_query"],
            chat_history=state.get("chat_history", ""),
        )
        resp = self.fast_llm.invoke([HumanMessage(content=formatted)])
        raw  = resp.content.strip().lower()

        if "conversational" in raw:
            route = "conversational"
        elif "rag" in raw:
            route = "rag"
        else:
            route = "general"

        print(f"[graph] route={route!r} for query={state['student_query']!r}")
        return {"route": route}

    def conversational_node(self, state: TutorState) -> TutorState:
        formatted = self.conv_prompt.format(
            question=state["student_query"],
            chat_history=state.get("chat_history", ""),
        )
        resp = self.fast_llm.invoke([HumanMessage(content=formatted)])
        return {"response": {
            "internal_thought_process": "Conversational message detected. Generating a direct, friendly reply.",
            "answer":    resp.content,
            "citations": [],
        }}

    def retrieve_node(self, state: TutorState) -> TutorState:
        """Condense query, retrieve docs, grade relevance."""
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

        # Quick relevance check — avoid answering general questions with
        # irrelevant uploaded document chunks
        docs_relevant = False
        if docs:
            try:
                # Grade the top document only (fast)
                grade_input = self.grade_prompt.format(
                    document=docs[0].page_content[:600],
                    question=standalone,
                )
                grade_resp = self.fast_llm.invoke([HumanMessage(content=grade_input)])
                docs_relevant = "relevant" in grade_resp.content.lower()
            except Exception:
                docs_relevant = len(docs) > 0  # fallback: trust retriever

        print(f"[graph] retrieve: {len(docs)} docs, relevant={docs_relevant}")
        return {
            "retrieved_documents": docs,
            "docs_are_relevant":   docs_relevant,
            "standalone_query":    standalone,
        }

    def web_search_node(self, state: TutorState) -> TutorState:
        """
        DuckDuckGo web search with a hard 5-second timeout.
        On Vercel, DDG is often rate-limited — the graph continues to
        general_knowledge_node regardless of whether this succeeds.
        """
        query = state.get("standalone_query", state["student_query"])
        web_context = ""

        try:
            import signal

            def _timeout_handler(signum, frame):
                raise TimeoutError("DDG timeout")

            # POSIX timeout (works on Linux Vercel containers)
            try:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(5)
            except (AttributeError, OSError):
                pass  # Windows — no SIGALRM, proceed without hard timeout

            from duckduckgo_search import DDGS
            results = DDGS().text(query, max_results=4)

            try:
                signal.alarm(0)  # cancel alarm
            except (AttributeError, OSError):
                pass

            if results:
                snippets = [
                    f"<b>{r.get('title', '')}</b>: {r.get('body', '')}"
                    for r in results if r.get("body")
                ]
                web_context = "WEB SEARCH RESULTS:\n" + "\n\n".join(snippets)
                print(f"[graph] web_search: {len(snippets)} results")
            else:
                print("[graph] web_search: no results returned")

        except Exception as e:
            print(f"[graph] web_search failed (expected on Vercel): {type(e).__name__}: {e}")

        return {"web_context": web_context}

    def general_knowledge_node(self, state: TutorState) -> TutorState:
        """
        Answer using LLM's built-in knowledge, supplemented by web context
        if available. This node ALWAYS produces an answer — never blocked.
        """
        query       = state.get("standalone_query", state["student_query"])
        web_context = state.get("web_context", "")
        history     = state.get("chat_history", "")

        web_section = (
            f"\nSUPPLEMENTARY WEB CONTEXT (use as additional reference):\n"
            f"---------------------\n{web_context}\n---------------------\n"
            if web_context else ""
        )

        formatted = self.general_prompt.format(
            question=query,
            chat_history=history,
            web_context=web_section,
        )

        citations = []
        if web_context:
            citations = [{"source": "Web Search (DuckDuckGo)", "chapter_number": "N/A", "concept_tags": "Web"}]

        try:
            response = self.structured_llm.invoke([HumanMessage(content=formatted)])
            resp_dict = response.model_dump()
            resp_dict["citations"] = citations
            return {"response": resp_dict}
        except Exception as e:
            # Structured output failed — raw fallback
            try:
                raw = self.llm.invoke([HumanMessage(content=formatted)])
                return {"response": {
                    "internal_thought_process": f"Structured output failed ({e}); used raw generation.",
                    "answer":    raw.content,
                    "citations": citations,
                }}
            except Exception as e2:
                return {"response": {
                    "internal_thought_process": f"Generation failed: {e2}",
                    "answer":    "<p>I encountered an error generating a response. Please try again.</p>",
                    "citations": [],
                }}

    def generate_node(self, state: TutorState) -> TutorState:
        """
        RAG generation — answers primarily from retrieved document chunks,
        supplemented by LLM knowledge where context is insufficient.
        """
        query    = state.get("standalone_query", state["student_query"])
        docs     = state["retrieved_documents"]
        history  = state.get("chat_history", "")

        if not docs:
            return self.general_knowledge_node(state)

        context_str = "\n\n".join([
            f"Source (Course: {d.metadata.get('course_name', 'N/A')}, "
            f"Ch: {d.metadata.get('chapter_number', 'N/A')}):\n{d.page_content}"
            for d in docs
        ])

        # Determine source type label for the prompt
        courses = set(d.metadata.get("course_name", "") for d in docs)
        source_type = "Uploaded Documents" if "User Uploaded Docs" in courses else "Retrieved Knowledge Base"

        formatted = self.system_prompt.format(
            context=context_str,
            question=query,
            chat_history=history,
            source_type=source_type,
        )

        try:
            response  = self.structured_llm.invoke([HumanMessage(content=formatted)])
            resp_dict = response.model_dump()
            resp_dict["citations"] = [d.metadata for d in docs]
            return {"response": resp_dict}
        except Exception as e:
            try:
                raw = self.llm.invoke([HumanMessage(content=formatted)])
                return {"response": {
                    "internal_thought_process": f"Structured output failed ({e}); raw generation used.",
                    "answer":    raw.content,
                    "citations": [d.metadata for d in docs],
                }}
            except Exception as e2:
                return {"response": {
                    "internal_thought_process": f"Generation failed: {e2}",
                    "answer":    "<p>An error occurred. Please try again.</p>",
                    "citations": [],
                }}

    # ── Graph wiring ─────────────────────────────────────────────────────────────

    def _route_after_classify(self, state: TutorState) -> str:
        return state.get("route", "general")

    def _route_after_retrieve(self, state: TutorState) -> str:
        return "generate" if state.get("docs_are_relevant", False) else "web_search"

    def _build_graph(self):
        wf = StateGraph(TutorState)

        wf.add_node("route_query",        self.route_query_node)
        wf.add_node("conversational",     self.conversational_node)
        wf.add_node("retrieve",           self.retrieve_node)
        wf.add_node("web_search",         self.web_search_node)
        wf.add_node("general_knowledge",  self.general_knowledge_node)
        wf.add_node("generate",           self.generate_node)

        wf.set_entry_point("route_query")

        # 3-way split after classification
        wf.add_conditional_edges(
            "route_query",
            self._route_after_classify,
            {
                "conversational": "conversational",
                "rag":            "retrieve",
                "general":        "web_search",    # web search first, then LLM
            },
        )

        wf.add_edge("conversational", END)

        # RAG branch: grade docs → generate or fall back to general
        wf.add_conditional_edges(
            "retrieve",
            self._route_after_retrieve,
            {
                "generate":   "generate",
                "web_search": "web_search",
            },
        )

        # web_search always feeds into general_knowledge (LLM answers regardless)
        wf.add_edge("web_search", "general_knowledge")
        wf.add_edge("general_knowledge", END)
        wf.add_edge("generate", END)

        return wf.compile()

    # ── Public API ────────────────────────────────────────────────────────────────

    def run(self, student_query: str, chat_history: str = "") -> dict:
        try:
            result = self.graph.invoke({
                "student_query":      student_query,
                "chat_history":       chat_history,
                "standalone_query":   student_query,
                "route":              "general",
                "retrieved_documents": [],
                "docs_are_relevant":  False,
                "web_context":        "",
                "response":           {},
            })
            return result.get("response", {
                "internal_thought_process": "Graph completed with no response.",
                "answer":    "<p>Unable to generate a response. Please try again.</p>",
                "citations": [],
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "internal_thought_process": f"Pipeline error: {type(e).__name__}: {e}",
                "answer":    f"<p>An internal error occurred: {e}. Please try again.</p>",
                "citations": [],
            }

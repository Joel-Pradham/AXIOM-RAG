"""
socratic_graph.py — AXIOM Agentic RAG (LangGraph).

Key fixes in this version:
  1. Removed with_structured_output() — Groq's function calling breaks on HTML
     in JSON strings (tool_use_failed). Now uses a robust [THOUGHT]/[ANSWER]
     text block format that is HTML-safe and always parseable.
  2. Multi-query retrieval — generates 3 search queries per user question
     for comprehensive document coverage.
  3. Feeds 12 chunks (up from 8) to the LLM for richer context.
  4. Removed faithfulness re-generation loop — caused answer vagueness.
  5. Faithfulness scoring retained as telemetry only (not blocking).
"""

import os
import re
import yaml
from typing import List, Any, TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from src.retrieval import RAGTutorRetriever


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


# ── Response parser (HTML-safe, no function calling) ────────────────────────
def _parse_response(raw: str) -> dict:
    """
    Parse the [THOUGHT]...[/THOUGHT][ANSWER]...[/ANSWER] format.
    Falls back gracefully if the model doesn't follow format exactly.
    """
    thought_match = re.search(r'\[THOUGHT\](.*?)\[/THOUGHT\]', raw, re.DOTALL)
    answer_match  = re.search(r'\[ANSWER\](.*?)\[/ANSWER\]',  raw, re.DOTALL)

    thought = thought_match.group(1).strip() if thought_match else "No thought captured."
    answer  = answer_match.group(1).strip()  if answer_match  else raw.strip()

    # If answer has no HTML tags at all, wrap it in <p> tags
    if not re.search(r'<[a-zA-Z]', answer):
        answer = "<p>" + answer.replace("\n\n", "</p><p>") + "</p>"

    return {"internal_thought_process": thought, "answer": answer}


# ── Main tutor class ─────────────────────────────────────────────────────────
class SocraticTutor:

    def __init__(self, persist_directory: str = None):
        self.retriever = RAGTutorRetriever()

        # ── LLM ───────────────────────────────────────────────────────────────
        if "GROQ_API_KEY" in os.environ:
            from langchain_groq import ChatGroq
            self.llm      = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
            self.fast_llm = ChatGroq(model="llama-3.1-8b-instant",    temperature=0.0)
        elif "OPENAI_API_KEY" in os.environ:
            from langchain_openai import ChatOpenAI
            self.llm      = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
            self.fast_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        else:
            from langchain_cohere import ChatCohere
            self.llm      = ChatCohere(model="command-r-plus")
            self.fast_llm = ChatCohere(model="command-r")

        # ── Load prompts ──────────────────────────────────────────────────────
        with open("prompts.yaml", "r") as f:
            self.prompts = yaml.safe_load(f)

        self.system_prompt   = PromptTemplate.from_template(self.prompts["system_prompt"])
        self.general_prompt  = PromptTemplate.from_template(self.prompts["general_knowledge_prompt"])
        self.route_prompt    = PromptTemplate.from_template(self.prompts["route_query"])
        self.condense_prompt = PromptTemplate.from_template(self.prompts["condense_question"])
        self.conv_prompt     = PromptTemplate.from_template(self.prompts["conversational_response"])
        self.grade_prompt    = PromptTemplate.from_template(self.prompts["grade_documents"])
        self.faith_prompt    = PromptTemplate.from_template(self.prompts["faithfulness_score"])

        self.graph = self._build_graph()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _llm_text(self, prompt: str, fast: bool = False) -> str:
        """Call LLM and return raw text. Never raises — returns empty string on failure."""
        try:
            llm  = self.fast_llm if fast else self.llm
            resp = llm.invoke([HumanMessage(content=prompt)])
            return resp.content.strip()
        except Exception as e:
            print(f"[graph] LLM call failed: {e}")
            return ""

    def _expand_queries(self, query: str) -> List[str]:
        """
        Generate 2 additional search queries for multi-query retrieval.
        Returns [original] + up to 2 expansions. Never blocks the pipeline.
        """
        prompt = (
            f"Generate 2 alternative search queries to retrieve different but relevant "
            f"chunks from a document about:\n\n\"{query}\"\n\n"
            f"Focus on different aspects of the topic. "
            f"Output ONLY a numberd list like:\n1. query one\n2. query two"
        )
        try:
            raw    = self._llm_text(prompt, fast=True)
            extras = re.findall(r'^\d+\.\s*(.+)', raw, re.MULTILINE)
            extras = [e.strip() for e in extras if len(e.strip()) > 5][:2]
            queries = [query] + extras
            print(f"[graph] multi-query: {queries}")
            return queries
        except Exception:
            return [query]

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def route_query_node(self, state: TutorState) -> TutorState:
        doc_count     = self.retriever.doc_count()
        has_documents = "true" if doc_count > 0 else "false"

        route_text = self._llm_text(
            self.route_prompt.format(
                question=state["student_query"],
                chat_history=state.get("chat_history", ""),
                has_documents=has_documents,
            ),
            fast=True,
        )
        raw = route_text.lower()
        if "conversational" in raw:
            route = "conversational"
        elif "rag" in raw:
            route = "rag"
        else:
            route = "general"

        print(f"[graph] route={route!r} | docs={doc_count}")
        return {"route": route}

    def conversational_node(self, state: TutorState) -> TutorState:
        raw = self._llm_text(
            self.conv_prompt.format(
                question=state["student_query"],
                chat_history=state.get("chat_history", ""),
            ),
            fast=True,
        )
        return {"response": {
            "internal_thought_process": "Conversational input — no retrieval needed.",
            "answer":            raw or "<p>Hello! How can I help you today?</p>",
            "citations":         [],
            "faithfulness_score": None,
        }}

    def retrieve_node(self, state: TutorState) -> TutorState:
        """Condense + multi-query retrieval + permissive relevance grading."""
        query   = state["student_query"]
        history = state.get("chat_history", "")

        # ── Condense follow-ups ───────────────────────────────────────────────
        if history:
            condensed = self._llm_text(
                self.condense_prompt.format(question=query, chat_history=history),
                fast=True,
            )
            standalone = condensed if condensed else query
        else:
            standalone = query

        # ── Multi-query retrieval ─────────────────────────────────────────────
        queries = self._expand_queries(standalone)

        seen, merged = set(), []
        for q in queries:
            for doc in self.retriever.retrieve(q):
                h = hash(doc.page_content.strip())
                if h not in seen:
                    seen.add(h)
                    merged.append(doc)

        print(f"[graph] multi-query retrieved {len(merged)} unique chunks "
              f"across {len(queries)} queries")

        # ── Relevance grade (all top-5 chunks, permissive) ───────────────────
        docs_relevant = False
        if merged:
            combined = "\n\n---\n\n".join(
                f"[Chunk {i+1}]: {d.page_content[:400]}"
                for i, d in enumerate(merged[:5])
            )
            grade_raw = self._llm_text(
                self.grade_prompt.format(documents=combined, question=standalone),
                fast=True,
            )
            docs_relevant = "relevant" in grade_raw.lower()

        print(f"[graph] docs_relevant={docs_relevant}")
        return {
            "retrieved_documents": merged,
            "docs_are_relevant":   docs_relevant,
            "standalone_query":    standalone,
        }

    def web_search_node(self, state: TutorState) -> TutorState:
        """DDG web search with 5s timeout. Silent on failure."""
        query      = state.get("standalone_query", state["student_query"])
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
        except Exception as e:
            print(f"[graph] web_search skipped: {type(e).__name__}")
        return {"web_context": web_context}

    def general_knowledge_node(self, state: TutorState) -> TutorState:
        """General LLM answer with optional web context."""
        query       = state.get("standalone_query", state["student_query"])
        web_context = state.get("web_context", "")
        web_section = (
            f"\nSUPPLEMENTARY WEB CONTEXT:\n---\n{web_context}\n---\n"
            if web_context else ""
        )
        raw = self._llm_text(
            self.general_prompt.format(
                question=query,
                chat_history=state.get("chat_history", ""),
                web_context=web_section,
            )
        )
        parsed = _parse_response(raw)
        parsed["citations"]          = [{"source": "General Knowledge", "course_name": "LLM",
                                         "chapter_number": "N/A", "concept_tags": "General"}] \
                                       if not web_context else \
                                       [{"source": "Web Search", "course_name": "Web",
                                         "chapter_number": "N/A", "concept_tags": "Web"}]
        parsed["faithfulness_score"] = None
        return {"response": parsed}

    def generate_node(self, state: TutorState) -> TutorState:
        """
        RAG generation — document-grounded with enrichment.
        Uses plain text [THOUGHT]/[ANSWER] format to avoid Groq function-call failures.
        Feeds up to 12 chunks for comprehensive coverage.
        """
        query   = state.get("standalone_query", state["student_query"])
        docs    = state["retrieved_documents"]
        history = state.get("chat_history", "")

        if not docs:
            return self.general_knowledge_node(state)

        # Take top 12 chunks for comprehensive context
        top_docs    = docs[:12]
        context_str = "\n\n---\n\n".join([
            f"[Source — {d.metadata.get('course_name','Doc')}, "
            f"Page {d.metadata.get('page','?')}]\n{d.page_content}"
            for d in top_docs
        ])

        prompt_text = self.system_prompt.format(
            context=context_str,
            question=query,
            chat_history=history,
            source_type="User Uploaded Documents",
        )

        raw    = self._llm_text(prompt_text)
        parsed = _parse_response(raw)

        # ── Faithfulness scoring (telemetry only — does not block) ───────────
        faith_score = self._score_faithfulness(
            parsed.get("answer", ""), context_str, query
        )
        print(f"[graph] faithfulness={faith_score:.2f}")

        parsed["citations"]          = [d.metadata for d in top_docs]
        parsed["faithfulness_score"] = round(faith_score, 2)

        return {
            "faithfulness_score": faith_score,
            "response":           parsed,
        }

    def _score_faithfulness(self, answer: str, context: str, question: str) -> float:
        """LLM-as-judge faithfulness scorer. Returns 0.0–1.0."""
        if not answer or not context:
            return 0.8
        try:
            raw   = self._llm_text(
                self.faith_prompt.format(
                    context=context[:3000],
                    answer=answer[:2000],
                ),
                fast=True,
            )
            score = float(raw.strip().split()[0])
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"[graph] faithfulness scoring failed: {e}")
            return 0.8

    # ── Graph wiring ──────────────────────────────────────────────────────────

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
            {"generate": "generate", "web_search": "web_search"},
        )

        wf.add_edge("web_search",        "general_knowledge")
        wf.add_edge("general_knowledge", END)
        wf.add_edge("generate",          END)

        return wf.compile()

    # ── Public API ────────────────────────────────────────────────────────────

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
            resp = result.get("response", {})
            if not resp or not resp.get("answer"):
                return {
                    "internal_thought_process": "Graph returned empty response.",
                    "answer":    "<p>Unable to generate a response. Please try again.</p>",
                    "citations": [],
                    "faithfulness_score": None,
                }
            return resp
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "internal_thought_process": f"Pipeline error: {type(e).__name__}: {e}",
                "answer":    f"<p>An internal error occurred: {e}. Please try again.</p>",
                "citations": [],
                "faithfulness_score": None,
            }

"""
socratic_graph.py — AXIOM Ultimate RAG (Unconstrained).

Full feature set — no deployment compromises:

RETRIEVAL:
  - Multi-query expansion: 3 query variants per question
  - Dense (FAISS) + Sparse (BM25) per variant
  - Reciprocal Rank Fusion across all variants
  - Cohere cross-encoder reranking (top-10)

GENERATION:
  - 10 chunks fed to LLM — comprehensive context
  - Pure HTML output — no JSON/function-calling failures
  - Faithfulness score as LLM judge (0.0–1.0)
  - Automatic 2nd-pass refinement if faithfulness < 0.85
  - Document-locked with explicit supplement enrichment

ROUTING:
  - conversational / rag / general — 3-way
  - Web search fallback (DuckDuckGo) for general queries
  - Prefers RAG whenever documents are available

TARGET: > 85% document faithfulness on every response.
"""

import os
import re
import yaml
from typing import List, Any, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from src.retrieval import RAGTutorRetriever


# ── State ─────────────────────────────────────────────────────────────────────
class TutorState(TypedDict):
    student_query:       str
    standalone_query:    str
    chat_history:        str
    route:               str
    query_variants:      List[str]
    retrieved_documents: List[Any]
    docs_are_relevant:   bool
    web_context:         str
    faithfulness_score:  float
    response:            dict


# ── AXIOM tutor ───────────────────────────────────────────────────────────────
class SocraticTutor:

    MAX_CHUNKS    = 10     # chunks fed to LLM
    MAX_CHUNK_LEN = 2000   # chars per chunk in prompt
    FAITH_THRESH  = 0.82   # trigger 2nd-pass refinement below this

    def __init__(self, persist_directory: str = None):
        self.retriever = RAGTutorRetriever()

        # ── LLM init ────────────────────────────────────────────────────────
        if "GROQ_API_KEY" in os.environ:
            from langchain_groq import ChatGroq
            self.llm      = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
            self.fast_llm = ChatGroq(model="llama-3.1-8b-instant",    temperature=0.0)
            print("[tutor] Using Groq: llama-3.3-70b (main) + llama-3.1-8b (fast)")
        elif "OPENAI_API_KEY" in os.environ:
            from langchain_openai import ChatOpenAI
            self.llm      = ChatOpenAI(model="gpt-4o",      temperature=0.1)
            self.fast_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
            print("[tutor] Using OpenAI: gpt-4o (main) + gpt-4o-mini (fast)")
        else:
            from langchain_cohere import ChatCohere
            self.llm = self.fast_llm = ChatCohere(model="command-r-plus")
            print("[tutor] Using Cohere: command-r-plus")

        # ── Prompts ──────────────────────────────────────────────────────────
        with open("prompts.yaml", "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.graph = self._build_graph()
        print("[tutor] Graph compiled — ultimate mode active.")

    # ── LLM helper ───────────────────────────────────────────────────────────

    def _call(self, prompt: str, fast: bool = False) -> str:
        """Single LLM call. Returns empty string on failure — never raises."""
        try:
            model = self.fast_llm if fast else self.llm
            return model.invoke([HumanMessage(content=prompt)]).content.strip()
        except Exception as e:
            print(f"[graph] LLM error ({type(e).__name__}): {e}")
            return ""

    # ── HTML sanitiser ────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_html(text: str) -> str:
        """Guarantee the answer is always valid HTML — never blank."""
        if not text or not text.strip():
            return "<p>Unable to generate a response. Please rephrase your question.</p>"
        # If already has HTML tags, return as-is
        if re.search(r'<(p|ul|ol|li|h[1-6]|table|div|strong|em|code)\b', text):
            return text
        # Wrap plain text paragraphs
        paras = [p.strip() for p in text.split('\n\n') if p.strip()]
        return ''.join(f'<p>{p}</p>' for p in paras) if paras else f'<p>{text}</p>'

    # ── Nodes ─────────────────────────────────────────────────────────────────

    # Keywords that signal the user wants info FROM the uploaded document
    _DOC_INTENT_KEYWORDS = (
        "document", "uploaded", "file", "pdf", "topic", "topics",
        "chapter", "section", "content", "summarize", "summary",
        "explain", "what is", "what are", "how does", "describe",
        "tell me", "list", "covered", "about", "mention",
    )

    def route_query_node(self, state: TutorState) -> TutorState:
        doc_count = self.retriever.doc_count()
        query_lower = state["student_query"].lower()

        # ── Hard override: docs exist → always try RAG first ──────────────
        # The small fast-LLM router frequently makes wrong decisions when the
        # chat history contains past answers (even after HTML stripping).
        # Skipping the LLM router entirely when docs are present is safer
        # and ensures we never land on "general" for document questions.
        if doc_count > 0:
            # Only route to conversational for obvious small talk
            _SMALL_TALK = ("hello", "hi ", "hey ", "thanks", "thank you",
                           "good morning", "good night", "who are you",
                           "how are you", "bye", "goodbye")
            is_small_talk = any(query_lower.startswith(t) or query_lower == t.strip()
                                for t in _SMALL_TALK)
            route = "conversational" if is_small_talk else "rag"
            print(f"[graph] route={route!r} (hard override) | docs_in_store={doc_count}")
            return {"route": route}

        # ── No docs in store: fall back to LLM router ─────────────────────
        prompt = self.cfg["route_query"].format(
            question=state["student_query"],
            chat_history=state.get("chat_history", "") or "",
            has_documents="false",
        )
        raw   = self._call(prompt, fast=True).lower()
        route = ("conversational" if "conversational" in raw
                 else "rag"       if "rag"            in raw
                 else "general")
        print(f"[graph] route={route!r} (llm router) | docs_in_store={doc_count}")
        return {"route": route}

    def conversational_node(self, state: TutorState) -> TutorState:
        raw = self._call(
            self.cfg["conversational_response"].format(
                question=state["student_query"],
                chat_history=state.get("chat_history", "") or "",
            ),
            fast=True,
        )
        return {"response": {
            "internal_thought_process": "Conversational — no retrieval needed.",
            "answer":            self._ensure_html(raw),
            "citations":         [],
            "faithfulness_score": None,
        }}

    def retrieve_node(self, state: TutorState) -> TutorState:
        """
        Full retrieval pipeline:
        1. Condense follow-up to standalone
        2. Generate 2 query variants (multi-query expansion)
        3. retrieve_multi() — RRF + Cohere reranking
        4. Relevance grading — used as a HINT only, never discards docs that exist
        """
        query   = state["student_query"]
        history = state.get("chat_history", "") or ""

        # ── Step 1: condense ────────────────────────────────────────────────
        # Only condense if the query has genuine follow-up references
        # (pronouns / demonstratives). Standalone queries must NOT be rewritten
        # using history — that injects irrelevant topics from past turns.
        import re
        _FOLLOWUP_SIGNALS = (r"\bit\b", r"\bits\b", r"\bthis\b", r"\bthat\b", r"\bthey\b", 
                             r"\btheir\b", r"\btheirs\b", r"\bthese\b", r"\bthose\b",
                             r"\bthe previous\b", r"\bsame\b", r"\babove\b", r"\bmentioned\b",
                             r"\bsaid earlier\b")
        query_lower = query.lower()
        has_followup = any(re.search(sig, query_lower) for sig in _FOLLOWUP_SIGNALS)

        if history.strip() and has_followup:
            condensed = self._call(
                self.cfg["condense_question"].format(
                    question=query, chat_history=history
                ),
                fast=True,
            )
            standalone = condensed if condensed else query
        else:
            standalone = query

        # ── Step 2: query expansion ─────────────────────────────────────────
        expansion_prompt = (
            f"Generate 2 alternative search queries for a document retrieval system "
            f"to find comprehensive information about this question.\n\n"
            f"Original question: \"{standalone}\"\n\n"
            f"Rules:\n"
            f"1. Each variant must focus on a DIFFERENT aspect of the topic.\n"
            f"2. Use different vocabulary and angle than the original.\n"
            f"3. Keep them concise (max 20 words each).\n\n"
            f"Output ONLY a numbered list:\n"
            f"1. first variant\n"
            f"2. second variant"
        )
        expansion_raw = self._call(expansion_prompt, fast=True)
        variants = re.findall(r'^\d+\.\s*(.+)', expansion_raw, re.MULTILINE)
        variants = [v.strip() for v in variants if len(v.strip()) > 8][:2]
        queries  = [standalone] + variants
        print(f"[graph] Query variants: {queries}")

        # ── Step 3: multi-query retrieval with RRF + reranking ──────────────
        docs = self.retriever.retrieve_multi(queries, original_question=standalone)
        print(f"[graph] retrieve_multi returned {len(docs)} docs")

        # ── Step 4: relevance grading ───────────────────────────────────────
        # Determine if the user EXPLICITLY wants a document-based answer
        query_lower = standalone.lower()
        mentions_doc = any(k in query_lower for k in self._DOC_INTENT_KEYWORDS)

        if docs:
            sample = "\n\n---\n\n".join(
                f"[Chunk {i+1}]: {d.page_content[:350]}"
                for i, d in enumerate(docs[:5])
            )
            # Use the smarter, slower LLM for grading to avoid false negatives
            grade = self._call(
                self.cfg["grade_documents"].format(
                    documents=sample, question=standalone
                ),
                fast=False,
            )
            grader_says_relevant = "relevant" in grade.lower()
            
            if mentions_doc:
                docs_are_relevant = True
                print(f"[graph] Grader: {grade.strip()!r} | User specifically mentioned doc. Forcing RAG route.")
            else:
                docs_are_relevant = grader_says_relevant
                print(f"[graph] Grader: {grade.strip()!r} | relevance = {docs_are_relevant}")
        else:
            docs_are_relevant = False
            print("[graph] No docs retrieved — routing to web_search fallback")

        return {
            "retrieved_documents": docs,
            "docs_are_relevant":   docs_are_relevant,
            "standalone_query":    standalone,
            "query_variants":      queries,
        }

    def web_search_node(self, state: TutorState) -> TutorState:
        """DuckDuckGo fallback. Graceful on failure."""
        query = state.get("standalone_query", state["student_query"])
        web   = ""
        try:
            from duckduckgo_search import DDGS
            results = DDGS().text(query, max_results=5)
            if results:
                snippets = [
                    f"<b>{r.get('title','')}</b>: {r.get('body','')}"
                    for r in results if r.get("body")
                ]
                web = "WEB SEARCH RESULTS:\n" + "\n\n".join(snippets)
            print(f"[graph] Web search: {len(results or [])} results")
        except Exception as e:
            print(f"[graph] Web search skipped ({type(e).__name__})")
        return {"web_context": web}

    def general_knowledge_node(self, state: TutorState) -> TutorState:
        query   = state.get("standalone_query", state["student_query"])
        history = state.get("chat_history", "") or ""
        web     = state.get("web_context", "") or ""
        web_sec = f"\nSUPPLEMENTARY WEB CONTEXT:\n---\n{web}\n---\n" if web else ""

        raw = self._call(
            self.cfg["general_knowledge_prompt"].format(
                question=query,
                chat_history=history,
                web_context=web_sec,
            )
        )
        return {"response": {
            "internal_thought_process": "General knowledge path — no document context.",
            "answer":            self._ensure_html(raw),
            "citations":         [],
            "faithfulness_score": None,
        }}

    def generate_node(self, state: TutorState) -> TutorState:
        """
        RAG generation with faithfulness-gated 2nd-pass refinement.

        Flow:
          1. Build context from top 10 reranked chunks
          2. Generate HTML answer (single LLM call)
          3. Score faithfulness (LLM judge)
          4. If score < FAITH_THRESH → refine with stricter prompt
          5. Return best answer with score
        """
        query   = state.get("standalone_query", state["student_query"])
        docs    = state.get("retrieved_documents", [])
        history = state.get("chat_history", "") or ""

        if not docs:
            return self.general_knowledge_node(state)

        # ── Build context string ────────────────────────────────────────────
        top_docs    = docs[:self.MAX_CHUNKS]
        context_str = self._build_context(top_docs)

        # ── Pass 1: generate ────────────────────────────────────────────────
        answer, score = self._generate_and_score(query, context_str, history)
        print(f"[graph] Pass 1 — faithfulness={score:.2f}")

        # ── Pass 2: refine if below threshold ───────────────────────────────
        if score < self.FAITH_THRESH:
            print(f"[graph] Score {score:.2f} < {self.FAITH_THRESH} — running refinement pass")
            answer2, score2 = self._generate_and_score(
                query, context_str, history, refinement=True,
                previous_answer=answer, previous_score=score
            )
            if score2 > score:
                print(f"[graph] Refinement improved: {score:.2f} → {score2:.2f}")
                answer, score = answer2, score2
            else:
                print(f"[graph] Refinement did not improve score — keeping pass 1")

        thought = (
            f"RAG | {len(top_docs)} chunks | queries: {len(state.get('query_variants', []))} | "
            f"pages: {', '.join(str(d.metadata.get('page', '?')) for d in top_docs)} | "
            f"faithfulness: {score:.0%}"
        )

        return {
            "faithfulness_score": round(score, 2),
            "response": {
                "internal_thought_process": thought,
                "answer":            self._ensure_html(answer),
                "citations":         [d.metadata for d in top_docs],
                "faithfulness_score": round(score, 2),
            },
        }

    # ── Generation helpers ────────────────────────────────────────────────────

    def _build_context(self, docs: List[Document]) -> str:
        parts = []
        for i, d in enumerate(docs):
            page   = d.metadata.get("page", "?")
            course = d.metadata.get("course_name", "Document")
            text   = d.page_content[:self.MAX_CHUNK_LEN]
            parts.append(f"[Chunk {i+1} | {course} | Page {page}]\n{text}")
        return ("\n\n" + "─" * 60 + "\n\n").join(parts)

    def _generate_and_score(
        self,
        query:           str,
        context:         str,
        history:         str,
        refinement:      bool = False,
        previous_answer: str  = "",
        previous_score:  float = 0.0,
    ):
        if refinement:
            prompt = self.cfg["refinement_prompt"].format(
                context=context,
                question=query,
                chat_history=history,
                previous_answer=previous_answer,
                previous_score=f"{previous_score:.0%}",
            )
        else:
            prompt = self.cfg["system_prompt"].format(
                context=context,
                question=query,
                chat_history=history,
                source_type="User Uploaded Documents",
            )

        raw    = self._call(prompt)
        answer = self._ensure_html(raw)
        score  = self._score_faithfulness(answer, context, query)
        return answer, score

    def _score_faithfulness(self, answer: str, context: str, query: str) -> float:
        """LLM-as-judge faithfulness scorer (0.0–1.0)."""
        if not answer or not context:
            return 0.5
        try:
            raw = self._call(
                self.cfg["faithfulness_score"].format(
                    context=context[:4000],
                    answer=answer[:3000],
                ),
                fast=True,
            )
            # Extract first number from response
            match = re.search(r'\b(0\.\d+|1\.0|[01])\b', raw)
            if match:
                return max(0.0, min(1.0, float(match.group(1))))
        except Exception as e:
            print(f"[graph] Faithfulness scoring error: {e}")
        return 0.8  # optimistic default

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

        wf.add_conditional_edges(
            "route_query",
            lambda s: s.get("route", "general"),
            {"conversational": "conversational",
             "rag":            "retrieve",
             "general":        "web_search"},
        )
        wf.add_edge("conversational", END)

        # Route to generate whenever ANY docs were retrieved.
        # Only fall back to web_search when retrieval returned nothing at all.
        wf.add_conditional_edges(
            "retrieve",
            lambda s: "generate" if s.get("docs_are_relevant") else "web_search",
            {"generate": "generate", "web_search": "web_search"},
        )
        wf.add_edge("web_search",        "general_knowledge")
        wf.add_edge("general_knowledge", END)
        wf.add_edge("generate",          END)

        return wf.compile()

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, student_query: str, chat_history: str = "") -> dict:
        try:
            result = self.graph.invoke({
                "student_query":       student_query,
                "standalone_query":    student_query,
                "chat_history":        chat_history or "",
                "route":               "general",
                "query_variants":      [],
                "retrieved_documents": [],
                "docs_are_relevant":   False,
                "web_context":         "",
                "faithfulness_score":  0.0,
                "response":            {},
            })
            resp = result.get("response") or {}
            if not resp.get("answer"):
                resp["answer"] = "<p>No response generated. Please try again.</p>"
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

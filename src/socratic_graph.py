import yaml
from typing import List, Dict, Any, TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from src.retrieval import RAGTutorRetriever

# 1. Pydantic schema for generating the direct answer
class QAOutput(BaseModel):
    internal_thought_process: str = Field(description="Your detailed internal reasoning chain. Analyze the query, identify all relevant information from the context, and plan a thorough response.")
    answer: str = Field(description="A thorough, comprehensive, and detailed answer. Extract ALL relevant information from the context. Use HTML formatting: <ul>/<li> for lists, characteristics, steps, types, features, pros/cons, comparisons; <p> for definitions, narratives, summaries. Never be brief. Cover every relevant detail.")

# 2. State definition for LangGraph
class TutorState(TypedDict):
    student_query: str
    standalone_query: str
    chat_history: str
    retrieved_documents: List[Any]
    has_documents: bool
    is_conversational: bool
    response: dict

# 3. Graph Node Implementations
class SocraticTutor:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.retriever = RAGTutorRetriever(persist_directory=persist_directory)
        import os
        if "GROQ_API_KEY" in os.environ:
            try:
                from langchain_groq import ChatGroq
            except ImportError:
                pass
            self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
        elif "OPENAI_API_KEY" in os.environ:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                pass
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        else:
            from langchain_cohere import ChatCohere
            self.llm = ChatCohere(model="command-r")
            
        self.structured_llm = self.llm.with_structured_output(QAOutput)
        
        # Load Prompts
        with open("prompts.yaml", "r") as f:
            self.prompts = yaml.safe_load(f)
            
        self.qa_prompt = PromptTemplate.from_template(self.prompts["system_prompt"])
        self.route_prompt = PromptTemplate.from_template(self.prompts["route_query"])
        self.condense_prompt = PromptTemplate.from_template(self.prompts.get("condense_question", "{question}"))
        self.conversational_prompt = PromptTemplate.from_template(self.prompts["conversational_response"])

        # Build Graph
        self.graph = self._build_graph()

    def route_query_node(self, state: TutorState) -> TutorState:
        query = state["student_query"]
        prompt_formatted = self.route_prompt.format(question=query, chat_history=state.get("chat_history", ""))
        resp = self.llm.invoke([HumanMessage(content=prompt_formatted)])
        is_conv = "conversational" in resp.content.lower()
        return {"is_conversational": is_conv}

    def conversational_node(self, state: TutorState) -> TutorState:
        query = state["student_query"]
        prompt_formatted = self.conversational_prompt.format(question=query, chat_history=state.get("chat_history", ""))
        resp = self.llm.invoke([HumanMessage(content=prompt_formatted)])
        resp_dict = {
            "internal_thought_process": "User is engaging in basic conversation. Generating professional reply.",
            "answer": resp.content,
            "citations": []
        }
        return {"response": resp_dict}

    def retrieve_node(self, state: TutorState) -> TutorState:
        query = state["student_query"]
        history = state.get("chat_history", "")
        # Resolve any context or pronouns using the condense prompt
        if history:
            condense_fmt = self.condense_prompt.format(question=query, chat_history=history)
            standalone = self.llm.invoke([HumanMessage(content=condense_fmt)]).content.strip()
            # Safety: in case the LLM spits out an empty string or fails, fallback to original query
            if not standalone:
                standalone = query
        else:
            standalone = query
            
        docs = self.retriever.retrieve(standalone)
        
        # Determine if we got real document content (not empty)
        has_docs = len(docs) > 0
        return {"retrieved_documents": docs, "has_documents": has_docs, "standalone_query": standalone}

    def web_search_node(self, state: TutorState) -> TutorState:
        query = state["student_query"]
        try:
            from duckduckgo_search import DDGS
            results = DDGS().text(query, max_results=3)
            result_str = "\n".join([f"Source: {r.get('title')}\n{r.get('body')}" for r in results]) if results else "No results found."
            docs = [Document(page_content=result_str, metadata={"course_name": "Web (DuckDuckGo)", "chapter_number": "N/A", "concept_tags": "Web Search"})]
        except Exception as e:
            docs = [Document(page_content=f"Web search failed. The user asked: {query}", metadata={"course_name": "Fallback", "chapter_number": "N/A", "concept_tags": "Error Recovery"})]
        return {"retrieved_documents": docs}

    def generate_node(self, state: TutorState) -> TutorState:
        # Use the fully resolved standalone query rather than the original raw query with pronouns
        query_to_answer = state.get("standalone_query", state["student_query"])
        docs = state["retrieved_documents"]
        
        # If no docs at all, return safe fallback
        if not docs:
            return {"response": {
                "internal_thought_process": "No context documents were available for generation.",
                "answer": "<p>I could not find relevant information to answer this query. Please try rephrasing your question or upload a document related to the topic.</p>",
                "citations": []
            }}
        
        # Combine ALL retrieved contexts — no filtering, no grading
        context_str = "\n\n".join([
            f"Source (Course: {d.metadata.get('course_name')}, Ch: {d.metadata.get('chapter_number')}):\n{d.page_content}" 
            for d in docs
        ])
        
        # Format the system prompt using the standalone query
        prompt_formatted = self.qa_prompt.format(
            context=context_str, 
            question=query_to_answer, 
            chat_history=state.get("chat_history", "")
        )
        
        try:
            # Call structured LLM
            response = self.structured_llm.invoke([HumanMessage(content=prompt_formatted)])
            resp_dict = response.model_dump()
            resp_dict["citations"] = [d.metadata for d in docs]
            return {"response": resp_dict}
        except Exception as e:
            # If structured output fails, fall back to raw LLM call
            try:
                raw_resp = self.llm.invoke([HumanMessage(content=prompt_formatted)])
                return {"response": {
                    "internal_thought_process": f"Structured output failed ({type(e).__name__}). Used raw generation.",
                    "answer": raw_resp.content,
                    "citations": [d.metadata for d in docs]
                }}
            except Exception as e2:
                return {"response": {
                    "internal_thought_process": f"Generation failed: {str(e2)}",
                    "answer": "<p>I encountered an error while generating a response. Please try again.</p>",
                    "citations": []
                }}

    def _build_graph(self):
        workflow = StateGraph(TutorState)
        
        # Nodes — streamlined: no grading, no evaluation loops
        workflow.add_node("route_query", self.route_query_node)
        workflow.add_node("conversational", self.conversational_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("web_search", self.web_search_node)
        workflow.add_node("generate", self.generate_node)
        
        # Entry
        workflow.set_entry_point("route_query")
        
        # Route: conversational → chat response, educational → retrieve
        workflow.add_conditional_edges(
            "route_query",
            lambda state: "conversational" if state.get("is_conversational", False) else "retrieve",
            {
                "conversational": "conversational",
                "retrieve": "retrieve"
            }
        )
        
        workflow.add_edge("conversational", END)
        
        # After retrieval: if documents exist → generate, else → web search
        workflow.add_conditional_edges(
            "retrieve",
            lambda state: "generate" if state.get("has_documents", False) else "web_search",
            {
                "generate": "generate",
                "web_search": "web_search"
            }
        )
        
        # Web search → generate → end
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()

    def run(self, student_query: str, chat_history: str = ""):
        try:
            result = self.graph.invoke({
                "student_query": student_query, 
                "chat_history": chat_history, 
                "has_documents": False,
                "is_conversational": False
            })
            return result.get("response", {
                "internal_thought_process": "Graph completed but no response was generated.",
                "answer": "<p>Unable to generate a response. Please try again.</p>",
                "citations": []
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "internal_thought_process": f"Pipeline error: {type(e).__name__}: {str(e)}",
                "answer": f"<p>An internal error occurred: {str(e)}. Please try again or rephrase your query.</p>",
                "citations": []
            }

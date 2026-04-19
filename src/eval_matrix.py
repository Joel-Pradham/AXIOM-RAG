import time
import re
import json
import os
from dotenv import load_dotenv
load_dotenv()

from src.socratic_graph import SocraticTutor

def grade_answer(tutor, question, expected, actual):
    """
    LLM-as-a-judge to grade answer accuracy.
    """
    prompt = f"""
    You are an objective scorer. Grade the following answer against the ground truth reference.
    Return only a single float number between 0.0 and 1.0 representing the accuracy.
    0.0 means completely wrong, 1.0 means perfectly covers the ground truth.

    Question: {question}
    Expected Answer: {expected}
    Actual Answer: {actual}

    Score (0.0 to 1.0):"""
    
    try:
        raw = tutor._call(prompt, fast=True)
        # Extract the first float from the response
        match = re.search(r'\b(0\.\d+|1\.0|1|0)\b', raw)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"Failed to grade answer: {e}")
    return 0.0

def run_evaluation():
    # Pre-defined test cases
    # Note: For the "rag" expected routes to succeed, there must be a document ingested!
    # If the vectorstore is empty, they will auto-degrade to 'general'.
    eval_set = [
        {
            "query": "What is the capital of France?",
            "expected_route": "general", 
            "expected_answer": "Paris is the capital of France."
        },
        {
            "query": "Who is the CEO of Tesla?",
            "expected_route": "general",
            "expected_answer": "Elon Musk is the CEO of Tesla."
        },
        {
            "query": "Summarize the key points of the document about RAG architecture.",
            "expected_route": "rag", # Should trigger document routing if documents exist
            "expected_answer": "RAG architecture involves retrieving relevant document chunks and generating an answer based on them."
        },
        {
            "query": "What did the uploaded text say about chunking sizes?",
            "expected_route": "rag",
            "expected_answer": "Chunking sizes should be optimal for the embedding model to avoid truncation."
        }
    ]

    print("Initializing SocraticTutor pipeline for Evaluation...")
    tutor = SocraticTutor()
    doc_count = tutor.retriever.doc_count()
    print(f"Store currently has {doc_count} real document chunks.")
    
    if doc_count == 0:
        print("WARNING: Vectorstore is empty! RAG-specific queries will automatically degrade to general/web routes.")

    total_latency = 0
    correct_routes = 0
    total_hit_rate = 0
    total_accuracy = 0
    num_queries = len(eval_set)

    print("\nStarting Evaluation...")
    print("-" * 50)

    for i, item in enumerate(eval_set):
        query = item["query"]
        expected_route = item["expected_route"]
        expected_answer = item["expected_answer"]

        # Dynamically adjust expected route if no documents exist to avoid unfair penalties
        if doc_count == 0 and expected_route == "rag":
           expected_route = "general"

        start_time = time.time()
        
        # Invoke the graph directly to get visibility into internal routing and state variables
        state = {
            "student_query":       query,
            "standalone_query":    query,
            "chat_history":        "",
            "route":               "general",
            "query_variants":      [],
            "retrieved_documents": [],
            "docs_are_relevant":   False,
            "web_context":         "",
            "faithfulness_score":  0.0,
            "response":            {},
        }
        
        res = tutor.graph.invoke(state)
        
        latency_ms = (time.time() - start_time) * 1000
        total_latency += latency_ms

        actual_route = res.get("route")
        
        # Routing Accuracy Check
        if actual_route == expected_route:
            correct_routes += 1
            
        # Retrieval Hit Rate Check
        # We consider a "hit" if the grader reviewed the retrieved docs and marked them as relevant.
        # If it bypassed retrieval (route = general), we consider it N/A for RAG hit rate metrics, 
        # or we just record the raw hit. Here we will strictly track how often 'docs_are_relevant' flipped True.
        doc_hit = 1 if res.get("docs_are_relevant") else 0
        total_hit_rate += doc_hit

        answer = res.get("response", {}).get("answer", "")
        
        # Answer Accuracy Check (Groq Fast LLM as Judge)
        acc = grade_answer(tutor, query, expected_answer, answer)
        total_accuracy += acc
        
        print(f"[{i+1}/{num_queries}] Query: '{query}'")
        print(f"   Latency           : {latency_ms:.1f}ms")
        print(f"   Actual Route      : {actual_route} (Expected: {expected_route})")
        print(f"   Retrieval Hit     : {'Yes' if doc_hit else 'No'}")
        print(f"   Answer Accuracy   : {acc * 100:.1f}%\n")

    # Final Calculatons
    avg_latency = total_latency / num_queries
    routing_acc = (correct_routes / num_queries) * 100
    # Hit rate is tricky; if we only had 2 RAG expected queries, dividing by num_queries dilutes it.
    # We will compute hit rate against the number of times it ATTEMPTED to be a RAG route.
    rag_attempts = sum(1 for item in eval_set if (doc_count > 0 and item["expected_route"] == "rag"))
    
    avg_hit_rate = 0.0
    if rag_attempts > 0:
        avg_hit_rate = (total_hit_rate / rag_attempts) * 100
        
    avg_answer_acc = (total_accuracy / num_queries) * 100

    print("=" * 50)
    print("           FINAL EVALUATION MATRIX")
    print("=" * 50)
    print(f"Average Latency      : {avg_latency:.1f} ms")
    print(f"Routing Accuracy     : {routing_acc:.1f} %")
    print(f"Retrieval Hit Rate   : {avg_hit_rate:.1f} %  (out of RAG queries)")
    print(f"Answer Accuracy      : {avg_answer_acc:.1f} %")
    print("=" * 50)

if __name__ == "__main__":
    run_evaluation()

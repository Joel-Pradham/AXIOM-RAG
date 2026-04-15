import os
import json
import logging
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# Suppress debug logs
logging.basicConfig(level=logging.WARNING)

def build_dummy_dataset() -> Dataset:
    """
    Constructs a dummy golden dataset.
    In a real CI pipeline, this should be loaded from a JSONL or CSV file 
    containing verified human question/answer pairs.
    """
    data = {
        "question": [
            "What is chunking in Retrieval-Augmented Generation?"
        ],
        "ground_truth": [
            "Chunking is the process of breaking down large documents into smaller, manageable pieces to improve retrieval accuracy."
        ],
        "contexts": [
            [
                "In Retrieval-Augmented Generation, the text needs to be processed before embedding. Chunking involves breaking down large documents into smaller passages, usually matching the model's context window."
            ]
        ],
        "answer": [
            "To answer your question, you should think about how large documents are broken down. Chunking is dividing huge texts into smaller passages before they are vectorized."
        ]
    }
    return Dataset.from_dict(data)

def run_evaluation():
    try:
        eval_dataset = build_dummy_dataset()
        
        if "OPENAI_API_KEY" not in os.environ and "COHERE_API_KEY" not in os.environ:
            print("WARNING: Neither OPENAI_API_KEY nor COHERE_API_KEY found. Ragas might fail if no LLM provider is explicitly configured.")
            
        print("Starting Ragas Evaluation Pipeline...")
        # Metrics defined in the objective
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision
        ]
        
        # Execute Evaluation
        # Note: If no custom LLM is passed, Ragas relies on OpenAI defaults
        result = evaluate(
            eval_dataset,
            metrics=metrics
        )
        
        # Convert to dictionary and print for the bash parser
        scores = {
            "faithfulness": result.get("faithfulness", 0.0),
            "answer_relevancy": result.get("answer_relevancy", 0.0),
            "context_precision": result.get("context_precision", 0.0)
        }
        
        print("--- RAGAS RESULTS ---")
        print(json.dumps(scores, indent=2))
        
    except Exception as e:
        print(f"Evaluation Failed: {e}")
        exit(1)

if __name__ == "__main__":
    run_evaluation()

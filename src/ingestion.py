import os
import glob
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document

class DataIngestionPipeline:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        import os
        if "OPENAI_API_KEY" in os.environ:
            try:
                from langchain_openai import OpenAIEmbeddings
            except ImportError:
                raise ImportError("Please run `pip install langchain-openai` to use OpenAI embeddings.")
            self.embeddings = OpenAIEmbeddings()
        elif "COHERE_API_KEY" in os.environ:
            from langchain_cohere import CohereEmbeddings
            self.embeddings = CohereEmbeddings(model="embed-english-v3.0")
        else:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                raise ImportError("Please run `pip install langchain-huggingface sentence-transformers`")
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize chroma db
        self.vectorstore = Chroma(
            collection_name="educational_material",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

        # Config splitter — larger chunks preserve more context per retrieval hit
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " "]
        )

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Sanitize raw PDF-extracted text.
        PyPDF preserves layout line-breaks as literal newlines, fragmenting
        words across lines (e.g. 'the\\n \\ndata' instead of 'the data').
        This collapses that noise back into readable sentences.
        """
        import re
        # Collapse sequences of whitespace/newlines between words into single spaces
        text = re.sub(r'\s*\n\s*', ' ', text)
        # Collapse multiple spaces into one
        text = re.sub(r' {2,}', ' ', text)
        # Remove non-printable characters except common ones
        text = re.sub(r'[^\x20-\x7E\n\t\u00A0-\u024F]', '', text)
        return text.strip()

    def load_documents(self, file_path: str) -> List[Document]:
        """Loads a single document depending on its extension."""
        if file_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            return loader.load()
        elif file_path.lower().endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path)
            return loader.load()
        elif file_path.lower().endswith(".txt"):
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(file_path, autodetect_encoding=True)
            return loader.load()
        else:
            print(f"Unsupported file format: {file_path}")
            return []

    def ingest_file(self, file_path: str, course_name: str, chapter_number: int, concept_tags: List[str]):
        """Ingests a single document file."""
        docs = self.load_documents(file_path)
        if not docs:
            return 0
        
        # Clean extracted text BEFORE chunking    
        for doc in docs:
            doc.page_content = self.clean_text(doc.page_content)
            doc.metadata["course_name"] = course_name
            doc.metadata["chapter_number"] = chapter_number
            doc.metadata["concept_tags"] = ", ".join(concept_tags)
            
        chunks = self.text_splitter.split_documents(docs)
        
        # Filter out garbage fragments (headers, footers, page numbers)
        chunks = [c for c in chunks if len(c.page_content.strip()) >= 100]
        
        if chunks:
            self.vectorstore.add_documents(chunks)
        return len(chunks)

    def ingest_directory(self, directory_path: str, course_name: str, chapter_number: int, concept_tags: List[str]):
        """Ingests all valid text documents from a directory and applies common metadata."""
        all_docs = []
        files = glob.glob(os.path.join(directory_path, "**", "*.*"), recursive=True)
        
        for file in files:
            docs = self.load_documents(file)
            for doc in docs:
                # Assign the requested metadata signature 
                doc.metadata["course_name"] = course_name
                doc.metadata["chapter_number"] = chapter_number
                doc.metadata["concept_tags"] = ", ".join(concept_tags) # chroma metadata values must be primitive
            all_docs.extend(docs)

        if not all_docs:
            print("No documents found to ingest.")
            return []

        # Chunk the docs
        chunks = self.text_splitter.split_documents(all_docs)
        print(f"Generated {len(chunks)} chunks from {len(all_docs)} pages/sections.")

        # Batch upload to Chroma
        self.vectorstore.add_documents(chunks)
        print("Ingestion complete. Changes persisted to ChromaDB.")
        return chunks

if __name__ == "__main__":
    # Example usage:
    # Ensure COHERE_API_KEY is in your environment variables
    # os.environ["COHERE_API_KEY"] = "your-api-key"
    
    # os.makedirs("./data", exist_ok=True)
    # with open("./data/sample.md", "w") as f:
    #     f.write("Welcome to the first chapter on RAG. RAG stands for Retrieval-Augmented Generation.")
        
    print("Ingestion script ready.")
    # pipeline = DataIngestionPipeline()
    # pipeline.ingest_directory("./data", "Intro to AI", 1, ["RAG", "LLMs", "Vector DBs"])

"""LLM Generation Module

Handles prompt templating and answer generation with context.
"""

import logging
import time
from typing import Dict, List

from langchain_core.prompts import PromptTemplate
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGGenerator:
    """
    Generates answers using retrieved context and an LLM.

    Features:
    - Prompt templating to reduce hallucination
    - OpenAI API integration
    - Latency and source tracking
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        base_url: str | None = None,
    ):
        """
        Initialize generator with model and parameters.
        
        Args:
            model: OpenAI model name (or Ollama model like 'llama3.2')
            temperature: Sampling temperature (0-1)
            base_url: Optional custom API base URL (for Ollama, LM Studio, etc.)
        """
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(base_url=base_url) if base_url else OpenAI()

        # Define RAG prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "query"],
            template="""You are a helpful assistant answering questions based on provided documents.

Context from documents:
{context}

Question: {query}

Answer based on the context above. If the context doesn't contain relevant information, say so.""",
        )

    def generate(self, query: str, context: List[str]) -> str:
        """
        Generate an answer using context.

        Args:
            query: User query
            context: List of context strings from retrieval

        Returns:
            Generated answer string
        """
        start = time.time()

        # Format context
        context_str = "\n\n".join([f"- {c}" for c in context])

        # Format prompt
        prompt = self.prompt_template.format(context=context_str, query=query)

        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )

        answer = response.choices[0].message.content
        elapsed = time.time() - start

        logger.info(f"Generated answer in {elapsed:.2f}s")

        return answer

    def rag_answer(self, query: str, retriever) -> Dict:
        """
        Full RAG pipeline: retrieve → generate → return.

        Args:
            query: User query
            retriever: RAGRetriever instance

        Returns:
            Dict with answer, sources, and latency
        """
        start = time.time()

        # Retrieve
        retrieved = retriever.retrieve(query, top_k=3)

        if not retrieved:
            return {
                "answer": "No relevant documents found.",
                "sources": [],
                "latency_ms": (time.time() - start) * 1000,
            }

        # Extract context
        context = [doc["text"] for doc in retrieved]
        sources = [doc["source"] for doc in retrieved]

        # Generate
        answer = self.generate(query, context)

        elapsed = (time.time() - start) * 1000

        return {
            "answer": answer,
            "sources": sources,
            "latency_ms": elapsed,
        }


# Example usage
if __name__ == "__main__":
    from ingestion import DocumentIngester
    from retrieval import RAGRetriever

    # Setup
    ingester = DocumentIngester()
    documents = ingester.ingest("data/sample.txt")

    retriever = RAGRetriever()
    retriever.index(documents)

    generator = RAGGenerator()

    # Generate answer
    query = "What is RAG and why is it useful?"
    result = generator.rag_answer(query, retriever)

    print(f"\nQuery: {query}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print(f"Latency: {result['latency_ms']:.0f}ms")

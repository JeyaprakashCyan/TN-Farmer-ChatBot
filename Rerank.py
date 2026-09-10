import argparse
import json
import os
import sys
from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI
from typing import List, Dict

from downloaded_schemes.IndexFix import Neo4jGraphManager

class AdvancedRAGPipeline:
    def __init__(self, neo4j_mgr, openai_api_key):
        self.neo4j = neo4j_mgr
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0.0)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def route_query(self, user_query: str) -> str:
        prompt = f"""
        Classify the intent of this user query:
        '{user_query}'
        
        Options:
        - GREETING: Conversational greetings (e.g., 'Hi', 'Hello', 'Vanakkam').
        - AGRI_SCHEME: Questions regarding Tamil Nadu government schemes, agriculture, crop insurance, subsidies, equipment, or benefits.
        - OUT_OF_SCOPE: Topics completely unrelated to farming or government schemes (e.g., 'How is the weather?', sports, entertainment).

        Return ONLY ONE word corresponding to the option.
        """
        return self.llm.invoke(prompt).content.strip()

    def rewrite_query(self, query: str, history: str) -> str:
        prompt = f"""
        Given the conversation context:
        {history}
        
        Rewrite the query to be self-contained for searching Tamil Nadu agriculture scheme documents:
        Query: {query}
        """
        return self.llm.invoke(prompt).content.strip()

    def retrieve_and_rerank(self, query: str, top_k: int = 5) -> List[Dict]:
        query_vec = self.neo4j.embedder.embed_query(query)
        cypher = """
        CALL db.index.vector.queryNodes('scheme_chunks', $k, $embedding)
        YIELD node, score
        MATCH (s:Scheme)-[:HAS_CHUNK]->(node)
        RETURN node.text AS text, node.id AS id, s.title AS scheme_title, s.url AS url, score
        """
        with self.neo4j.driver.session() as session:
            result = session.run(cypher, k=top_k, embedding=query_vec)
            docs = [dict(rec) for rec in result]
            
        if not docs:
            return []
            
        # Re-ranking using CrossEncoder
        pairs = [[query, doc["text"]] for doc in docs]
        scores = self.reranker.predict(pairs)
        for idx, doc in enumerate(docs):
            doc["rerank_score"] = float(scores[idx])
            
        return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Retrieve and rerank Tamil Nadu agriculture schemes.")
    parser.add_argument(
        "query",
        nargs="?",
        help="Question or search query to retrieve. If omitted, you will be prompted.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve.")
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")

    query = args.query or input("Enter your question: ").strip()
    if not query:
        parser.error("A question is required")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    manager = Neo4jGraphManager.from_environment()
    pipeline = AdvancedRAGPipeline(manager, openai_api_key)
    try:
        manager.verify_connectivity()
        results = pipeline.retrieve_and_rerank(query, top_k=args.top_k)
        print(json.dumps(results, indent=2, ensure_ascii=False))
    finally:
        manager.close()


if __name__ == "__main__":
    main()
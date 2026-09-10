import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import OpenAIEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Neo4jGraphManager:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        openai_api_key: str,
        model: str,
        trust_all_certs: bool = False,
    ):
        if trust_all_certs and uri.startswith("neo4j+s://"):
            uri = uri.replace("neo4j+s://", "neo4j+ssc://", 1)
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.embedder = OpenAIEmbeddings(api_key=openai_api_key, model=model)

    @classmethod
    def from_environment(cls):
        required = {
            "NEO4J_URI": os.getenv("NEO4J_URI"),
            "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME"),
            "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(
                f"Missing required settings in {PROJECT_ROOT / '.env'}: {', '.join(missing)}"
            )

        trust_all_certs = os.getenv("NEO4J_TRUST_ALL_CERTS", "false").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        return cls(
            uri=required["NEO4J_URI"],
            user=required["NEO4J_USERNAME"],
            password=required["NEO4J_PASSWORD"],
            openai_api_key=required["OPENAI_API_KEY"],
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            trust_all_certs=trust_all_certs,
        )

    def verify_connectivity(self) -> None:
        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()

    def init_schema(self):
        with self.driver.session() as session:
            session.run("""
                CREATE VECTOR INDEX scheme_chunks IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {indexConfig: {
                 `vector.dimensions`: 1536,
                 `vector.similarity_function`: 'cosine'
                }}
            """)

    def ingest_chunks(self, chunks: list):
        self.init_schema()
        cypher = """
        MERGE (s:Scheme {title: $scheme_title})
        ON CREATE SET s.url = $url, s.department = $department, s.state = $state
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text, c.embedding = $embedding
        MERGE (s)-[:HAS_CHUNK]->(c)
        """
        with self.driver.session() as session:
            for chunk in chunks:
                embedding = self.embedder.embed_query(chunk["text"])
                session.run(
                    cypher,
                    scheme_title=chunk["scheme_title"],
                    url=chunk["url"],
                    department=chunk["department"],
                    state=chunk["state"],
                    chunk_id=chunk["chunk_id"],
                    text=chunk["text"],
                    embedding=embedding
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Index extracted scheme chunks in Neo4j.")
    parser.add_argument(
        "input",
        nargs="?",
        default=str(PROJECT_ROOT / "raw_scheme_data" / "extracted_chunks.json"),
        help="Path to the extracted chunks JSON file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        parser.error(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as input_file:
        chunks = json.load(input_file)
    if not isinstance(chunks, list):
        parser.error("Input JSON must contain a list of chunks")

    manager = Neo4jGraphManager.from_environment()
    try:
        manager.verify_connectivity()
        manager.ingest_chunks(chunks)
        print(f"Indexed {len(chunks)} chunks in Neo4j.")
    finally:
        manager.close()


if __name__ == "__main__":
    main()
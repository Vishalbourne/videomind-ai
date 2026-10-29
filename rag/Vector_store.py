from pathlib import Path
from langchain_chroma import Chroma
from rag.Chunk_text import splitter_text
from rag.Embeddings import get_embedding

ROOT_PATH = Path(__file__).resolve().parent.parent

DIRECTORY = ROOT_PATH / "database"
DIRECTORY.mkdir(exist_ok=True)

print("Database directory:", DIRECTORY)


def build_vector_database():
    try:
        print("Building vector database...")

        embeddings = get_embedding()

        vector_store = Chroma(
            collection_name="youtube_transcripts_embedding",
            embedding_function=embeddings,
            persist_directory=str(DIRECTORY),
        )

        print("Vector database created successfully")
        return vector_store

    except Exception as e:
        print("Could not build vector database:", e)


def add_to_vector(path):
    """Split the transcript and save the chunks."""
    try:
        print("Adding transcript to vector database:", path)

        vector_store = build_vector_database()

        if vector_store is None:
            print("Vector database initialization failed")
            return None

        chunks = splitter_text(path=path)

        if not chunks:
            print("No chunks to add")
            return None

        print("Total chunks:", len(chunks))

        ids = [f"{Path(path).stem}_{i}" for i in range(len(chunks))]

        vector_store.add_documents(chunks, ids=ids)

        print("Chunks added successfully")
        return vector_store

    except Exception as e:
        print("Could not add chunks:", e)


def retriever_chunk(k=2):
    """Return a retriever that finds the k most similar chunks."""
    try:
        print("Creating retriever...")

        vector_store = build_vector_database()

        if vector_store is None:
            print("Vector database is unavailable")
            return None

        retriever = vector_store.as_retriever(
            search_kwargs={"k": k}
        )

        print(f"Retriever created successfully. Top k: {k}")
        return retriever

    except Exception as e:
        print("Could not create retriever:", e)
        return None
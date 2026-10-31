from pathlib import Path
from langchain_chroma import Chroma

from rag.Chunk_text import splitter_text
from rag.Embeddings import get_embedding


# Database location
DB_PATH = Path(__file__).resolve().parent.parent / "database"

# Collection name
COLLECTION_NAME = "youtube_transcripts_embedding"


# Connect to ChromaDB
def get_vector_store():

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding(),
        persist_directory=str(DB_PATH)
    )


# Store transcript chunks
def add_to_vector(path, video_id):

    vector_store = get_vector_store()

    # Split transcript into chunks
    chunks = splitter_text(path)

    # Add video information to every chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata["video_id"] = video_id
        chunk.metadata["chunk_index"] = i

    # Store chunks in ChromaDB
    ids = [f"{video_id}_{i}" for i in range(len(chunks))]

    vector_store.add_documents(
        documents=chunks,
        ids=ids
    )

    print("Transcript stored successfully!")

    return vector_store


# Retrieve chunks from a particular video
def retriever_chunk(video_id, k=4):

    vector_store = get_vector_store()

    return vector_store.as_retriever(
        search_kwargs={
            "k": k,
            "filter": {"video_id": video_id}
        }
    )

def video_exists(video_id):

    vector_store = get_vector_store()

    result = vector_store.get(
        where={"video_id": video_id},
        limit=1
    )

    return len(result["ids"]) > 0

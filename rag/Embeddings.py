from langchain_ollama import OllamaEmbeddings


def get_embedding(dimensions: int = 512):
    try:
        print("Loading embedding model...")

        embeddings = OllamaEmbeddings(
            model="embeddinggemma-2:270m",
            dimensions=dimensions,
        )

        print("Embedding model loaded successfully")
        print("Dimensions:", dimensions)

        return embeddings

    except Exception as e:
        print("Could not load embeddings:", e)
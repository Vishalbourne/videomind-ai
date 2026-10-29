from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.Document_loader import doc_loader


def load_splitter():
    try:
        print("Creating text splitter...")
        return RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
    except ValueError as e:
        print("Splitter error:", e)


def splitter_text(path):
    try:
        print("Loading transcript:", path)
        documents = doc_loader(path=path)

        if not documents or not documents[0].page_content.strip():
            print("Transcript is empty")
            return []

        print("Documents loaded:", len(documents))

        splitter = load_splitter()
        chunks = splitter.split_documents(documents)

        print("Total chunks created:", len(chunks))

        return chunks

    except Exception as e:
        print("Error while splitting:", e)
        return []
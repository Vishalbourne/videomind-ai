from langchain_community.document_loaders import TextLoader


def doc_loader(path):
    try:
        print("Loading file:", path)

        loader = TextLoader(file_path=path, encoding="utf-8")
        docs = loader.load()

        print("File loaded successfully")
        print("Total documents:", len(docs))

        return docs

    except FileNotFoundError:
        print("File not found:", path)

    except Exception as e:
        print("Error loading file:", e)


if __name__ == "__main__":
    path = r"E:\video ai agent\transcripts\Human in the Loop Is Not a Governance Strategy for Agentic AI.txt"

    docs = doc_loader(path)

    if docs:
        print(docs[0].page_content[:300])
        print(docs[0].metadata)
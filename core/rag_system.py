
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_ollama import ChatOllama

from rag.Vector_store import retriever_chunk


def format_docs(docs):
    """Combine retrieved document chunks into one context."""
    return "\n\n".join(
        doc.page_content for doc in docs
    )


def chain_define():

    try:
        print("Creating RAG chain...")

        # Get retriever from vector database
        retriever = retriever_chunk(k=4)

        if retriever is None:
            raise RuntimeError("Could not create retriever")

        # Initialize local Ollama model
        print("Loading Mistral model...")

        model = ChatOllama(
            model="mistral:latest",
            temperature=0
        )

        # Define prompt
        prompt = ChatPromptTemplate.from_template("""
You are VideoMind AI, an intelligent video assistant.

Answer the user's question using only the provided video context.

Rules:
1. Give accurate and relevant answers.
2. Do not make up information.
3. If the answer is not available in the context, say:
   "I couldn't find that information in the video."
4. Keep answers clear and simple.

Video Context:
{context}

User Question:
{question}

Answer:
""")

        # Build RAG chain
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | model
            | StrOutputParser()
        )

        print("RAG chain created successfully!")

        return rag_chain

    except Exception as e:
        print("Could not create RAG chain:", e)
        raise


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_ollama import ChatOllama

from rag.Vector_store import retriever_chunk


# Combine retrieved chunks into one text
def format_docs(docs):

    return "\n\n".join(
        doc.page_content for doc in docs
    )


# Create RAG chain
def chain_define(video_id):

    print("Creating RAG chain...")

    # Get chunks from the selected video
    retriever = retriever_chunk(
        video_id=video_id,
        k=4
    )

    # Load Mistral model
    model = ChatOllama(
        model="mistral:latest",
        temperature=0
    )

    # Create prompt
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

    # Build RAG pipeline
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

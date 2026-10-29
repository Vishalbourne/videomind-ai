from core.rag_system import chain_define
from core.transcribe import transcribe_video
from rag.Vector_store import add_to_vector


def make_transcript(url,break_min=2):
    texts, outpath = transcribe_video(url=url,chunk_min=break_min)
    return outpath


def rag_engine(url, question):
    outpath = make_transcript(url=url)

    add_to_vector(path=outpath)

    chain = chain_define()

    for chunk in chain.stream(question):
        print(chunk, end="", flush=True)

    print()


if __name__ == "__main__":
    # url = input("Enter YouTube URL: ")
    url="https://www.youtube.com/watch?v=UgEA5ictENM"
    question="How to Avoid These Slowdowns in Compiled Models"

    # Use the provided path when saving the transcript.
    # This assumes transcribe_video supports an output-path argument.
    # question = input("Enter your question about the video: ")

    rag_engine(url=url, question=question)
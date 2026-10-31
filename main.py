from core.rag_system import chain_define
from core.transcribe import transcribe_video
from rag.Vector_store import add_to_vector, video_exists



def get_video_id(url):

    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]

    return url.split("v=")[1].split("&")[0]


# Process video
def rag_engine(url, question):

    video_id = get_video_id(url)

    if not video_id:
        raise ValueError("Invalid YouTube URL")

    # Check whether video already exists
    if video_exists(video_id):

        print("Video already exists!")
        print("Using existing vector database...")

    else:

        print("New video detected!")
        print("Starting transcription...")

        texts, outpath = transcribe_video(
            url=url,
            chunk_min=2
        )

        add_to_vector(
            path=outpath,
            video_id=video_id
        )

    # Create RAG chain
    chain = chain_define(video_id)

    print("\nAnswer:")

    for chunk in chain.stream(question):
        print(chunk, end="", flush=True)

    print()


if __name__ == "__main__":

    url = "https://www.youtube.com/watch?v=UgEA5ictENM"

    question = "How to Avoid These Slowdowns in Compiled Models"

    rag_engine(url=url, question=question)
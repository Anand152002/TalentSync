import openai
from config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL
from langchain.text_splitter import RecursiveCharacterTextSplitter

openai.api_key = OPENAI_API_KEY

def chunk_text(text, max_length=5000):
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_length, chunk_overlap=100)
    return splitter.split_text(text)

def generate_openai_embeddings(text_chunks):
    embeddings = []
    for chunk in text_chunks:
        try:
            response = openai.Embedding.create(
                input=chunk,
                model=OPENAI_EMBEDDING_MODEL
            )
            embedding = response['data'][0]['embeddings']
            embeddings.append(embedding)
        except Exception as e:
            print(f"ERROR: Failed to process chunk '{chunk}'. Reason: {e}")
            continue
    return embeddings

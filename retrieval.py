import deeplake
import openai
import numpy as np
import os
from config import OPENAI_API_KEY, DEEPLAKE_DATASET_PATH, ACTIVELOOP_TOKEN
from embedding_generator import generate_openai_embeddings

# Set OpenAI API Key
openai.api_key = OPENAI_API_KEY
os.environ['ACTIVELOOP_TOKEN'] = ACTIVELOOP_TOKEN

# Function to compute cosine similarity
def cosine_similarity(embedding1, embeddings_matrix):
    embedding1 = np.array(embedding1)
    embeddings_matrix = np.array(embeddings_matrix)

    embedding1_norm = embedding1 / np.linalg.norm(embedding1)
    embeddings_matrix_norm = embeddings_matrix / np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)

    return np.dot(embeddings_matrix_norm, embedding1_norm)

# Function to retrieve the most relevant resumes based on a query
# Function to retrieve the most relevant resumes based on a query
def retrieve_resumes(query, dataset_path, top_k=5):
    try:
        # Load the dataset
        ds = deeplake.load(dataset_path)
        if ds is None:
            return []

        # Generate embeddings for the query
        query_embedding = generate_openai_embeddings([query])[0]
        if query_embedding is None:
            return []

        # Retrieve all embeddings and metadata
        embeddings = ds['embedding'][:]
        metadatas = ds['metadata'][:]

        embeddings = embeddings.numpy() if hasattr(embeddings, 'numpy') else np.array(embeddings)

        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, embeddings)

        # Get indices of the top_k most similar embeddings
        top_k_indices = similarities.argsort()[-top_k:][::-1]

        # Retrieve top_k resumes
        top_k_resumes = []
        for i in top_k_indices:
            try:
                index = int(i)  
                # Safely retrieve metadata
                metadata_entry = metadatas[index].numpy() if hasattr(metadatas[index], 'numpy') else metadatas[index]
                metadata_entry = metadata_entry.item() if isinstance(metadata_entry, np.ndarray) else metadata_entry

                if isinstance(metadata_entry, dict):
                    resume_data = {
                        "file_name": metadata_entry.get("file_name", "Unknown"),
                        "candidate_name": metadata_entry.get("candidate_name", "Unknown"),
                        "skills": metadata_entry.get("skills", []),
                        "years_of_experience": metadata_entry.get("years_of_experience", "N/A")
                    }
                    top_k_resumes.append(resume_data)
            except Exception as e:
                print(f"Skipping index {i} due to error: {e}")

        return top_k_resumes

    except Exception as e:
        print(f"Error retrieving resumes: {e}")
        return []



if __name__ == "__main__":
    query = input("Please enter your query (e.g., desired skills or job requirements): ")
    top_k_resumes = retrieve_resumes(query, DEEPLAKE_DATASET_PATH, top_k=5)
    
    if top_k_resumes:
        print("Top matching resumes:")
        for resume in top_k_resumes:
            print(resume)
    else:
        print("No relevant resumes found.")

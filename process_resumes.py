import os
from pdf_reader import read_pdf
from embedding_generator import chunk_text, generate_openai_embeddings
from llm_query import ask_llm_for_experience_and_graduation_year
from deeplake_storage import store_in_deeplake
from config import RESUMES_FOLDER_PATH, REQUIRED_SKILLS, DEEPLAKE_DATASET_PATH, ACTIVELOOP_TOKEN

os.environ['ACTIVELOOP_TOKEN'] = ACTIVELOOP_TOKEN

def find_matching_skills(text, required_skills):
    # Convert text to lowercase to ensure case-insensitive matching
    lower_text = text.lower()
    
    # Check which skills are present in the resume text
    matching_skills = [skill for skill in required_skills if skill.lower() in lower_text]
    
    return matching_skills

def process_resumes_in_folder(folder_path=RESUMES_FOLDER_PATH, required_skills=REQUIRED_SKILLS, deeplake_dataset_path=DEEPLAKE_DATASET_PATH):
    all_embeddings, all_metadatas = [], []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        if file_name.endswith('.pdf'):
            text = read_pdf(file_path)
            if text:
                # Find skills that match with the resume text
                matching_skills = find_matching_skills(text, required_skills)
                
                # Always chunk the text and generate embeddings
                chunks = chunk_text(text)
                embeddings = generate_openai_embeddings(chunks)
                
                # Get the experience and graduation year from the LLM
                llm_response = ask_llm_for_experience_and_graduation_year(text)
                years_of_experience = llm_response.get("years_of_experience", "N/A")  # Default to "N/A" if not found
                
                # Create metadata regardless of matching skills
                metadata = {
                    "file_name": file_name,
                    "skills": matching_skills,  # Include matching skills, may be empty
                    "years_of_experience": years_of_experience
                }
                
                # Extend the lists with embeddings and metadata for this resume
                all_embeddings.extend(embeddings)
                all_metadatas.append(metadata)

    # Store the embeddings and metadata in Deep Lake if there are any
    if all_embeddings and all_metadatas:
        store_in_deeplake(all_embeddings, all_metadatas, deeplake_dataset_path, overwrite=True)
        print("Stored the embeddings and metadata in Deep Lake.")
    else:
        print("No valid documents found or no embeddings generated.")

if __name__ == "__main__":
    process_resumes_in_folder()

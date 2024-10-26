from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
from retrieval import retrieve_resumes  # Import your retrieve_resumes function

# Define the FastAPI app
app = FastAPI()

# Define a request model for the query and required skills
class QueryRequest(BaseModel):
    query: str
    required_skills: List[str]
    top_k: int = 5  # Default value for top_k

# Define a response model for structured resume data
class ResumeResponse(BaseModel):
    file_name: str
    candidate_name: Optional[str]  # Use Optional for fields that may not be present
    skills: List[str]
    years_of_experience: Optional[Union[str, int]]  # Union to accept both string and int

# Create a route for resume retrieval
@app.post("/retrieve_resumes/", response_model=List[ResumeResponse])
async def get_resumes(request: QueryRequest):
    try:
        dataset_path = "hub://smruthisumanthrao/text_embed"  # Specify your dataset path
        # Retrieve resumes based on the query and required skills
        results = retrieve_resumes(request.query, dataset_path, top_k=request.top_k)
        
        # Filter results based on required skills provided by the user
        filtered_resumes = []
        for resume in results:
            # Check if the resume has all the required skills
            if all(skill in resume['skills'] for skill in request.required_skills):
                filtered_resumes.append(resume)

        return filtered_resumes
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

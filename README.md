# Talentsync
This is a Generative AI-based solution designed to rank resumes based on a job description using vector similarity search. It leverages Deep Lake as the vector database to store resume embeddings along with metadata, such as years of experience, for efficient filtering. Metadata is extracted from resumes using a large language model (LLM), enhancing ranking and filtering based on relevant criteria.


## Workflow
1) Resume and Job Description Input

Users upload a folder of resumes and a job description (as text or file).
```pdf_reader.py ```extracts text for processing.

2) Embedding Generation

``` embedding_generator.py``` generates vector embeddings for both resumes and job descriptions, capturing semantic content.

3) Metadata Extraction

``` llm_query.py``` uses an LLM to extract metadata (e.g., years of experience) from resumes, stored in Deep Lake for filtering.

4) Vector Similarity Search

```deeplake_storage.py``` ranks resumes based on similarity to the job description in vector space.

5) Efficient Filtering

Metadata filtering (like years of experience) is applied before similarity search for targeted results.

6) Result Presentation

Ranked resumes are displayed based on their relevance to the job description.

## Getting Started
1) Clone the repository:

 ```bash

https://github.com/Anand152002/Talentsync.git
 ```

2) Install dependencies:
 ```bash
pip install -r requirements.txt
 ```
3) Set Up Deep Lake and API Keys Configuration:

Configure your Deep Lake account  and storage by adding the credentials in ```config.py```

4) Running the App
 ```bash

python main.py
 ```


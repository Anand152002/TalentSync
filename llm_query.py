import openai
import json
import time
from config import OPENAI_API_KEY, CURRENT_YEAR, RETRY_COUNT, INITIAL_DELAY

openai.api_key = OPENAI_API_KEY

def is_valid_json(response_text):
    try:
        json.loads(response_text)
        return True
    except json.JSONDecodeError:
        return False

def ask_llm_for_experience_and_graduation_year(text, retries=RETRY_COUNT, delay=INITIAL_DELAY):
    prompt = f"""
    Based on the following resume text, extract the graduation year and/or the first year the person started working.
    Use the most relevant year to calculate the total years of experience (as of the current year: {CURRENT_YEAR}). 

    Respond in JSON format with the following field:
    - "years_of_experience": Total years of work experience based on the context of the resume (e.g., graduation year or work start year).

    Resume text:
    {text}
    """
    
    attempt = 0
    while attempt < retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in resume parsing."},
                    {"role": "user", "content": prompt}
                ]
            )

            content = response['choices'][0]['message']['content'].strip()
            print(f"Raw LLM Response: {content}")

            if is_valid_json(content):
                return json.loads(content)
            else:
                print("Invalid JSON format in LLM response.")
        except openai.error.RateLimitError:
            print(f"Rate limit hit, retrying in {delay} seconds...")
            time.sleep(delay)
            attempt += 1
            delay *= 2
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return {"years_of_experience": "N/A"}

    print("Exceeded maximum retries. Failed to process.")
    return {"years_of_experience": "N/A"}

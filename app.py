import json
from pydantic import BaseModel
from typing import List, Union
import ast
from process_doc import extract_text_from_pdf
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, UploadFile, File
import shutil
import psycopg2


GROQ_API_KEY = "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"

groq_client = Groq(
    api_key=GROQ_API_KEY,
)


sys_instruct = """You are a CV parser. You take input text from a cv in the
                user prompt and extract the following fields:
                 name, job title of the person on the CV, email, phone,
                 address, education, experience, skills, languages,
                 certifications, and projects. If there is any field not
                 found in the text, you should write Not provided, do not
                 leave it blank. for education, experience, skills,
                 languages and certifications make  Return a json format
                 of the extracted fields."""


# Data model for LLM to generate
class CvDetails(BaseModel):
    name: str
    job_title: str
    email: str
    phone: str
    address: str
    education: Union[str, List[str]]
    experience: Union[str, List[str]]
    skills: Union[str, List[str]]
    languages: Union[str, List[str]]
    certifications: Union[str, List[str]]


def get_cv_details(text: str) -> CvDetails:
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": sys_instruct + f"the cv text is: {text}" +
                # Pass the json schema to the model.
                f" The JSON object must use the schema: {json.dumps(
                    CvDetails.model_json_schema(), indent=2)}",
            },
            {
                "role": "user",
                "content": """extract the details from the cv text and
                export the result in the specified json format""",
            },
        ],
        model="deepseek-r1-distill-llama-70b",
        temperature=0,
        # Streaming is not supported in JSON mode
        stream=False,
        # Enable JSON mode by setting the response format
        response_format={"type": "json_object"},
    )
    return CvDetails.model_validate_json(
                    chat_completion.choices[0].message.content)


def get_sentence_embedding(text, model):
    return model.encode(text, normalize_embeddings=True)


def canditate_fitness(score):
    if score > 60:
        return "pass"
    else:
        return "not a fit"


conn = psycopg2.connect("dbname=cv_agent user=postgres password=layo")
cursor = conn.cursor()


def save_candidate_to_db(candidate, score):
    if score >= 60:
        query = """
        INSERT INTO candidates (name, job_title, email, score)
        VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query, (candidate["name"], candidate["job_title"],
                               candidate["email"], score))
        conn.commit()


app = FastAPI()


@app.post("/upload/")
async def upload_files(cv: UploadFile = File(...),
                       job_desc: UploadFile = File(...)):
    # Save CV and Job Description temporarily
    cv_path = f"./uploads/{cv.filename}"
    job_desc_path = f"./uploads/{job_desc.filename}"

    with open(cv_path, "wb") as buffer:
        shutil.copyfileobj(cv.file, buffer)

    with open(job_desc_path, "wb") as buffer:
        shutil.copyfileobj(job_desc.file, buffer)

    # Extract text from PDFs
    cv_text = extract_text_from_pdf(cv_path)
    job_desc_text = extract_text_from_pdf(job_desc_path)

    # get LLM output json for cv details
    output_from_llm = get_cv_details(cv_text)
    result = ast.literal_eval(output_from_llm.model_dump_json())

    # get the needed fields
    cv_fields = ["education",
                 "experience",
                 "skills",
                 "languages",
                 "certifications"]

    # get cv details
    cv_details_text = json.dumps({key: result[key] for key in cv_fields})

    # Load pre-trained model (optimized for similarity tasks)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    cv_embedding = get_sentence_embedding(cv_details_text, model)
    job_desc_embedding = get_sentence_embedding(job_desc_text, model)

    # Compute cosine similarity
    similarity_score = cosine_similarity([cv_embedding],
                                         [job_desc_embedding])[0][0]
    # Compute similarity score
    score = similarity_score * 100  # Scale to 0-100

    # Save candidate to database
    save_candidate_to_db(result, round(score))

    return {
        "candidate_name": result["name"],
        "candidate_email": result["email"],
        "fitness score": round(score, 2),
        "candidate_fitness": canditate_fitness(score)
    }

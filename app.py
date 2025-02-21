import json
from pydantic import BaseModel
from typing import List, Union
import ast
from process_doc import extract_text_from_pdf
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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


pdf_path = "C:/Users/PC/Documents/ai_agent/Omolayo  Ipinsanmi_ CV.pdf"
text = extract_text_from_pdf(pdf_path)

job_description = extract_text_from_pdf(
    "C:/Users/PC/Documents/ai_agent/job_description.pdf")

# get LLM output json for cv details
output_from_llm = get_cv_details(text)
result = ast.literal_eval(output_from_llm.model_dump_json())

# get the needed fields
cv_fields = ["education",
             "experience",
             "skills",
             "languages",
             "certifications"]

# get cv details
cv_details_text = json.dumpts({key: result[key] for key in cv_fields})

# get embeddings from texts
# Load pre-trained model (optimized for similarity tasks)
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_sentence_embedding(text):
    return model.encode(text, normalize_embeddings=True)  # Normalize


cv_embedding = get_sentence_embedding(cv_details_text)
job_desc_embedding = get_sentence_embedding(job_description)

# Compute cosine similarity
similarity_score = cosine_similarity([cv_embedding],
                                     [job_desc_embedding])[0][0]


def canditate_fitness(similarity_score):
    score = round(similarity_score*100, 2)
    if score > 60:
        return "pass"
    else:
        return "not a fit"

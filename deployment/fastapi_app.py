from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import PyPDF2
import io
import matplotlib.pyplot as plt
import base64
from scipy.special import softmax
import re
import numpy as np
from collections import defaultdict

app = FastAPI()
templates = Jinja2Templates(directory="deployment/templates")

# Load model
model = joblib.load("models/best_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


# extract text
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# validation in cv
def is_cv(text):
    text_lower = text.lower()


    cv_keywords = [
        "experience", "education", "skills", "resume", "cv", "curriculum vitae",
        "work", "employment", "qualification", "professional", "summary",
        "objective", "contact", "phone", "email", "address", "projects",
        "certifications", "achievements", "references", "bachelor", "master",
        "degree", "university", "college", "internship", "responsibilities",
        "job", "position", "career", "profile", "background", "history"
    ]
    keyword_count = sum(1 for keyword in cv_keywords if keyword in text_lower)


    if len(text) < 500:
        return False


    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    has_email = bool(re.search(email_pattern, text))

  
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\(\d{3}\)\s*\d{3}[-.]?\d{4}'
    has_phone = bool(re.search(phone_pattern, text))

    lines = text.split('\n')
    has_name = any(line.strip() and line[0].isupper() and len(line.split()) <= 4 for line in lines[:5])

    date_pattern = r'\b(19|20)\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b'
    has_dates = len(re.findall(date_pattern, text)) >= 2

 
    degree_keywords = ["bachelor", "master", "phd", "msc", "bsc", "diploma"]
    has_degree = any(deg in text_lower for deg in degree_keywords)

    sections = ["experience", "education", "skills", "projects", "certifications", "references", "contact", "summary", "objective"]
    section_count = sum(1 for sec in sections if sec in text_lower)

    has_personal = " i " in text_lower or " my " in text_lower or " i'm " in text_lower

    not_job = not ("we are hiring" in text_lower or "apply now" in text_lower or "job description" in text_lower)

    score = 0
    score += min(keyword_count / 10, 3) 
    if has_email: score += 1
    if has_phone: score += 1
    if has_name: score += 1
    if has_dates: score += 1
    if has_degree: score += 1
    score += min(section_count / 3, 2) 
    if has_personal: score += 1
    if not_job: score += 1

    return score >= 6 

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Prediction 
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):

    contents = await file.read()
    pdf_url = base64.b64encode(contents).decode()
    text = extract_text_from_pdf(contents)

    if not is_cv(text):
        error = "The uploaded PDF does not appear to be a CV. Please upload a valid CV in PDF format."
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": error,
            "pdf_url": pdf_url
        })

    X = tfidf.transform([text])

    probabilities = softmax(model.decision_function(X), axis=1)[0]
    classes = model.classes_

    # 3 job predic
    top_indices = probabilities.argsort()[-3:][::-1]
    percentages = [(classes[i], float(probabilities[i]) * 100) for i in top_indices]



    # job links
    job_links = []
    for role, _ in percentages:
        job_links.append({
            "role": role,
            "linkedin": f"https://www.linkedin.com/jobs/search/?keywords={role}"
        })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "percentages": percentages,
        "job_links": job_links,
        "pdf_url": pdf_url
    })

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import PyPDF2
import io
import matplotlib.pyplot as plt
import base64
from scipy.special import softmax

app = FastAPI()
templates = Jinja2Templates(directory="deployment/templates")

# Load model and vectorizer
model = joblib.load("models/best_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Prediction endpoint
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):

    contents = await file.read()
    pdf_url = base64.b64encode(contents).decode()
    text = extract_text_from_pdf(contents)

    X = tfidf.transform([text])

    probabilities = softmax(model.decision_function(X), axis=1)[0]
    classes = model.classes_

    # Get top 2 predictions
    top_indices = probabilities.argsort()[-3:][::-1]
    percentages = [(classes[i], float(probabilities[i]) * 100) for i in top_indices]

    # Keyword importance (TF-IDF values)
    feature_names = tfidf.get_feature_names_out()
    tfidf_values = X.toarray()[0]

    top_keywords_indices = tfidf_values.argsort()[-10:][::-1]
    keywords = [feature_names[i] for i in top_keywords_indices]
    keyword_scores = [tfidf_values[i] for i in top_keywords_indices]

    # Create pie chart
    plt.figure(figsize=(6,6))
    plt.pie(keyword_scores, labels=keywords, autopct='%1.1f%%')
    plt.title("Technical Skills Analysis")

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()

    # Sample job links
    job_links = []
    for role, _ in percentages:
        job_links.append({
            "role": role,
            "linkedin": f"https://www.linkedin.com/jobs/search/?keywords={role}",
            "topjobs": f"https://www.topjobs.lk/applicant/vacancy/search?keywords={role}"
        })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "percentages": percentages,
        "graph_url": graph_url,
        "job_links": job_links,
        "pdf_url": pdf_url
    })

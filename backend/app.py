from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import os
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

app = Flask(__name__)
CORS(app)

# -------------------------------
# LOAD CSV
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "DiseaseAndSymptoms.csv")

df = pd.read_csv(csv_path)
df = df.fillna("")  # fix NaN issue

df.columns = df.columns.str.strip()

for col in df.columns:
    df[col] = df[col].astype(str).str.replace("_", " ").str.lower()

df["all"] = df.drop(columns=["Disease"]).agg(" ".join, axis=1)

print("✅ Dataset loaded")

# -------------------------------
# MATCH DISEASE
# -------------------------------
def match_diseases(user_input):
    words = set(user_input.lower().split())
    scores = []

    for _, row in df.iterrows():
        symptoms_set = set(row["all"].split())
        score = len(words & symptoms_set)

        if score > 0:
            scores.append((row["Disease"], score))

    scores.sort(key=lambda x: x[1], reverse=True)
    total = sum(score for _, score in scores) or 1

    return [
        {"disease": d, "confidence": round(score / total, 2)}
        for d, score in scores[:3]
    ]

# -------------------------------
# GROQ LLM
# -------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not set")

llm = ChatGroq(
    model="llama3-70b-8192",
    api_key=GROQ_API_KEY,
    temperature=0.3
)

# -------------------------------
# AI RESPONSE
# -------------------------------
def generate_response(symptoms, diseases):
    disease_names = [d["disease"] for d in diseases] or ["Unknown"]

    template = """
You are a medical assistant.

User Symptoms:
{symptoms}

Possible Diseases:
{diseases}

Give response in this format:

Condition:
Medicines:
Home Care:
Diet Advice:
When to See Doctor:
Disclaimer:
This is not a medical diagnosis.
"""

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm

    res = chain.invoke({
        "symptoms": symptoms,
        "diseases": ", ".join(disease_names)
    })

    return res.content

# -------------------------------
# ROUTES
# -------------------------------
@app.route("/")
def home():
    return "✅ Backend Running"

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    symptoms = data.get("symptoms", "")

    diseases = match_diseases(symptoms)
    response = generate_response(symptoms, diseases)

    return jsonify({
        "diseases": diseases,
        "response": response
    })

@app.route("/download", methods=["POST"])
def download():
    data = request.get_json()

    os.makedirs("static", exist_ok=True)

    file = f"static/report_{datetime.now().timestamp()}.pdf"

    doc = SimpleDocTemplate(file, pagesize=A4)
    styles = getSampleStyleSheet()

    content = [
        Paragraph("Sympto AI Report", styles["Title"]),
        Spacer(1, 10),
        Paragraph(data["response"], styles["Normal"])
    ]

    doc.build(content)

    return send_file(file, as_attachment=True)

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import os
import re
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

app = Flask(__name__)
CORS(app)

# -------------------------------
# LOAD DATASET SAFELY
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

main_path = os.path.join(BASE_DIR, "DiseaseAndSymptoms.csv")
custom_path = os.path.join(BASE_DIR, "custom_data.csv")

def load_dataset(path):
    try:
        df = pd.read_csv(path).fillna("")
        df.columns = df.columns.str.strip()
        return df
    except:
        print(f"⚠ Could not load {path}")
        return pd.DataFrame()

main_df = load_dataset(main_path)
custom_df = load_dataset(custom_path)

df = pd.concat([main_df, custom_df], ignore_index=True)

# normalize text
for col in df.columns:
    df[col] = df[col].astype(str).str.replace("_", " ").str.lower()

# create combined symptoms safely
def combine_row(row):
    return " ".join([str(x) for x in row.values if isinstance(x, str)])

df["all"] = df.apply(combine_row, axis=1)

print("✅ Dataset loaded safely")

# -------------------------------
# MATCHING
# -------------------------------
def match_diseases(user_input):
    words = set(re.findall(r'\b\w+\b', user_input.lower()))
    results = []

    text = user_input.lower()

    # 🚨 smart rules
    if "missed" in words and "period" in words and ("vomiting" in words or "nausea" in words):
        return [{"disease": "Possible Pregnancy", "confidence": 0.95}]

    if "chest" in words and "pain" in words:
        return [{"disease": "Possible Heart Issue", "confidence": 0.95}]

    for _, row in df.iterrows():
        symptoms = set(row["all"].split())
        score = len(words & symptoms)

        if score > 0:
            results.append((row.get("Disease", "Unknown"), score))

    results.sort(key=lambda x: x[1], reverse=True)
    total = sum(score for _, score in results) or 1

    return [
        {"disease": d, "confidence": round(score / total, 2)}
        for d, score in results[:3]
    ]

# -------------------------------
# GROQ AI
# -------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("⚠ GROQ_API_KEY missing")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=0.3
)

# -------------------------------
# RESPONSE
# -------------------------------
def generate_response(symptoms, diseases):
    disease_names = [d["disease"] for d in diseases]

    template = """
You are a medical assistant.

Rules:
- Suggest only safe medicines (paracetamol, ORS)
- No strong drugs
- Keep answer simple

Symptoms:
{symptoms}

Diseases:
{diseases}

Format:

Condition:
Medicines:
Home Care:
Diet:
When to See Doctor:
Disclaimer:
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
    return "✅ Running"

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        symptoms = data.get("symptoms", "")

        diseases = match_diseases(symptoms)
        response = generate_response(symptoms, diseases)

        return jsonify({
            "diseases": diseases,
            "response": response
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/download", methods=["POST"])
def download():
    data = request.get_json()

    os.makedirs("static", exist_ok=True)
    filename = f"static/report_{datetime.now().timestamp()}.pdf"

    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()

    content = [
        Paragraph("Sympto AI Report", styles["Title"]),
        Spacer(1, 10),
        Paragraph(data["response"], styles["Normal"])
    ]

    doc.build(content)

    return send_file(filename, as_attachment=True)

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
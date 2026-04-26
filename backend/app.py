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
# LOAD DATASETS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

main_path = os.path.join(BASE_DIR, "DiseaseAndSymptoms.csv")
custom_path = os.path.join(BASE_DIR, "custom_data.csv")

main_df = pd.read_csv(main_path).fillna("")
custom_df = pd.read_csv(custom_path).fillna("")

df = pd.concat([main_df, custom_df], ignore_index=True)
df.columns = df.columns.str.strip()

# normalize text
for col in df.columns:
    df[col] = df[col].astype(str).str.replace("_", " ").str.lower()

# combine all symptom text
df["all"] = df.drop(columns=["Disease"]).agg(" ".join, axis=1)

print("✅ Dataset loaded & merged")

# -------------------------------
# SMART MATCHING
# -------------------------------
def match_diseases(user_input):
    words = set(re.findall(r'\b\w+\b', user_input.lower()))
    results = []

    text = user_input.lower()

    # 🚨 HIGH PRIORITY RULES
    if "missed" in words and "period" in words and ("vomiting" in words or "nausea" in words):
        return [{"disease": "Possible Pregnancy", "confidence": 0.95}]

    if "chest" in words and "pain" in words:
        return [{"disease": "Possible Heart Issue", "confidence": 0.95}]

    if "seizure" in words or "unconscious" in words:
        return [{"disease": "Possible Neurological Emergency", "confidence": 0.95}]

    # 🔍 dataset matching
    for _, row in df.iterrows():
        symptoms_set = set(row["all"].split())
        score = len(words & symptoms_set)

        # weighted improvements
        if "vomiting" in words and "nausea" in symptoms_set:
            score += 2

        if "fever" in words and "chills" in symptoms_set:
            score += 2

        if "cough" in words and "breathing" in symptoms_set:
            score += 2

        # pregnancy boost
        if "missed" in words and "period" in words:
            if "pregnancy" in row["Disease"]:
                score += 5

        if score > 0:
            results.append((row["Disease"], score))

    results.sort(key=lambda x: x[1], reverse=True)
    total = sum(score for _, score in results) or 1

    return [
        {"disease": d, "confidence": round(score / total, 2)}
        for d, score in results[:3]
    ]

# -------------------------------
# GROQ LLM
# -------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not set")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
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

STRICT RULES:
- Suggest only SAFE and COMMON medicines (paracetamol, ORS, etc.)
- DO NOT suggest antibiotics or prescription drugs
- Highlight serious conditions clearly
- If symptoms indicate pregnancy, mention it clearly
- Keep response simple and practical

User Symptoms:
{symptoms}

Possible Diseases:
{diseases}

Format:

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
    try:
        data = request.get_json()
        symptoms = data.get("symptoms", "").strip()

        if not symptoms:
            return jsonify({"error": "Enter symptoms"}), 400

        diseases = match_diseases(symptoms)
        response = generate_response(symptoms, diseases)

        return jsonify({
            "symptoms": symptoms,
            "diseases": diseases,
            "response": response
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download", methods=["POST"])
def download():
    try:
        data = request.get_json()

        os.makedirs("static", exist_ok=True)

        filename = f"static/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        doc = SimpleDocTemplate(filename, pagesize=A4)
        styles = getSampleStyleSheet()

        content = [
            Paragraph("AI Medical Report", styles["Title"]),
            Spacer(1, 10),
            Paragraph(f"Symptoms: {data['symptoms']}", styles["Normal"]),
            Spacer(1, 10),
            Paragraph(data["response"], styles["Normal"]),
            Spacer(1, 10),
            Paragraph("⚠ This is not a medical diagnosis. Consult a doctor.", styles["Normal"])
        ]

        doc.build(content)

        return send_file(filename, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
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

if not os.path.exists(csv_path):
    raise FileNotFoundError("❌ CSV not found")

df = pd.read_csv(csv_path)
df = df.fillna("")   # fix NaN issue

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
    raise ValueError("❌ GROQ_API_KEY not set in Render")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=0.3
)

# -------------------------------
# AI RESPONSE
# -------------------------------
def generate_response(symptoms, diseases):
    try:
        disease_names = [d["disease"] for d in diseases] or ["General condition"]

        template = """
You are a helpful AI medical assistant.

User Symptoms:
{symptoms}

Possible Conditions:
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

    except Exception as e:
        return f"⚠️ AI service error: {str(e)}"

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

        if not data or "symptoms" not in data:
            return jsonify({"error": "No symptoms provided"}), 400

        symptoms = data["symptoms"].strip()

        if not symptoms:
            return jsonify({"error": "Empty symptoms"}), 400

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

        symptoms = data.get("symptoms", "")
        response = data.get("response", "")

        os.makedirs("static", exist_ok=True)

        file_path = f"static/report_{datetime.now().timestamp()}.pdf"

        doc = SimpleDocTemplate(file_path, pagesize=A4)
        styles = getSampleStyleSheet()

        content = [
            Paragraph("<b>Sympto AI Report</b>", styles["Title"]),
            Spacer(1, 15),
            Paragraph(f"<b>Symptoms:</b> {symptoms}", styles["Normal"]),
            Spacer(1, 10),
            Paragraph(response.replace("\n", "<br/>"), styles["Normal"])
        ]

        doc.build(content)

        return send_file(file_path, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import os
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

# LLM
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

app = Flask(__name__)
CORS(app)

# -------------------------------
# 📁 LOAD DATASET (SAFE)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "DiseaseAndSymptoms.csv")

print("📂 Reading CSV from:", csv_path)

if not os.path.exists(csv_path):
    raise FileNotFoundError("❌ DiseaseAndSymptoms.csv not found")

if os.path.getsize(csv_path) == 0:
    raise ValueError("❌ CSV file is empty")

df = pd.read_csv(csv_path)

# ✅ CLEAN DATA (FIXED)
df.columns = df.columns.str.strip()

df = df.fillna("")  # 🔥 FIX NaN issue

for col in df.columns:
    df[col] = df[col].astype(str).str.replace("_", " ").str.lower()

df["all"] = df.drop(columns=["Disease"]).astype(str).agg(" ".join, axis=1)

print("✅ Dataset loaded successfully")

# -------------------------------
# 🧠 DISEASE MATCHING
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
# 🤖 LLM (GROQ SAFE)
# -------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not set in environment")

llm = ChatGroq(
    model="llama3-70b-8192",
    api_key=GROQ_API_KEY,
    temperature=0.3
)

def generate_response(symptoms, diseases):
    disease_names = [d["disease"] for d in diseases] or ["No strong match"]

    template = """
You are a professional AI medical assistant.

User Symptoms:
{symptoms}

Possible Conditions:
{diseases}

STRICT FORMAT:

Condition:
Severity:
Advice:
Warning Signs:
Disclaimer:
This is not a medical diagnosis. Consult a doctor.
"""

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm

    response = chain.invoke({
        "symptoms": symptoms,
        "diseases": ", ".join(disease_names)
    })

    return response.content

# -------------------------------
# 🏠 HOME ROUTE
# -------------------------------
@app.route("/")
def home():
    return "✅ Sympto AI Backend Running"

# -------------------------------
# 🚀 ANALYZE API
# -------------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        symptoms = data.get("symptoms", "").strip()

        if not symptoms:
            return jsonify({"error": "Please enter symptoms"}), 400

        diseases = match_diseases(symptoms)
        response = generate_response(symptoms, diseases)

        return jsonify({
            "symptoms": symptoms,
            "diseases": diseases,
            "response": response
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------
# 📄 DOWNLOAD PDF
# -------------------------------
@app.route("/download", methods=["POST"])
def download():
    try:
        data = request.get_json()
        symptoms = data["symptoms"]
        response = data["response"]

        os.makedirs("static", exist_ok=True)

        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join("static", filename)

        doc = SimpleDocTemplate(filepath, pagesize=A4)
        styles = getSampleStyleSheet()

        content = [
            Paragraph("<b>AI Medical Report</b>", styles["Title"]),
            Spacer(1, 15),
            Paragraph(f"<b>Symptoms:</b> {symptoms}", styles["Normal"]),
            Spacer(1, 10),
            Paragraph(response, styles["Normal"])
        ]

        doc.build(content)

        return send_file(filepath, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------
# ▶ RUN (RENDER SAFE)
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
const API_URL = "https://symptoai.onrender.com"; // 🔁 replace this

async function analyzeSymptoms() {
    const input = document.getElementById("symptoms").value;

    if (!input) {
        alert("Please enter symptoms");
        return;
    }

    // Show loading
    const responseBox = document.getElementById("response");
    responseBox.innerHTML = "⏳ Analyzing...";

    try {
        const res = await fetch(`${API_URL}/analyze`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ symptoms: input })
        });

        const data = await res.json();

        if (data.error) {
            responseBox.innerHTML = "❌ " + data.error;
            return;
        }

        // Show response
        responseBox.innerHTML = `
            <h3>🧠 AI Analysis</h3>
            <pre>${data.response}</pre>

            <button onclick="downloadReport()">📄 Download Report</button>
        `;

        // Save for download
        window.lastResponse = data.response;
        window.lastSymptoms = input;

    } catch (err) {
        responseBox.innerHTML = "❌ Server error. Try again.";
    }
}

async function downloadReport() {
    const res = await fetch(`${API_URL}/download`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            symptoms: window.lastSymptoms,
            response: window.lastResponse
        })
    });

    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "medical_report.pdf";
    a.click();
}
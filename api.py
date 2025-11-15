import os
import base64
import torch
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from flask import Flask, request, jsonify, render_template
from transformers import BartTokenizer, BartForConditionalGeneration
from collections import Counter
from wordcloud import WordCloud
import google.generativeai as genai

# -----------------------------
# 🔧 Flask App Setup
# -----------------------------
app = Flask(__name__)


GEMINI_API_KEY = "AIzaSyBTlGO2KQTAQSGw9rjqc2goHesgeS8ErPU"
GEMINI_MODEL = "gemini-2.5-flash"
genai.configure(api_key=GEMINI_API_KEY)

# -----------------------------
# 🧠 Load Summarizer Model (Local)
# -----------------------------
def load_summarizer():
    """Loads local BART summarizer model."""
    try:
        print("📂 Loading summarizer model from local path...")
        local_path = r"C:\Users\rsuma\OneDrive\Desktop\MiniProject\models\bart-summarizer"

        tokenizer = BartTokenizer.from_pretrained(local_path)
        model = BartForConditionalGeneration.from_pretrained(local_path)

        device = torch.device("cpu")
        model.to(device)

        print("✅ Summarizer model loaded successfully.")
        return model, tokenizer, device
    except Exception as e:
        print(f"❌ Error loading summarizer: {e}")
        return None, None, torch.device("cpu")

summarizer_model, tokenizer, device = load_summarizer()


def analyze_with_gemini(text: str):
    """Uses Gemini API to detect sentiment & sarcasm."""
    try:
        prompt = f"""
        Analyze the following comment for both sentiment and sarcasm:

        "{text}"

        Respond strictly in JSON format:
        {{
          "sentiment": "Positive" or "Negative" or "Neutral",
          "sarcasm": true or false
        }}
        """

        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt, request_options={"timeout": 15})
        result_text = response.text.strip().lower()

        sentiment = "Neutral"
        sarcasm = False

        if "positive" in result_text:
            sentiment = "Positive"
        elif "negative" in result_text:
            sentiment = "Negative"
        elif "neutral" in result_text:
            sentiment = "Neutral"

        if '"sarcasm": true' in result_text or "sarcastic" in result_text:
            sarcasm = True

        return {"sentiment": sentiment, "sarcasm": sarcasm}

    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return {"sentiment": "Error", "sarcasm": False}

# -----------------------------
# ☁️ Generate Word Cloud
# -----------------------------
def create_wordcloud(texts):
    """Generates base64-encoded word cloud image."""
    try:
        combined_text = " ".join(texts)
        wc = WordCloud(width=800, height=400, background_color="white").generate(combined_text)
        buffer = BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        return img_base64
    except Exception as e:
        print(f"❌ Error generating word cloud: {e}")
        return None

# -----------------------------
# 📝 Summarize Texts
# -----------------------------
def summarize_texts(texts):
    """Summarizes the combined text using local BART model."""
    if not summarizer_model or not tokenizer:
        return "Summarizer model unavailable."

    try:
        full_text = " ".join(texts)[:4000]
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            summary_ids = summarizer_model.generate(
                **inputs,
                max_length=120,
                min_length=20,
                num_beams=4,
                early_stopping=True
            )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        print(f"❌ Summarization error: {e}")
        return "Error generating summary."

# -----------------------------
# 🏠 Home Route → Serve Frontend
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# -----------------------------
# 🚀 Prediction Endpoint
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # --- Case 1: CSV Upload ---
        if "file" in request.files:
            uploaded_file = request.files["file"]
            df = pd.read_csv(uploaded_file)

            if "text" not in df.columns:
                return jsonify({"error": "CSV must contain a 'text' column"}), 400

            sentiments, sarcasms = [], []
            for text in df["text"]:
                result = analyze_with_gemini(str(text))
                sentiments.append(result["sentiment"])
                sarcasms.append("Yes" if result["sarcasm"] else "No")

            df["sentiment"] = sentiments
            df["sarcasm_detected"] = sarcasms

            counts = dict(Counter(sentiments))
            wordcloud_b64 = create_wordcloud(df["text"].astype(str).tolist())
            overall_summary = summarize_texts(df["text"].astype(str).tolist())

            # Create downloadable CSV
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            csv_base64 = base64.b64encode(csv_buffer.read()).decode("utf-8")

            return jsonify({
                "counts": counts,
                "summary": overall_summary,
                "wordcloud": wordcloud_b64,
                "csv_file": csv_base64
            })

        # --- Case 2: Single text ---
        data = request.get_json(force=True)
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400

        result = analyze_with_gemini(text)
        sentiment = result["sentiment"]
        if result["sarcasm"]:
            sentiment = f"{sentiment} (sarcasm detected)"

        return jsonify({
            "text": text,
            "predicted_sentiment": sentiment
        })

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------------
# 🏁 Run Flask App
# -----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)

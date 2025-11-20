import os
import base64
import torch
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BartTokenizer, BartForConditionalGeneration
from collections import Counter
from wordcloud import WordCloud
import google.generativeai as genai

# -------------------------------------
# 🔧 Flask App Setup
# -------------------------------------
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------
# 🔑 Gemini API Key
# -------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("⚠ WARNING: GEMINI_API_KEY not found in environment!")

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"

# -------------------------------------
# 🧠 Summarizer Model (Recommended Fix)
# -------------------------------------
print("📂 Loading BART summarizer from HuggingFace...")

MODEL_NAME = "facebook/bart-large-cnn"   # ✔ public + works everywhere

try:
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    summarizer_model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    summarizer_model.to("cpu")
    summarizer_loaded = True
    print("✅ Summarizer loaded.")
except Exception as e:
    print(f"❌ Error loading summarizer: {e}")
    tokenizer = None
    summarizer_model = None
    summarizer_loaded = False

# -------------------------------------
# 🤖 Gemini analysis
# -------------------------------------
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
        result = response.text.lower()

        sentiment = "Neutral"
        sarcasm = False

        if "positive" in result:
            sentiment = "Positive"
        elif "negative" in result:
            sentiment = "Negative"

        if "sarcasm" in result and "true" in result:
            sarcasm = True

        return {"sentiment": sentiment, "sarcasm": sarcasm}

    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return {"sentiment": "Error", "sarcasm": False}

# -------------------------------------
# ☁ Wordcloud
# -------------------------------------
def create_wordcloud(texts):
    try:
        combined = " ".join(texts)
        wc = WordCloud(width=800, height=400, background_color="white").generate(combined)

        buffer = BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode("utf-8")
    except:
        return None

# -------------------------------------
# 📝 Summarizer
# -------------------------------------
def summarize_texts(texts):
    if not summarizer_loaded:
        return "Summarizer unavailable."

    try:
        combined = " ".join(texts)[:4000]
        inputs = tokenizer(combined, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            ids = summarizer_model.generate(
                **inputs,
                max_length=120,
                min_length=20,
                num_beams=4,
                early_stopping=True
            )

        return tokenizer.decode(ids[0], skip_special_tokens=True)

    except Exception as e:
        print("❌ Summarization error:", e)
        return "Error generating summary."

# -------------------------------------
# 🏠 Home Route
# -------------------------------------
@app.route("/", methods=["GET"])
def home():
    return "Backend is running successfully!"

# -------------------------------------
# 🚀 Prediction API
# -------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # CASE 1: CSV Upload
        if "file" in request.files:
            file = request.files["file"]
            df = pd.read_csv(file)

            if "text" not in df.columns:
                return jsonify({"error": "CSV must contain a 'text' column"}), 400

            sentiments, sarcasms = [], []

            for text in df["text"].astype(str):
                result = analyze_with_gemini(text)
                sentiments.append(result["sentiment"])
                sarcasms.append("Yes" if result["sarcasm"] else "No")

            df["sentiment"] = sentiments
            df["sarcasm"] = sarcasms

            summary = summarize_texts(df["text"].tolist())
            wordcloud = create_wordcloud(df["text"].tolist())
            counts = dict(Counter(sentiments))

            # Encode CSV
            buffer = BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)
            csv_base64 = base64.b64encode(buffer.read()).decode("utf-8")

            return jsonify({
                "summary": summary,
                "wordcloud": wordcloud,
                "counts": counts,
                "csv_file": csv_base64
            })

        # CASE 2: Single text
        data = request.get_json(force=True)
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "Text is required"}), 400

        result = analyze_with_gemini(text)
        sentiment = result["sentiment"]
        if result["sarcasm"]:
            sentiment += " (sarcasm detected)"

        return jsonify({
            "text": text,
            "predicted_sentiment": sentiment
        })

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({"error": str(e)}), 500

# -------------------------------------
# LOCAL RUN
# -------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

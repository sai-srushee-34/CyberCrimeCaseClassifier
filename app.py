"""
Cybercrime Complaint Classifier - Web Application Backend
Serves the Clarity Public ML / Stitch Frontend and exposes the /api/predict endpoint
connecting directly to the trained scikit-learn Pipeline (models/best_model.joblib).
"""

import os
import sys
import time
import logging
from typing import Dict, Any, List
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory

# Ensure src directory is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import joblib
from text_pipeline import CANONICAL_CATEGORIES, SoftmaxLinearSVC

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.joblib")

# Load model pipeline at startup
logger.info(f"Loading trained pipeline from {MODEL_PATH}...")
try:
    pipeline = joblib.load(MODEL_PATH)
    logger.info("Pipeline loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load pipeline: {e}")
    pipeline = None

# Category Icons for Material Symbols
CATEGORY_ICONS = {
    "Phishing": "mark_email_unread",
    "Online Financial Fraud": "payments",
    "Identity Theft": "badge",
    "Account Hacking": "lock_reset",
    "Cyberbullying / Online Harassment": "record_voice_over",
    "Fake Social Media Account": "face",
    "Malware / Ransomware": "terminal",
    "Online Shopping Fraud": "shopping_bag",
}

# Thematic Color Mapping for Probability Bars
CATEGORY_COLORS = {
    "Phishing": "bg-primary",
    "Online Financial Fraud": "bg-primary",
    "Identity Theft": "bg-secondary",
    "Account Hacking": "bg-secondary",
    "Cyberbullying / Online Harassment": "bg-error",
    "Fake Social Media Account": "bg-primary",
    "Malware / Ransomware": "bg-tertiary",
    "Online Shopping Fraud": "bg-secondary",
}

# NLP Explanations tailored to each crime category
CATEGORY_EXPLANATIONS = {
    "Phishing": "The complaint exhibits strong vector weights for deceptive communication, credential harvesting portals, spoofed banking links, and artificial account urgency.",
    "Online Financial Fraud": "The narrative features prominent indicators of unauthorized fund transfers, deceptive payment requests, fraudulent UPI/card settlements, and OTP manipulation.",
    "Identity Theft": "Strong correlation detected with forged identity credentials, unauthorized document submissions, fraudulent loan originations, and personal impersonation.",
    "Account Hacking": "High lexical affinity with unauthorized account takeover, compromised login credentials, bypassed security verifications, and lockout mechanisms.",
    "Cyberbullying / Online Harassment": "Strong pattern resonance for persistent intimidation, abusive language, non-consensual contact, threatening messages, and online defamation.",
    "Fake Social Media Account": "Significant indicator density for entity cloning, stolen photographs, duplicate profile creation, and impersonation of public or personal identity.",
    "Malware / Ransomware": "Pronounced technical indicators for malicious executable payloads, unauthorized data encryption, device lockouts, and extortion or ransom demands.",
    "Online Shopping Fraud": "Strong characteristics of deceptive e-commerce platforms, unfulfilled product orders, bogus courier tracking, and seller non-responsiveness after payment.",
}


def extract_top_tfidf_tokens(text: str, vectorizer, top_k: int = 4) -> List[Dict[str, str]]:
    """Extracts the top weighted TF-IDF vocabulary tokens for the input complaint."""
    try:
        feature_names = vectorizer.get_feature_names_out()
        tfidf_vec = vectorizer.transform([text]).toarray()[0]
        top_indices = tfidf_vec.argsort()[::-1]
        tokens = []
        for idx in top_indices:
            if tfidf_vec[idx] > 0 and len(tokens) < top_k:
                tokens.append({
                    "word": feature_names[idx],
                    "score": f"{tfidf_vec[idx]:.2f}",
                })
        return tokens
    except Exception as e:
        logger.warning(f"Token extraction error: {e}")
        return []


@app.route("/")
def index():
    """Renders the Stitch HTML frontend."""
    if os.path.exists(os.path.join(app.template_folder, "index.html")):
        return render_template("index.html")
    return send_from_directory(".", "index.html")


@app.route("/api/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "online",
        "model_loaded": pipeline is not None,
        "categories_count": len(CANONICAL_CATEGORIES),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Classifies a natural-language cybercrime complaint narrative.
    JSON Request: {"text": "I received an email asking for password..."}
    """
    if pipeline is None:
        return jsonify({"error": "Model pipeline is not loaded on server."}), 500

    data = request.get_json(silent=True) or {}
    raw_text = str(data.get("text", "")).strip()

    if not raw_text:
        return jsonify({"error": "Complaint narrative text is required."}), 400

    start_time = time.perf_counter()

    try:
        # Run inference through the scikit-learn pipeline
        predicted_category = pipeline.predict([raw_text])[0]

        # Calculate class probabilities
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba([raw_text])[0]
        elif hasattr(pipeline.named_steps.get("clf"), "decision_function"):
            df_scores = pipeline.named_steps["clf"].decision_function(
                pipeline.named_steps["tfidf"].transform([raw_text])
            )[0]
            exp_scores = np.exp(df_scores - np.max(df_scores))
            probs = exp_scores / np.sum(exp_scores)
        else:
            probs = np.zeros(len(pipeline.classes_))
            top_idx = list(pipeline.classes_).index(predicted_category)
            probs[top_idx] = 1.0

        elapsed_ms = max(1, int((time.perf_counter() - start_time) * 1000))
        top_prob = float(np.max(probs))

        # Build sorted multi-class distribution
        classes = list(pipeline.classes_)
        sorted_indices = np.argsort(probs)[::-1]
        distribution = []
        for rank, idx in enumerate(sorted_indices[:3]):
            cat = classes[idx]
            p = float(probs[idx])
            color = CATEGORY_COLORS.get(cat, "bg-primary") if rank == 0 else ("bg-secondary" if rank == 1 else "bg-outline-variant")
            distribution.append({
                "name": cat,
                "score": f"{p * 100:.1f}%",
                "width": f"{p * 100:.1f}%",
                "color": color,
            })

        # Extract salient TF-IDF vocabulary tokens
        tokens = []
        if "tfidf" in pipeline.named_steps:
            tokens = extract_top_tfidf_tokens(raw_text, pipeline.named_steps["tfidf"], top_k=4)

        icon = CATEGORY_ICONS.get(predicted_category, "security")
        explanation = CATEGORY_EXPLANATIONS.get(
            predicted_category,
            f"Classified into {predicted_category} based on semantic n-gram feature weights."
        )

        return jsonify({
            "category": predicted_category,
            "icon": icon,
            "confidence": f"{top_prob * 100:.1f}%",
            "confidenceNum": round(top_prob * 100, 1),
            "model": "LinearSVC + TF-IDF (N-gram 1,2)",
            "latency": f"{elapsed_ms}ms",
            "explanation": explanation,
            "tokens": tokens,
            "distribution": distribution,
        })

    except Exception as e:
        logger.exception("Inference failed")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Cybercrime Complaint Classifier - Web Application")
    print("  Frontend: Clarity Public ML (Stitch)")
    print("  Server running at: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)

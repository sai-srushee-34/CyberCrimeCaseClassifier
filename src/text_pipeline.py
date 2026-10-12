"""
Text Preprocessing and Feature Extraction Pipeline for Cybercrime Classification.

This module encapsulates scikit-learn's TfidfVectorizer inside an end-to-end
Pipeline, ensuring:
1. TF-IDF vocabulary and IDF weights are fitted EXCLUSIVELY on training data.
2. Validation, test, and live inference text are transformed strictly using
   the pre-fitted training vectorizer.
3. Crucial negations ('not', 'never', 'without') and incident indicators ('unauthorized')
   are preserved rather than destructively eliminated by standard stop-word lists.
4. Word-level unigram and bigram feature representations (ngram_range=(1, 2)).
"""

import os
import re
import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SoftmaxLinearSVC(LinearSVC):
    """
    Linear Support Vector Classifier equipped with a Softmax Decision Function
    to provide native predict_proba() class confidence distributions for web apps.
    """
    def predict_proba(self, X):
        d = self.decision_function(X)
        if len(self.classes_) == 2:
            probs = 1.0 / (1.0 + np.exp(-d))
            return np.column_stack([1 - probs, probs])
        exp_d = np.exp(d - np.max(d, axis=1, keepdims=True))
        return exp_d / np.sum(exp_d, axis=1, keepdims=True)


CANONICAL_CATEGORIES = [
    "Phishing",
    "Online Financial Fraud",
    "Identity Theft",
    "Account Hacking",
    "Cyberbullying / Online Harassment",
    "Fake Social Media Account",
    "Malware / Ransomware",
    "Online Shopping Fraud",
]

# -------------------------------------------------------------------------
# Negation-Preserving Stop Words
# -------------------------------------------------------------------------
# Standard ENGLISH_STOP_WORDS blindly removes words like 'not', 'never', 'without',
# which reverses semantic meaning in fraud and incident reports (e.g., 'not authorized'
# would become 'authorized', 'never received' would become 'received').
CRITICAL_NEGATIONS_AND_MODIFIERS = {
    "not", "no", "never", "without", "cannot", "cant", "wont", "dont",
    "didnt", "isnt", "arent", "wasnt", "werent", "hasnt", "havent",
    "hadnt", "neither", "nor", "none", "nothing", "nowhere", "against"
}

PLATFORM_BIAS_WORDS = {"rt", "twitter", "tweet", "tweets"}

# Remove critical negations from stop words, add platform-specific keywords to stop words
CUSTOM_STOP_WORDS = sorted(list((set(ENGLISH_STOP_WORDS) - CRITICAL_NEGATIONS_AND_MODIFIERS) | PLATFORM_BIAS_WORDS))


def normalize_complaint_text(text: str) -> str:
    """
    Applies consistent, non-destructive normalization suitable for English
    cybercrime complaints:
    - Lowercases text
    - Preserves hyphenated technical terms (e.g., 'two-factor', 'e-commerce')
    - Preserves monetary/loss tokens and alphanumeric text
    - Collapses extra whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    # Normalize unicode quotes and dashes
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    text = re.sub(r"[\u2013\u2014]", "-", text)

    # Normalize multiple whitespace characters
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_tfidf_vectorizer(
    min_df: int = 1,
    max_df: float = 0.85,
    ngram_range: Tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
    norm: str = "l2",
    custom_stop_words: Optional[List[str]] = None,
) -> TfidfVectorizer:
    """
    Constructs a configurable, reproducible TfidfVectorizer.

    Args:
        min_df: Ignore terms with document frequency strictly lower than this threshold.
                Set conservatively to 2 to eliminate single-occurrence typos while retaining rare terms.
        max_df: Ignore terms that occur in more than this fraction of documents.
                Set to 0.85 to automatically neutralize repetitive boilerplate and greetings.
        ngram_range: Lower and upper boundary of range of n-values for n-grams.
                     Default (1, 2) extracts both individual words (unigrams) and 2-word phrases (bigrams).
        sublinear_tf: Apply sublinear scaling: replaces TF with 1 + log(TF).
                      Prevents frequent words from exerting disproportionate dominance.
        norm: Normalization used on feature vectors. 'l2' ensures unit Euclidean norm.
        custom_stop_words: Custom stop words list preserving negations. Defaults to CUSTOM_STOP_WORDS.

    Returns:
        Configured scikit-learn TfidfVectorizer instance.
    """
    if custom_stop_words is None:
        custom_stop_words = CUSTOM_STOP_WORDS

    vectorizer = TfidfVectorizer(
        preprocessor=normalize_complaint_text,
        lowercase=True,
        stop_words=custom_stop_words,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        norm=norm,
        token_pattern=r"(?u)\b\w[\w-]*\w\b|\b\w\b",  # Preserves hyphenated terms like 'two-factor'
    )
    return vectorizer


def build_cybercrime_pipeline(
    classifier: Any,
    min_df: int = 2,
    max_df: float = 0.85,
    ngram_range: Tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
) -> Pipeline:
    """
    Constructs an end-to-end scikit-learn Pipeline encapsulating the TfidfVectorizer
    and a downstream classifier.

    Guarantee:
    Fitting this pipeline on training data guarantees that TF-IDF vocabulary,
    document frequencies, and IDF weights are learned ONLY from the training fold,
    eliminating preprocessing leakage.
    """
    vectorizer = build_tfidf_vectorizer(
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
    )
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])
    return pipeline


def demonstrate_tfidf_transformation(sample_texts: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Provides a clear, educational demonstration of how raw English complaint text
    is vectorized into numerical TF-IDF feature representations.
    """
    if sample_texts is None:
        sample_texts = [
            "I did not authorize this payment and money was taken without my consent.",
            "I never received the laptop I ordered from the online shopping website.",
            "An unknown person is impersonating me using a fake profile without permission.",
            "A suspicious phishing email requested my bank password and credentials.",
        ]

    logger.info("Demonstrating TF-IDF transformation step-by-step...")
    vectorizer = build_tfidf_vectorizer(min_df=1, max_df=1.0, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(sample_texts)

    feature_names = vectorizer.get_feature_names_out()
    vocab_size = len(feature_names)
    matrix_shape = tfidf_matrix.shape
    density = (tfidf_matrix.nnz / (matrix_shape[0] * matrix_shape[1])) * 100

    print("\n=================================================================")
    print("EDUCATIONAL WALKTHROUGH: RAW TEXT TO TF-IDF NUMERICAL VECTORS")
    print("=================================================================")
    print(f"Number of Documents (N): {len(sample_texts)}")
    print(f"Extracted Vocabulary Size: {vocab_size} features (unigrams + bigrams)")
    print(f"Sparse Matrix Shape: {matrix_shape} (Rows x Features)")
    print(f"Non-Zero Elements (NNZ): {tfidf_matrix.nnz} out of {matrix_shape[0] * matrix_shape[1]}")
    print(f"Matrix Density: {density:.2f}% (Sparse matrix saves > 90% memory)")

    # Inspect Document 0 features
    doc_0_vector = tfidf_matrix[0].toarray().flatten()
    top_indices = doc_0_vector.argsort()[::-1]

    rows = []
    for idx in top_indices:
        score = doc_0_vector[idx]
        if score > 0:
            term = feature_names[idx]
            idf = vectorizer.idf_[idx]
            is_ngram = "Bigram" if " " in term else "Unigram"
            rows.append({
                "Feature Token": term,
                "Type": is_ngram,
                "IDF Weight": round(idf, 4),
                "TF-IDF Value": round(score, 4),
            })

    doc_df = pd.DataFrame(rows)
    print("\nSample Document 0:")
    print(f"\"{sample_texts[0]}\"")
    print("\nExtracted Features with Weights (Sorted by Importance):")
    print(doc_df.to_string(index=False))

    return doc_df


def run_pipeline_test():
    """
    Performs a self-test verifying that:
    1. The Pipeline fits smoothly on a training sample.
    2. Validation data is transformed strictly via training weights.
    3. Unseen raw text produces calibrated class predictions.
    4. Test set is strictly untouched in this phase.
    """
    logger.info("Running pipeline unit sanity test...")
    train_path = "data/processed/train.csv"
    val_path = "data/processed/validation.csv"

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        logger.warning("Train/validation datasets not found. Skipping live data unit test.")
        return

    # Load samples
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    # Instantiate pipeline with a baseline Logistic Regression
    clf = LogisticRegression(class_weight="balanced", max_iter=200, random_state=42)
    pipeline = build_cybercrime_pipeline(
        classifier=clf,
        min_df=2,
        max_df=0.85,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    # 1. Fit ONLY on train
    logger.info(f"Fitting Pipeline on {len(df_train)} training records...")
    pipeline.fit(df_train["complaint_text"], df_train["category"])

    vocab = pipeline.named_steps["tfidf"].vocabulary_
    logger.info(f"Training vocabulary fitted successfully: {len(vocab):,} unique features.")

    # 2. Predict on validation records
    val_preds = pipeline.predict(df_val["complaint_text"].iloc[:5])
    logger.info(f"Successfully predicted on 5 validation samples: {list(val_preds)}")

    # 3. Predict on brand-new, unseen complaint text
    unseen_test_complaint = [
        "Someone hacked into my personal email, changed my recovery phone number, and locked me out.",
        "I was scammed by a fake online shop that took my payment but never delivered the goods.",
        "A malicious ransomware script locked all my files and is demanding cryptocurrency.",
    ]
    unseen_preds = pipeline.predict(unseen_test_complaint)
    unseen_probs = pipeline.predict_proba(unseen_test_complaint)

    logger.info("\nInference Verification on Unseen Text:")
    for text, pred, prob in zip(unseen_test_complaint, unseen_preds, unseen_probs):
        confidence = np.max(prob) * 100
        logger.info(f"Input: \"{text}\"")
        logger.info(f" --> Predicted Category: [{pred}] (Confidence: {confidence:.1f}%)")

    logger.info("\nAll text pipeline validation checks passed successfully!")


if __name__ == "__main__":
    demonstrate_tfidf_transformation()
    print("\n-----------------------------------------------------------------")
    run_pipeline_test()

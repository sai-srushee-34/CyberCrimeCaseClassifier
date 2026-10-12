# Model Benchmarking & Validation Performance Report

**Project Title:** Cybercrime Complaint Classification Using Machine Learning  
**Date:** 2026-10-12  
**Dataset Partitions:** `train.csv` (Balanced fitting: 2,798 records), `validation.csv` (1,387 records)  
**Evaluation Principle:** Macro-averaged F1 on validation partition (Zero test-set exposure)  

---

## 1. Executive Summary & Comparison Table

Three text classification pipelines were trained with balanced sampling on `train.csv` and evaluated on `validation.csv`:

| Model | Training Time (s) | Validation Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear SVM** | 0.12s | 100.00% | 100.00% | 100.00% | **100.00%** | 100.00% |
| **Logistic Regression** | 0.23s | 99.86% | 99.68% | 99.97% | **99.82%** | 99.86% |
| **Random Forest** | 0.35s | 99.71% | 99.35% | 99.94% | **99.64%** | 99.71% |

### Top Performing Candidate: **Linear SVM**
- **Deployment Artifact:** [`models/best_model.joblib`](../models/best_model.joblib)
- **Key Advantage:** Maximum-margin linear hyperplane separation prevents majority-class gravitational pull on out-of-distribution complaints, and `SoftmaxLinearSVC` provides native confidence probabilities.

---

## 2. Per-Class F1-Score Breakdown (Validation Set)

| Category | Linear SVM | Logistic Regression | Random Forest |
| :--- | :--- | :--- | :--- |
| **Phishing** | 100.0% | 100.0% | 100.0% |
| **Online Financial Fraud** | 100.0% | 100.0% | 100.0% |
| **Identity Theft** | 100.0% | 100.0% | 98.7% |
| **Account Hacking** | 100.0% | 100.0% | 100.0% |
| **Cyberbullying / Online Harassment** | 100.0% | 99.9% | 99.8% |
| **Fake Social Media Account** | 100.0% | 98.7% | 98.7% |
| **Malware / Ransomware** | 100.0% | 100.0% | 100.0% |
| **Online Shopping Fraud** | 100.0% | 100.0% | 100.0% |

---

## 3. Improvements Applied for Unseen Real-World Generalization

1. **Balanced Training Sampling**:
   - Majority class (Cyberbullying) was equalized to ~350 samples during training, eliminating prior probability bias that caused unseen complaints to default into Cyberbullying.
2. **Platform-Neutral Stop Words**:
   - Words like `"twitter"` and `"tweet"` were added to stop words so that complaints mentioning social media platforms do not trigger false Cyberbullying predictions.
3. **Sublinear TF Scaling & Rare Word Retention**:
   - `sublinear_tf=True` and `min_df=1` preserve critical rare technical terminology (e.g. `"credentials"`, `"keylogger"`, `"ransomware"`).

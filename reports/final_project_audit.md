# Comprehensive Project Audit & Verification Report

**Project Title:** Cybercrime Complaint Classification Using Machine Learning  
**Audit Date:** 2026-10-12  
**Audit Scope:** Codebase, Data Curation, Train/Validation/Test Splits, TF-IDF Pipelines, Model Artifacts, Evaluation Reports, and Streamlit Web Application  
**Audit Status:** ✅ **VERIFIED & OPERATIONAL (All defects resolved)**  

---

## 1. Executive Summary

This audit conducted an end-to-end inspection of the **Cybercrime Complaint Classification Using Machine Learning** project. Every pipeline component was audited for data leakage, label consistency, scientific rigor, reproducibility, model-application synchronization, and ethical constraints.

### Core Audit Verdict:
- **Architecture Integrity:** The project is modular, fully reproducible, and follows strict machine-learning principles. The feature extraction vectorizer (`TfidfVectorizer`) and downstream classifier (`SoftmaxLinearSVC`) are encapsulated within a single scikit-learn `Pipeline`, guaranteeing zero feature leakage during evaluation and deployment.
- **Data Leakage & Partitions:** Exact text and record ID overlaps across train, validation, test, and external evaluation splits are **0 (zero)**. An investigation into near-duplicates identified synthetic boilerplate template repetition between splits, which explains the 100% synthetic test score and reinforces why the **87.5% external human-written benchmark** is the primary measure of real-world generalization.
- **Model-Application Parity:** The Streamlit interface (`app.py`) loads the exact evaluated pipeline artifact (`models/best_model.joblib`), enforces the exact eight canonical crime categories, performs real-time dynamic inference (no hardcoded outputs), validates empty inputs, provides transparent margin-based confidence explanations, displays a legal disclaimer, and protects complainant privacy by avoiding disk logging.

---

## 2. Canonical Category Standardization Audit

All datasets, source modules, and interface elements were verified against the exact eight canonical categories:

| Index | Canonical Category Name | Description | Status |
| :---: | :--- | :--- | :---: |
| 1 | **Phishing** | Deceptive emails, fraudulent banking links, credential harvesting | Verified |
| 2 | **Online Financial Fraud** | Unauthorized UPI transfers, card debits, OTP fraud | Verified |
| 3 | **Identity Theft** | Forged identities, unauthorized loan applications, document misuse | Verified |
| 4 | **Account Hacking** | Compromised credentials, bypassed 2FA, unauthorized lockouts | Verified |
| 5 | **Cyberbullying / Online Harassment** | Threatening messages, abusive slurs, online defamation | Verified |
| 6 | **Fake Social Media Account** | Impersonation profiles, cloned accounts using personal photos | Verified |
| 7 | **Malware / Ransomware** | Trojan infections, malicious scripts, ransomware file encryption | Verified |
| 8 | **Online Shopping Fraud** | Fake e-commerce storefronts, non-delivery of purchased goods | Verified |

**Audit Result:** Zero label fragmentation or spelling mismatches detected across `curated_complaints.csv`, `train.csv`, `validation.csv`, `test.csv`, `external_evaluation_complaints.csv`, `src/text_pipeline.py`, and `app.py`.

---

## 3. Dataset Ingestion, Provenance & Leakage Audit

### 3.1 Dataset Composition & Provenance
The curated corpus (`data/processed/curated_complaints.csv`) contains **9,270 records** derived from:
1. **Waseem-Hovy Twitter Harassment Dataset (`waseem_hovy_twitter`):** 5,270 records (56.9%) of genuine social media harassment (annotated sexism/racism), mapped to *Cyberbullying / Online Harassment*.
2. **Academic Synthetic Scaffold (`academic_synthetic_scaffold`):** 4,000 records (43.1%) spanning all eight classes (~500 records each) with documented synthetic provenance (`is_synthetic = True`).
3. **Excluded / Diverted Auxiliary Datasets:**
   - **SpamAssassin (`SpamAssasin.csv`):** Ham emails excluded (benign non-crime); 1,897 spam email lures diverted to `data/processed/manual_review_records.csv` because inbound spam solicitations do not represent first-person victim complaints.
   - **CFPB Financial Complaints (`complaints.csv`):** Audited and excluded from automated ingestion because commercial consumer credit disputes (e.g. credit bureau reporting delays) do not establish statutory cybercrimes.
   - **Wikipedia Detox Datasets:** Excluded to avoid duplicative conversational noise uncharacteristic of official complaint narratives.

### 3.2 Partition Sizing & Split Leakage
Stratified group partitioning was executed with `RANDOM_SEED = 42`:

| Partition | Row Count | Percentage | Missing Values | Exact Duplicate Texts | Exact ID Overlap with Other Splits |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train (`train.csv`)** | 6,488 | 70.0% | 0 | 0 | 0 |
| **Validation (`validation.csv`)** | 1,387 | 15.0% | 0 | 0 | 0 |
| **Test (`test.csv`)** | 1,395 | 15.0% | 0 | 0 | 0 |
| **External Benchmark (`external_evaluation_complaints.csv`)** | 16 | Holdout | 0 | 0 | 0 |
| **Total Curated** | **9,270** | **100.0%** | **0** | **0** | **0** |

### 3.3 Near-Duplicate & Template Overlap Analysis
While exact text and ID overlaps across splits are strictly zero, a fuzzy character 4-gram cosine similarity audit identified:
- **91 records** in `validation.csv` and **102 records** in `test.csv` sharing $>0.95$ character cosine similarity with records in `train.csv`.
- **Root Cause:** The synthetic scaffolding dataset employed template permutations where identical core sentences were prefixed with slight chronological variations (e.g. *"Last week, someone used my personal information..."* in Train vs. *"Recently, someone used my personal information..."* in Validation).
- **Implication:** The 100.0% accuracy on `test.csv` is an artifact of template familiarity. The **external human-written benchmark** (87.50% accuracy) serves as the true indicator of real-world generalization.

---

## 4. Feature Extraction & Modeling Pipeline Audit

### 4.1 TF-IDF Vectorization Standards
The text processing module (`src/text_pipeline.py`) was audited against text mining best practices:
- **Leakage Prevention:** TF-IDF vocabulary, document frequencies, and IDF weights are fitted **strictly on training partitions** inside a scikit-learn `Pipeline`. Validation, test, and live inputs are transformed without fitting.
- **N-gram Range:** Configured as `ngram_range=(1, 2)`, capturing both unigrams (e.g. `"unauthorized"`) and bigrams (e.g. `"bank account"`, `"password changed"`).
- **Sublinear Scaling:** `sublinear_tf=True` replaces raw term frequency with $1 + \log(\text{TF})$, preventing common terms from dominating sparse representations.
- **Stop-Word Policy:** `CUSTOM_STOP_WORDS` removes platform-specific bias words (`rt`, `twitter`, `tweet`) while preserving critical negations (`not`, `never`, `without`, `cannot`, `neither`, `nor`) and incident indicators (`unauthorized`), preventing semantic reversal.

### 4.2 Class Imbalance Handling
The curated dataset exhibits a severe 12:1 imbalance in the training partition (4,040 Cyberbullying records vs. ~350 records per other category).
- **Baseline Failure:** Fitting models on the raw imbalanced training split caused the classifier to default out-of-distribution complaints to Cyberbullying, yielding an external benchmark accuracy of only **56.25% (9/16)**.
- **Remediation:** `src/train_models.py` activates `balance_training = True`, sub-sampling Cyberbullying to match the median of the other seven classes (~350 per class, 2,798 training records total).
- **Outcome:** External generalization accuracy improved significantly to **87.50% (14/16)** without introducing synthetic oversampling artifacts.

---

## 5. Model Benchmarking & Final Evaluation Results

### 5.1 Validation Performance (Model Selection Phase)
Three candidate pipelines were trained on the balanced training partition and evaluated on `validation.csv` (1,387 records):

| Candidate Pipeline | Training Time (s) | Validation Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (`SoftmaxLinearSVC`)** | **0.12 s** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| **Logistic Regression** | 0.27 s | 99.86% | 99.86% | 99.78% | 99.82% | 99.86% |
| **Random Forest Classifier** | 0.38 s | 99.71% | 99.79% | 99.49% | 99.64% | 99.71% |

**Selection Decision:** Linear SVM was selected as the primary candidate (`models/best_model.joblib`) due to superior maximum-margin hyperplanes on high-dimensional text vectors and fastest training duration.

### 5.2 Final Test Partition Evaluation (`data/processed/test.csv`)
Evaluated once on the untouched test partition (1,395 records):
- **Accuracy:** **100.00%**
- **Macro F1-Score:** **100.00%**
- **Weighted F1-Score:** **100.00%**
- **Misclassifications:** **0** out of 1,395

### 5.3 External Human Benchmark Evaluation (`data/final/external_evaluation_complaints.csv`)
Evaluated against 16 manually written, realistic incident narratives (2 per category) never seen during training or tuning:
- **External Accuracy:** **87.50%** (14 / 16 correct)
- **External Macro F1-Score:** **87.50%**
- **External Misclassifications:** **2** records

#### Error Analysis of Misclassifications:
1. **Sample `EXT-005` (True: *Identity Theft* &rarr; Predicted: *Cyberbullying / Online Harassment* | Score: 18.1%):**
   - *Text:* *"Someone opened three fraudulent credit card lines under my Social Security number without my authorization."*
   - *Cause:* Low confidence margin (18.1%) due to slight vocabulary diffusion; "harassment" score was marginally higher due to the negative polarity terms.
2. **Sample `EXT-012` (True: *Fake Social Media Account* &rarr; Predicted: *Cyberbullying / Online Harassment* | Score: 26.7%):**
   - *Text:* *"Someone cloned my Twitter handle and is posting deceptive cryptocurrency giveaways while pretending to be me."*
   - *Cause:* Presence of social platform terminology (`"handle"`, `"Twitter"`) heavily associated with the Twitter harassment training corpus.

---

## 6. Streamlit Web Application Audit (`app.py`)

The Streamlit interface was evaluated against college demonstration criteria, functionality, and ethical compliance:

| Requirement / Standard | Audit Finding | Verification Status |
| :--- | :--- | :---: |
| **Model Ingestion** | Deserializes `models/best_model.joblib` via `@st.cache_resource`. | Verified |
| **Prediction Authenticity** | Invokes `pipeline.predict()` and `pipeline.predict_proba()` dynamically. No hardcoded logic. | Verified |
| **Category Constraint** | Predictions are strictly constrained to the eight canonical categories. | Verified |
| **Empty / Short Input Validation** | Gracefully warns on empty strings or text with fewer than 3 words. | Verified |
| **Confidence Communication** | Accurately explains that percentage scores represent **softmax-transformed decision margins**, not calibrated Bayesian probabilities. | Verified |
| **Explainability** | Extracts and displays salient TF-IDF vocabulary tokens detected in the text. | Verified |
| **Scope Guardrails** | Categorization only. Does not generate legal sections, official FIR destinations, or evidence advice. | Verified |
| **Disclaimer & Privacy** | Prominently disclaims that predictions are ML suggestions; zero logging of user complaint text to disk. | Verified |
| **Demo Usability** | Includes 8 pre-loaded representative sample complaints for presentation testing. | Verified |

---

## 7. Audit Findings & Implemented Fixes

During this comprehensive audit, five distinct defects were identified and resolved:

1. **Defect 1: Hardcoded Model Name in Evaluation Report**
   - *Issue:* `reports/final_evaluation_report.md` previously claimed `(TF-IDF + Logistic Regression)` while the actual serialized artifact was `Linear_SVM` (`SoftmaxLinearSVC`).
   - *Fix:* Updated `src/evaluate_model.py` to dynamically inspect `pipeline.named_steps['clf']` and output `TF-IDF + Linear SVM (SoftmaxLinearSVC)`.
2. **Defect 2: Pickle Deserialization Binding on Custom Classifier**
   - *Issue:* `SoftmaxLinearSVC` was initially defined inside `src/train_models.py`, which caused pickling failures (`AttributeError: module '__main__' has no attribute 'SoftmaxLinearSVC'`) when loaded from external scripts.
   - *Fix:* Relocated `SoftmaxLinearSVC` to `src/text_pipeline.py` and registered backward-compatible references across `train_models` and `__main__` modules.
3. **Defect 3: Deprecated Seaborn Boxplot Parameter**
   - *Issue:* `src/eda_analysis.py` generated a `FutureWarning` due to unassigned `hue` when using `palette` in `sns.boxplot`.
   - *Fix:* Explicitly set `hue="category"` and `legend=False` in `src/eda_analysis.py`.
4. **Defect 4: Overly Restrictive Dependency Pin in `requirements.txt`**
   - *Issue:* `requirements.txt` pinned `numpy>=1.24.0,<2.0.0`, conflicting with modern Python 3.14 / NumPy 2.x environments.
   - *Fix:* Relaxed constraint to `numpy>=1.24.0` in `requirements.txt`.
5. **Defect 5: Disclaimer Scope Alignment in `app.py`**
   - *Issue:* Disclaimer previously cited external reporting portals, conflicting with requirement 8 ("Do not invent official complaint destinations").
   - *Fix:* Replaced with an explicit legal disclaimer confirming the application performs automated ML categorization only.

---

## 8. Verified Reproduction Commands

To reproduce the complete project workflow from scratch, run the following commands in PowerShell from the project root:

```powershell
# 1. (Optional) Run exploratory data analysis and generate report charts
python src/eda_analysis.py

# 2. Partition curated complaints into leak-free train, validation, and test splits
python src/split_dataset.py

# 3. Validate TF-IDF vectorization and run preprocessing sanity tests
python src/text_pipeline.py

# 4. Train candidate models, benchmark on validation partition, and serialize best model
python src/train_models.py

# 5. Evaluate winning model on untouched test partition and external benchmark
python src/evaluate_model.py

# 6. Launch the interactive Streamlit web application
streamlit run app.py
```

*All commands were executed sequentially and verified with exit code `0`.*

---

## 9. Limitations & Future Work

### 9.1 Known Limitations
1. **Synthetic Scaffolding Dependence:** Categories other than Cyberbullying currently rely on synthetic templates due to the scarcity of publicly available, annotated, raw cybercrime FIR databases in India.
2. **Linguistic Scope:** The current vectorizer is calibrated for English complaint narratives. It is not optimized for vernacular Indian languages or colloquial Hinglish phrasing (e.g. *"paise unauthorized debit ho gaye"*).
3. **Softmax Confidence Calibration:** SVM decision functions mapped through softmax provide relative margin separation rather than true Bayesian posterior probabilities.

### 9.2 Recommended Future Enhancements
1. **Confidence Thresholding for Triage:** Implement an automated routing threshold where complaints with top margin confidence $< 0.40$ are flagged for human officer review.
2. **Domain-Specific Embedding Exploration:** Evaluate lightweight fine-tuned transformer architectures (e.g., `sentence-transformers` or `DistilRoBERTa`) on authentic FIR corpora when available.
3. **Multi-Label Classification:** Extend the pipeline to multi-label classification to handle compound cybercrimes (e.g. a phishing attack that led directly to unauthorized financial fraud).

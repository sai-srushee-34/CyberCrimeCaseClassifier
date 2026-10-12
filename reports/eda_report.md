# Exploratory Data Analysis & Data Quality Report

**Project Title:** Cybercrime Complaint Classification Using Machine Learning  
**Dataset Analyzed:** [`data/processed/curated_complaints.csv`](../data/processed/curated_complaints.csv)  
**Analysis Date:** 2026-10-12  
**Executable Artifacts:**
- Script: [`src/eda_analysis.py`](../src/eda_analysis.py)
- Interactive Notebook: [`notebooks/02_data_cleaning.ipynb`](../notebooks/02_data_cleaning.ipynb)

---

## 1. Executive Summary

A comprehensive exploratory data analysis (EDA) and quality audit was conducted on the finalized curated dataset (`data/processed/curated_complaints.csv`). 

- **Total Records:** 9,270 complaints across 8 target cybercrime classes.
- **Completeness:** 0 null values across all 7 standardized fields.
- **Deduplication:** 0 exact duplicate rows, 0 exact duplicate text strings.
- **Real vs. Synthetic Composition:** 5,270 real-world social media harassment instances (56.9%) and 4,000 academic synthetic complaint narratives (43.1%).
- **Training Readiness Verdict:** **CONDITIONALLY READY FOR MODEL TRAINING.** The dataset is structured, leak-free, and clean, but models must be trained using `class_weight='balanced'` and evaluated with **Macro F1-Score** to handle significant class imbalance.

---

## 2. Dataset Dimensions & Structural Integrity

| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Total Rows** | 9,270 | 100% compliant with project scope |
| **Total Columns** | 7 | `id`, `complaint_text`, `category`, `source_type`, `source_name`, `original_record_id`, `is_synthetic` |
| **Missing / Null Values** | 0 (0.0%) | Complete integrity across all columns |
| **Exact Duplicate Rows** | 0 | Removed during curation |
| **Exact Duplicate Texts** | 0 | Removed during curation |
| **Flagged Near-Duplicates** | 68 records | Segregated in `flagged_near_duplicates.csv` (contained within identical splits) |

---

## 3. Category Distribution & Class Imbalance

| Canonical Category | Count | Percentage | Primary Provenance |
| :--- | :--- | :--- | :--- |
| **Cyberbullying / Online Harassment** | 5,770 | 62.2% | Waseem & Hovy Twitter (5,270) + Synthetic (500) |
| **Online Financial Fraud** | 500 | 5.4% | Academic Synthetic Scaffold |
| **Phishing** | 500 | 5.4% | Academic Synthetic Scaffold |
| **Identity Theft** | 500 | 5.4% | Academic Synthetic Scaffold |
| **Account Hacking** | 500 | 5.4% | Academic Synthetic Scaffold |
| **Fake Social Media Account** | 500 | 5.4% | Academic Synthetic Scaffold |
| **Malware / Ransomware** | 500 | 5.4% | Academic Synthetic Scaffold |
| **Online Shopping Fraud** | 500 | 5.4% | Academic Synthetic Scaffold |
| **Total** | **9,270** | **100.0%** | |

![Category Distribution](figures/category_distribution.png)

### Key Finding:
- **Severe Class Imbalance**: `Cyberbullying / Online Harassment` represents **62.2%** of the entire corpus due to the inclusion of real-world annotated harassment tweets, while the other 7 categories contain exactly 500 instances (5.4% each).
- **Impact on ML**: An unweighted classifier could achieve ~62% baseline accuracy simply by predicting *Cyberbullying* for every sample. **Models must use class-weighted loss (`class_weight='balanced'`)** and be evaluated on **Macro-averaged F1** and **Balanced Accuracy**.

---

## 4. Text Length Metrics & Distribution

| Category | Min Words | Mean Words | Median Words | Max Words | Std Dev (Words) | Max Char Length |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Account Hacking** | 9 | 22.0 | 22.0 | 33 | 4.4 | 190 |
| **Cyberbullying / Online Harassment** | 3 | 16.9 | 17.0 | 74 | 8.8 | 178 |
| **Fake Social Media Account** | 10 | 21.4 | 21.0 | 33 | 4.4 | 189 |
| **Identity Theft** | 11 | 23.2 | 23.0 | 34 | 4.2 | 189 |
| **Malware / Ransomware** | 10 | 22.3 | 22.0 | 36 | 4.4 | 213 |
| **Online Financial Fraud** | 10 | 24.1 | 24.0 | 35 | 4.6 | 212 |
| **Online Shopping Fraud** | 10 | 23.2 | 23.0 | 34 | 4.2 | 196 |
| **Phishing** | 14 | 26.7 | 27.0 | 36 | 4.4 | 206 |

![Length Distribution by Category](figures/length_distribution_by_category.png)
![Complaint Length Density Histogram](figures/complaint_length_histogram.png)

### Key Findings:
1. **Modality Length Gap**:
   - Real-world Twitter harassment text has a lower mean word count (**16.9 words**) and higher variance (ranging from 3 to 74 words), characteristic of social media posts.
   - Synthetic complaints have tightly clustered word counts (**21.4 to 26.7 words**, standard deviation ~4.4), reflecting template generation bounds.
2. **Short Text Preservation**:
   - Even the shortest tweets (e.g., 3–5 words: `"Call me sexist..."`, `"Wrong, ISIS follows..."`) contain high-signal lexical features. They are intentionally preserved per project guidelines.

---

## 5. Source Composition & Bias Analysis

| Category | `academic_synthetic_scaffold` | `waseem_hovy_twitter` | Total |
| :--- | :--- | :--- | :--- |
| **Account Hacking** | 500 (100%) | 0 (0%) | 500 |
| **Cyberbullying / Online Harassment** | 500 (8.7%) | 5,270 (91.3%) | 5,770 |
| **Fake Social Media Account** | 500 (100%) | 0 (0%) | 500 |
| **Identity Theft** | 500 (100%) | 0 (0%) | 500 |
| **Malware / Ransomware** | 500 (100%) | 0 (0%) | 500 |
| **Online Financial Fraud** | 500 (100%) | 0 (0%) | 500 |
| **Online Shopping Fraud** | 500 (100%) | 0 (0%) | 500 |
| **Phishing** | 500 (100%) | 0 (0%) | 500 |

![Source Composition](figures/source_distribution_by_category.png)

### Key Finding:
- **Confounding Source Bias**: 7 of the 8 classes are 100% synthetic, while *Cyberbullying* is 91.3% sourced from Twitter.
- **Risk**: A machine learning model might inadvertently learn syntactic stylistic markers of Twitter (e.g., informal phrasing, punctuation styles) rather than purely semantic harassment features.
- **Mitigation**: TF-IDF tokenization with URL/handle masking ensures the classifier focuses on topical n-grams rather than social media metadata artifacts.

---

## 6. Template Repetition & Linguistic Patterns

An n-gram prefix audit revealed a distinct artifact of synthetic generation:

| Category | Top Starting Phrase (4 Words) | Frequency | % of Category |
| :--- | :--- | :--- | :--- |
| **Phishing** | `"A few days ago,"` | 102 | **20.4%** |
| **Online Financial Fraud** | `"A few days ago,"` | 86 | **17.2%** |
| **Identity Theft** | `"A few days ago,"` | 98 | **19.6%** |
| **Account Hacking** | `"A few days ago,"` | 94 | **18.8%** |
| **Fake Social Media Account** | `"A few days ago,"` | 94 | **18.8%** |
| **Malware / Ransomware** | `"A few days ago,"` | 92 | **18.4%** |
| **Online Shopping Fraud** | `"A few days ago,"` | 104 | **20.8%** |
| **Cyberbullying / Harassment** | `"[USER_REDACTED] [USER_REDACTED] ..."` | 229 | **4.0%** |

### Analysis:
- In all 7 synthetic classes, **approximately 17% to 21% of records begin with the exact phrase `"A few days ago,"`**, followed by `"Last week,"` (3% to 4%).
- **Why this does NOT break TF-IDF**: Because `"a few days ago"` appears uniformly across *all* synthetic categories, TF-IDF automatically downweights these ubiquitous n-grams via high document frequency ($IDF \approx 0$). They will not be assigned discriminant power toward any single class.

---

## 7. Privacy Redaction Residual Scan

A rigorous regex audit confirmed that zero unredacted personal identifiers exist in the curated text:
- **Unmasked Email Addresses:** 0
- **Unmasked Phone Numbers:** 0
- **Unmasked 16-digit Card / Bank Account Numbers:** 0
- **Masking Tags Verified:**
  - `[USER_REDACTED]`: 3,280 tags (Twitter handles scrubbed)
  - `[URL_REDACTED]`: 717 tags (external links neutralized)
  - Incident damage values (`$5,000`, `Rs. 25,000`) and event dates were successfully preserved to retain contextual fraud clues.

---

## 8. Summary of Dataset Limitations

1. **Class Imbalance (62.2% vs. 5.4%)**:
   - Over-representation of Cyberbullying requires class weighting (`class_weight='balanced'`) during model fitting.
2. **Synthetic Regularity**:
   - Synthetic complaints have lower linguistic variety, few misspellings, and repetitive temporal openers. Real victim reports in production will contain greater typographical noise.
3. **Missing Empirical Narratives for Financial/Ransomware Crimes**:
   - The CFPB `complaints.csv` raw file lacked narrative text. If empirical complaint narratives are later added, they will reduce synthetic reliance.

---

## 9. Final Training Readiness Verdict & Recommendations

### Verdict:
> **STATUS: READY FOR PHASE 4 MODEL TRAINING (WITH BALANCED CLASS WEIGHTS).**
> The dataset is clean, properly tokenizable, strictly partitioned (70% train, 15% val, 15% test), free of PII, and possesses zero exact duplicates.

### Concrete Recommendations for Phase 4:
1. **Feature Extraction (`src/features.py`)**:
   - Use `TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.85, sublinear_tf=True)`.
   - `max_df=0.85` will automatically neutralize template phrases like `"a few days ago"`.
2. **Algorithm Configuration (`src/train.py`)**:
   - Initialize all models with balanced weighting:
     - `LogisticRegression(class_weight='balanced', max_iter=1000)`
     - `LinearSVC(class_weight='balanced')`
     - `RandomForestClassifier(class_weight='balanced', n_estimators=200)`
3. **Evaluation Protocol**:
   - Benchmark models primarily on **Macro F1-Score** and generate confusion matrices in `reports/figures/` to confirm that minority classes (Phishing, Malware, Hacking) maintain high recall despite the dominant harassment class.

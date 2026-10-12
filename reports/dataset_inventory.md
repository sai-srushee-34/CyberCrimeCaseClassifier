# Comprehensive Dataset Inventory & Feasibility Assessment Report

**Project Title:** Cybercrime Complaint Classification Using Machine Learning  
**Last Updated:** 2026-10-12  
**Target Canonical Categories (8):**
1. Phishing
2. Online Financial Fraud
3. Identity Theft
4. Account Hacking
5. Cyberbullying / Online Harassment
6. Fake Social Media Account
7. Malware / Ransomware
8. Online Shopping Fraud

---

## 1. Executive Summary

A comprehensive inventory audit was performed across all datasets located under `data/raw/` (including all newly supplied files) and auxiliary benchmarking data in `data/final/`.

| Dataset Name | File Path | Format / Size | Records | Modality / Data Type | 8-Class Suitability | Pipeline Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CFPB Consumer Complaints** | `data/raw/complaints.csv` | CSV (5.55 GB) | > 15 Million rows | 15 Categorical metadata fields (no text) | ❌ 0 / 8 categories (Narrative column missing) | **EXCLUDE (METADATA-ONLY)** |
| **Twitter Harassment Dataset** | `data/raw/twitter_parsed_dataset.csv` | CSV (2.51 MB) | 16,851 rows | Social media tweets (sexism, racism, none) | ✅ 5,283 tweets for Cyberbullying / Online Harassment | **CURATE (HARASSMENT SUBSET)** |
| **Apache SpamAssassin** | `data/raw/SpamAssasin.csv` | CSV (14.8 MB) | 5,809 emails | Full emails (ham developer emails & spam lures) | ⚠️ 1,718 spam lures (no 8-class labels) | **DIVERT SPAM TO MANUAL REVIEW** |
| **Wikipedia Detox (Attack)** | `data/raw/attack_parsed_dataset.csv` | CSV (50.3 MB) | 115,864 rows | Wikipedia editor talk-page edit disputes | ⚠️ Talk page editor debates (Domain mismatch) | **EXCLUDE (DOMAIN MISMATCH)** |
| **Wikipedia Detox (Aggression)**| `data/raw/aggression_parsed_dataset.csv` | CSV (50.4 MB) | 115,864 rows | Wikipedia editor talk-page edit disputes | ⚠️ Talk page editor debates (Domain mismatch) | **EXCLUDE (DOMAIN MISMATCH)** |
| **Wikipedia Detox (Toxicity)** | `data/raw/toxicity_parsed_dataset.csv` | CSV (66.7 MB) | 159,686 rows | Wikipedia editor talk-page edit disputes | ⚠️ Talk page editor debates (Domain mismatch) | **EXCLUDE (DOMAIN MISMATCH)** |
| **Phishing Websites Data Set** | `data/raw/phishing+websites.zip` | Weka ARFF (1.0 MB) | 11,055 instances | Tabular web telemetry (`-1, 0, 1`) | ❌ 0 / 8 categories (Non-NLP) | **EXCLUDE (TABULAR NON-TEXT)** |
| **SMS Spam Collection v.1** | `data/raw/sms+spam+collection.zip` | TSV (203 KB) | 5,574 messages | Short SMS mobile messages (`ham`/`spam`) | ⚠️ 747 spam lures (no 8-class labels) | **DIVERT SPAM TO MANUAL REVIEW** |
| *(Auxiliary)* **Synthetic Complaints** | `data/final/cybercrime_complaints_4000_synthetic.csv` | CSV (839 KB) | 4,000 complaints | First-person incident narratives (500 per class) | ✅ 8 / 8 categories (Synthetic scaffold) | **CURATE AS SYNTHETIC SCAFFOLD** |

---

## 2. Deep Technical Audits of Newly Provided Datasets

### A. CFPB `complaints.csv` (5.55 GB)
* **Columns (15):** `Date received`, `Product`, `Sub-product`, `Issue`, `Sub-issue`, `Company public response`, `Company`, `State`, `ZIP code`, `Tags`, `Submitted via`, `Date sent to company`, `Company response to consumer`, `Timely response?`, `Complaint ID`.
* **Critical Finding:** The `Consumer complaint narrative` text column is **completely missing**.
* **Implication:** The file only contains administrative categorical flags. Silently synthesizing or concatenating issue/product fields into fake complaint text would violate academic integrity.
* **Verdict:** **EXCLUDED.** If CFPB complaints are desired for Online Financial Fraud and Identity Theft, the dataset must be re-exported with the "Consumer complaint narrative" checkbox selected.

### B. Twitter Harassment Dataset (`twitter_parsed_dataset.csv`)
* **Origin:** Waseem & Hovy (2016) Twitter Hate Speech / Online Harassment corpus.
* **Columns:** `index`, `id`, `Text`, `Annotation`, `oh_label`.
* **Distribution:**
  - `oh_label = 0` (`Annotation = none`): 11,501 benign tweets $\rightarrow$ **Excluded**.
  - `oh_label = 1` (`sexism`: 3,377, `racism`: 1,970): 5,347 real-world harassment tweets.
* **Cleaning & Redaction:** User handles (`@username` $\rightarrow$ `[USER_REDACTED]`) and URLs (`[URL_REDACTED]`) are scrubbed. Deduplication yields 5,283 unique records.
* **Verdict:** **CURATED.** Mapped defensibly to **Cyberbullying / Online Harassment** with provenance `source_type="real_social_media_harassment"`, `source_name="waseem_hovy_twitter"`, `is_synthetic=False`.

### C. Apache SpamAssassin Corpus (`SpamAssasin.csv`)
* **Columns:** `sender`, `receiver`, `date`, `subject`, `body`, `label`, `urls`.
* **Distribution:**
  - `label = 0` (`ham`): 4,091 developer mailing-list emails $\rightarrow$ **Excluded as non-crime**.
  - `label = 1` (`spam`): 1,718 unsolicited spam messages (commercial adverts, financial scams, phishing lures).
* **Perspective & Label Mismatch:** These represent incoming attack emails, not reports filed by victims. Furthermore, they are only labeled with binary `spam`, lacking 8-class distinction.
* **Verdict:** **DIVERTED TO MANUAL REVIEW.** 1,718 spam records exported to `data/processed/manual_review_records.csv` with `source_type="scam_lure_email"`.

### D. Wikipedia Detox Datasets (`attack`, `aggression`, `toxicity`)
* **Origin:** Wulczyn et al. (2017) Wikipedia Talk Page Detox project (~390,000 comments).
* **Domain Mismatch:** These comments are arguments between Wikipedia volunteer editors over article wording and talk-page reverts, rather than cybercrime complaints.
* **Volume Distortion:** Ingesting 14,000+ personal attack comments from Wikipedia would inflate the harassment category to over 20,000 samples while the other 7 cybercrime classes have only 500 samples, inducing massive model distortion.
* **Verdict:** **EXCLUDED.**

---

## 3. Curated Corpus Summary (`data/processed/curated_complaints.csv`)

| Canonical Category | Total Curated | Train (70%) | Val (15%) | Test (15%) | Provenance Sources |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phishing** | 500 | 350 | 75 | 75 | Synthetic Scaffold |
| **Online Financial Fraud** | 500 | 350 | 75 | 75 | Synthetic Scaffold |
| **Identity Theft** | 500 | 350 | 75 | 75 | Synthetic Scaffold |
| **Account Hacking** | 500 | 350 | 75 | 75 | Synthetic Scaffold |
| **Cyberbullying / Online Harassment** | 5,770 | 4,036 | 864 | 870 | Synthetic (497) + Real Twitter Harassment (5,273) |
| **Fake Social Media Account** | 500 | 350 | 75 | 75 | Synthetic Scaffold |
| **Malware / Ransomware** | 500 | 350 | 75 | 75 | Synthetic Scaffold |
| **Online Shopping Fraud** | 500 | 350 | 75 | 75 | Synthetic Scaffold |
| **Total** | **9,270** | **6,486** | **1,389** | **1,395** | |

> **Manual Review Pipeline:** 2,465 candidate scam lures (747 SMS + 1,718 SpamAssassin emails) are segregated in `data/processed/manual_review_records.csv` to allow human verification without corrupting the training set.

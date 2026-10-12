# Comprehensive Data Curation Report

**Project Title:** Cybercrime Complaint Classification Using Machine Learning  
**Execution Date:** 2026-10-12  
**Target Output Categories (8):** Canonical Cybercrime Taxonomy  

---

## 1. Executive Summary

This report documents the curation of the updated `data/raw/` repository, which now includes additional external corpora:
- `complaints.csv` (5.55 GB CFPB export)
- `twitter_parsed_dataset.csv` (Waseem & Hovy Twitter harassment dataset)
- `SpamAssasin.csv` (Apache SpamAssassin email corpus)
- `aggression_parsed_dataset.csv`, `attack_parsed_dataset.csv`, `toxicity_parsed_dataset.csv` (Wikipedia Talk Page Detox)
- `phishing+websites.zip` (UCI Phishing ARFF)
- `sms+spam+collection.zip` (UCI SMS Spam)
- `data/final/cybercrime_complaints_4000_synthetic.csv` (Auxiliary multi-class scaffold)

### Key Metrics:
- **Curated Dataset:** [`data/processed/curated_complaints.csv`](../data/processed/curated_complaints.csv)
- **Total Valid Curated Records:** 9,270
- **Stratified Partition:** Train = 6,486 (70%), Val = 1,389 (15%), Test = 1,395 (15%)
- **Manual Review Candidates (Lures):** 2,465 ([`data/processed/manual_review_records.csv`](../data/processed/manual_review_records.csv))
- **Flagged Near-Duplicate Records:** 68 ([`data/processed/flagged_near_duplicates.csv`](../data/processed/flagged_near_duplicates.csv))
- **Raw Files Altered:** 0 (All raw archives strictly preserved)

---

## 2. Ingestion & Exclusion Accounting Across All Raw Sources

| Source Dataset | Initial Records / Size | Curated Records | Excluded / Diverted | Technical Rationale & Decision |
| :--- | :--- | :--- | :--- | :--- |
| **CFPB_complaints** | 5.55 GB file (estimated > 15M records) | 0 | All records excluded | Missing 'Consumer complaint narrative' column. The provided CSV contains only 15 categorical metadata fields (Date, Product, Issue, Company, etc.) with no text. |
| **Twitter_Harassment** | 16851 | 5331 | 11504 benign non-harassment tweets (oh_label=0) | Annotated sexism/racism harassment tweets mapped to Cyberbullying / Online Harassment. |
| **SpamAssassin** | 5809 | 0 | 4091 ham developer emails excluded; 1718 spam lures diverted to manual review | Ham is non-crime; spam lures lack victim complaint narrative and multi-class target labels. |
| **aggression_parsed_dataset.csv** | > 115,000 talk-page comments | 0 | All comments excluded from primary corpus | Domain mismatch: represents Wikipedia talk-page editorial edit disputes rather than formal cybercrime complaints. Ingesting 14k+ talk comments would also severely distort the 8-class balance (40:1 ratio against other crimes). |
| **attack_parsed_dataset.csv** | > 115,000 talk-page comments | 0 | All comments excluded from primary corpus | Domain mismatch: represents Wikipedia talk-page editorial edit disputes rather than formal cybercrime complaints. Ingesting 14k+ talk comments would also severely distort the 8-class balance (40:1 ratio against other crimes). |
| **toxicity_parsed_dataset.csv** | > 115,000 talk-page comments | 0 | All comments excluded from primary corpus | Domain mismatch: represents Wikipedia talk-page editorial edit disputes rather than formal cybercrime complaints. Ingesting 14k+ talk comments would also severely distort the 8-class balance (40:1 ratio against other crimes). |
| **SMS_Spam_Collection** | 5574 | 0 | 4827 ham messages excluded; 747 spam lures diverted to manual review | Ham is non-crime; spam lures lack victim perspective and 8-class labels. |
| **Phishing_Websites_ARFF** | 11,055 instances (31 attributes) | 0 | 11,055 instances excluded | Incompatible modality: discrete tabular website telemetry (-1, 0, 1). No text. |
| **Synthetic_Scaffold** | 4000 | 4000 | 0 excluded | Scaffolding baseline covering all 8 classes with explicit synthetic provenance. |

---

## 3. Class Distribution Across Stratified Splits

| Canonical Category | Total Curated | Train (70%) | Validation (15%) | Test (15%) | Primary Sources |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phishing** | 500 | 350 | 76 | 74 | Synthetic Scaffold |
| **Online Financial Fraud** | 500 | 349 | 76 | 75 | Synthetic Scaffold |
| **Identity Theft** | 500 | 348 | 74 | 78 | Synthetic Scaffold |
| **Account Hacking** | 500 | 350 | 75 | 75 | Synthetic Scaffold |
| **Cyberbullying / Online Harassment** | 5,770 | 4,038 | 865 | 867 | Synthetic + Twitter Harassment |
| **Fake Social Media Account** | 500 | 350 | 75 | 75 | Synthetic Scaffold |
| **Malware / Ransomware** | 500 | 350 | 74 | 76 | Synthetic Scaffold |
| **Online Shopping Fraud** | 500 | 351 | 74 | 75 | Synthetic Scaffold |
| **Total** | **9,270** | **6,486** | **1,389** | **1,395** | |

---

## 4. Source Composition & Provenance

| Source Name | Source Type | Curated Count | Percentage |
| :--- | :--- | :--- | :--- |
| **waseem_hovy_twitter** | `waseem_hovy_twitter` | 5,270 | 56.9% |
| **academic_synthetic_scaffold** | `academic_synthetic_scaffold` | 4,000 | 43.1% |

---

## 5. Privacy Redaction Metrics

The PII redaction engine parsed all incoming records before export:
- **Twitter User Handles (`@user`):** 5,756 masked
- **Hyperlinks & URLs (`https://...`):** 769 masked
- **Email Addresses:** 0 masked
- **Phone Numbers:** 0 masked
- **Account / Card Numbers:** 0 masked
- **IP Addresses:** 0 masked

---

## 6. Critical Technical Findings & Next Steps

1. **CFPB `complaints.csv` Finding**:
   - The user provided a 5.55 GB CSV (`complaints.csv`), but upon inspection, it **does not contain the `Consumer complaint narrative` column**.
   - It only includes categorical metadata (`Date received`, `Product`, `Issue`, `Company`, `State`, etc.).
   - To utilize real-world complaints for **Online Financial Fraud**, **Identity Theft**, and **Online Shopping Fraud**, download the CFPB extract that includes `Consumer complaint narrative` (the narrative checkbox on consumerfinance.gov).
2. **Scam Lures in Manual Review**:
   - 2,465 incoming scam messages (747 SMS from UCI and 1,718 SpamAssassin emails) are preserved in `manual_review_records.csv`. They represent attack lures rather than victim complaint reports.
3. **Leakage Prevention**:
   - Stratified splits prevent near-duplicate leakage by grouping near-duplicate clusters together.

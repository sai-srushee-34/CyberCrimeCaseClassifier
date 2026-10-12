# Dataset Partitioning & Leakage Verification Report

**Project Title:** Cybercrime Complaint Classification Using Machine Learning  
**Execution Date:** 2026-10-12  
**Curated Source:** [`data/processed/curated_complaints.csv`](../data/processed/curated_complaints.csv)  
**Partition Strategy:** 70% Training / 15% Validation / 15% Test  
**Random Seed:** `42` (Deterministic & Documented)  
**Grouping Identifier:** Text cluster hash derived from normalized complaint text  

---

## 1. Executive Summary

The curated cybercrime complaint dataset (9,270 records) was partitioned into strictly isolated partitions with zero data leakage:
- **Training Set (`train.csv`):** 6,488 records (69.99%)
- **Validation Set (`validation.csv`):** 1,387 records (14.96%)
- **Test Set (`test.csv`):** 1,395 records (15.05%)
- **External Evaluation Set:** 16 unseen test records isolated at [`data/final/external_evaluation_complaints.csv`](../data/final/external_evaluation_complaints.csv) (never used for training or tuning)

---

## 2. Leakage Prevention Protocol

To prevent optimistic bias, the following leakage checks were enforced:
1. **Cluster Containment**:
   - Total records: 9,270
   - Unique cluster groups: 9,230
   - Any complaints with near-identical wording or derivative origin share a cluster group and are strictly assigned to the same partition.
2. **Partition Overlap Audit**:
   - `Train` $\cap$ `Validation` Record Overlap: **0 records**
   - `Train` $\cap$ `Test` Record Overlap: **0 records**
   - `Validation` $\cap$ `Test` Record Overlap: **0 records**
3. **Feature Preprocessing Separation**:
   - **TF-IDF is NOT fitted across the whole corpus.**
   - TF-IDF vectorizers must be fit **exclusively** on `train.csv` inside an end-to-end `scikit-learn` Pipeline.

---

## 3. Class Distribution Across Splits

| Canonical Category | Train (70.0%) | Validation (15.0%) | Test (15.0%) | Total Curated |
| :--- | :--- | :--- | :--- | :--- |
| **Phishing** | 350 | 76 | 74 | 500 |
| **Online Financial Fraud** | 349 | 76 | 75 | 500 |
| **Identity Theft** | 348 | 74 | 78 | 500 |
| **Account Hacking** | 350 | 75 | 75 | 500 |
| **Cyberbullying / Online Harassment** | 4,040 | 863 | 867 | 5,770 |
| **Fake Social Media Account** | 350 | 75 | 75 | 500 |
| **Malware / Ransomware** | 350 | 74 | 76 | 500 |
| **Online Shopping Fraud** | 351 | 74 | 75 | 500 |
| **Total** | **6,488** | **1,387** | **1,395** | **9,270** |

---

## 4. Source Composition Across Splits

| Source Name | Train Count | Val Count | Test Count | Total |
| :--- | :--- | :--- | :--- | :--- |
| `academic_synthetic_scaffold` | 2,799 | 596 | 605 | 4,000 |
| `waseem_hovy_twitter` | 3,689 | 791 | 790 | 5,270 |

---

## 5. External Evaluation Benchmark

- **File Path:** [`data/final/external_evaluation_complaints.csv`](../data/final/external_evaluation_complaints.csv)
- **Record Count:** 16 manually written complaints (covering all 8 classes)
- **Role:** Serves as an untouched out-of-distribution evaluation benchmark for final project demonstration. It is excluded from cross-validation, hyperparameter tuning, and training.

---

## 6. Ready-to-Use Artifacts

| Artifact | File Path | Records |
| :--- | :--- | :--- |
| **Training Partition** | [`data/processed/train.csv`](../data/processed/train.csv) | 6,488 |
| **Validation Partition** | [`data/processed/validation.csv`](../data/processed/validation.csv) | 1,387 |
| **Test Partition** | [`data/processed/test.csv`](../data/processed/test.csv) | 1,395 |
| **External Benchmark** | [`data/final/external_evaluation_complaints.csv`](../data/final/external_evaluation_complaints.csv) | 16 |
| **Split Verification Report** | [`reports/split_report.md`](split_report.md) | Audit Documentation |

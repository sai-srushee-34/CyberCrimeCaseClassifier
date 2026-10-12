# Cybercrime Complaint Classification Using Machine Learning

An end-to-end Machine Learning course project designed to categorize incoming natural-language cybercrime complaints into predefined crime categories to streamline incident triage and automated routing.

---

## 🎯 Project Objective

Given unstructured, natural-language text from a complainant (e.g., an email, incident description, or FIR narrative), the system classifies the complaint into one of **8 distinct cybercrime categories**:

1. **Phishing** (Deceptive emails, fake banking portals, credential harvesting)
2. **Online Financial Fraud** (UPI scams, credit/debit card fraud, unauthorized transactions)
3. **Identity Theft** (Impersonation, forged credentials, Aadhaar/SSN misuse)
4. **Account Hacking** (Compromised email, social media, or portal accounts)
5. **Cyberbullying / Online Harassment** (Threatening messages, cyberstalking, online defamation)
6. **Fake Social Media Account** (Cloned profiles, fake celebrity/brand handles)
7. **Malware / Ransomware** (Ransomware lockouts, spyware, Trojan infections)
8. **Online Shopping Fraud** (Fake e-commerce stores, non-delivery of items, counter-party fraud)

---

## 🛠️ Technology Stack

| Layer | Tools & Libraries |
| :--- | :--- |
| **Language** | Python (3.9+) |
| **Data Manipulation** | Pandas, NumPy |
| **Text Vectorization** | TF-IDF (`TfidfVectorizer` from Scikit-Learn) |
| **Machine Learning Algorithms** | Logistic Regression, Linear Support Vector Machine (Linear SVM), Random Forest |
| **Evaluation & Metrics** | Scikit-Learn (`classification_report`, `confusion_matrix`, `f1_score`, `precision_score`, `recall_score`) |
| **Data Visualization** | Matplotlib, Seaborn |
| **Model Serialization** | Joblib (`joblib.dump`, `joblib.load`) |
| **User Interface / Demo** | Flask + Stitch Modern Responsive Web UI |

---

## 📁 Project Structure

```text
ML_CBP/
├── app.py               # Flask backend web server & live inference API
├── templates/           # Web application frontend templates
│   └── index.html       # Clean incident input & classification output UI
├── data/
│   ├── processed/       # Active model datasets (train.csv, validation.csv, test.csv, curated_complaints.csv)
│   ├── final/           # External evaluation benchmark (external_evaluation_complaints.csv)
│   └── raw/             # Raw ingestion placeholder directory (.gitkeep)
├── src/                 # Modular Python pipeline & training scripts
│   ├── text_pipeline.py # TF-IDF vectorizer & SoftmaxLinearSVC pipeline
│   ├── train_models.py  # Model benchmarking, training & joblib export
│   ├── evaluate_model.py# Final test set evaluation & metric reporting
│   ├── curate_dataset.py# Data cleaning, deduplication & normalization
│   ├── split_dataset.py # Stratified train/val/test data splitting
│   └── eda_analysis.py  # Exploratory data analysis & figure generator
├── models/              # Serialized scikit-learn models (best_model.joblib)
├── notebooks/           # Jupyter exploration & data cleaning notebooks
├── reports/             # Performance evaluation reports, metrics & figures
│   └── figures/         # Confusion matrices and distribution plots
├── requirements.txt     # Python package dependencies
├── SETUP_GUIDE.md       # Step-by-step Windows setup and run guide
├── .gitignore           # Git ignore rules for virtual environments & models
└── README.md            # Project overview and documentation
```

---

## ⚡ Quick Start: Windows Setup

For a detailed step-by-step walkthrough, refer to [SETUP_GUIDE.md](SETUP_GUIDE.md).

### 1. Create and Activate Virtual Environment

Open PowerShell inside the project directory:

```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

*(If using Command Prompt `cmd.exe`, run `.\venv\Scripts\activate.bat` instead)*

### 2. Install Project Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch the Interactive Web Application

To start the local web application, run:

```powershell
python app.py
```

The application will launch at: **http://127.0.0.1:5000**.

---

## 🖥️ Interactive Web Frontend Features

The web frontend provides an ultra-clean, focused incident triage interface:

- **Incident Situation Input Textarea**: Responsive multi-line text input with live character counting for entering or pasting natural-language cybercrime complaints.
- **Analyse Situation Action**: Triggers the machine learning inference pipeline (`POST /api/predict`) connecting directly to `models/best_model.joblib`.
- **Dynamic Output Card**: Shown on demand once analysis completes, displaying:
  - Top predicted cybercrime category with corresponding icon
  - Model confidence score percentage with animated progress bar
  - Plain-language NLP linguistic explanation
  - Salient TF-IDF feature keyword badges with weights
  - Top candidate category probability breakdown
- **In-Memory Privacy**: Complaints are processed strictly in-memory during inference without being logged or stored to disk.

---

## 🔄 Recommended Course Workflow

1. **Data Ingestion**: Place raw complaint datasets into `data/raw/`.
2. **Exploratory Data Analysis (EDA)**: Use `notebooks/` to inspect class distributions and sample complaints.
3. **Text Preprocessing**: Lowercasing, removing special characters/URLs, tokenization, stop-word removal.
4. **Feature Extraction**: Convert cleaned complaint text to numerical feature vectors using **TF-IDF**.
5. **Model Training & Comparison**:
   - Train baseline **Logistic Regression** model.
   - Train **Linear SVM** (effective for high-dimensional sparse text vectors).
   - Train **Random Forest** classifier.
6. **Evaluation**: Compare models using Accuracy, Weighted Precision, Recall, and F1-Score on test data. Generate confusion matrices in `reports/figures/`.
7. **Model Export**: Save the best performing pipeline into `models/` using `joblib`.
8. **Interactive Demo**: Build a clean user interface with Streamlit for live testing.

---

## 📜 Academic Integrity Note

This workspace contains standard scaffolding and modular layout for course development. No synthetic or falsified data has been pre-populated, and no preliminary evaluation metrics are claimed. All model results should be generated on genuine course-approved data.

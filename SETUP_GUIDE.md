# Windows Environment Setup Guide

This guide walks you through setting up an isolated Python environment for the **Cybercrime Complaint Classification** project on Windows (PowerShell or Command Prompt).

---

## 1. Prerequisites

Make sure Python (version 3.9 to 3.11 recommended) is installed and added to your system PATH:

```powershell
python --version
```

If `python` is not recognized, install Python from [python.org](https://www.python.org/) and ensure **"Add Python to PATH"** is checked during installation.

---

## 2. Open Terminal in the Project Directory

Navigate to your project root folder (`ML_CBP`):

```powershell
cd c:\Users\raksh\OneDrive\Desktop\ML_CBP
```

---

## 3. Create a Virtual Environment

Create an isolated virtual environment named `venv`:

```powershell
python -m venv venv
```

---

## 4. Activate the Virtual Environment

### Option A: Using Windows PowerShell (Recommended)
If script execution is disabled on PowerShell, enable it first for your current user:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the environment:

```powershell
.\venv\Scripts\Activate.ps1
```

> You should now see `(venv)` at the beginning of your command prompt.

### Option B: Using Command Prompt (cmd.exe)
```cmd
.\venv\Scripts\activate.bat
```

### Option C: Using Git Bash
```bash
source venv/Scripts/activate
```

---

## 5. Upgrade pip and Install Dependencies

With the `(venv)` active, upgrade `pip` and install all required libraries:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 6. (Optional) Register Virtual Environment in Jupyter

If you plan to use Jupyter Notebooks in the `notebooks/` directory, register the virtual environment kernel:

```powershell
python -m ipykernel install --user --name=cybercrime_env --display-name "Python (Cybercrime ML)"
```

To launch Jupyter Notebook:
```powershell
jupyter notebook
```

---

## 7. Running the Web Application

To launch the web application and machine learning inference API:

```powershell
python app.py
```

The application will start on: **http://127.0.0.1:5000**

Open `http://127.0.0.1:5000` in any modern web browser to interact with the classifier.


---

## 8. Deactivate the Environment

When you are done working:

```powershell
deactivate
```

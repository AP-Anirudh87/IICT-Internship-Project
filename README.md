<div align="center">

# 🛡️ Dual AI Defense Systems: Fake News & Phishing Detection
### **Indian Institute of Computing and Technology (IICT) Internship Project**
**Natural Language Processing • Machine Learning • Cybersecurity Engineering**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=for-the-badge)](https://github.com/AP-Anirudh87/IICT-Internship)

<p align="center">
  <b>A comprehensive dual-pipeline machine learning framework engineered for automated digital deception mitigation: Combating disinformation through linguistic verification (TruthGuard AI) and neutralizing cyber social engineering through hybrid structural-textual telemetry (PhishGuard AI).</b>
</p>

[Key Innovations](#-key-innovations--engineering-highlights) •
[Project 1: TruthGuard AI](#-project-1-truthguard-ai--fake-news-classification) •
[Project 2: PhishGuard AI](#-project-2-phishguard-ai--hybrid-phishing-detection) •
[Model Benchmarks](#-comparative-model-benchmarks) •
[Quickstart Guide](#-quickstart--execution-guide) •
[Test Examples](#-live-verification--test-examples) •
[Codespaces Setup](#option-c-running-in-github-codespaces-1-click-cloud-container) •
[Cloud Deployment](#option-d-free-public-web-deployment-streamlit-community-cloud) •
[Multi-PC Setup](#-zero-configuration-portability-cloning-to-any-other-pc)

---
</div>

## 📌 Executive Summary

Developed during the internship at the **Indian Institute of Computing and Technology (IICT)**, this repository implements two end-to-end Machine Learning systems designed to detect and counter malicious digital threats:

1. **TruthGuard AI (Fake News Detection):** An advanced Natural Language Processing system analyzing over 6,300 long-form political articles. Integrates both ground-up mathematical implementations (custom Bag-of-Words & TF-IDF) and industry-standard vectorizers to benchmark KNN, Logistic Regression, Random Forest, and Multi-Layer Perceptrons (MLP).
2. **PhishGuard AI (Phishing Email Detection):** An enterprise-grade cybersecurity detection engine trained on over 18,600 emails. Features a novel **Hybrid Feature Fusion Architecture** combining 10,000 sparse TF-IDF semantic features with 12 structural/behavioral heuristic signals via sparse matrix concatenation (`scipy.sparse.hstack`).

Both solutions feature full automated ETL pipelines, cross-model diagnostic suites (Confusion Matrices, ROC-AUC curves, Gini feature importance, log-odds feature coefficients), and production-grade **Streamlit web applications** offering sub-second real-time inference.

---

## 🚀 Key Innovations & Engineering Highlights

| Feature | TruthGuard AI (Fake News) | PhishGuard AI (Phishing Emails) |
| :--- | :--- | :--- |
| **Primary Domain** | NLP & Misinformation Forensics | Cybersecurity & Threat Intelligence |
| **Dataset Scale** | 6,335 articles (Title + Body) | 18,650 real-world emails |
| **Feature Space** | 10,000 Unigram/Bigram TF-IDF with sublinear scaling | **10,012-Dimensional Hybrid Matrix** (10,000 TF-IDF + 12 Structural Signals) |
| **Algorithmic Breadth** | KNN (k=5), Logistic Regression, Random Forest, MLP | Complement Naive Bayes, Logistic Regression, Random Forest, MLP |
| **Top Performing Model** | **MLP Neural Network (93.8% F1-Score / 99.4% Benchmark)** | **MLP Neural Network (96.6% F1-Score / 98.9% Benchmark)** |
| **Explainability** | Top discriminatory n-grams & prediction probability | Gini importance ranking, LR odds ratios, behavioral breakdown |
| **Deployment** | Responsive Streamlit Web Application | SOC-style Cybersecurity Dashboard with Risk Telemetry |

---

## 📰 Project 1: TruthGuard AI — Fake News Classification

### 1. Architectural Pipeline
```
[Raw Article: Title + Body]
            │
            ▼
[Zero-Dependency Regex Cleaner] ──► (Strip HTML, Normalize URLs/Emails, Custom Stopwords)
            │
            ▼
[TF-IDF Feature Extraction]    ──► (Unigram + Bigram, Sublinear TF, Top 10,000 Vocabulary)
            │
            ▼
[Multi-Model Tournament]        ──► [KNN (k=5)]  [Logistic Regression]  [Random Forest]  [MLP Neural Net]
            │
            ▼
[Best Model Serialization]     ──► `models/best_model.pkl` + Auto-Generated Diagnostic Plots
            │
            ▼
[Interactive Web Dashboard]    ──► Real-time Sentiment, Confidence Gauges, Linguistic Forensics
```

### 2. Algorithmic Rigor: From-Scratch & Scikit-Learn Formulations
To demonstrate deep algorithmic comprehension beyond high-level wrappers:
* **Custom Bag-of-Words & TF-IDF Vectorizers:** Implemented from scratch using pure Python and NumPy (`feature_extractor.py`), proving exact mathematical mastery of Term Frequency and Inverse Document Frequency calculations:
  $$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}, \quad \text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
* **Production TF-IDF:** Leverages Scikit-Learn's optimized C-extensions with sublinear term frequency scaling ($1 + \log(\text{TF})$) to penalize repetitive spam words.

### 3. Model Evaluation Results
*Evaluated on independent held-out test sets with 5-fold cross-validation:*

| Model | Accuracy | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MLP Neural Network (Hidden Layers: 64, ReLU)** | **93.76%** | **92.10%** | **95.73%** | **0.9388** | 🏆 **Champion** |
| **Logistic Regression (L2 Regularization)** | 93.29% | 91.39% | 95.58% | 0.9344 | Runner Up |
| **Random Forest (100 Trees, Gini Impurity)** | 91.63% | 90.98% | 92.42% | 0.9169 | Robust |
| **K-Nearest Neighbors (k=5, Cosine Distance)** | 90.61% | 95.57% | 85.15% | 0.9006 | Baseline |

---

## 🎣 Project 2: PhishGuard AI — Hybrid Phishing Detection

### 1. The Core Innovation: Hybrid Feature Fusion
Standard NLP approaches fail against zero-day phishing attacks that use polite or novel wording. PhishGuard AI addresses this vulnerability through a **multi-modal fusion strategy**:

```
Raw Email Text
   │
   ├──► [12 Hand-Crafted Structural Features] ──► [MinMax Normalization] ──┐
   │    (URL count, IP hosts, urgency index, capitalization ratio, etc.)    │
   │                                                                        ├─► [scipy.sparse.hstack] ──► 10,012-Dim Matrix
   └──► [Text Preprocessor & TF-IDF Vectorizer] ───────────────────────────┘
        (Sublinear TF-IDF, Top 10,000 N-Grams)
```

#### The 12 Behavioral & Structural Heuristics:
1. `url_count`: Hyperlink density across the email payload.
2. `has_ip_url`: Binary flag detecting raw IPv4 hostnames (e.g., `http://192.168.1.1/verify`).
3. `exclamation_count` & `question_count`: Syntactic indicators of artificial urgency.
4. `uppercase_ratio`: Proportion of capitalized tokens indicating coercive authority.
5. `urgent_word_count`: Domain lexicon frequency ("urgent", "suspended", "immediately", "action required").
6. `money_word_count`: Financial lure lexicon ("wire", "invoice", "payroll", "bitcoin", "refund").
7. `avg_word_len`: Morphological metric identifying lexical obfuscation.
8. `char_count` & `word_count`: Payload volume metrics.
9. `digit_ratio`: Proportion of numeric characters indicating fraud-associated codes/dates.
10. `has_html_tags`: Detects hidden tracking pixels, zero-point fonts, and obfuscated CSS.

### 2. Model Evaluation Results

| Model | Accuracy | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MLP Neural Network (Multi-Layer Perceptron)** | **96.60%** | **94.54%** | **96.93%** | **0.9572** | 🏆 **Champion** |
| **Logistic Regression (Class-Balanced)** | 96.30% | 94.86% | 95.77% | 0.9532 | Runner Up |
| **Random Forest (Ensemble Trees)** | 96.17% | 94.79% | 95.50% | 0.9514 | High Precision |
| **Complement Naive Bayes** | 95.47% | 93.26% | 95.36% | 0.9430 | Fast Baseline |

---

## 📊 Comparative Model Benchmarks

```
TruthGuard AI (Fake News)
  MLP Neural Net   ████████████████████████████████ 93.76% (F1: 0.9388)
  Logistic Reg     ██████████████████████████████   93.29% (F1: 0.9344)
  Random Forest    ████████████████████████████     91.63% (F1: 0.9169)
  KNN (k=5)        ████████████████████████         90.61% (F1: 0.9006)

PhishGuard AI (Phishing Emails)
  MLP Neural Net   ████████████████████████████████ 96.60% (F1: 0.9572)
  Logistic Reg     ███████████████████████████████  96.30% (F1: 0.9532)
  Random Forest    ██████████████████████████████   96.17% (F1: 0.9514)
  Naive Bayes      ████████████████████████████     95.47% (F1: 0.9430)
```

All models generate automated diagnostic artifacts stored in the `plots/` folder of each project:
* `01_class_distribution.png` — Dataset class balance validation.
* `04_model_comparison.png` — Side-by-side Accuracy, Precision, Recall, and F1 metrics.
* `05_confusion_matrices.png` — False Positive vs False Negative forensic breakdowns.
* `06_roc_curves.png` — Multi-model Area Under Curve (AUC) discrimination analysis.
* `07_feature_importance.png` / `08_lr_coefficients.png` — Explainable AI visualizations.

---

## 💻 Tech Stack & Engineering Competencies

* **Languages & Core:** Python 3.9+, NumPy, SciPy (Sparse matrix manipulation & mathematical concatenation).
* **Machine Learning & NLP:** Scikit-Learn (`TfidfVectorizer`, `MLPClassifier`, `RandomForestClassifier`, `LogisticRegression`, `KNeighborsClassifier`, `ComplementNB`), Regular Expressions (`re`).
* **Data Engineering & ETL:** Pandas, Automated streaming dataset downloaders (`urllib.request`).
* **Visualizations & Diagnostics:** Matplotlib, Seaborn.
* **Full-Stack ML Deployment:** Streamlit (Custom responsive CSS, reactive widgets, dynamic charting).
* **Environment & Tools:** GitHub Codespaces, VS Code, Git, Jupyter Notebooks.

---

## 📂 Repository Organization

```
IICT-Internship/
│
├── .devcontainer/                      # GitHub Codespaces automated cloud configuration
│   └── devcontainer.json               # Auto-setup container with dual-port forwarding (8501 & 8502)
├── .streamlit/                         # Global Streamlit cloud & headless server settings
│   └── config.toml                     # Headless mode, CORS & reverse-proxy configuration
│
├── requirements.txt                    # Pinned core production dependencies
├── verify_all.py                       # Automated dual-pipeline integrity test suite (PC & Codespaces)
├── .gitignore                          # Git hygiene (ignores cache, checkpoints & venvs)
├── setup_local.bat                     # Windows 1-click automated virtual environment setup
├── run_truthguard.bat                  # Windows 1-click launcher for Project 1 (Port 8501)
├── run_phishguard.bat                  # Windows 1-click launcher for Project 2 (Port 8502)
├── setup_local.sh                      # Linux / Codespaces automated environment setup
├── run_truthguard.sh                   # Linux / Codespaces launcher for Project 1 (Port 8501)
├── run_phishguard.sh                   # Linux / Codespaces launcher for Project 2 (Port 8502)
├── README.md                           # Enterprise-level repository documentation
│
├── Project 1/                          # TruthGuard AI: Fake News Detection
│   ├── app.py                          # Interactive Streamlit Web Application
│   ├── fake_news_detection.py          # Complete ML Pipeline (ETL, Train, Evaluate)
│   ├── train_all.py                    # Multi-model batch training script
│   ├── text_preprocessor.py            # Zero-dependency regex cleaning module
│   ├── feature_extractor.py            # From-scratch BoW & TF-IDF implementations
│   ├── download_data.py                # Automated dataset fetcher & validator
│   ├── fake_news_detection.ipynb       # Research notebook with step-by-step EDA
│   ├── Presentation.pdf                # Technical presentation delivered at IICT
│   ├── Report.pdf                      # Comprehensive technical research report
│   ├── data/                           # Training dataset directory (train.csv)
│   ├── models/                         # Serialized best model & benchmark CSV
│   └── plots/                          # Generated ROC, Confusion Matrix & Metric plots
│
└── Project 2/                          # PhishGuard AI: Phishing Email Detection
    ├── app.py                          # SOC-Style Streamlit Cybersecurity Dashboard
    ├── phishing_email_detection.py     # Complete ML Pipeline with Hybrid Concatenation
    ├── train_all.py                    # Multi-model batch training script
    ├── metadata_features.py            # 12-dimensional structural feature extractor
    ├── download_data.py                # Automated dataset fetcher & validator
    ├── phishing_email_detection.ipynb  # Research & Exploratory Data Analysis notebook
    ├── Presentation.pdf                # Technical presentation delivered at IICT
    ├── Report.pdf                      # Comprehensive technical research report
    ├── data/                           # Training dataset directory (Phishing_Email.csv)
    ├── models/                         # Serialized best model & benchmark CSV
    └── plots/                          # ROC curves, confusion matrices, feature rankings
```

---

## ⚡ Quickstart & Execution Guide

### Option A: Local Execution on Windows (One-Click Automated)

If you are running on Windows, you can use the pre-configured batch scripts:

1. **One-Click Environment Setup:**
   Double-click or run in terminal:
   ```cmd
   setup_local.bat
   ```
   *Automatically creates an isolated `.venv`, upgrades pip, and installs all packages from `requirements.txt`.*

2. **Fast Automated Local Verification:**
   Run the automated test suite in PowerShell using your local `.venv`:
   ```powershell
   .\.venv\Scripts\python.exe verify_all.py
   ```
   *Expected Output:*
   ```text
   [1/2] Testing Project 1: TruthGuard AI ... ✅ PASS (REAL NEWS)
   [2/2] Testing Project 2: PhishGuard AI ... ✅ PASS (PHISHING ATTACK - Threat Score: 99.1%)
   🏆 ALL DUAL DEFENSE PIPELINES ARE VERIFIED & OPERATIONAL!
   ```

3. **Launch Project 1 (TruthGuard AI — Fake News Detector):**
   ```cmd
   run_truthguard.bat
   ```
   👉 *Starts the web server and opens [http://localhost:8501](http://localhost:8501).*

4. **Launch Project 2 (PhishGuard AI — Phishing Email SOC Dashboard):**
   ```cmd
   run_phishguard.bat
   ```
   👉 *Starts the web server and opens [http://localhost:8502](http://localhost:8502).*

---

### Option B: Local Execution via Terminal (PowerShell / Linux / macOS)

1. **Clone the repository and install dependencies:**
   ```bash
   git clone https://github.com/AP-Anirudh87/IICT-Internship.git
   cd IICT-Internship
   python -m venv .venv
   
   # Activate on Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   # Or activate on Linux/macOS:
   # source .venv/bin/activate

   pip install -r requirements.txt
   ```

2. **Run Project 1 (TruthGuard AI):**
   ```bash
   cd "Project 1"
   streamlit run app.py --server.port 8501
   ```

3. **Run Project 2 (PhishGuard AI):**
   ```bash
   cd "Project 2"
   streamlit run app.py --server.port 8502
   ```

---

### Option C: Running in GitHub Codespaces (1-Click Cloud Container)

This repository includes a native `.devcontainer/devcontainer.json` configuration for GitHub Codespaces.

1. Open **[github.com/AP-Anirudh87/IICT-Internship](https://github.com/AP-Anirudh87/IICT-Internship)**.
2. Click **Code** → **Codespaces** tab → **Create codespace on Internship**.
3. **Install Dependencies in Codespaces:**
   In your Codespaces terminal, run:
   ```bash
   pip install -r requirements.txt
   ```
   *(Or if running in a bare container without requirements.txt uploaded yet:)*
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn streamlit scipy
   ```
   > 💡 *Note: If you ever see `ModuleNotFoundError: No module named 'sklearn'`, running the command above installs all required libraries in ~30 seconds.*

4. **Fast Automated Verification Test (Both Projects):**
   Run the unified automated test suite directly in the Codespaces terminal:
   ```bash
   python3 verify_all.py
   ```
   *Expected Terminal Output:*
   ```text
   =================================================================
     🛡️  IICT AI DEFENSE SYSTEMS: FAST INTEGRITY TEST
   =================================================================
   [1/2] Testing Project 1: TruthGuard AI (Fake News Detection)...
     ✅ Model loaded successfully : MLP Classifier
     ✅ Test Headline Input       : WASHINGTON (Reuters) - The Senate passed...
     ✅ System Prediction         : REAL NEWS
     🎉 Project 1 Test Status     : PASS (Correctly identified as Legitimate)

   [2/2] Testing Project 2: PhishGuard AI (Phishing Email Detection)...
     ✅ Model loaded successfully : MLP Classifier
     ✅ Test Email Input          : URGENT ACTION REQUIRED! Your bank account...
     ✅ Threat Score              : 99.1%
     ✅ System Prediction         : PHISHING ATTACK
     🎉 Project 2 Test Status     : PASS (Correctly identified as Threat)

   =================================================================
     🏆  ALL DUAL DEFENSE PIPELINES ARE VERIFIED & OPERATIONAL!
   =================================================================
   ```

5. **Launch the Web Apps:**
   * **To Launch Project 1 (TruthGuard AI — Port 8501):**
     ```bash
     cd "Project 1" && streamlit run app.py --server.port 8501
     ```
   * **To Launch Project 2 (PhishGuard AI — Port 8502):**
     *(Open a second terminal tab by clicking `+` in the terminal panel)*
     ```bash
     cd "Project 2" && streamlit run app.py --server.port 8502
     ```
6. When the VS Code notification **"Your application running on port 8501/8502 is available"** pops up, click **Open in Browser** to view the live dashboard!

---

### Option D: Free Public Web Deployment (Streamlit Community Cloud)

To deploy these applications permanently to the web for free with a public URL:

1. Push your repository to GitHub (`https://github.com/AP-Anirudh87/IICT-Internship`).
2. Visit **[share.streamlit.io](https://share.streamlit.io/)** and log in with your GitHub account.
3. Click **New app**:
   - **Repository:** `AP-Anirudh87/IICT-Internship`
   - **Branch:** `Internship`
   - **Main file path:** `Project 1/app.py` *(or `Project 2/app.py` for PhishGuard)*
4. Click **Deploy!** — Streamlit Cloud detects `requirements.txt` and `.streamlit/config.toml` automatically and hosts your app with a public `https://...streamlit.app` link for your resume and portfolio!

---

### 🌐 Zero-Configuration Portability: Cloning to Any Other PC

> [!IMPORTANT]
> **Do I need to modify any code, filenames, or filepaths when downloading this repository onto another PC?**  
> **NO.** The codebase has been engineered with 100% cross-platform portability:
> * **Zero Hardcoded Absolute Paths:** All Python modules use dynamic path anchoring (`os.path.dirname(__file__)` and `os.path.abspath`) rather than fixed drive letters (`C:`, `D:`, `E:`).
> * **Dynamic Multi-User Windows Batch Launchers:** `setup_local.bat`, `run_truthguard.bat`, and `run_phishguard.bat` do not contain hardcoded usernames. They automatically probe the system `PATH`, the Windows `py` launcher, and `%LOCALAPPDATA%\Programs\Python` across Python 3.9 through 3.13 on any Windows machine.
> * **Self-Contained Fallbacks:** Both web apps detect missing datasets and model bundles gracefully, triggering automated streaming downloads via `download_data.py` or training lightweight fallback classifiers on-the-fly.

#### 3 Steps to Run on a Fresh Windows Machine:
1. **Clone or Download ZIP:**
   ```cmd
   git clone -b Internship https://github.com/AP-Anirudh87/IICT-Internship.git
   cd IICT-Internship
   ```
2. **Double-Click Setup:**
   Run `setup_local.bat` — it detects Python, initializes an isolated `.venv`, and installs all dependencies from `requirements.txt`.
3. **Launch:**
   Double-click `run_truthguard.bat` (Fake News) or `run_phishguard.bat` (Phishing Detection).

#### Setup on a Fresh Linux or macOS Machine:
```bash
git clone -b Internship https://github.com/AP-Anirudh87/IICT-Internship.git
cd IICT-Internship
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd "Project 1" && streamlit run app.py --server.port 8501
```

---

## 🧪 Live Verification & Test Examples

Both web applications include 1-click **"Load Test Samples"** buttons in their sidebars. You can also manually copy and paste the benchmark test cases below to verify that your models and web dashboards are functioning with 100% precision:

### 📰 Project 1: TruthGuard AI (Fake News Detection)

#### Test Case 1: Legitimate Wire News Article
* **Input Text to Paste:**
  ```text
  WASHINGTON (Reuters) - The Senate passed a $1.2 trillion infrastructure bill on Tuesday, sending it to the House for a final vote. The legislation includes funding for roads, bridges, public transit, clean water, and broadband internet expansion across the country.
  ```
* **Expected System Output:**
  | Metric / Field | Expected Value | Status |
  | :--- | :--- | :---: |
  | **Verdict Banner** | `✅ VERIFIED REAL NEWS` (Green Badge) | 🟢 PASS |
  | **Fake Probability** | `< 5.0%` (Real News Confidence: > 95%) | 🟢 PASS |
  | **Linguistic Profile** | Objective tone, attribution to reputable wire service (`Reuters`), factual syntactic structure | 🟢 PASS |

#### Test Case 2: Fabricated Sensationalist Disinformation
* **Input Text to Paste:**
  ```text
  SHOCKING PROOF: Leaked internal documents reveal that secret shadow organization is controlling global food supplies to force citizens into digital compliance! Share this story IMMEDIATELY before the mainstream media deletes it from the web!!!
  ```
* **Expected System Output:**
  | Metric / Field | Expected Value | Status |
  | :--- | :--- | :---: |
  | **Verdict Banner** | `🚨 DECEPTIVE / FAKE NEWS DETECTED` (Pulsing Red Card) | 🔴 PASS |
  | **Fake Probability** | `> 95.0%` (High Malicious Confidence) | 🔴 PASS |
  | **Flagged Patterns** | Excessive exclamation marks (`!!!`), all-caps urgency (`SHOCKING`, `IMMEDIATELY`), conspiracy terminology (`shadow organization`, `deletes it`) | 🔴 PASS |

---

### 🎣 Project 2: PhishGuard AI (Phishing Email Detection)

#### Test Case 1: Legitimate Internal Workplace Email
* **Input Text to Paste:**
  ```text
  Hi John,

  Please find attached the meeting notes from yesterday's quarterly review.
  The next meeting is scheduled for Thursday 3 PM in Conference Room B.

  Let me know if you have any questions.

  Best regards,
  Sarah Johnson
  Project Manager
  ```
* **Expected System Output:**
  | Metric / Field | Expected Value | Status |
  | :--- | :--- | :---: |
  | **Verdict Banner** | `✅ SAFE EMAIL` (Green Shield) | 🟢 PASS |
  | **Threat Risk Score** | `< 5.0% Phishing Risk` | 🟢 PASS |
  | **12-Point Heuristics** | `url_count: 0`, `has_ip_url: 0.0`, `exclamation_count: 0`, `urgent_word_count: 0`, standard lexical diversity | 🟢 PASS |

#### Test Case 2: Zero-Day Credential Harvesting Phishing Attack
* **Input Text to Paste:**
  ```text
  URGENT ACTION REQUIRED!

  Your account has been SUSPENDED due to unusual activity.
  Click the link IMMEDIATELY to verify your account and restore access:
  http://192.168.1.104/verify-now?user=you

  Failure to act within 24 HOURS will result in PERMANENT account closure.

  Account Security Team
  support@bankofamerica-secure.verify-login.com
  ```
* **Expected System Output:**
  | Metric / Field | Expected Value | Status |
  | :--- | :--- | :---: |
  | **Verdict Banner** | `🚨 PHISHING ATTACK DETECTED` (Pulsing Crimson SOC Alert) | 🔴 PASS |
  | **Threat Risk Score** | `> 95.0% Critical Phishing Risk` | 🔴 PASS |
  | **Flagged Heuristic Telemetry** | • **Raw IP Address in URL:** Flagged (`http://192.168.1.104`)<br>• **Urgency Lexicon:** High (`urgent`, `immediately`, `suspended`, `verify`, `action`)<br>• **Coercive Capitalization:** High (`URGENT`, `SUSPENDED`, `IMMEDIATELY`, `PERMANENT`) | 🔴 PASS |

---

### 🏋️ Retraining All Machine Learning Models from Scratch

If you wish to re-execute the automated ETL, model tournaments, and regenerate all diagnostic plots:

```bash
# Project 1: Downloads dataset if missing, trains 4 models, generates plots & best_model.pkl
cd "Project 1"
python download_data.py
python train_all.py

# Project 2: Downloads dataset, computes 12 structural heuristics + TF-IDF, trains 4 models
cd "../Project 2"
python download_data.py
python train_all.py
```

---

## 👨‍💻 Author & Acknowledgements

* **Researcher / Developer:** [AP-Anirudh87](https://github.com/AP-Anirudh87)
* **Organization:** Indian Institute of Computing and Technology (IICT)
* **Domain:** Artificial Intelligence, Natural Language Processing & Cyber Intelligence

*Special thanks to the mentors and project guides at the Indian Institute of Computing and Technology (IICT) for their continuous support and guidance.*
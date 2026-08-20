# IICT Internship Projects

This repository contains two Machine Learning projects focused on cybersecurity and misinformation detection. Both projects feature complete pipelines: automated data handling, text preprocessing, training multiple AI models (Naive Bayes, Logistic Regression, Random Forest, MLP Neural Networks), and beautiful interactive Web Apps.

---

## 📰 Project 1: Fake News Detection
**TruthGuard AI** is an advanced Natural Language Processing (NLP) system designed to detect fabricated political news articles and clickbait. 

### 🧠 Comprehensive Project Details & Methodology

#### 1. Project Objective
The primary objective of TruthGuard AI is to combat the spread of digital misinformation by automatically classifying news articles as either factual or fabricated. This is achieved using a robust Natural Language Processing (NLP) pipeline combined with classical and deep machine learning algorithms.

#### 2. Dataset Architecture
* **Source:** ISOT Fake News Dataset (accessed via Kaggle API).
* **Volume:** Over 40,000 full-length articles.
* **Composition:** 
  * **Real News:** Authentic political and world news articles scraped from Reuters.com.
  * **Fake News:** Fabricated articles collected from flagged unreliable websites and clickbait domains by PolitiFact.
* **Labeling:** Binary classification (0 = Real News, 1 = Fake News).

#### 3. Preprocessing Pipeline
To ensure the models learn from semantic meaning rather than structural noise, a rigorous, zero-dependency custom regex pipeline cleans the raw text:
* **HTML Stripping:** Removes stray web tags.
* **Entity Redaction:** Replaces all URLs with the token `URL` and email addresses with `EMAIL`.
* **Alphanumeric Normalization:** Strips all punctuation and special characters, retaining only alphabetical characters and spaces.
* **Lowercasing & Whitespace Reduction:** Standardizes capitalization and collapses multiple spaces.
* **Stopword Filtering:** Removes extremely common English words (e.g., "the", "and", "is") using a custom hardcoded dictionary to reduce dimensional noise without relying on heavy external libraries like NLTK or spaCy.

#### 4. Feature Engineering Strategy
* **Vectorization:** Implements **TF-IDF (Term Frequency - Inverse Document Frequency)** vectorization using `scikit-learn`. 
* **Parameters:** Captures both unigrams and bigrams (`ngram_range=(1,2)`) to understand two-word contextual phrasing (e.g., "white house", "fake news").
* **Dimensionality:** Capped at the top 10,000 most significant features (`max_features=10000`) to balance computational efficiency with predictive power.

#### 5. Model Architectures & Evaluation
The project trains and evaluates four distinct algorithms to find the optimal decision boundary:
1. **Multinomial Naive Bayes:** A fast, probabilistic baseline model assuming feature independence.
2. **Logistic Regression:** A linear model optimized with balanced class weights, excellent for text classification.
3. **Random Forest Classifier:** An ensemble of decision trees capturing complex, non-linear relationships and interactions between specific words.
4. **MLP Neural Network (Multi-Layer Perceptron):** A deep learning architecture with a 64-node hidden layer, ReLU activation, and early stopping. This model consistently achieves the highest evaluation metrics.
* **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrices, and ROC-AUC curves are generated and saved as visualizations in the `/plots` directory.

#### 6. Deployment & Interface
* **Web Application:** Features a sleek, dark-themed, responsive web application built with Streamlit (`app.py`).
* **Real-time Inference:** Automatically loads the highest-performing serialized model (`best_model.pkl`) and applies the exact same preprocessing and TF-IDF transformations to user-pasted text, returning a confidence score and risk level assessment instantly.

### How to Run Project 1

> **📌 Note for all users:** The commands below use generic placeholders. Replace them with your own values before running:
> - Replace `YOUR_DRIVE` with the drive where you saved the project (e.g. `C`, `D`, `E`)
> - Replace `YOUR_FOLDER_PATH` with the full folder path on your PC (e.g. `Documents\IICT Internship Project`)
> - Replace `YOUR_USERNAME` with your Windows username (found in `C:\Users\`)
> - Replace `YOUR_PYTHON_VERSION` with your installed Python version (e.g. `Python312`, `Python313`)

**Step 1: Open Terminal and navigate to the Project 1 folder**

Replace the path below with the actual location of your project:
```powershell
cd "YOUR_DRIVE:\YOUR_FOLDER_PATH\Project 1"
```
**Example:**
```powershell
# If saved on D: drive
cd "D:\IICT Internship Project\Project 1"

# If saved on C: drive inside Documents
cd "C:\Users\john\Documents\IICT Internship Project\Project 1"
```

**Step 2: Download the Dataset**
*(This script securely connects to Kaggle and downloads the ISOT Fake News Dataset.)*

Replace the Python path with your own:
```powershell
& "C:\Users\YOUR_USERNAME\AppData\Local\Programs\Python\YOUR_PYTHON_VERSION\python.exe" download_data.py
```
**Example:**
```powershell
& "C:\Users\john\AppData\Local\Programs\Python\Python313\python.exe" download_data.py
```
**Tip:** Not sure of your Python path? Run this in terminal to find it automatically:
```powershell
python --version
(Get-Command python).Source
```

**Step 3: Train the AI Models**
*(Trains 4 ML models, generates charts in `/plots`, and saves the best model.)*
```powershell
& "C:\Users\YOUR_USERNAME\AppData\Local\Programs\Python\YOUR_PYTHON_VERSION\python.exe" fake_news_detection.py
```

**Step 4: Launch the Web App**
*(Opens the TruthGuard AI website in your browser automatically.)*
```powershell
& "C:\Users\YOUR_USERNAME\AppData\Local\Programs\Python\YOUR_PYTHON_VERSION\python.exe" -m streamlit run app.py
```

**✅ Shortcut (if `python` is in your system PATH):**
```powershell
python download_data.py
python fake_news_detection.py
python -m streamlit run app.py
```

---

## 🎣 Project 2: Phishing Email Detection
**PhishGuard AI** is a highly accurate cybersecurity tool designed to detect malicious phishing attempts, fraudulent requests, and deceptive emails before they compromise users.

### 🧠 Comprehensive Project Details & Methodology

#### 1. Project Objective
PhishGuard AI is an advanced cybersecurity tool engineered to detect sophisticated phishing attempts, fraudulent requests, and deceptive social engineering emails before they compromise end-users or corporate networks.

#### 2. Dataset Architecture
* **Source:** A curated cybersecurity dataset containing over 18,000 emails.
* **Labeling:** Binary classification mapping text to "Safe Email" or "Phishing Email".
* **Imbalance Handling:** Handled inherently during training using balanced class weights and stratified train-test splitting to ensure minority attack vectors are learned.

#### 3. Hybrid Feature Extraction (The Core Innovation)
Unlike standard NLP projects that only look at words, PhishGuard AI uses a hybrid, two-pronged approach that mimics how a human security analyst evaluates an email:

**Part A: Structural & Behavioral Metadata (12 Features)**
Before the email text is cleaned, the system extracts critical behavioral indicators using regular expressions:
1. `url_count`: Total number of hyperlinks.
2. `has_ip_url`: Flag for suspicious URLs containing raw IP addresses (e.g., `http://192.168.1.1/login`).
3. `exclamation_count` & `question_count`: High punctuation counts often indicate false urgency.
4. `uppercase_ratio`: The proportion of words written entirely in ALL CAPS.
5. `urgent_word_count`: Frequency of words like "urgent", "immediate", "suspended", "verify".
6. `money_word_count`: Frequency of words like "invoice", "payment", "bank", "wire".
7. `avg_word_len`, `char_count`, `word_count`: Basic length metrics.
8. `digit_ratio`: The proportion of numbers to alphabetical text.
9. `has_html_tags`: Detects hidden tracking pixels or obfuscated web forms.

**Part B: TF-IDF Vectorization**
After metadata extraction, the email undergoes standard NLP cleaning (HTML stripping, tokenization) and is vectorized using TF-IDF (10,000 features).

**Part C: Matrix Concatenation**
The 12 numerical metadata features are scaled using `StandardScaler` and mathematically concatenated to the 10,000 sparse TF-IDF features using `scipy.sparse.hstack`, creating a massive 10,012-dimension hybrid matrix.

#### 4. Model Architectures & Evaluation
The hybrid matrix is fed into four classifiers:
1. **Multinomial Naive Bayes**
2. **Logistic Regression:** (Analyzed for feature coefficients to understand which words/metadata most strongly predict phishing).
3. **Random Forest Classifier:** (Used for Gini Feature Importance rankings to prove the value of the custom metadata).
4. **MLP Neural Network:** A deep perceptron that excels at finding patterns across the 10,012 features, ultimately achieving >98% accuracy.

#### 5. Deployment & Interface
* **Web Application:** A responsive Streamlit dashboard (`app.py`) designed for Security Operations Center (SOC) style analysis.
* **Real-time Pipeline:** When a user pastes a suspicious email, the app invisibly extracts the 12 metadata features, scales them, cleans the text, vectorizes it, concatenates the matrices, and feeds it to the Neural Network—returning a sub-second "Safe" or "Phishing" verdict with confidence percentages.
### How to Run Project 2

> **📌 Note for all users:** Same rules as above — replace `YOUR_DRIVE`, `YOUR_FOLDER_PATH`, `YOUR_USERNAME`, and `YOUR_PYTHON_VERSION` with your actual values.

**Step 1: Open Terminal and navigate to the Project 2 folder**
```powershell
cd "YOUR_DRIVE:\YOUR_FOLDER_PATH\Project 2"
```
**Example:**
```powershell
# If saved on D: drive
cd "D:\IICT Internship Project\Project 2"

# If saved on C: drive inside Documents
cd "C:\Users\john\Documents\IICT Internship Project\Project 2"
```

**Step 2: Download the Dataset**
*(Downloads the Phishing Email dataset from Kaggle.)*
```powershell
& "C:\Users\YOUR_USERNAME\AppData\Local\Programs\Python\YOUR_PYTHON_VERSION\python.exe" download_data.py
```

**Step 3: Train the AI Models**
*(Trains the hybrid TF-IDF + Metadata models, saves charts in `/plots`, and saves the best model.)*
```powershell
& "C:\Users\YOUR_USERNAME\AppData\Local\Programs\Python\YOUR_PYTHON_VERSION\python.exe" phishing_email_detection.py
```

**Step 4: Launch the Web App**
*(Opens the PhishGuard AI website in your browser automatically.)*
```powershell
& "C:\Users\YOUR_USERNAME\AppData\Local\Programs\Python\YOUR_PYTHON_VERSION\python.exe" -m streamlit run app.py
```

**✅ Shortcut (if `python` is in your system PATH):**
```powershell
python download_data.py
python phishing_email_detection.py
python -m streamlit run app.py
```

---

## ⚙️ Requirements / Prerequisites

If you are running this on a **brand new Windows PC**, follow these steps first:

**Step 1: Install Python**
Download and install Python from: https://www.python.org/downloads/
> ⚠️ During installation, tick **"Add Python to PATH"** — this makes the shortcut commands work.

**Step 2: Install required libraries**

Using the full Python path (replace with your own):
```powershell
& "C:\Users\YOUR_USERNAME\AppData\Local\Programs\Python\YOUR_PYTHON_VERSION\python.exe" -m pip install pandas numpy scikit-learn matplotlib seaborn kaggle streamlit scipy
```

OR using the shortcut (if Python is in PATH):
```powershell
pip install pandas numpy scikit-learn matplotlib seaborn kaggle streamlit scipy
```

**Step 3: Set up your Kaggle API Key** *(required for downloading datasets)*
1. Go to https://www.kaggle.com/settings → Account → API → **"Create New Token"**
2. A file called `kaggle.json` will download — move it to: `C:\Users\YOUR_USERNAME\.kaggle\kaggle.json`

---

## ☁️ Running Online (GitHub Codespaces / VS Code Web)

If you are uploading this project to GitHub and running it in an online cloud environment (like **GitHub Codespaces**), the commands are much simpler because you will be on a Linux server instead of Windows!

**Step 1: Install Required Packages**
Open the VS Code terminal inside Codespaces and run:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn kaggle streamlit
```

**Step 2: Run Project 1 (Fake News)**
```bash
cd "Project 1"
python download_data.py
python fake_news_detection.py
python -m streamlit run app.py
```

**Step 3: Run Project 2 (Phishing Email)**
```bash
cd "../Project 2"
python phishing_email_detection.py
python -m streamlit run app.py
```

*(Note: When you run `streamlit run` in GitHub Codespaces, a small popup will appear in the bottom right corner of your VS Code window asking to "Open in Browser". Click that popup to view your website!)*

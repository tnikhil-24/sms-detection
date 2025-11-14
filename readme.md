# 📩 SMS Spam Detection — Streamlit + Machine Learning

This project classifies SMS messages as **Spam** or **Ham (Safe)** using **Natural Language Processing (NLP)** and **Machine Learning**.  
The application features a **Streamlit-based UI** for real-time message testing and model visualization.

---

## 🧠 Project Overview

This project uses:

- Text preprocessing & normalization
- **TF-IDF Vectorization** to convert message text into numerical features
- Machine Learning classifiers (trained and saved in `artifacts/`)
- A Streamlit web interface for:
  - Live message classification
  - Model comparison & evaluation
  - Displaying tuned hyperparameters (if applicable)

---

## 📂 Folder Structure

```
SMS_Spam_Detector/
│
├── app.py                        # Streamlit App
├── readme.md                     # (This file)
├── SMS_Spam_Detector.ipynb       # Training & evaluation notebook
├── SMSSpamCollection             # Dataset (Raw SMS data)
│
├── artifacts/                    # Saved artifacts (used directly by app)
│   ├── all_models.pkl            # Dictionary of trained models
│   ├── spam_model.pkl            # Selected / best performing model
│   └── tfidf.pkl                 # TF-IDF Vectorizer (must not be retrained)
│
└── .venv/                        # Virtual environment (not required for deployment)
```

---

## 🗃 Dataset Information

Dataset Used: **SMSSpamCollection Dataset**

| Label | Meaning |
|------|---------|
| `ham` | Safe, normal message |
| `spam` | Fraud / Scam / Promotional message |

Example messages:

```
ham: "I'll call you later, meeting now."
spam: "You won $500! Click here to claim → http://scam-link.com"
```

---

## ⚙️ Setup Instructions

### 1) Install Dependencies
```
pip install -r requirements.txt
```

### 2) Run the Web Interface
```
streamlit run app.py
```

---

## 💬 How to Use the App

| Feature | Description |
|--------|-------------|
| Message Input Box | Enter any text message |
| Model Selection | Choose which trained model to test |
| Output | Displays SPAM or HAM with confidence score |
| Performance View | Compares accuracy & confusion matrix (if implemented) |

---

## 🔧 Models Included (Inside `all_models.pkl`)

| Model | Description |
|------|-------------|
| Multinomial Naive Bayes | Fast & efficient for text |
| Logistic Regression | Balanced performance |
| Linear SVM | **Typically best performer** |
| Random Forest | Included for performance comparison |

The file `spam_model.pkl` is the **best performing** selected model.

---

## 🚀 Possible Improvements

- Deploy to **Streamlit Cloud / HuggingFace Spaces**
- Add **LSTM / BERT** for advanced NLP performance
- Integrate **Twilio API** to classify live SMS messages

---

### ✨ Built with Passion for Machine Learning & Clean UI Design

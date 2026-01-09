# 📄 Resume Category Prediction App 

This project is a Machine Learning–based web application that predicts the job category of a resume using Natural Language Processing (NLP). The app is built with **Python**, **Scikit-learn**, and **Streamlit**.

---

## 🚀 Features

- Resume text cleaning using NLP techniques
- TF-IDF vectorization
- Logistic Regression (One-vs-Rest) classifier
- Predicts **job category** with **confidence probabilities**
- Interactive **Streamlit web interface**
- Visual probability distribution (bar chart)
- Uses saved preprocessing and model artifacts for consistency

---

## 🛠️ Tech Stack

- Python  
- Scikit-learn  
- NLP (Regex, TF-IDF)  
- Logistic Regression (OvR)  
- Streamlit  
- Pickle (Model Persistence)

---


## ⚙️ How It Works

1. Resume text is cleaned using a custom NLP preprocessing function  
2. Cleaned text is transformed using TF-IDF  
3. Logistic Regression model predicts the encoded class  
4. LabelEncoder converts encoded class back to category name  
5. Probabilities are displayed for all categories  

---

## 📂 Project Structure

├── app.py

├── log.pkl # Trained Logistic Regression model

├── tfidf.pkl # TF-IDF Vectorizer

├── label_encoder.pkl # Label Encoder

├── requirements.txt

└── README.md

## ▶️ Run the App Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Sample Output

Predicted Category: Data Scientist

Confidence: 31.8%

Probability distribution shown as a bar chart

## 👩‍💻 Author

Sonia Firdous
Aspiring Machine Learning Engineer


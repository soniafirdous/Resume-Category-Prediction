import pickle
import streamlit as st
import pandas as pd
import re


# Load model artifacts

@st.cache_resource
def load_artifacts():
    log = pickle.load(open("log.pkl", "rb"))
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))
    return log, tfidf, le

log, tfidf, le= load_artifacts()


# Resume cleaning function

def cleanResume(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"RT|cc", " ", text)
    text = re.sub(r"#\S+", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"[^A-Za-z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

# Streamlit UI


st.set_page_config(page_title="Resume Category Predictor", layout="centered")


st.title("📄 Resume Category Prediction App")
st.write("Paste a resume below to predict its job category")


resume_text = st.text_area("Paste Resume Text Here", height=250)


if st.button("Predict Category"):
  if resume_text.strip() == "":
     st.warning("Please paste resume text before predicting.")
  else:

    # Clean & vectorize
    cleaned_resume = cleanResume(resume_text)
    input_feature = tfidf.transform([cleaned_resume])

    # Predict encoded label
    pred_encoded = log.predict(input_feature)[0]
    pred_category = le.inverse_transform([pred_encoded])[0]


    # Predict probabilities
    proba = log.predict_proba(input_feature)[0]


    st.success(f"✅ Predicted Category: **{pred_category}**")


    st.subheader("Prediction Probabilities")
    
    prob_dict = {
    le.inverse_transform([cls])[0]: float(p)
    for cls, p in zip(log.classes_, proba)
}



    st.bar_chart(prob_dict)


    st.write("### Detailed Probabilities")
    for cat, p in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
       st.write(f"{cat}: {p:.4f}")
       
       
    top_cat, top_prob = max(prob_dict.items(), key=lambda x: x[1])

    st.metric(
        label="Predicted Category",
        value=top_cat,
        delta=f"{top_prob*100:.2f}% confidence"
    )



st.markdown("---")
st.caption("Built with Streamlit • Logistic Regression • TF-IDF • NLP")

st.markdown("---")
st.write("Developed by Sonia Firdous")
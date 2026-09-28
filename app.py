import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Fake Review Detector", layout="wide")

# -------------------- STYLE --------------------
st.markdown("""
<style>
.big-title {font-size:40px; font-weight:800; text-align:center;}
.section {padding:20px; margin-top:20px; border-radius:10px; background:#f5f5f5;}
</style>
""", unsafe_allow_html=True)

# -------------------- FUNCTION --------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# -------------------- HEADER --------------------
st.markdown("<div class='big-title'>Fake Review Detection System</div>", unsafe_allow_html=True)
st.write("### AI-powered system to detect Fake vs Genuine reviews")

# -------------------- ABOUT SECTION --------------------
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.header("About Project")
st.write("""
This system uses Machine Learning (TF-IDF + XGBoost) to detect fake reviews.
It helps e-commerce platforms identify misleading or spam reviews.
""")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------- LOAD DATA --------------------
data = pd.read_excel("reviews.xlsx", engine="openpyxl")
data['label'] = data['label'].map({'CG':1,'OR':0})
data['text_'] = data['text_'].apply(clean_text)

# -------------------- MODEL --------------------
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(data['text_'])
y = data['label']

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)

# -------------------- MODEL INFO --------------------
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.header("Model Information")

acc = accuracy_score(y, model.predict(X))
st.success(f"Accuracy: {acc*100:.2f}%")

st.write("""
- Algorithm: XGBoost  
- Vectorization: TF-IDF  
- Labels: Fake / Genuine  
""")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------- FILE UPLOAD --------------------
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.header("Upload Dataset")

file = st.file_uploader("Upload Excel File", type=["xlsx"])

if file:
    df = pd.read_excel(file, engine="openpyxl")

    if 'text_' not in df.columns:
        st.error("File must contain 'text_' column")
    else:
        df['text_'] = df['text_'].apply(clean_text)
        vec = tfidf.transform(df['text_'])
        preds = model.predict(vec)

        df["Prediction"] = preds
        df["Prediction"] = df["Prediction"].replace({0:"Genuine",1:"Fake"})

        st.subheader("Results Table")
        st.dataframe(df, use_container_width=True)

        # -------------------- METRICS --------------------
        fake_count = (df["Prediction"] == "Fake").sum()
        genuine_count = (df["Prediction"] == "Genuine").sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(df))
        col2.metric("Fake", fake_count)
        col3.metric("Genuine", genuine_count)

        # -------------------- CHART --------------------
        st.subheader("Chart")
        st.bar_chart(df["Prediction"].value_counts())

        # -------------------- FAKE REVIEWS --------------------
        st.subheader("Fake Reviews")
        st.write(df[df["Prediction"] == "Fake"])

        # -------------------- DOWNLOAD --------------------
        st.download_button("Download CSV", df.to_csv(index=False), file_name="results.csv")

st.markdown("</div>", unsafe_allow_html=True)

# -------------------- SINGLE REVIEW --------------------
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.header("Single Review Checker")

review = st.text_area("Enter review here")

if st.button("Analyze"):
    if review.strip():
        clean = clean_text(review)
        vec = tfidf.transform([clean])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.error("Fake Review")
        else:
            st.success("Genuine Review")
    else:
        st.warning("Enter a review")

st.markdown("</div>", unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.write("© 2026 Fake Review Detection Project | Built with Streamlit")
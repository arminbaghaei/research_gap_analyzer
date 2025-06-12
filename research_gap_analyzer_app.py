import streamlit as st
import requests
import pandas as pd
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from fpdf import FPDF
import matplotlib.pyplot as plt
from io import BytesIO

# Optional: BERTopic
try:
    from bertopic import BERTopic
    BER_TOPIC_AVAILABLE = True
except ImportError:
    BER_TOPIC_AVAILABLE = False

st.set_page_config(page_title="Research Gap Analyzer – Lite Edition", layout="wide")

st.title("🔬 Research Gap Analyzer – Lite (TF-IDF + KMeans + BERTopic)")
st.markdown("This version uses TF-IDF and KMeans clustering and optionally BERTopic for advanced topic modeling.")

def generate_pdf_report(suggestion):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Research Gap Analyzer Report", ln=1, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", 'B', size=10)
    pdf.cell(200, 10, txt="Suggested Research Gap:", ln=1)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 10, suggestion)

    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return BytesIO(pdf_bytes)

def cluster_texts(texts, n_clusters=5):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8)
    X = vectorizer.fit_transform(texts)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    km.fit(X)
    clusters = km.predict(X)
    top_terms = []
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    for i in range(n_clusters):
        cluster_keywords = [terms[ind] for ind in order_centroids[i, :5]]
        top_terms.append(", ".join(cluster_keywords))
    return top_terms

def run_bertopic(texts):
    if not BER_TOPIC_AVAILABLE:
        return "BERTopic not available. Install it with: pip install bertopic", None
    model = BERTopic()
    topics, probs = model.fit_transform(texts)
    topic_df = model.get_topic_info()
    return None, topic_df

query = st.text_input("🔍 Enter your research topic or keywords")

if query and st.button("Run Analysis"):
    st.info("Fetching abstracts from Semantic Scholar...")

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "limit": 30, "fields": "title,abstract"}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        papers = data.get("data", [])
        docs = [p.get("abstract", "") for p in papers if p.get("abstract")]

        if not docs:
            st.warning("No abstracts found.")
        else:
            st.subheader("📊 Clusters from Abstracts (TF-IDF + KMeans)")
            top_terms = cluster_texts(docs)
            for i, terms in enumerate(top_terms):
                st.markdown(f"**Cluster {i+1}**: {terms}")

            st.subheader("📚 Advanced Topic Modeling (BERTopic)")
            err, topic_df = run_bertopic(docs)
            if err:
                st.warning(err)
            else:
                st.dataframe(topic_df)

            # Suggest a dummy research gap based on cluster 1
            if top_terms:
                suggestion = (
                    f"While existing research clusters around terms like **{top_terms[0]}**, "
                    f"other thematic areas may be underrepresented and require further exploration."
                )
                st.subheader("🧭 Suggested Research Gap")
                st.markdown(suggestion)
                pdf_buffer = generate_pdf_report(suggestion)
                st.download_button("📄 Download PDF Report", data=pdf_buffer, file_name="research_gap_report.pdf")
    else:
        st.error("Semantic Scholar API failed.")
        st.code(response.text)

st.markdown("---")
st.markdown("Developed by **Abdollah Baghaei Daemei** – [ResearchMate.org](https://www.researchmate.org)")

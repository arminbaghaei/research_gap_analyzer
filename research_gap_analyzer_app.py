import streamlit as st
import requests
import pandas as pd
from keybert import KeyBERT
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

def extract_keywords(text, top_n=15):
    kw_model = KeyBERT()
    return kw_model.extract_keywords(text, top_n=top_n, stop_words='english')

def generate_wordcloud(freq_dict):
    wordcloud = WordCloud(width=800, height=300, background_color="white").generate_from_frequencies(freq_dict)
    return wordcloud

def generate_pdf_report(keywords, suggestion):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Research Gap Analyzer Report", ln=1, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="Top Extracted Keywords:", ln=1)
    for kw, score in keywords:
        pdf.cell(200, 10, txt=f"- {kw} ({score:.2f})", ln=1)

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
            all_text = " ".join(docs)
            keywords = extract_keywords(all_text)
            freq_dict = {kw: score for kw, score in keywords}

            st.subheader("☁️ Keyword Cloud")
            wordcloud = generate_wordcloud(freq_dict)
            st.image(wordcloud.to_array(), use_container_width=True)

            st.subheader("🧠 Top Extracted Keywords")
            for kw, score in keywords:
                st.markdown(f"- {kw} ({score:.2f})")

            st.subheader("🧭 Suggested Research Gap")
            if len(keywords) >= 3:
                strong = keywords[0][0]
                weak = keywords[-1][0]
                suggestion = (
                    f"Despite growing interest in **{strong}**, "
                    f"the role of **{weak}** remains underexplored in the context of **{query}**."
                )
                st.markdown(suggestion)

                pdf_buffer = generate_pdf_report(keywords, suggestion)
                st.download_button("📄 Download PDF Report", data=pdf_buffer, file_name="research_gap_report.pdf")
            else:
                st.warning("Not enough keywords for suggestion or export.")

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
    else:
        st.error("Semantic Scholar API failed.")
        st.code(response.text)

st.markdown("---")
st.markdown("Developed by **Abdollah Baghaei Daemei** – [ResearchMate.org](https://www.researchmate.org)")

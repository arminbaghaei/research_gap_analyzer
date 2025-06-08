import streamlit as st
import requests
import pandas as pd
from keybert import KeyBERT
from wordcloud import WordCloud
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF

st.set_page_config(page_title="Research Gap Analyzer – Topic Modeling Edition", layout="wide")

st.title("🔬 Research Gap Analyzer – Topic Modeling + PDF Export")
st.markdown("Uses BERTopic to cluster abstracts and extract potential research gaps. You can also export the results as a PDF report.")

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

    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

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
            st.success(f"Retrieved {len(docs)} abstracts. Running BERTopic...")
            topic_model = BERTopic(verbose=False)
            topics, probs = topic_model.fit_transform(docs)

            fig = topic_model.visualize_barchart(top_n_topics=5)
            st.plotly_chart(fig, use_container_width=True)

            all_text = " ".join(docs)
            keywords = extract_keywords(all_text)
            freq_dict = {kw: score for kw, score in keywords}

            st.subheader("☁️ Keyword Cloud")
            wordcloud = generate_wordcloud(freq_dict)
            st.image(wordcloud.to_array(), use_container_width=True)

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
    else:
        st.error("Semantic Scholar API failed.")
else:
    st.info("Enter a topic and click 'Run Analysis'.")

st.markdown("---")
st.markdown("Developed by **Abdollah Baghaei Daemei** – [ResearchMate.org](https://www.researchmate.org)")

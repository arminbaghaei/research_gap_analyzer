import streamlit as st
import requests
import pandas as pd
from keybert import KeyBERT
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO, StringIO

st.set_page_config(page_title="Research Gap Analyzer (Advanced)", layout="centered")

st.title("📡 Research Gap Analyzer – Advanced Edition")
st.markdown("Analyze research topics via live academic search or uploaded text.")

kw_model = KeyBERT()

def extract_keywords(text, top_n=15):
    return kw_model.extract_keywords(text, top_n=top_n, stop_words='english')

def generate_wordcloud(freq_dict):
    wordcloud = WordCloud(width=800, height=300, background_color="white").generate_from_frequencies(freq_dict)
    return wordcloud

def display_results(text):
    keywords = extract_keywords(text)
    freq_dict = {kw: score for kw, score in keywords}
    
    st.subheader("🧠 Top Extracted Keywords")
    for kw, score in keywords:
        st.markdown(f"- {kw} ({score:.2f})")
    
    if freq_dict:
        st.subheader("☁️ Keyword Cloud")
        wordcloud = generate_wordcloud(freq_dict)
        st.image(wordcloud.to_array(), use_container_width=True)

        st.download_button(
            "📥 Download Keywords as CSV",
            data=pd.DataFrame(keywords, columns=["Keyword", "Score"]).to_csv(index=False),
            file_name="keywords.csv",
            mime="text/csv"
        )

        # Gap suggestion
        st.subheader("🧭 Suggested Research Gap")
        if len(keywords) >= 3:
            strong = keywords[0][0]
            weak = keywords[-1][0]
            suggestion = (
                f"Despite growing interest in **{strong}**, "
                f"the role of **{weak}** remains underexplored in the context of **your topic**."
            )
            st.markdown(suggestion)
        else:
            st.markdown("Not enough keywords to suggest a research gap.")
    else:
        st.warning("⚠️ No keywords found to generate insights.")

tab1, tab2 = st.tabs(["🔗 Live Search via Semantic Scholar", "📁 Upload Local CSV (Abstracts)"])

with tab1:
    query = st.text_input("🔍 Enter your research topic or keywords")

    if query and st.button("Analyze via API"):
        st.info("Searching Semantic Scholar and analyzing abstracts...")

        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {"query": query, "limit": 20, "fields": "title,abstract"}
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            papers = data.get("data", [])
            if not papers:
                st.warning("No papers found. Try a broader query.")
            else:
                all_text = " ".join(p.get("abstract", "") for p in papers if p.get("abstract"))
                display_results(all_text)
        else:
            st.error("API Error. Could not retrieve results.")
            st.code(response.text)

with tab2:
    uploaded_file = st.file_uploader("Upload CSV with Abstracts column", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'abstract' in df.columns:
            combined_text = " ".join(df['abstract'].dropna().astype(str))
            display_results(combined_text)
        else:
            st.error("CSV must contain a column named 'abstract'.")

st.markdown("---")
st.markdown("Developed by **Abdollah Baghaei Daemei** – [ResearchMate.org](https://www.researchmate.org)")

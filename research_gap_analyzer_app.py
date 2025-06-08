import streamlit as st
import requests
from keybert import KeyBERT
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Research Gap Analyzer (API)", layout="centered")

st.title("📡 Research Gap Analyzer – Live Literature Mode")
st.write("""
This version connects to the Semantic Scholar API to search recent papers,
extract keywords from their abstracts, and suggest possible research gaps.
""")

query = st.text_input("🔍 Enter your research topic or keywords")

if query and st.button("Analyze Research Gaps"):
    st.info("Searching Semantic Scholar and analyzing abstracts...")

    # Semantic Scholar API request
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": 20,
        "fields": "title,abstract"
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        papers = data.get("data", [])

        if not papers:
            st.warning("No papers found. Try a broader query.")
        else:
            all_abstracts = " ".join(paper.get("abstract", "") for paper in papers if paper.get("abstract"))
            kw_model = KeyBERT()
            keywords = kw_model.extract_keywords(all_abstracts, top_n=15, stop_words='english')

            st.subheader("🧠 Top Extracted Keywords")
            for kw, score in keywords:
                st.markdown(f"- {kw} ({score:.2f})")

            # Prepare frequencies
            freq_dict = {kw: score for kw, score in keywords}

            # Generate Word Cloud
            if freq_dict:
                st.subheader("☁️ Keyword Cloud")
                wordcloud = WordCloud(width=800, height=300, background_color="white").generate_from_frequencies(freq_dict)
                st.image(wordcloud.to_array(), use_container_width=True)
            else:
                st.warning("⚠️ No keywords found to generate a word cloud.")

            # Suggest a gap
            st.subheader("🧭 Suggested Research Gap")
            if len(keywords) >= 3:
                strong = keywords[0][0]
                weak = keywords[-1][0]
                suggestion = f"""
                - **{strong}** is frequently studied.  
                - **{weak}** is less discussed, yet related.  
                - You may explore how **{weak}** interacts with **{strong}** as a potential research angle.
                """
                st.markdown(suggestion)
            else:
                st.markdown("Not enough keywords to suggest a research gap.")

    else:
        st.error("Failed to retrieve data from Semantic Scholar API.")
        st.code(response.text)

st.markdown("---")
st.markdown("Developed by **Abdollah Baghaei Daemei** – [ResearchMate.org](https://www.researchmate.org)")

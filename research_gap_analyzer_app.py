import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="Research Gap Analyzer", layout="centered")

st.title("🔍 Research Gap Analyzer")
st.write("""
Paste your research abstract or topic below. This tool will analyze recent literature trends,
extract key themes, and suggest potentially underexplored areas for research.
""")

abstract = st.text_area("✍️ Enter your abstract or research topic", height=250)

if abstract and st.button("Analyze Research Gaps"):
    st.info("Extracting key terms and analyzing trends...")

    # Use KeyBERT for keyword extraction
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(abstract, top_n=10, stop_words='english')

    # Show top keywords
    st.subheader("🧠 Top Keywords")
    for kw, score in keywords:
        st.markdown(f"- {kw} ({score:.2f})")

    # Generate Word Cloud
    freq_dict = {kw: score for kw, score in keywords}
    st.subheader("☁️ Keyword Cloud")
    wordcloud = WordCloud(width=800, height=300, background_color="white").generate_from_frequencies(freq_dict)
    st.image(wordcloud.to_array(), use_column_width=True)

    # Suggest research gaps based on keyword spread
    st.subheader("🧭 Suggested Research Gaps")
    if len(keywords) >= 3:
        strong = keywords[0][0]
        weak = keywords[-1][0]
        suggestion = (
            f"- While **{strong}** is well-represented, **{weak}** appears less frequently.\n"
            f"- Consider exploring the relationship between **{strong}** and **{weak}** as a novel direction in your field."
        )
        st.markdown(suggestion)
    else:
        st.markdown("- Not enough keywords to infer a gap. Try a longer abstract.")

    st.success("This is a prototype. Future versions will fetch papers, perform topic modeling, and visualize trends.")

st.markdown("---")
st.markdown("Developed by **Abdollah Baghaei Daemei** – [ResearchMate.org](https://www.researchmate.org)")
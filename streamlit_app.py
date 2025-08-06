import streamlit as st
from engine import generate_content

st.set_page_config(page_title="SkillNestEdu Course Engine", layout="centered")

st.title("📘 SkillNestEdu Course Engine")
st.subheader("Your AI-powered interactive lesson builder")
st.write(
    "Select a subject, level, and topic to generate content dynamically — aligned with IB Economics and Business Management."
)

# New subject dropdown
subject = st.selectbox("Choose Subject", ["Economics", "Business Management"])

# Level dropdown
level = st.selectbox("Choose IB Level", ["IB1", "IB2"])

# Topic dropdown
topics = [
    "Foundations of Economics",
    "Demand and Supply",
    "Elasticity",
    "Market Failure",
    "Government Intervention",
    "Market Structures",
    "Macroeconomic Goals",
    "Aggregate Demand and Supply",
    "Monetary and Fiscal Policy",
    "International Trade",
    "Development Economics"
]

topic = st.selectbox("Choose a Topic", topics)

# Generate content if all selections are made
if subject and level and topic:
    st.info(f"📘 Generating content for {subject} | {level} - {topic}")
    with st.spinner("Creating your lesson..."):
        content = generate_content(subject, level, topic)
    st.markdown(content, unsafe_allow_html=True)

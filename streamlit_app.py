st.set_page_config(page_title="SkillNestEdu Course Engine", layout="centered")

st.title("📘 SkillNestEdu Course Engine")
st.subheader("Your AI-powered interactive lesson builder")

# Subject selection
subject = st.selectbox("Choose Subject", ["Economics", "Business Management"])

# Level selection
level = st.selectbox("Choose IB Level", ["IB1", "IB2"])

# Topic options
topic_dict = {
    "Economics": [
        "Foundations of Economics",
        "Elasticity",
        "Market Failure"
    ],
    "Business Management": [
        "Business Organization"
    ]
}

# Subtopics for each topic
subtopic_dict = {
    "Foundations of Economics": ["Production Possibility Curve", "None"],
    "Elasticity": ["Price Elasticity Interactive", "None"],
    "Market Failure": ["Externalities", "None"],
    "Business Organization": ["None"]
}

topic = st.selectbox("Choose a Topic", topic_dict.get(subject, []))
subtopic = st.selectbox("Choose a Subtopic (Optional)", subtopic_dict.get(topic, ["None"]))

# Normalize subtopic
subtopic = None if subtopic == "None" else subtopic

# Generate content
if subject and level and topic:
    st.info(f"📘 Generating content for {subject} - {level} - {topic}{' - ' + subtopic if subtopic else ''}")
    with st.spinner("Creating your lesson..."):
        content = generate_content(subject, level, topic, subtopic)
    st.markdown(content, unsafe_allow_html=True)
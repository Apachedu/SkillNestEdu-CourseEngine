from diagrams import ppc_diagram, elasticity_diagram
import streamlit as st

def generate_content(subject, level, topic, subtopic=None):
    if subject == "Economics":

        if topic == "Foundations of Economics":
            if subtopic == "Production Possibility Curve":
                st.markdown(f"""
                ## 📘 {level} - {topic}: {subtopic}
                ### 📊 Interactive Diagram
                """)
                ppc_diagram()
                return ""
            else:
                return f"""
                ## 📘 {level} - {topic}

                ### 🧠 What is Economics?
                Economics is the study of how scarce resources are allocated among competing needs.

                #### Key Concepts:
                - **Scarcity and Choice**
                - **Opportunity Cost**
                - **Positive vs Normative Statements**
                - **Economic Systems**

                #### ✅ Quick Quiz:
                1. What is scarcity?
                2. How do opportunity costs arise?
                """

        elif topic == "Elasticity":
            if subtopic == "Price Elasticity Interactive":
                st.markdown(f"""
                ## 📘 {level} - {topic}: {subtopic}
                ### 📊 Interactive Simulation
                """)
                elasticity_diagram()
                return ""
            else:
                return f"""
                ## 📘 {level} - {topic}

                ### 📏 What is Price Elasticity of Demand?
                It measures how quantity demanded changes in response to price changes.

                #### Key Concepts:
                - **PED Formula**
                - **Determinants of Elasticity**
                - **Elastic vs Inelastic Demand**
                - **Total Revenue Relationship**

                #### ✅ Quick Quiz:
                1. What does it mean if PED > 1?
                2. How does PED affect total revenue?
                """

    elif subject == "Business Management":
        if topic == "Business Organization":
            return f"""
            ## 📘 {level} - {topic}

            ### 🏢 What is a Business?
            A business is an organization that provides goods or services to satisfy needs and wants.

            #### Key Concepts:
            - **Profit vs Non-profit**
            - **Sole Traders, Partnerships, Corporations**
            - **Social Enterprises**

            #### ✅ Quick Quiz:
            1. Name three types of business ownership.
            2. What is a stakeholder?
            """

    return "🚧 Content loading..."

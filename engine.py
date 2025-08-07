from diagrams import ppc_diagram, elasticity_diagram
import streamlit as st

def generate_content(subject, level, topic, subtopic=None):
    if subject == "Economics":

        if topic == "Foundations of Economics":
            if subtopic == "Production Possibility Curve":
                st.markdown(f"""
                ## 📘 {level} - {topic}: {subtopic}
                ### 📊 Interactive Diagram with Explanation
                """)
                ppc_diagram()
                st.markdown("""
                **Explanation:**
                - The curve is concave due to increasing opportunity cost.
                - Points **inside** the curve = underutilization (inefficiency).
                - Points **on** the curve = optimal utilization.
                - Points **outside** = currently unattainable.
                """)
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

                #### 🔍 Real-World Example:
                Consider healthcare in a pandemic. Limited vaccines mean choices must be made.

                #### ✅ Quick Quiz:
                1. What is scarcity?
                2. How do opportunity costs arise?
                """

        elif topic == "Elasticity":
            if subtopic == "Price Elasticity Interactive":
                st.markdown(f"""
                ## 📘 {level} - {topic}: {subtopic}
                ### 📊 Elastic vs Inelastic Demand (Interactive)
                """)
                elasticity_diagram()
                st.markdown("""
                **Explanation:**
                - Green = Elastic → Small price change → large quantity change
                - Red = Inelastic → Large price change → small quantity change

                **Try It:**
                - Identify which curve represents luxury goods vs. necessities.
                - Discuss implications for total revenue.

                #### ✅ Quiz:
                1. Where is demand more responsive?
                2. What does a steeper curve imply?
                """)
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
            ## 📘 {level} - Business Organization

            ### 🏢 What is a Business?
            A business is an organization that provides goods or services to satisfy needs and wants.

            #### Key Concepts:
            - **Profit vs Non-profit**
            - **Sole Traders, Partnerships, Corporations**
            - **Social Enterprises**

            #### 🧠 Critical Thinking Prompt:
            Compare and contrast for-profit and social enterprises.

            #### ✅ Quick Quiz:
            1. Name three types of business ownership.
            2. What is a stakeholder?
            """

    return "🚧 Content loading..."
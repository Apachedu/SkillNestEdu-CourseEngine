def generate_content(subject, level, topic):
    if subject == "Economics":
        if topic == "Foundations of Economics":
            return f"""
            ## 📘 {level} - Foundations of Economics

            ### 🧠 What is Economics?
            Economics is the study of how scarce resources are allocated among competing needs.

            #### Key Concepts:
            - **Scarcity and Choice**
            - **Opportunity Cost**
            - **Positive vs Normative Statements**
            - **Economic Systems**

            #### 🔍 Real-World Example:
            Consider healthcare in a pandemic. Limited vaccines mean choices must be made.

            #### 📊 Diagram Prompt:
            Draw a Production Possibility Curve showing opportunity cost.

            #### ✅ Quick Quiz:
            1. What is scarcity?
            2. How do opportunity costs arise?
            """
        
        elif topic == "Elasticity":
            return f"""
            ## 📘 {level} - Elasticity

            ### 📏 What is Price Elasticity of Demand?
            It measures how quantity demanded changes in response to price changes.

            #### Key Concepts:
            - **PED Formula**
            - **Determinants of Elasticity**
            - **Elastic vs Inelastic Demand**
            - **Total Revenue Relationship**

            #### 🧪 Simulation Idea:
            Input price and quantity changes to see the curve shift dynamically.

            #### ✅ Quick Quiz:
            1. What does it mean if PED > 1?
            2. How does PED affect total revenue?
            """

        # Add more Econ topics...

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

        # Add more BM topics...

    else:
        return "🚧 Content loading..."


        
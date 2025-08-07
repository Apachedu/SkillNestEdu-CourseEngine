import plotly.graph_objects as go
import streamlit as st
import numpy as np

def ppc_diagram():
    x = np.linspace(0, 10, 100)
    y = 10 - 0.5 * x**2 / 10  # Concave PPC

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='PPC Curve', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=[2, 6, 8],
                             y=[10 - 0.5 * 2**2 / 10, 10 - 0.5 * 6**2 / 10, 10 - 0.5 * 8**2 / 10],
                             mode='markers+text',
                             name='Choices',
                             text=["A", "B", "C"],
                             textposition="top center",
                             marker=dict(size=10, color='orange')))
    
    fig.update_layout(
        title="Production Possibility Curve (PPC)",
        xaxis_title="Units of Good A",
        yaxis_title="Units of Good B",
        showlegend=True,
        width=700,
        height=500
    )

    st.plotly_chart(fig)
import streamlit as st
import plotly.graph_objects as go

def elasticity_diagram():
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[10, 8, 6, 4, 2], 
        y=[2, 4, 6, 8, 10],
        mode='lines+markers',
        name='Elastic Demand',
        line=dict(color='green')
    ))

    fig.add_trace(go.Scatter(
        x=[10, 9, 8, 7, 6], 
        y=[2, 3, 4, 5, 6],
        mode='lines+markers',
        name='Inelastic Demand',
        line=dict(color='red')
    ))

    fig.update_layout(
        title='Elastic vs Inelastic Demand',
        xaxis_title='Price',
        yaxis_title='Quantity',
        legend=dict(x=0.7, y=0.95)
    )

    st.plotly_chart(fig)

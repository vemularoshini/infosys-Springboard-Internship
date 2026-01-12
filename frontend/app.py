import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from backend.knowledge_engine import KnowledgeEngine

engine = KnowledgeEngine("data/knowledge.csv")

st.title("AI Powered Knowledge Engine - Smart Support & Ticket Resolution")
st.markdown("""
This is an **AI-powered Knowledge Engine**.
- Type your question in the box below.
- Known questions get **instant answers**.
- Unknown questions are submitted as **support tickets**.
""")

query = st.text_input("Ask your question:")
if query:
    response = engine.answer_query(query)
    st.write("**Answer:**", response)

# Optional: show tickets
if st.checkbox("Show submitted tickets"):
    with open(engine.ticket_file, "r") as f:
        tickets = f.read()
    st.text_area("Tickets:", tickets, height=200)

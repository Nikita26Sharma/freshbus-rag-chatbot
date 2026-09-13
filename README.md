# FreshBus RAG Assistant

A deliberately simple RAG-based chatbot prototype for the FreshBus interview assessment.

## Architecture

FreshBus public information
→ curated knowledge records
→ local sentence embeddings
→ cosine similarity retrieval
→ top 3 relevant records
→ constrained Gemini prompt
→ grounded response in Streamlit

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Set `GEMINI_API_KEY` in the environment, or add it to Streamlit secrets.

## Deployment

Designed for Streamlit Community Cloud. Put this project in a GitHub repository and deploy `app.py`.

## Important prototype limitation

This uses a small, curated knowledge base and an in-memory vector index rather than a production database. It does not connect to FreshBus booking, tracking, payment, seat inventory, or customer-account systems.

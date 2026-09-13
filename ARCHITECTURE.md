# FreshBus RAG Chatbot Architecture

## Assessment objective

Build a simple RAG-based chatbot using publicly available FreshBus information, explain the architecture, and provide a working chatbot link.

## Prototype architecture

1. **Knowledge source**
   Public FreshBus FAQ, Terms & Conditions, Fresh Card, Fresh Promise and Group Booking pages.

2. **Knowledge preparation**
   Relevant customer-facing facts are curated into small semantic records. Each record retains category and source URL metadata.

3. **Embedding**
   `all-MiniLM-L6-v2` converts each knowledge record and the user's question into vector representations.

4. **Retrieval**
   Cosine similarity is used to retrieve the top 3 relevant records. A relevance threshold prevents every query from automatically receiving context.

5. **Augmentation**
   The retrieved records are inserted into a constrained prompt.

6. **Generation**
   Gemini 2.5 Flash generates a concise answer using only the retrieved FreshBus context.

7. **Interface**
   Streamlit provides the chat interface and session-level conversation history.

## Why this is intentionally simple

The assessment asks for a simplistic RAG chatbot. The prototype therefore avoids unnecessary production infrastructure such as a managed vector database, authentication, agent orchestration, live APIs and complex evaluation services.

## Production evolution

A production FreshBus implementation could replace the curated local knowledge file with governed sources such as a CMS, policy database and operational APIs. A managed vector database could replace the local in-memory index. Live questions such as bus tracking, seat availability and booking status should be handled by operational APIs rather than RAG.

## Key limitation

The prototype knowledge base is intentionally limited and is not a complete representation of FreshBus operational data. Information that changes frequently should be sourced from live systems in production.

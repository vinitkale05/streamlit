import streamlit as st
import pandas as pd
import numpy as np
import torch
import faiss
import re
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ─── Configuration ──────────────────────────────────────────────────────────────

GEMINI_API_KEY = "AIzaSyDjmJZB7jp92jiKQHHu9gn86gpjLD_30NQ"
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash")

# ─── Load Model ─────────────────────────────────────────────────────────────────

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# ─── Load Data and Build Index ──────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_data_embed_faiss():
    df = pd.read_csv("vehicle_specs_cleaned.csv").fillna("")
    texts = [" | ".join(f"{k}: {v}" for k, v in row.items()) for _, row in df.iterrows()]
    embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")

    # Build FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return df, texts, embeddings, index

df, texts, embeddings, faiss_index = load_data_embed_faiss()

# ─── Helpers ─────────────────────────────────────────────────────────────────────

def extract_years(text):
    return re.findall(r'\b(19|20)\d{2}\b', text)

def filter_indices_by_years(df, years):
    if not years:
        return list(df.index)
    return df.index[
        df.apply(lambda row: any(year in str(val) for val in row.values for year in years), axis=1)
    ].tolist()

def rerank_mmr(embeddings_subset, query_embedding, k=10, diversity=0.7):
    selected, candidates = [], list(range(len(embeddings_subset)))
    sim = torch.nn.functional.cosine_similarity(
        torch.tensor(query_embedding).unsqueeze(0),
        torch.tensor(embeddings_subset),
        dim=1
    ).numpy()

    for _ in range(min(k, len(candidates))):
        if not selected:
            idx = int(np.argmax(sim))
        else:
            mmr_scores = []
            for i in candidates:
                rel = sim[i]
                div = max(torch.nn.functional.cosine_similarity(
                    torch.tensor(embeddings_subset[i]).unsqueeze(0),
                    torch.tensor([embeddings_subset[j] for j in selected])
                ).numpy())
                mmr = diversity * rel - (1 - diversity) * div
                mmr_scores.append((i, mmr))
            idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(idx)
        candidates.remove(idx)
    return selected

# ─── Streamlit UI ────────────────────────────────────────────────────────────────

st.title("⚡ Ultra-Fast Vehicle Specs Q&A (with Gemini & FAISS)")
st.markdown("Ask about any year/model like: `Specs of 2022 Swift` or `What is 2020 Honda City mileage?`")

question = st.text_input("❓ Ask your question:")
diversity = st.slider("Diversity (MMR rerank)", 0.0, 1.0, 0.7, 0.05)
top_k = st.slider("Number of Specs to Use", 1, 15, 10)

if st.button("🔍 Submit Question"):
    if not question.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Processing under 1s..."):
            query_vec = model.encode(question, convert_to_numpy=True).astype("float32")
            years = extract_years(question)
            year_filtered_indices = filter_indices_by_years(df, years)

            if not year_filtered_indices:
                st.error("❌ No matching specs found for your query.")
            else:
                # Filter FAISS index by precomputed indices
                emb_subset = embeddings[year_filtered_indices]
                index_subset = faiss.IndexFlatL2(emb_subset.shape[1])
                index_subset.add(emb_subset)

                _, faiss_ids = index_subset.search(np.array([query_vec]), top_k)
                top_indices = faiss_ids[0].tolist()
                reranked = rerank_mmr(emb_subset[top_indices], query_vec, k=top_k, diversity=diversity)

                final_indices = [year_filtered_indices[top_indices[i]] for i in reranked]
                top_specs = [texts[i] for i in final_indices]

                context = "\n\n".join(f"Spec {i+1}:\n{spec}" for i, spec in enumerate(top_specs))
                prompt = f"""You are an expert in automobile specifications.

Here are some vehicle specs:

{context}

Question: {question}
Answer clearly and concisely, with technical detail but in simple language."""

                try:
                    response = gemini.generate_content(prompt)
                    st.success("✅ Answer:")
                    st.markdown(response.text)
                    with st.expander("📂 View Specs Used"):
                        st.text(context)
                except Exception as e:
                    st.error(f"Gemini API error: {e}")

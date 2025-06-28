import streamlit as st
import pandas as pd
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# For production, use environment variables for API keys!
GEMINI_API_KEY = "AIzaSyDjmJZB7jp92jiKQHHu9gn86gpjLD_30NQ"  
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash")

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

@st.cache_resource(show_spinner=False)
def load_data_embed_faiss():
    df = pd.read_csv("vehicle_specs_train.csv").fillna("")
    texts = [
        " | ".join(f"{k}: {v}" for k, v in row.items())
        for _, row in df.iterrows()
    ]
    embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return df, texts, embeddings, index

df, texts, embeddings, faiss_index = load_data_embed_faiss()

def rerank_mmr(embeddings_subset, query_embedding, k=5, diversity=0.7):
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

st.title("🛠️ Vehicle Tag & Info Predictor (via User Input + MMR + Gemini)")
st.markdown("Enter vehicle specifications manually and get AI-generated tags and summary.")

with st.form("input_form"):
    user_input = st.text_area("🔧 Enter Specs (Format: key: value | key: value ...)")
    top_k = st.slider("Top similar specs to retrieve (context)", 1, 10, 5)
    diversity = st.slider("MMR Diversity (0 = focused, 1 = diverse)", 0.0, 1.0, 0.7, 0.05)
    submitted = st.form_submit_button("🔍 Predict Tags & Info")

if submitted and user_input.strip():
    try:
        query_vec = model.encode(user_input, convert_to_numpy=True).astype("float32")
        _, faiss_ids = faiss_index.search(np.array([query_vec]), top_k)
        subset_indices = faiss_ids[0].tolist()
        subset_embeddings = embeddings[subset_indices]
        reranked = rerank_mmr(subset_embeddings, query_vec, k=top_k, diversity=diversity)
        context_texts = [texts[subset_indices[i]] for i in reranked]
        context = "\n".join(f"Spec {i+1}: {t}" for i, t in enumerate(context_texts))

        prompt = f"""You are an expert in vehicle specifications.

Given these examples:

{context}

And the following user specs:

{user_input}

Predict:
Tags: <comma-separated list>
Additional Info: <brief summary>
"""

        response = gemini.generate_content(prompt)
        st.success("✅ Predicted Output")
        st.markdown(response.text)

        with st.expander("🧠 Context Used from Training Data"):
            st.text(context)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

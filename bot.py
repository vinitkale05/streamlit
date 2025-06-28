import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
import torch
import re


GEMINI_API_KEY = "AIzaSyDjmJZB7jp92jiKQHHu9gn86gpjLD_30NQ"
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash")
model = SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource(show_spinner=False)
def load_and_embed_specs():
    df = pd.read_csv("vehicle_specs_cleaned.csv")
    specs = df.to_dict(orient="records")
    texts = [" | ".join(f"{k}: {v}" for k, v in item.items()) for item in specs]
    embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")
    return df, texts, embeddings


def mmr(doc_embeddings, query_embedding, texts, top_k=3, diversity=0.7):
    if not isinstance(doc_embeddings, torch.Tensor):
        doc_embeddings = torch.tensor(doc_embeddings)
    if not isinstance(query_embedding, torch.Tensor):
        query_embedding = torch.tensor(query_embedding)
    similarity = util.cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()
    selected, candidates = [], list(range(len(texts)))
    for _ in range(top_k):
        if not candidates:
            break
        if not selected:
            selected_idx = int(np.argmax(similarity))
        else:
            mmr_scores = []
            for idx in candidates:
                relevance = similarity[idx]
                selected_embeds = doc_embeddings[selected]
                sim_to_selected = util.cos_sim(doc_embeddings[idx].unsqueeze(0), selected_embeds)[0].cpu().numpy()
                diversity_score = float(sim_to_selected) if sim_to_selected.ndim == 0 else max(sim_to_selected)
                mmr_score = diversity * relevance - (1 - diversity) * diversity_score
                mmr_scores.append((idx, mmr_score))
            selected_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(selected_idx)
        candidates.remove(selected_idx)
    return [texts[i] for i in selected]


st.title("🚘 Vehicle Specs Assistant ")
st.markdown("Ask your vehicle-related questions. This assistant uses memory, specs to answer like specialist in cars.")

df, texts, doc_embeddings = load_and_embed_specs()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


user_input = st.chat_input("Ask something about vehicle specs...")

def extract_years(text):
    
    return [int(y) for y in re.findall(r"\b(19[5-9]\d|20[0-4]\d|2050)\b", text)]

if user_input:
    if len(texts) == 0 or doc_embeddings.shape[0] == 0:
        st.error("No specs data available. Please check your CSV file.")
    else:
        with st.spinner("Processing your question..."):
            
            years = extract_years(user_input)
            df_search = df
            if years and 'year' in df.columns:
                df_search = df[df['year'].isin(years)]
            if df_search.empty:
                st.error("No vehicles found for the specified year(s).")
            else:
                specs = df_search.to_dict(orient="records")
                texts_search = [" | ".join(f"{k}: {v}" for k, v in item.items()) for item in specs]
                doc_embeddings_search = model.encode(texts_search, convert_to_numpy=True).astype("float32")

                query_embedding = model.encode(user_input, convert_to_numpy=True).astype("float32")
                top_specs = mmr(doc_embeddings_search, query_embedding, texts_search, top_k=3, diversity=0.7)
                context = "\n".join(f"Spec {i+1}: {spec}" for i, spec in enumerate(top_specs))

                conversation = ""
                for turn in st.session_state.chat_history:
                    conversation += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

                if years:
                    prompt = f"""
You are a helpful AI assistant trained on vehicle specifications.

If the user asks about years, focus on the 'year' field in the specs.

Context vehicle specs:
{context}

Ongoing conversation:
{conversation}
User: {user_input}
Assistant:"""
                else:
                    prompt = f"""
You are a helpful AI assistant trained on vehicle specifications.

Context vehicle specs:
{context}

Ongoing conversation:
{conversation}
User: {user_input}
Assistant:"""

                try:
                    response = gemini.generate_content(prompt)
                    answer = response.text.strip()
                except Exception as e:
                    answer = f"Error from Gemini: {e}"

                st.session_state.chat_history.append({
                    "user": user_input,
                    "assistant": answer,
                    "specs": context
                })


for turn in st.session_state.chat_history:
    st.chat_message("user").write(turn["user"])
    st.chat_message("assistant").write(turn["assistant"])
    with st.expander("🔍 Specs used"):
        st.text(turn["specs"])

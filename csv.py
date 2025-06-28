import pandas as pd
import openai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your key

# 1. Load and clean data
df = pd.read_csv('vehicle_specs.csv').dropna()

# 2. Prepare text chunks
chunks = df.apply(lambda row: ', '.join([f"{col}: {row[col]}" for col in df.columns]), axis=1).tolist()

# 3. Embed chunks
def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

# 4. Embed query and run MMR
def mmr(doc_embeddings, query_embedding, top_k=3, lambda_param=0.5):
    doc_embeddings = np.array(doc_embeddings)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    similarities = cosine_similarity(doc_embeddings, query_embedding).flatten()
    selected = []
    candidates = list(range(len(doc_embeddings)))
    for _ in range(top_k):
        if not candidates:
            break
        if not selected:
            idx = np.argmax(similarities)
            selected.append(idx)
            candidates.remove(idx)
        else:
            mmr_scores = []
            for candidate in candidates:
                sim_to_query = similarities[candidate]
                sim_to_selected = max([cosine_similarity(
                    doc_embeddings[candidate].reshape(1, -1),
                    doc_embeddings[s].reshape(1, -1)
                )[0][0] for s in selected])
                mmr_score = lambda_param * sim_to_query - (1 - lambda_param) * sim_to_selected
                mmr_scores.append(mmr_score)
            idx = candidates[np.argmax(mmr_scores)]
            selected.append(idx)
            candidates.remove(idx)
    return selected

user_query = "Which vehicle has the highest horsepower?"
query_embedding = get_embedding(user_query)
selected_indices = mmr(chunk_embeddings, query_embedding, top_k=3)
retrieved_specs = [chunks[i] for i in selected_indices]

# 5. Pass as context to LLM and ask question
context = "\n".join(retrieved_specs)
prompt = f"Given the following vehicle specs:\n{context}\n\nAnswer this question: {user_query}"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}]
)

print("LLM Answer:", response['choices'][0]['message']['content'])
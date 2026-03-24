import pickle

from openai import OpenAI
# client = OpenAI(api_key="sk-...")
client = OpenAI()

def embed_chunks(chunks):
    response = client.embeddings.create(
        input=chunks,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]

def chunk_text(text, chunk_size=100, overlap=20):
    text = text.split()
    return [' '.join(text[i:i+chunk_size]) for i in range(0, len(text), chunk_size-overlap)]

def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a ** 2 for a in vec1) ** 0.5
    magnitude2 = sum(b ** 2 for b in vec2) ** 0.5
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

def retrieve(query, chunk_embedding_pairs, top_k=5):
    query_embedding = embed_chunks([query])[0]
    similarities = [(chunk, cosine_similarity(query_embedding, embedding)) for chunk, embedding in chunk_embedding_pairs]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def generate_answer(retrieval_prompt):
    response = client.responses.create(
        model="gpt-3.5-turbo",
        input=[{"role": "system", "content": "You are a helpful RAG assistant."}, {"role": "user", "content": retrieval_prompt}]
    )
    return response.output_text

try:
    with open('chunk_embeddings.pkl', 'rb') as f:
        chunk_embedding_pairs = pickle.load(f)
except FileNotFoundError:
    with open('book1.txt', 'r') as f:
        text = f.read()
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    chunk_embedding_pairs = list(zip(chunks, embeddings))
    with open('chunk_embeddings.pkl', 'wb') as f:
        pickle.dump(chunk_embedding_pairs, f)

print("Enter your query:")
query = input()
results = retrieve(query, chunk_embedding_pairs)
retrieval_prompt = f"Relevant chunks: {" // ".join([chunk for chunk, _ in results[:5]])}, Answer the following query: {query}"
# print(retrieval_prompt)
answer = generate_answer(retrieval_prompt)
print("\nGenerated Answer:")
print(answer)
# retrieval_prompt = f"Relevant chunks: {" // ".join([chunk for chunk, _ in results[:5]])}"

# print("\nTop relevant chunks:")
# for i, (chunk, similarity) in enumerate(results):
#     print(f'Chunk {i+1} (Similarity: {similarity:.4f}):\n{chunk}\n')

# for i, (chunk, embedding) in enumerate(chunk_embedding_pairs):
#     print(f'Embedding for Chunk {i+1}:\n{embedding[:10]}... (truncated)\n')
#     # print(f'Embedding for Chunk {i+1}:\n{embedding}\n')

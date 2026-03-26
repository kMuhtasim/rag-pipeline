import chromadb

from openai import OpenAI
# client = OpenAI(api_key="sk-...")
client = OpenAI()
chroma_client = chromadb.PersistentClient(path="./chroma_db")

TOP_K = 5

def embed_chunks(chunks):
    response = client.embeddings.create(
        input=chunks,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]

def chunk_text(text, chunk_size=100, overlap=20):
    text = text.split()
    return [' '.join(text[i:i+chunk_size]) for i in range(0, len(text), chunk_size-overlap)]

### Legacy snippet, which did ENN, replaced by ANN version using ChromaDB
def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a ** 2 for a in vec1) ** 0.5
    magnitude2 = sum(b ** 2 for b in vec2) ** 0.5
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

### Legacy version, replaced by ChromaDB version below
def retrieve(query, chunk_embedding_pairs, top_k=TOP_K):
    query_embedding = embed_chunks([query])[0]
    similarities = [(chunk, cosine_similarity(query_embedding, embedding)) for chunk, embedding in chunk_embedding_pairs]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def retrieve_chroma(query, collection, top_k=TOP_K):
    query_embedding = embed_chunks([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances"]
    )
    # for i, (document, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
    #     print(f"Retrieved Document {i+1}:\n{document}\n")
    #     print(f"Distance: {distance}\n")
    return list(zip(results['documents'][0], results['distances'][0]))

def generate_answer(instructions):
    response = client.responses.create(
        model="gpt-3.5-turbo",
        input=instructions
    )
    return response.output_text

### Legacy version, replaced by ChromaDB version below
# try:
#     with open('chunk_embeddings.pkl', 'rb') as f:
#         chunk_embedding_pairs = pickle.load(f)
# except FileNotFoundError:
#     with open('book1.txt', 'r') as f:
#         text = f.read()
#     chunks = chunk_text(text)
#     embeddings = embed_chunks(chunks)
#     chunk_embedding_pairs = list(zip(chunks, embeddings))
#     with open('chunk_embeddings.pkl', 'wb') as f:
#         pickle.dump(chunk_embedding_pairs, f)

collection = chroma_client.get_or_create_collection(
    name="books",
    metadata={"hnsw:space": "cosine"}
)
if collection.count() == 0:
    with open('book1.txt', 'r') as f:
        text = f.read()
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    collection.add(
        ids=[f"chunk_{i+1}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings
    )

print("Enter your query:")
query = input()
# query = "What metaphysical conclusion does he arrive at?"
# results = retrieve(query, chunk_embedding_pairs)
results = retrieve_chroma(query, collection)
for i, (chunk, distance) in enumerate(results):
    print(f"Retrieved Chunk {i+1}:\n{chunk}\n")
    print(f"Distance: {distance}\n")
instructions = []
instructions.append({"role": "system", "content": f"You are a helpful Retrieval Augmented Generation (RAG) assistant. Use the following relevant chunks to answer the user's query.\n{"\n".join([f"Chunk {i+1}: {chunk}" for i, (chunk, _) in enumerate(results)])}"})
instructions.append({"role": "user", "content": query})

# print("\nInstructions for RAG model:")
# for instruction in instructions:
#     print(f"{instruction['role'].capitalize()}:\n{instruction['content']}\n")

answer = generate_answer(instructions)
print("\nGenerated Answer:")
print(answer)

# print("\nTop relevant chunks:")
# for i, (chunk, similarity) in enumerate(results):
#     print(f'Chunk {i+1} (Similarity: {similarity:.4f}):\n{chunk}\n')

# for i, (chunk, embedding) in enumerate(chunk_embedding_pairs):
#     print(f'Embedding for Chunk {i+1}:\n{embedding[:10]}... (truncated)\n')
#     # print(f'Embedding for Chunk {i+1}:\n{embedding}\n')

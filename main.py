import chromadb

from openai import OpenAI
# client = OpenAI(api_key="sk-...")
client = OpenAI()
chroma_client = chromadb.PersistentClient(path="./.chroma_db")

TOP_K = 5
MODEL = "gpt-4o-mini"
TOKEN_LIMIT = 100000

def embed_chunks(chunks):
    response = client.embeddings.create(
        input=chunks,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]

def chunk_text(text, chunk_size=300, overlap=20):
    text = text.split()
    return [' '.join(text[i:i+chunk_size]) for i in range(0, len(text), chunk_size-overlap)]

### Legacy snippet, which did ENN, replaced by ANN version using ChromaDB
# def cosine_similarity(vec1, vec2):
#     dot_product = sum(a * b for a, b in zip(vec1, vec2))
#     magnitude1 = sum(a ** 2 for a in vec1) ** 0.5
#     magnitude2 = sum(b ** 2 for b in vec2) ** 0.5
#     if magnitude1 == 0 or magnitude2 == 0:
#         return 0.0
#     return dot_product / (magnitude1 * magnitude2)

### Legacy version, replaced by ChromaDB version below
# def retrieve(query, chunk_embedding_pairs, top_k=TOP_K):
#     query_embedding = embed_chunks([query])[0]
#     similarities = [(chunk, cosine_similarity(query_embedding, embedding)) for chunk, embedding in chunk_embedding_pairs]
#     similarities.sort(key=lambda x: x[1], reverse=True)
#     return similarities[:top_k]

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

def dummy_answer_from_HyDE(query, retrieved_chunks):
    instructions = []
    instructions.append({"role": "system", "content": f"You are a helpful Retrieval Augmented Generation (RAG) assistant. Use ONLY the following relevant chunks to answer the user's query. If you don't have relevant chunks, provide a dummy answer.\n{"\n".join([f"Chunk {i+1}: {chunk}" for i, (chunk, _) in enumerate(retrieved_chunks)])}"})
    instructions.append({'role': 'user', 'content': query})
    # print("\nInstructions for HyDE:")
    # for instruction in instructions:
    #     print(f"{instruction['role'].capitalize()}:\n{instruction['content']}\n")
    response = client.chat.completions.create(
        model=MODEL,
        messages=instructions
    )
    dummy_answer = response.choices[0].message.content
    # print("\nDummy Answer from HyDE:")
    # print(dummy_answer)
    return dummy_answer

def generate_answer(instructions):
    # response = client.responses.create(
    #     model=MODEL,
    #     input=instructions
    # )
    # return response.output_text
    response = client.chat.completions.create(
        model=MODEL,
        messages=instructions
    )
    return response.choices[0].message.content

### Legacy version, replaced by ChromaDB version below
# try:
#     with open('chunk_embeddings.pkl', 'rb') as f:
#         chunk_embedding_pairs = pickle.load(f)
# except FileNotFoundError:
#     with open('18637-8.txt', 'r') as f:
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
    with open('18637-8.txt', 'r') as f:
        full_text = f.read()
    word_count = len(full_text.split())
    token_count = word_count * 2
    # remaining_tokens = token_count
    how_many_rounds = int((token_count // TOKEN_LIMIT) + 1)
    text_chunk = []
    chunks = []
    embeddings = []
    for round in range(how_many_rounds):
        print(f"Processing round {round+1}/{how_many_rounds}...")
        start_index = round * TOKEN_LIMIT
        end_index = min((round + 1) * TOKEN_LIMIT, len(full_text.split()))
        text_round = ' '.join(full_text.split()[start_index:end_index])
        if not text_round.strip():
            print("Empty text round, skipping...")
            continue
        chunk_round = chunk_text(text_round)
        embedding_round = embed_chunks(chunk_round)
        chunks.extend(chunk_round)
        embeddings.extend(embedding_round)


    # chunks = chunk_text(text_chunk)
    # embeddings = embed_chunks(chunks)
    collection.add(
        ids=[f"chunk_{i+1}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings
    )

print("Enter your query:")
query = input()
# query = "What are the main themes explored in the book?"
# print(query)
# results = retrieve(query, chunk_embedding_pairs)
retrieval_results = retrieve_chroma(query, collection)
# dummy_answer = dummy_answer_from_HyDE(query, retrieval_results)
# retrieval_results = retrieve_chroma(dummy_answer, collection)
for i, (chunk, distance) in enumerate(retrieval_results):
    print(f"Retrieved Chunk {i+1}:\n{chunk}\n")
    print(f"Distance: {distance}\n")

instructions = []
instructions.append({"role": "system", "content": f"You are a helpful Retrieval Augmented Generation (RAG) assistant. Use ONLY the following relevant chunks to answer the user's query.\n{"\n".join([f"Chunk {i+1}: {chunk}" for i, (chunk, _) in enumerate(retrieval_results)])}"})
instructions.append({"role": "user", "content": query})

answer = generate_answer(instructions)
print("\nGenerated Answer:")
print(answer)
from sentence_transformers import SentenceTransformer
import faiss
import json

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def load_knowledge(file_path='data/knowledge_base.json'):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def build_index(questions):
    embeddings = model.encode(questions)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    return index, embeddings

def search_answer(query, data, index, questions, top_k=1):
    query_vec = model.encode([query])
    D, I = index.search(query_vec, top_k)
    return data[I[0][0]] if D[0][0] < 1.0 else {"question": "", "answer": "很抱歉，我还没有相关知识。"}
import json

def load_knowledge(file_path='data/knowledge_base.json'):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_knowledge(data, file_path='data/knowledge_base.json'):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def add_entry(question, answer, tags, file_path='data/knowledge_base.json'):
    data = load_knowledge(file_path)
    data.append({"question": question, "answer": answer, "tags": tags})
    save_knowledge(data, file_path)
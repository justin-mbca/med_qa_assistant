import streamlit as st
import json
import os
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# 页面配置
st.set_page_config(page_title="医疗问答助手", page_icon="💊")

# 文本标准化处理（清除多余符号、空格）
def clean_text(text):
    text = text.strip()
    text = re.sub(r"[，,]", ",", text)
    text = re.sub(r"[？?]", "?", text)
    text = re.sub(r"[！!]", "!", text)
    text = re.sub(r"\s+", "", text)  # 删除所有空格
    return text

# 加载模型
@st.cache_resource
def load_model():
    return SentenceTransformer("shibing624/text2vec-base-chinese")

model = load_model()

# 知识库路径
KB_PATH = "data/knowledge_base.json"

# 加载知识库
def load_knowledge():
    if not os.path.exists(KB_PATH):
        return []
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# 保存知识库
def save_knowledge(kb):
    with open(KB_PATH, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)

# 构建向量索引
def build_index(kb):
    if not kb:
        return None, []
    cleaned_questions = [clean_text(item["question"]) for item in kb]
    embeddings = model.encode(cleaned_questions, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

# 加载知识库并构建索引
knowledge_base = load_knowledge()
index, embeddings = build_index(knowledge_base)

# 查询最相似问题
def search_answer(query, top_k=1):
    if index is None or not knowledge_base:
        return None, 0.0
    cleaned_query = clean_text(query)
    query_vec = model.encode([cleaned_query], convert_to_numpy=True)
    D, I = index.search(query_vec, top_k)
    if D[0][0] < 1.2:  # 设置容忍度（距离越小越相似）
        return knowledge_base[I[0][0]], D[0][0]
    return None, 0.0

# 主界面
tab1, tab2 = st.tabs(["🤖 问答助手", "📚 专家知识录入"])

with tab1:
    st.title("🧠 医疗问答助手")
    st.markdown("输入您的健康问题，我们将根据知识库提供参考建议。")

    query = st.text_input("请输入您的问题：", key="query_input")

    if st.button("🔍 查询"):
        if not query.strip():
            st.warning("请输入有效的问题。")
        else:
            result, score = search_answer(query)
            if result:
                st.success("找到相关答案：")
                st.markdown(f"**Q：** {result['question']}")
                st.markdown(f"**A：** {result['answer']}")
                if result.get("tags"):
                    st.caption(f"标签：{'、'.join(result['tags'])}")
                st.caption(f"相似度：{1.0 - score:.2f}")
            else:
                st.error("很抱歉，我还没有相关知识。欢迎添加！")

with tab2:
    st.subheader("📚 添加专家知识")
    new_q = st.text_input("问题：")
    new_a = st.text_area("答案：")
    new_tags = st.text_input("标签（用逗号分隔）")

    if st.button("📥 添加新条目"):
        if new_q and new_a:
            new_entry = {
                "question": new_q.strip(),
                "answer": new_a.strip(),
                "tags": [t.strip() for t in new_tags.split(",") if t.strip()]
            }
            knowledge_base.append(new_entry)
            save_knowledge(knowledge_base)
            index, embeddings = build_index(knowledge_base)
            st.success("✅ 新知识已添加！")
        else:
            st.warning("请填写完整的问题和答案。")

import streamlit as st
import json
import os
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# 页面设置
st.set_page_config(page_title="医疗问答助手", page_icon="🩺")

# 清洗文本：去标点、空白，统一格式
def clean_text(text):
    text = text.strip()
    text = re.sub(r"[，,]", ",", text)
    text = re.sub(r"[？?]", "?", text)
    text = re.sub(r"[！!]", "!", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[,.!?？！。，、]", "", text)
    return text

# 加载模型（使用缓存避免重复加载）
@st.cache_resource
def load_model():
    return SentenceTransformer("shibing624/text2vec-base-chinese")

model = load_model()

# 数据路径
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

# 构建FAISS索引
def build_index(kb):
    if not kb:
        return None, []
    cleaned_questions = [clean_text(item["question"]) for item in kb]
    embeddings = model.encode(cleaned_questions, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

# 加载知识库 & 构建索引
knowledge_base = load_knowledge()
index, embeddings = build_index(knowledge_base)

# 查询函数
def search_answer(query, top_k=1):
    if index is None or not knowledge_base:
        return None, 0.0
    cleaned_query = clean_text(query)
    query_vec = model.encode([cleaned_query], convert_to_numpy=True)
    D, I = index.search(query_vec, top_k)
    if D[0][0] < 1.5:  # 设置相似度阈值
        return knowledge_base[I[0][0]], D[0][0]
    return None, 0.0

# 主页面
tab1, tab2 = st.tabs(["💬 问答助手", "📚 添加知识"])

with tab1:
    st.title("🧠 医疗问答助手")
    st.markdown("请选择一个样例问题，或直接输入您的健康问题，我们将为您提供建议。")

    # 选择样例问题（前几个）
    sample_questions = [item["question"] for item in knowledge_base[:5]]
    sample_questions.insert(0, "🔽 请选择一个样例问题")

    selected = st.selectbox("常见问题：", sample_questions, index=0)
    default_query = "" if selected == "🔽 请选择一个样例问题" else selected

    # 手动输入问题
    query = st.text_input("或者输入其他问题：", value=default_query, key="query_input")

    if st.button("🔍 查询"):
        if not query.strip():
            st.warning("请输入有效问题。")
        else:
            result, score = search_answer(query)
            if result:
                st.success("✅ 找到相关答案：")
                st.markdown(f"**Q：** {result['question']}")
                st.markdown(f"**A：** {result['answer']}")
                if result.get("tags"):
                    st.caption(f"标签：{'、'.join(result['tags'])}")
                st.caption(f"匹配相似度分数：{1.0 - score:.2f}")
            else:
                st.error("❌ 很抱歉，当前知识库中没有相关答案。欢迎录入新知识！")

with tab2:
    st.subheader("📥 添加专家知识")
    new_q = st.text_input("问题：")
    new_a = st.text_area("答案：")
    new_tags = st.text_input("标签（逗号分隔，可选）")

    if st.button("➕ 添加"):
        if new_q and new_a:
            new_entry = {
                "question": new_q.strip(),
                "answer": new_a.strip(),
                "tags": [tag.strip() for tag in new_tags.split(",") if tag.strip()]
            }
            knowledge_base.append(new_entry)
            save_knowledge(knowledge_base)
            index, embeddings = build_index(knowledge_base)
            st.success("✅ 成功添加新知识！")
        else:
            st.warning("请输入完整的问题和答案。")

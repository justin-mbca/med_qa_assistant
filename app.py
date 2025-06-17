import streamlit as st
from utils.qa_engine import load_knowledge, build_index, search_answer
from utils.editor import add_entry

st.set_page_config(page_title="医疗问答原型", layout="centered")
st.title("🩺 医疗智能问答系统（中文原型）")

data = load_knowledge()
questions = [d["question"] for d in data]
index, _ = build_index(questions)

tab1, tab2 = st.tabs(["👩‍⚕️ 病人咨询", "📚 专家知识录入"])

with tab1:
    st.subheader("请输入您的问题或症状")
    query = st.text_input("例如：我最近胸闷, 头晕，是怎么回事？我有咳嗽、发烧，是不是得了新冠？高血压应该注意什么?")

    if st.button("🔍 查询"):
        if query.strip():
            result = search_answer(query, data, index, questions)
            st.markdown(f"**可能相关问题：** {result['question']}")
            st.success(result["answer"])
        else:
            st.warning("请输入一个问题。")

with tab2:
    st.subheader("添加或修改知识条目")
    q = st.text_area("问题")
    a = st.text_area("答案")
    tags = st.text_input("标签（可选，用逗号分隔）")

    if st.button("📥 添加新条目"):
        if q.strip() and a.strip():
            add_entry(q, a, tags.split(","))
            st.success("已添加成功！刷新后生效。")
        else:
            st.error("请输入完整的问题和答案。")
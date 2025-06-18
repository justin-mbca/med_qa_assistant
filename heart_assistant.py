import streamlit as st
from symptom_tree import symptom_flow

st.title("🩺 多步问诊助手")

# 初始选择症状
selected_symptom = st.selectbox("请选择您目前最明显的不适症状：", ["-- 请选择 --"] + list(symptom_flow.keys()))

if selected_symptom != "-- 请选择 --":
    st.markdown(f"### 🔍 关于 **{selected_symptom}** 的进一步问题：")
    
    node = symptom_flow[selected_symptom]
    selections = st.multiselect(node["问诊"], node["选项"], key="symptom_detail")

    if selections:
        st.markdown("### 📋 初步建议：")
        for sel in selections:
            if sel in node["下一步"]:
                st.info(f"🩻 **{sel}：** {node['下一步'][sel]}")
        
        if st.button("📚 想了解其中某个建议的检查内容？"):
            st.markdown("欢迎使用右侧 AI 知识库助手进一步查询。")

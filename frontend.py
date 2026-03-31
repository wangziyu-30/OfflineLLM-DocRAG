import streamlit as st
from rag.chain import rag_chain_with_memory
st.title("RAG 知识库助手")
question = st.text_input("请输入问题")
if st.button("提问"):
    res = rag_chain_with_memory.invoke(
        {"question": question},
        config={"configurable": {"session_id": "user1"}}
    )
    st.write(res)
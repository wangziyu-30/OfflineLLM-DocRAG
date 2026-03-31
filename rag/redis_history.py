import streamlit as st
# 🔥 加这3行：屏蔽所有无用警告（解决无限刷屏）
import warnings
warnings.filterwarnings("ignore")
from transformers.utils import logging
logging.set_verbosity_error()

# 导入你的RAG链
from rag.chain import rag_chain_with_memory

# 页面配置
st.title("RAG智能问答系统")
st.subheader("混合检索+重排序 | 已就绪")

# 输入框
question = st.text_input("输入你的问题：")

# 🔥 核心修复：只有点击按钮才执行！绝不自动运行
if st.button("开始提问"):
    if st.button("提问"):
        try:
            res = rag_chain_with_memory.invoke(
                {"question": question},
                config={"configurable": {"session_id": "user1"}}
            )
            st.write(res)
        except Exception as e:
            # 🔥 直接显示真实错误！你就知道模型为啥没响应
            st.error(f"本地模型调用失败：{str(e)}")
    if not question:
        st.warning("请输入问题！")
    else:
        with st.spinner("检索中..."):
            try:
                # 仅点击时调用
                answer = rag_chain_with_memory.invoke({"question": question})
                st.success("回答完成！")
                st.write(answer)
            except Exception as e:
                st.error(f"错误：{str(e)}")
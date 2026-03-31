# 导入核心依赖
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
# Redis 会话存储
from langchain_community.chat_message_histories import RedisChatMessageHistory
from loguru import logger

# 导入配置 + 向量库管理器
from rag.config import settings
from rag.vector_store import VectorStoreManager

# ===================== Redis 会话持久化（重启不丢记忆） =====================
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6380/0",
        ttl=86400
    )

# ===================== Redis 版 清除会话历史 =====================
def clear_session_history(session_id: str) -> bool:
    """
    重写：清除 Redis 中的会话历史
    彻底解决爆红问题，功能和原来完全一致
    """
    try:
        # 连接Redis并删除对应会话键
        history = RedisChatMessageHistory(
            session_id=session_id,
            url="redis://localhost:6379/0"
        )
        history.clear()
        logger.info(f"✅ Redis会话 {session_id} 已清空")
        return True
    except Exception as e:
        logger.error(f"❌ 清空会话失败：{str(e)}")
        return False

# ===================== 初始化 32B 大模型 =====================
llm = ChatOllama(
    model="qwen:32b",  # ，必须和ollama list一致
    base_url="http://127.0.0.1:11434",
    temperature=0.1,
    timeout=30,      # 👈 加超时，防止无限等待
    streaming=False  # 👈 关闭流式输出，直接返回完整结果
)
# ===================== 少样本 + 引用溯源 Prompt =====================
RAG_PROMPT = """
你是一个严格基于知识库的专业问答助手，**只允许使用检索到的上下文回答**，禁止编造任何信息！

【核心规则】
1. 无相关内容时，必须回答：「抱歉，知识库中未找到相关信息，无法为你解答。」
2. 回答必须简洁准确，**必须标注来源文件 + 引用原文**
3. 输出格式：回答正文 【来源：xxx】【引用：xxx】
4. 结合历史对话，不重复、不跑偏

【少样本示例】
示例1：
问：什么是RAG？
答：RAG是检索增强生成，通过检索知识库辅助大模型生成可靠回答。【来源：AI知识库.txt】【引用：RAG=Retrieval-Augmented Generation】

示例2：
问：16G显卡能跑什么模型？
答：16G显存可流畅运行30B/32B INT4量化大模型，显存占用约13G。【来源：模型部署.txt】【引用：16G显卡适配32B级INT4模型】

【检索上下文】
{context}

请按规则回答用户问题：
"""

# 对话模板
prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# ===================== 文档格式化（溯源） =====================
def format_docs(docs):
    context = ""
    for idx, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知文档")
        content = doc.page_content
        context += f"文档{idx+1} | 来源：{source} | 内容：{content}\n\n"
    return context

# ===================== 最终RAG链 =====================
retriever = VectorStoreManager().get_retriever()

# 基础RAG链
base_rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(retriever.invoke(x["question"]))
    )
    | prompt
    | llm
    | StrOutputParser()
)

# 带 Redis 记忆的最终链
rag_chain_with_memory = RunnableWithMessageHistory(
    runnable=base_rag_chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

logger.info(" RAG链启动完成！Redis会话持久化已启用")

# 测试代码
if __name__ == "__main__":
    result = rag_chain_with_memory.invoke(
        {"question": "test.txt里写了什么？"},
        config={"configurable": {"session_id": "test_user"}}
    )
    print("AI回答：", result)
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from loguru import logger
from rag.config import settings
from rag.vector_store import vector_store_manager

# ========== 1. 全局会话历史存储 ==========
# 用字典存储不同session_id对应的对话历史，生产环境可替换为Redis/MySQL持久化
session_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    根据session_id获取对应的对话历史，实现多会话隔离
    :param session_id: 会话ID，建议用用户ID+随机字符串，保证唯一
    """
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

def clear_session_history(session_id: str) -> bool:
    """清空指定会话的历史记录"""
    if session_id in session_store:
        del session_store[session_id]
        logger.info(f"会话{session_id}的历史记录已清空")
        return True
    return False

# ========== 2. 初始化模型 ==========
llm = ChatOllama(
    model=settings.CHAT_MODEL,    # 本地模型名（从.env读取 qwen:32b）
    temperature=0.3,              # 保持不变
    streaming=True,              # 开启流式输出支持（不变）
    num_gpu=1                    # 强制调用你的 RTX 5070Ti 显卡
)
# ========== 3. 优化后的RAG提示词模板 ==========
RAG_PROMPT = """
你是一个专业的知识库问答助手，必须严格基于以下提供的【检索上下文】回答用户的问题，禁止编造任何不在上下文中的信息。

【核心规则】
1.  如果检索上下文中没有与问题相关的内容，必须直接回答："抱歉，我在知识库中没有找到与该问题相关的内容，无法为你解答。"，绝对禁止使用你自身的知识编造信息。
2.  回答必须准确、简洁、有条理，优先使用上下文中的原文表述，禁止过度引申。
3.  回答结束后，必须标注引用的文档来源，格式为：[来源：文档名称]。
4.  结合对话历史理解用户的问题，保证上下文连贯，不要重复之前已经回答过的内容。

【检索上下文】
{context}
"""

# 对话提示词模板，包含系统提示、对话历史、用户问题
prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_PROMPT),
    MessagesPlaceholder(variable_name="history"),  # 对话历史占位符
    ("human", "{question}")
])

# ========== 4. 文档格式化函数 ==========
def format_docs(docs):
    """把检索到的文档列表格式化为字符串，同时记录来源"""
    context = ""
    for i, doc in enumerate(docs):
        context += f"文档{i+1}（来源：{doc.metadata.get('source', '未知')}）：\n{doc.page_content}\n\n"
    return context

# ========== 5. 构建基础RAG链（LCEL写法） ==========
# 获取检索器
retriever = vector_store_manager.get_retriever()

base_rag_chain = (
    # 并行执行：获取上下文和用户问题
    RunnablePassthrough.assign(
        context=lambda x: format_docs(retriever.invoke(x["question"]))
    )
    # 传入提示词模板
    | prompt
    # 传入大模型生成回答
    | llm
    # 解析输出
    | StrOutputParser()
)

# ========== 6. 带记忆的RAG链（核心最终版） ==========
rag_chain_with_memory = RunnableWithMessageHistory(
    # 要执行的基础链
    runnable=base_rag_chain,
    # 获取会话历史的函数
    get_session_history=get_session_history,
    # 用户输入的消息对应的key
    input_messages_key="question",
    # 对话历史对应的key，和模板中的MessagesPlaceholder一致
    history_messages_key="history"
)
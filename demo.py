import os
from dotenv import load_dotenv
# 替换：用本地 Ollama 模型，替代在线 ChatOpenAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 第一步：加载环境配置，初始化模型
load_dotenv()

# 初始化 本地大模型（调用 5070Ti 显卡）
llm = ChatOllama(
    model="qwen:7b",    # 你用 Ollama 下载的模型名
    temperature=0.3,    # 保持原有参数，严谨回答
    num_gpu=1          # 强制调用你的 RTX 5070Ti 显卡
)

# 初始化嵌入模型，用于把文本转为向量
embeddings = OllamaEmbeddings(
    model=os.getenv("OLLAMA_MODEL")
)

# ====================================================
# ====================================================
# ====================================================

# 第二步：这是加载本地文档，完成分块与索引构建
# 1. 加载本地文档
loader = TextLoader("test.txt", encoding="utf-8")
documents = loader.load()
print(f"加载到的文档数量：{len(documents)}")

# 2. 文档分块：把长文档拆分为小块，提升检索准确率
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # 每个分块的最大长度
    chunk_overlap=20,  # 分块之间的重叠率，避免上下文丢失
    separators=["\n", "。", "！", "？", " ", ""],  # 分隔符，优先按完整句子拆分
)
split_docs = text_splitter.split_documents(documents)
print(f"分块后的文档数量：{len(split_docs)}")

# 3. 构建向量库：把分块后的文本转为向量，存储到FAISS中
vector_store = FAISS.from_documents(split_docs, embeddings)
# 把向量库转为检索器，用于后续召回相关文档
retriever = vector_store.as_retriever(search_kwargs={"k": 2})  # k=2，每次召回最相关的2个分块

# ====================================================
# ====================================================
# ====================================================

#第三步：构建 RAG 链，实现问答
# 1. 定义RAG提示词模板：核心是约束大模型严格基于检索到的上下文回答
rag_prompt = ChatPromptTemplate.from_template("""
你是一个专业的知识库问答助手，必须严格基于以下提供的上下文内容回答用户的问题，禁止编造任何不在上下文中的信息。

【规则】
1. 如果上下文中没有相关内容，直接回答："抱歉，我在知识库中没有找到相关内容，无法解答。"
2. 回答必须准确、简洁，优先使用上下文中的原文表述。

【检索到的上下文】
{context}

【用户问题】
{question}
""")

# 2. 定义文档格式化函数：把检索到的多个文档拼接为一个字符串
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# 3. 构建RAG链（LCEL写法，官方标准）
rag_chain = (
    # 第一步：获取用户问题，并行检索相关文档，格式化上下文
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    # 第二步：把上下文和问题传入提示词模板
    | rag_prompt
    # 第三步：把提示词传入大模型生成回答
    | llm
    # 第四步：解析大模型的输出，提取字符串内容
    | StrOutputParser()
)

# 4. 测试RAG问答
if __name__ == "__main__":
    # 测试问题1：知识库中有的内容
    print("问题1：千问大模型是什么？")
    response1 = rag_chain.invoke("千问大模型是什么？")
    print("回答：", response1)
    print("-" * 50)

    # 测试问题2：知识库中没有的内容
    print("问题2：Python是什么？")
    response2 = rag_chain.invoke("Python是什么？")
    print("回答：", response2)
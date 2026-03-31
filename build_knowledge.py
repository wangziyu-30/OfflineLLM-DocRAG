from rag.vector_store import vector_store_manager
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.config import settings

# 1. 加载你的test.txt文件
loader = TextLoader("test.txt", encoding="utf-8")
documents = loader.load()

# 2. 文档分块（和你之前的配置一致）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
    separators=["\n", "。", "！", "？", " ", ""],
)
split_docs = text_splitter.split_documents(documents)

# 3. 🔥 关键：把分块后的文档 添加到本地向量库（永久保存）
vector_store_manager.add_documents(split_docs)

# 打印验证
print(f"成功向向量库添加 {len(split_docs)} 个文档分块")
print(f"当前向量库总分块数：{vector_store_manager.get_document_count()}")
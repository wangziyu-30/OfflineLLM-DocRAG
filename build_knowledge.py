from rag.vector_store import VectorStoreManager
from langchain_community.document_loaders import TextLoader
# 导入全局配置
from rag.config import get_semantic_text_splitter

# 1. 加载test.txt文件
loader = TextLoader("test.txt", encoding="utf-8")
documents = loader.load()

# 2. 语义分块（传入配置参数）
text_splitter = get_semantic_text_splitter()
split_docs = text_splitter.split_documents(documents)

# 3. 【只创建一个实例】 核心修改
vs_manager = VectorStoreManager()
# 4. 用这个实例添加文档
vs_manager.add_documents(split_docs)

# 打印验证（同一个实例，数量正常显示）
print(f"成功向向量库添加 {len(split_docs)} 个文档分块")
print(f"当前向量库总分块数：{vs_manager.get_document_count()}")
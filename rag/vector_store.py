import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from loguru import logger
from rag.config import settings

class VectorStoreManager:
    """向量库管理器，单例模式，全局唯一实例，避免重复初始化"""
    # 单例实例
    _instance = None
    # 向量库实例（完全不变）
    _vector_store: FAISS | None = None
    # 【修改2】替换嵌入模型类型注解
    _embeddings: OllamaEmbeddings = None

    def __new__(cls, *args, **kwargs):
        """单例模式：保证整个项目只有一个向量库管理器实例"""
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._init_embeddings()
            cls._instance._load_local_vector_store()
        return cls._instance

    def _init_embeddings(self):
        """【修改3】初始化本地Ollama嵌入模型，全局复用一个实例"""
        # 本地模型：无需API Key、无需BaseURL，仅需模型名称
        self._embeddings = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL  # 直接读取.env中的本地模型名
        )
        logger.info(" 本地嵌入模型初始化完成")

    def _load_local_vector_store(self):
        """加载本地已保存的向量库，不存在则初始化空库（完全不变）"""
        if os.path.exists(settings.VECTOR_STORE_PATH) and os.listdir(settings.VECTOR_STORE_PATH):
            try:
                # 加载本地向量库
                self._vector_store = FAISS.load_local(
                    folder_path=settings.VECTOR_STORE_PATH,
                    embeddings=self._embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"本地向量库加载成功，当前文档分块数量：{self._vector_store.index.ntotal}")
            except Exception as e:
                logger.error(f"本地向量库加载失败：{str(e)}，初始化空向量库")
                self._init_empty_vector_store()
        else:
            # 没有本地向量库，初始化空库
            self._init_empty_vector_store()

    def _init_empty_vector_store(self):
        """初始化空的向量库（完全不变）"""
        self._vector_store = FAISS.from_texts(
            texts=["初始化向量库"],
            embedding=self._embeddings
        )
        # 删除初始化的文本
        self._vector_store.delete([self._vector_store.index_to_docstore_id[0]])
        logger.info("空向量库初始化完成")

    def add_documents(self, documents: list[Document]):
        """增量添加文档到向量库，并自动持久化到本地（完全不变）"""
        if not documents:
            logger.warning("无有效文档可添加")
            return
        # 添加文档到向量库
        self._vector_store.add_documents(documents)
        # 保存到本地磁盘，实现持久化
        self._vector_store.save_local(settings.VECTOR_STORE_PATH)
        logger.info(f"文档添加成功，新增分块数：{len(documents)}，当前总分块数：{self._vector_store.index.ntotal}")

    def get_retriever(self):
        """获取检索器，用于RAG链（完全不变）"""
        return self._vector_store.as_retriever(search_kwargs={"k": settings.RETRIEVE_TOP_K})

    def get_document_count(self) -> int:
        """获取当前向量库中的分块数量（完全不变）"""
        return self._vector_store.index.ntotal if self._vector_store else 0

    def clear_vector_store(self):
        """清空向量库，并删除本地文件（完全不变）"""
        self._init_empty_vector_store()
        # 删除本地文件
        if os.path.exists(settings.VECTOR_STORE_PATH):
            for file in os.listdir(settings.VECTOR_STORE_PATH):
                os.remove(os.path.join(settings.VECTOR_STORE_PATH, file))
        self._vector_store.save_local(settings.VECTOR_STORE_PATH)
        logger.info("向量库已全部清空")

# 全局单例实例，整个项目直接导入使用（完全不变）
vector_store_manager = VectorStoreManager()
import os
from typing import List
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from loguru import logger
from rag.config import settings
from langchain_community.retrievers import BM25Retriever
# 只保留安全的依赖，删除所有爆红的LangChain压缩器
from sentence_transformers import CrossEncoder


class VectorStoreManager:
    """向量库管理器：混合检索 + BGE重排序"""
    _instance = None
    _vector_store: FAISS | None = None
    _embeddings: OllamaEmbeddings = None
    _all_documents: List[Document] = []
    _reranker_model = None  # 重排序模型

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._init_embeddings()
            cls._instance._init_reranker()
            cls._instance._load_local_vector_store()
        return cls._instance

    def _init_embeddings(self):
        self._embeddings = OllamaEmbeddings(model=settings.EMBEDDING_MODEL)
        logger.info(" 嵌入模型加载完成")

    def _init_reranker(self):
        """BGE重排序"""
        try:
            self._reranker_model = CrossEncoder("BAAI/bge-reranker-base")
            logger.info("BGE-Reranker 重排序加载完成")
        except Exception as e:
            logger.warning(f"重排序模型加载失败：{str(e)}")
            self._reranker_model = None

    def _load_local_vector_store(self):
        if os.path.exists(settings.VECTOR_STORE_PATH) and os.listdir(settings.VECTOR_STORE_PATH):
            try:
                self._vector_store = FAISS.load_local(
                    folder_path=settings.VECTOR_STORE_PATH,
                    embeddings=self._embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"向量库加载成功，总分块：{self._vector_store.index.ntotal}")
            except:
                self._init_empty_vector_store()
        else:
            self._init_empty_vector_store()

    def _init_empty_vector_store(self):
        self._vector_store = FAISS.from_texts(["init"], self._embeddings)
        self._vector_store.delete([self._vector_store.index_to_docstore_id[0]])
        logger.info("空向量库初始化完成")

    def add_documents(self, documents: List[Document]):
        if not documents:
            return
        self._all_documents.extend(documents)
        self._vector_store.add_documents(documents)
        self._vector_store.save_local(settings.VECTOR_STORE_PATH)
        logger.info(f"文档添加成功：{len(documents)} 块")

    # ===================== 1. 混合检索（向量+BM25） =====================
    def _hybrid_retrieve(self, query: str) -> List[Document]:
        k = settings.RETRIEVE_TOP_K
        # 向量检索
        faiss_docs = self._vector_store.similarity_search(query, k=k)
        # BM25关键词检索
        bm25_retriever = BM25Retriever.from_documents(self._all_documents)
        bm25_docs = bm25_retriever.get_relevant_documents(query, k=k)

        # 加权融合去重
        doc_map = {}
        for i, d in enumerate(faiss_docs):
            doc_map[d.page_content] = (d, 0.6 / (i + 1))
        for i, d in enumerate(bm25_docs):
            score = 0.4 / (i + 1)
            if d.page_content in doc_map:
                doc_map[d.page_content] = (d, doc_map[d.page_content][1] + score)
            else:
                doc_map[d.page_content] = (d, score)
        return [d for d, _ in sorted(doc_map.values(), key=lambda x: x[1], reverse=True)[:k]]

    # ===================== 2.BGE重排序 =====================
    def _rerank_documents(self, query: str, docs: List[Document]) -> List[Document]:
        if not self._reranker_model or not docs:
            return docs

        # 手写重排序逻辑
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self._reranker_model.predict(pairs)

        # 按分数排序
        ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs[:3]]

    # ===================== 最终检索接口：混合检索 + 重排序 =====================
    def get_retriever(self):
        def retrieve(query: str):
            # 1. 混合检索召回
            docs = self._hybrid_retrieve(query)
            # 2. 手写重排序精排
            docs = self._rerank_documents(query, docs)
            return docs

        # 兼容原有调用，无任何报错
        return type('Retriever', (), {'invoke': retrieve})()

    def get_document_count(self):
        return self._vector_store.index.ntotal if self._vector_store else 0

    def clear_vector_store(self):
        self._init_empty_vector_store()
        self._all_documents = []
        self._vector_store.save_local(settings.VECTOR_STORE_PATH)
        logger.info("向量库已清空")


vector_store_manager = VectorStoreManager()
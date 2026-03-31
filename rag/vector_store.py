from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
import os


class VectorStoreManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # 初始化所有属性
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = None
        self.reranker = CrossEncoder("BAAI/bge-reranker-base")
        self.docs = []
        self.save_path = "./vector_store"

    def add_documents(self, docs):
        self.load_documents(docs)
        self.build_vector_store()
    def get_document_count(self):
        # 返回分块数量
        return len(self.docs) if self.docs else 0
    def load_documents(self, docs):
        """加载分块文档"""
        self.docs = docs

    def build_vector_store(self):
        """构建向量库"""
        if not self.docs:
            raise ValueError("未加载文档")
        self.vector_store = FAISS.from_documents(self.docs, self.embeddings)
        self.vector_store.save_local(self.save_path)

    def load_vector_store(self):
        """【自动加载】本地向量库（修复报错核心）"""
        if os.path.exists(self.save_path):
            self.vector_store = FAISS.load_local(
                self.save_path, self.embeddings, allow_dangerous_deserialization=True
            )
            return self.vector_store
        raise Exception("请先运行 build_knowledge.py 构建向量库")

    def get_retriever(self, top_k: int = 5):
        """混合检索 + 重排序（自动加载向量库，无属性报错）"""
        from langchain_core.runnables import RunnableLambda

        def retrieve(query: str, config=None):
            # 🔥 关键：自动加载向量库，不存在就报错
            if self.vector_store is None:
                self.load_vector_store()

            # 向量检索
            vector_docs = self.vector_store.similarity_search(query, k=top_k)
            # BM25检索
            bm25_retriever = BM25Retriever.from_documents(self.docs) if self.docs else []
            bm25_docs = bm25_retriever.get_relevant_documents(query) if self.docs else []

            # 合并去重
            unique_docs = []
            seen = set()
            for doc in vector_docs + bm25_docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    unique_docs.append(doc)

            # 重排序
            if unique_docs and self.reranker:
                scores = self.reranker.predict([(query, d.page_content) for d in unique_docs])
                unique_docs = [x for _, x in sorted(zip(scores, unique_docs), key=lambda x: -x[0])]

            return unique_docs[:top_k]

        return RunnableLambda(retrieve)
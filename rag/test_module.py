from rag.document_processor import document_processor
from rag.vector_store import vector_store_manager

if __name__ == "__main__":
    # 测试：加载本地PDF文件，添加到向量库
    with open("test.pdf", "rb") as f:
        split_docs = document_processor.load_and_split_document(f, "测试文档.pdf")

    # 添加到向量库
    vector_store_manager.add_documents(split_docs)

    # 测试检索
    retriever = vector_store_manager.get_retriever()
    docs = retriever.invoke("测试问题")
    print("检索到的相关内容：", docs)
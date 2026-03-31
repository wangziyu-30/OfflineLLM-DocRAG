from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from loguru import logger
from rag.config import settings
import os
from typing import BinaryIO

class DocumentProcessor:
    """文档处理器，负责文档加载、清洗、分块"""
    def __init__(self):
        # 初始化文本分块器，配置分块参数
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            # 分隔符优先级：先按段落拆分，再按句子拆分，保证语义完整
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
            is_separator_regex=False
        )

    def _get_loader(self, file_path: str, file_ext: str):
        """根据文件后缀选择对应的加载器"""
        if file_ext == ".pdf":
            return PyPDFLoader(file_path)
        elif file_ext == ".txt":
            return TextLoader(file_path, autodetect_encoding=True)
        elif file_ext in [".md", ".markdown"]:
            return UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"不支持的文件格式：{file_ext}")

    def load_and_split_document(self, file_content: BinaryIO, filename: str) -> list[Document]:
        """
        加载文档并完成分块，返回处理后的文档列表
        :param file_content: 上传的文件二进制内容
        :param filename: 文件名，用于判断格式和记录来源
        :return: 分块后的Document对象列表
        """
        # 获取文件后缀
        file_ext = os.path.splitext(filename)[1].lower()
        # 临时文件路径
        temp_path = f"./temp_{filename}"
        try:
            # 把上传的文件写入临时文件
            with open(temp_path, "wb") as f:
                f.write(file_content.read())

            # 加载文档
            loader = self._get_loader(temp_path, file_ext)
            documents = loader.load()
            logger.info(f"文档{filename}加载成功，原始段落数：{len(documents)}")

            # 文档清洗：去除空白内容，添加来源元数据
            for doc in documents:
                # 去除首尾空白
                doc.page_content = doc.page_content.strip()
                # 记录文档来源，后续用于标注引用
                doc.metadata["source"] = filename
            # 过滤掉空内容的文档
            documents = [doc for doc in documents if doc.page_content]

            # 文档分块
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"文档{filename}分块完成，总分块数：{len(split_docs)}")
            return split_docs

        except Exception as e:
            logger.error(f"文档{filename}处理失败：{str(e)}")
            raise Exception(f"文档处理失败：{str(e)}")
        finally:
            # 无论成功失败，都删除临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)

# 全局单例实例
document_processor = DocumentProcessor()
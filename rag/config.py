from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
# 新增导入
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载.env文件
load_dotenv()

class Settings(BaseSettings):
    """全局配置类，适配【本地Ollama大模型】，类型安全"""
    # 本地大模型配置
    CHAT_MODEL: str
    EMBEDDING_MODEL: str

    # 向量库与分块配置
    VECTOR_STORE_PATH: str = "./vector_store"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    RETRIEVE_TOP_K: int = 3

    # 服务配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# 全局单例配置
settings = Settings()

# ===================== 【语义分块函数】 =====================
def get_semantic_text_splitter():
    """
    语义化文本分块器
    优先按段落、完整句子拆分，保证语义完整性
    直接复用全局配置，无需传参，无报错
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        length_function=len,
    )
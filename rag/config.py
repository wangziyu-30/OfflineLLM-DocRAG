from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# 加载.env文件（功能不变）
load_dotenv()

class Settings(BaseSettings):
    """全局配置类，适配【本地Ollama大模型】，类型安全"""
    # 本地聊天大模型（比如 qwen:32b）
    CHAT_MODEL: str
    # 本地向量嵌入模型（和聊天模型用同一个即可）
    EMBEDDING_MODEL: str

    # ===================== 向量库与分块配置（通用配置） =====================
    VECTOR_STORE_PATH: str = "./vector_store"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    RETRIEVE_TOP_K: int = 3

    # ===================== 服务配置 =====================
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # 自动从.env文件加载，忽略额外的环境变量
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# 全局单例配置，整个项目直接导入使用
settings = Settings()
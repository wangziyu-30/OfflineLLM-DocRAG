import redis
from langchain_community.chat_message_histories import RedisChatMessageHistory

redis_client = redis.Redis(host="localhost", port=6379, db=0)

def get_redis_history(session_id: str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379/0"
    )
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
import time

logger.add("logs/app.log", rotation="1 day", level="INFO")

# 全局异常捕获
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"错误：{str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "服务器异常", "msg": str(exc)}
    )

# 接口限流
async def rate_limit_middleware(request: Request, call_next):
    time.sleep(0.1)
    response = await call_next(request)
    return response
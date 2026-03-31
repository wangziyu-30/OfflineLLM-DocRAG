from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from loguru import logger
import json

# 导入我们开发的RAG核心模块
from rag.config import settings
from rag.vector_store import vector_store_manager
from rag.document_processor import document_processor
from rag.chain import rag_chain_with_memory, clear_session_history

# ========== 1. 初始化FastAPI应用 ==========
app = FastAPI(
    title="RAG知识库问答系统",
    description="基于LangChain + FastAPI开发的企业级知识库问答服务",
    version="1.0.0"
)

# ========== 2. 配置CORS跨域 ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境替换为你的前端域名，比如["https://your-frontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 3. 统一响应模型定义 ==========
class BaseResponse(BaseModel):
    """所有接口的统一返回格式"""
    code: int = 200
    message: str = "success"
    data: Optional[dict | list | str | int] = None

# ========== 4. 请求模型定义 ==========
class ChatRequest(BaseModel):
    """问答接口的请求参数"""
    session_id: str  # 会话ID，用于隔离不同用户的对话历史
    question: str    # 用户的问题
    stream: bool = False  # 是否开启流式输出

# ========== 5. 工具类接口 ==========
@app.get("/health", summary="健康检查接口", response_model=BaseResponse, tags=["工具接口"])
async def health_check():
    """检查服务是否正常运行"""
    return BaseResponse(
        data={
            "status": "running",
            "document_count": vector_store_manager.get_document_count()
        }
    )

@app.get("/index-info", summary="获取知识库索引信息", response_model=BaseResponse, tags=["工具接口"])
async def get_index_info():
    """获取当前知识库的文档数量、存储路径等信息"""
    return BaseResponse(
        data={
            "document_count": vector_store_manager.get_document_count(),
            "vector_store_path": settings.VECTOR_STORE_PATH,
            "chunk_size": settings.CHUNK_SIZE,
            "retrieve_top_k": settings.RETRIEVE_TOP_K
        }
    )

@app.delete("/clear-index", summary="清空知识库索引", response_model=BaseResponse, tags=["工具接口"])
async def clear_index():
    """清空知识库中的所有文档和索引"""
    try:
        vector_store_manager.clear_vector_store()
        return BaseResponse(message="知识库清空成功")
    except Exception as e:
        logger.error(f"清空知识库失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"清空失败：{str(e)}")

@app.delete("/clear-session", summary="清空会话历史", response_model=BaseResponse, tags=["会话管理"])
async def clear_session(session_id: str):
    """清空指定会话的对话历史"""
    success = clear_session_history(session_id)
    if success:
        return BaseResponse(message="会话历史清空成功")
    else:
        raise HTTPException(status_code=404, detail="会话ID不存在")

# ========== 6. 核心业务接口 ==========
@app.post("/upload", summary="上传文档并构建知识库", response_model=BaseResponse, tags=["文档管理"])
async def upload_documents(files: List[UploadFile] = File(..., description="支持PDF/TXT/Markdown格式的文档")):
    """
    上传多个文档，系统会自动解析、分块、添加到知识库，增量更新索引
    """
    if not files:
        raise HTTPException(status_code=400, detail="请选择要上传的文件")

    success_count = 0
    failed_files = []
    all_split_docs = []

    for file in files:
        try:
            # 校验文件格式
            file_ext = file.filename.split(".")[-1].lower()
            support_ext = ["pdf", "txt", "md", "markdown"]
            if file_ext not in support_ext:
                failed_files.append(f"{file.filename}（不支持的文件格式）")
                continue

            # 处理文档：加载、分块
            split_docs = document_processor.load_and_split_document(file.file, file.filename)
            all_split_docs.extend(split_docs)
            success_count += 1
        except Exception as e:
            failed_files.append(f"{file.filename}（处理失败：{str(e)}）")
            continue

    # 批量添加到向量库
    if all_split_docs:
        vector_store_manager.add_documents(all_split_docs)

    # 返回处理结果
    return BaseResponse(
        data={
            "success_count": success_count,
            "failed_files": failed_files,
            "total_chunk_count": len(all_split_docs),
            "current_document_count": vector_store_manager.get_document_count()
        }
    )

@app.post("/chat", summary="知识库问答核心接口", tags=["问答服务"])
async def chat(request: ChatRequest):
    """
    知识库问答核心接口，支持同步非流式响应和SSE流式输出
    - session_id: 会话ID，用于隔离不同用户的对话历史，必须保证唯一
    - question: 用户的问题
    - stream: 是否开启流式输出，开启后返回SSE流式响应，前端用EventSource接收
    """
    # 参数校验
    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id不能为空")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 校验知识库是否为空
    if vector_store_manager.get_document_count() == 0:
        raise HTTPException(status_code=400, detail="知识库为空，请先上传文档构建索引")

    try:
        # 流式输出模式
        if request.stream:
            async def event_generator():
                """SSE事件生成器，逐块返回大模型的输出"""
                async for chunk in rag_chain_with_memory.astream(
                    input={"question": request.question},
                    config={"configurable": {"session_id": request.session_id}}
                ):
                    if chunk:
                        # SSE标准格式，前端用EventSource.onmessage接收
                        yield f"data: {json.dumps({'content': chunk})}\n\n"
                # 结束标记，告诉前端回答已完成
                yield f"data: {json.dumps({'content': '', 'is_end': True})}\n\n"

            # 返回流式响应
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )

        # 非流式同步模式
        else:
            response = await rag_chain_with_memory.ainvoke(
                input={"question": request.question},
                config={"configurable": {"session_id": request.session_id}}
            )
            return BaseResponse(data={"answer": response, "session_id": request.session_id})

    except Exception as e:
        logger.error(f"问答接口调用失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"问答失败：{str(e)}")

# ========== 7. 启动服务 ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
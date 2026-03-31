基于本地大模型的私有化离线文档问答系统无需联网、数据不上云，支持多格式文档上传与 RAG 智能检索问答，提供可视化界面 + API 接口双重使用方式。
项目简介
本项目是一款纯本地部署的 RAG 知识库问答工具，基于 Ollama 本地大模型、LangChain 框架与 FAISS 向量库实现。无需依赖云端 AI 服务，支持上传多种办公文档，自动构建私有知识库，实现精准的文档语义检索与智能问答，兼顾数据隐私与使用便捷性。
核心特性
🌐 全程离线运行：基于本地大模型推理，数据不上传任何云端，隐私安全
📄 多格式文档支持：支持 TXT / MD / DOC / DOCX 格式文件上传解析
🚀 双端使用方式：Streamlit 可视化交互界面 + FastAPI 后端 API 接口
🧠 RAG 智能检索：文档分块、向量存储、语义检索，回答更精准
🧹 知识库管理：一键清空知识库，灵活切换 / 重建文档知识库
🪟 Windows 友好适配：适配 Windows 系统，依赖极简、开箱即用
技术栈
大模型推理：Ollama + 本地开源大模型
RAG 框架：LangChain
向量数据库：FAISS-CPU
后端服务：FastAPI + Uvicorn
前端界面：Streamlit
文档解析：python-docx / pywin32
编程语言：Python 3.9+
快速开始
1. 环境依赖安装
将项目根目录下的 requirements.txt 安装完成：
bash
运行
pip install -r requirements.txt
2. 前置准备（必须）
安装 Ollama 并保持后台运行
拉取所需本地大模型（例如：qwen2、llama3、gemma 等）
bash
运行
ollama pull qwen2
3. 启动项目
方式一：启动可视化交互界面（推荐普通用户使用）
bash
运行
streamlit run frontend.py
启动后访问浏览器给出的本地地址，即可上传文档、进行问答。
方式二：启动 FastAPI 后端接口（适合开发者 / 二次开发）
bash
运行
python main.py
接口文档地址：http://127.0.0.1:8000/docs支持：文件上传、智能问答、清空知识库等 API 调用。
项目结构
plaintext
NativeLLM-DocChat/
├── frontend.py          # Streamlit可视化前端界面
├── main.py              # FastAPI后端接口服务
├── rag/                 # 项目核心模块
│   ├── document_processor.py   # 文档处理、分块模块
│   ├── vector_store.py         # 向量库管理、检索模块
│   └── ...
├── test_module.py       # 核心功能测试脚本
├── requirements.txt     # 项目完整依赖
└── README.md            # 项目说明文档
功能使用流程
启动对应服务（前端界面 / 后端接口）
上传本地文档（TXT/MD/DOC/DOCX）
系统自动完成文档解析→分块→向量入库
输入问题，基于本地大模型 + 检索结果生成回答
如需更换文档，可一键清空知识库后重新上传
注意事项
Windows 系统仅支持 faiss-cpu，请勿安装 faiss-gpu，否则无法运行
老版 DOC 格式文档仅支持 Windows 系统解析
首次使用需确保 Ollama 服务正常启动，大模型已拉取完成
知识库文件默认本地存储，无需额外配置数据库

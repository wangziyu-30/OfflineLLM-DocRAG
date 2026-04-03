# NativeLLM-DocChat
基于本地大模型的私有化离线文档问答系统

无需联网、数据不上云，支持多格式文档上传与 RAG 智能检索问答，提供可视化界面 + API 接口双重使用方式。

---

## 项目简介
本项目是一款纯本地部署的 RAG 知识库问答工具，基于 Ollama 本地大模型、LangChain 框架与 FAISS 向量库实现。
无需依赖云端 AI 服务，支持上传多种办公文档，自动构建私有知识库，实现精准的文档语义检索与智能问答，兼顾数据隐私与使用便捷性。

## 核心特性
- 🌐 全程离线运行：基于本地大模型推理，数据不上传任何云端，隐私安全
- 📄 多格式文档支持：支持 TXT / MD / DOC / DOCX 格式文件上传解析
- 🚀 双端使用方式：Streamlit 可视化交互界面 + FastAPI 后端 API 接口
- 🧠 RAG 智能检索：文档分块、向量存储、语义检索，回答更精准
- 🧹 知识库管理：一键清空知识库，灵活切换/重建文档知识库
- 🪟 Windows 友好适配：适配 Windows 系统，依赖极简、开箱即用

## 技术栈
- 大模型推理：Ollama + 本地开源大模型
- RAG 框架：LangChain
- 向量数据库：FAISS-CPU
- 后端服务：FastAPI + Uvicorn
- 前端界面：Streamlit
- 文档解析：python-docx / pywin32
- 编程语言：Python 3.10+

### 快速开始

### 1. 环境依赖安装
将项目根目录下的 `requirements.txt` 安装完成：
### 2. 前置准备（必须）
1. 安装 Ollama 并保持后台运行
2. 拉取所需本地大模型（例如：qwen2、llama3、gemma 等）
### 3. 项目结构
```text
NativeLLM-DocChat/
├── frontend.py          # Streamlit 可视化前端界面
├── main.py              # FastAPI 后端接口服务
├── rag/                 # 项目核心模块
│   ├── document_processor.py  # 文档处理、分块模块
│   ├── vector_store.py        # 向量库管理、检索模块
│   └── ...
├── test_module.py       # 核心功能测试脚本
├── requirements.txt     # 项目完整依赖
└── README.md            # 项目说明文档

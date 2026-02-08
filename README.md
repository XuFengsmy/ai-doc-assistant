# 🎓 垂直领域 AI 智能助手 (RAG)

这是一个基于 RAG (检索增强生成) 技术的 AI 文档助手。用户可以上传 PDF 文档，AI 会自动学习文档内容，并提供精准的问答服务，同时支持原文溯源。

## 🛠️ 技术栈
- **前端**: Streamlit (左右分栏布局 + PDF 预览)
- **核心框架**: LangChain
- **向量数据库**: ChromaDB
- **大模型 API**: DeepSeek-V3 (via SiliconFlow)
- **Embedding**: BAAI/bge-m3

## ✨ 主要功能
1. **多模态上传**: 支持用户上传任意 PDF 文件。
2. **智能索引**: 自动切分文本并向量化存储。
3. **精准问答**: 基于文档内容回答，杜绝幻觉。
4. **思维链展示**: 显示 AI 参考的原文片段及页码。

## 🚀 如何运行
1. 克隆项目到本地
2. 安装依赖: `pip install -r requirements.txt`
3. 运行: `streamlit run app.py`

---
*Created by 孙蒙源*

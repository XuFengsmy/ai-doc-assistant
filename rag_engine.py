import os
import uuid
import streamlit as st  # 必须导入 streamlit
from openai import AuthenticationError, PermissionDeniedError
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ================= 配置区 =================
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "deepseek-ai/DeepSeek-V3"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"


def clean_config_value(value):
    return str(value).strip().strip('"').strip("'")


def clean_api_key(value):
    value = clean_config_value(value)
    if value.lower().startswith("bearer "):
        value = value[7:].strip()
    return value


def read_config(*names, default=None, secret=False):
    for name in names:
        value = os.getenv(name)
        if value:
            return clean_api_key(value) if secret else clean_config_value(value)

    try:
        for name in names:
            value = st.secrets.get(name)
            if value:
                return clean_api_key(value) if secret else clean_config_value(value)
    except Exception:
        pass

    return default


def provider_auth_error(exc):
    detail = str(exc).replace("\n", " ").strip()
    return (
        "SiliconFlow 请求被拒绝。请检查：1. Secrets 里的 SILICON_API_KEY 是否是硅基流动 "
        "API Key，且不要包含 Bearer 前缀；2. 账号余额/额度是否可用；3. 账号是否有当前模型权限；"
        "4. 是否开启了 IP 白名单。服务商返回："
        f"{detail}"
    )


class RAGPro:
    def __init__(self):
        base_url = read_config(
            "SILICON_BASE_URL",
            "SILICONFLOW_BASE_URL",
            "BASE_URL",
            default=DEFAULT_BASE_URL,
        )
        api_key = read_config(
            "SILICON_API_KEY",
            "SILICONFLOW_API_KEY",
            "API_KEY",
            secret=True,
        )

        if not api_key:
            raise ValueError(
                "缺少 SiliconFlow API Key。请在 Streamlit Cloud 的 Secrets 中配置 "
                "SILICON_API_KEY。"
            )

        # -------------------------------------------------------
        # 👇 初始化模型 👇
        # -------------------------------------------------------
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=api_key,       # 使用刚才定义的变量
            openai_api_base=base_url,     # 使用刚才定义的变量
            check_embedding_ctx_length=False,
            chunk_size=50
        )
        
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=api_key,       # 使用刚才定义的变量
            openai_api_base=base_url,     # 使用刚才定义的变量
            temperature=0.1
        )
        
        self.vector_store = None
        self.config_summary = f"Base URL: {base_url}；API Key: 已读取"

    def load_and_index(self, pdf_path):
        print(f"📚 正在处理文件: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        splits = [
            doc for doc in text_splitter.split_documents(docs)
            if doc.page_content.strip()
        ]

        if not splits:
            raise ValueError("PDF 中没有可检索的文字内容，请上传文字版 PDF。")

        # Chroma only accepts simple scalar metadata values.
        # PyPDFLoader can include extra PDF metadata that is not needed for Q&A.
        for doc in splits:
            page = doc.metadata.get("page", 0)
            doc.metadata = {
                "page": int(page) if isinstance(page, int) or str(page).isdigit() else 0,
                "source": os.path.basename(str(doc.metadata.get("source", pdf_path))),
            }

        collection_name = f"doc_{uuid.uuid4().hex}"
        try:
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                collection_name=collection_name,
            )
        except (AuthenticationError, PermissionDeniedError) as exc:
            raise RuntimeError(provider_auth_error(exc)) from exc
        print("✅ 知识库构建完成！")

    def query(self, question):
        if not self.vector_store:
            raise ValueError("请先上传 PDF 并等待知识库构建完成。")
        
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(question)

        context_str = "\n\n".join([doc.page_content for doc in relevant_docs])
        source_pages = sorted(list(set([doc.metadata.get('page', 0) + 1 for doc in relevant_docs])))
        
        template = """
        你是一个严谨的文档助手。请根据下面的【参考资料】回答问题。
        规则：1. 必须完全基于参考资料回答。2. 资料里没有就说不知道。
        【参考资料】：{context}
        【问题】：{question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": lambda x: context_str, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        try:
            answer = chain.invoke(question)
        except (AuthenticationError, PermissionDeniedError) as exc:
            raise RuntimeError(provider_auth_error(exc)) from exc

        return {
            "answer": answer,
            "sources": source_pages,
            "context": context_str
        }


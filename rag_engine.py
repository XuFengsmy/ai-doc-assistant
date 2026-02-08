import os
import streamlit as st  # 必须导入 streamlit
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ================= 配置区 =================
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "deepseek-ai/DeepSeek-V3"

class RAGPro:
    def __init__(self):
        # -------------------------------------------------------
        # 🛡️ 安全区：在函数内部定义变量，防止 NameError
        # -------------------------------------------------------
        
        # 1. 定义 Base URL (直接写死在这里，绝对不会找不到)
        base_url = "https://api.siliconflow.cn/v1"

        # 2. 获取 API Key
        # 优先读取 Streamlit Secrets，如果没有就用空字符串占位
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
        elif "SILICON_API_KEY" in st.secrets:
            api_key = st.secrets["SILICON_API_KEY"]
        else:
            api_key = "key_not_found"

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
        
        self.db_path = "./chroma_db_pro"
        self.vector_store = None

    def load_and_index(self, pdf_path):
        print(f"📚 正在处理文件: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(docs)

        # 尝试清理旧数据库 (防止权限错误)
        if os.path.exists(self.db_path):
            try:
                import shutil
                shutil.rmtree(self.db_path)
            except:
                pass 
        
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        print("✅ 知识库构建完成！")

    def query(self, question):
        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory=self.db_path, 
                embedding_function=self.embeddings
            )
        
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
        
        answer = chain.invoke(question)

        return {
            "answer": answer,
            "sources": source_pages,
            "context": context_str
        }

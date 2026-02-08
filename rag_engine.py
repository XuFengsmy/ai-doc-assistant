import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # 升级使用新版库
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ================= 配置区 =================
# 填入你的 Key
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
elif "SILICON_API_KEY" in st.secrets:
    api_key = st.secrets["SILICON_API_KEY"]
else:
    api_key = "local_test_key" # 本地测试用
SILICON_BASE_URL = "https://api.siliconflow.cn/v1"

EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "deepseek-ai/DeepSeek-V3"


class RAGPro:
    def __init__(self):
        # 1. 初始化 Embedding 模型 (记得加 chunk_size 防止报错)
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=api_key,
            openai_api_base=base_url,
            check_embedding_ctx_length=False,
            chunk_size=50  # 关键修正
        )

        # 2. 初始化 LLM
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.1
        )

        self.db_path = "./chroma_db_pro"
        self.vector_store = None

    def load_and_index(self, pdf_path):
        """加载PDF -> 切分 -> 向量化 -> 存入数据库"""
        print(f"📚 正在处理文件: {pdf_path}")

        # 加载
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        print(f"📄 共加载 {len(docs)} 页")

        # 切分
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # 加大一点，保证语义完整
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(docs)
        print(f"✂️ 切分为 {len(splits)} 个片段")

        # 入库 (强制刷新数据库)
        if os.path.exists(self.db_path):
            try:
                import shutil
                shutil.rmtree(self.db_path) # 删除旧库
            except:
                pass

        print("💾 正在建立向量索引 (可能需要几十秒)...")
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        print("✅ 知识库构建完成！")

    def query(self, question):
        """核心问答逻辑：返回答案 + 来源页码"""
        if not self.vector_store:
            # 如果内存里没有，就尝试从硬盘加载
            self.vector_store = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )

        # 1. 检索 (Retrieval)
        # k=3 表示找最相关的3个片段
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(question)

        # 2. 构建上下文 (Context)
        # 我们不仅要拼合文字，还要提取页码
        context_str = "\n\n".join([doc.page_content for doc in relevant_docs])

        # 提取来源信息 (去重)
        # metadata['page'] 是从0开始的，所以要+1
        source_pages = sorted(list(set([doc.metadata.get('page', 0) + 1 for doc in relevant_docs])))

        # 3. Prompt
        template = """
        你是一个严谨的文档助手。请根据下面的【参考资料】回答问题。

        规则：
        1. 必须完全基于参考资料回答。
        2. 如果资料里没有提到的，请直接说“文档中未找到相关内容”。
        3. 保持回答简洁明了。

        【参考资料】：
        {context}

        【问题】：
        {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        # 4. 生成回答
        chain = (
                {"context": lambda x: context_str, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        answer = chain.invoke(question)

        # 5. 返回结构化结果 (答案 + 来源)
        return {
            "answer": answer,
            "sources": source_pages,
            "context": context_str
        }


# ================= 测试代码 =================
if __name__ == "__main__":
    # 第一次运行请取消注释下面这行来构建库
    # 假设你放了一个 handbook.pdf 在 data 文件夹
    bot = RAGPro()
    bot.load_and_index("./data/handbook.pdf")

    # 测试问答
    q = "旷课会有什么后果?"  # 请根据你的 PDF 内容提问
    print(f"\n❓ 问题: {q}")

    try:
        result = bot.query(q)
        print(f"🤖 回答: {result['answer']}")
        print(f"📖 来源页码: 第 {result['sources']} 页")
    except Exception as e:

        print(f"❌ 报错: {e}")



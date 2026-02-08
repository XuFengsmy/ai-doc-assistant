import streamlit as st
import os
from streamlit_pdf_viewer import pdf_viewer  # 昨天装好的神器
from rag_engine import RAGPro

# ================= 1. 页面配置 =================
st.set_page_config(
    page_title="万能 AI 文档助手",
    page_icon="📂",
    layout="wide"
)

st.title("📂 毕业设计：万能 AI 文档助手")


# ================= 2. 辅助函数 =================
@st.cache_resource
def load_bot():
    return RAGPro()


def save_uploaded_file(uploaded_file):
    """把用户上传的内存文件，保存到磁盘上，方便 RAG 读取"""
    # 确保 data 目录存在
    if not os.path.exists("./data"):
        os.makedirs("./data")

    # 保存路径：为了简单，我们统一保存为 uploaded_temp.pdf
    # 这样新文件会自动覆盖旧文件，不占空间
    file_path = os.path.join("./data", "uploaded_temp.pdf")

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


# ================= 3. 侧边栏：上传区 =================
with st.sidebar:
    st.header("📤 上传文档")
    st.caption("请上传 PDF 文件，AI 将自动学习内容。")

    # Streamlit 的文件上传组件
    uploaded_file = st.file_uploader("选择文件", type=["pdf"])

    # 逻辑判断：如果有文件被上传
    if uploaded_file is not None:
        st.success(f"文件名: {uploaded_file.name}")

        # 1. 保存文件到本地
        saved_path = save_uploaded_file(uploaded_file)

        # 2. 检查是否需要重建知识库
        # 我们用 session_state 记录上一次处理的文件名
        # 如果当前上传的文件名 != 上一次的文件名，说明换新书了，需要重新学习
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            with st.spinner("🤖 AI 正在阅读并消化文档 (RAG处理中)..."):
                # 调用后端引擎
                bot = load_bot()
                bot.load_and_index(saved_path)

                # 更新状态
                st.session_state.last_uploaded_file = uploaded_file.name
                st.session_state.bot_ready = True
                st.toast("✅ 知识库构建完成！可以开始提问了。", icon="🎉")
    else:
        # 如果没上传文件，提示用户
        st.session_state.bot_ready = False

     # 在 Sidebar 上传文件的下面，加一个清空按钮
    if st.button("🗑️ 清空聊天记录"):
        st.session_state.messages = []  # 清空列表
        st.rerun()  # 刷新页面

# ================= 4. 主界面：左右分栏 =================
if not st.session_state.get("bot_ready"):
    # 欢迎界面
    st.info("👈 请在左侧上传一个 PDF 文件开始使用！")
    st.markdown("""
    ### ✨ 功能介绍
    1. **上传**：支持任意 PDF 文档（建议传文字版，非扫描件）。
    2. **阅读**：左侧窗口原样展示文档。
    3. **问答**：右侧 AI 智能回答，并标注**页码来源**。
    """)

else:
    # 只有当 bot_ready 为 True 时，才显示主界面
    col1, col2 = st.columns([1.2, 1])  # 左边稍微宽一点

    # --- 左侧：PDF 预览 ---
    with col1:
        st.subheader("📄 文档原文")
        # 读取刚才保存的临时文件
        pdf_viewer("./data/uploaded_temp.pdf", height=800)

    # --- 右侧：聊天界面 ---
    with col2:
        st.subheader("🤖 AI 问答")

        # 初始化聊天历史
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "我已经读完了这份文档，您想问什么？"}]

        # 当切换文件时，清空聊天记录 (可选体验优化)
        # 这里为了简单，我们暂不清空，你可以尝试自己加逻辑

        chat_container = st.container(height=650)
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        if prompt := st.chat_input("请输入问题..."):
            # 显示用户问题
            with chat_container:
                with st.chat_message("user"):
                    st.write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # AI 回答
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("🔍 正在检索文档并生成回答..."):
                        try:
                            bot = load_bot()
                            response = bot.query(prompt)

                            # 1. 展示核心回答
                            answer_text = response['answer']
                            sources_text = f"\n\n---\n**📖 来源页码：第 {response['sources']} 页**"
                            full_response = answer_text + sources_text

                            st.markdown(full_response)

                            # 2. 【新增】展示思维链/引用来源 (折叠状态)
                            # 这就叫 "White Box" (白盒) AI，让用户看到证据
                            with st.expander("🕵️ 查看 AI 参考的原文片段 (思维过程)"):
                                st.info("以下是 AI 从文档中检索到的原始素材，AI 根据这些内容生成了回答：")
                                st.text(response['context'])  # 使用 st.text 显示纯文本，防止格式乱掉

                            # 存入历史 (注意：历史记录里通常只存回答，不存那个折叠框，保持整洁)
                            st.session_state.messages.append({"role": "assistant", "content": full_response})

                        except Exception as e:
                            st.error(f"出错: {e}")
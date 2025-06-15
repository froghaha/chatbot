import os, json, base64
from dotenv import load_dotenv
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain.docstore.document import Document




openai_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
USER_CREDENTIALS = json.loads(
    st.secrets.get("USER_CREDENTIALS", os.getenv("USER_CREDENTIALS")))
    # Authentication
def authenticate(username, password):
    return USER_CREDENTIALS.get(username) == password

def login_form():
    st.title("🔐 請先登入")
    with st.form("login"):
        username = st.text_input("使用者名稱")
        password = st.text_input("密碼", type="password")
        if st.form_submit_button("登入"):
            if authenticate(username, password):
                st.session_state.authenticated = True
                st.success("✅ 登入成功")
                st.rerun()
            else:
                st.error("❌ 使用者名稱或密碼錯誤")

if not st.session_state.get("authenticated", False):
    login_form()
    st.stop()

# Page config
st.set_page_config(page_title="Intern Q&A ChatBot", page_icon="🤖")
st.title("🤖 Intern Q&A ChatBot")
st.caption("請直接輸入你的問題，我會根據 TXT 和 PDF 知識庫回答（含圖片）")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query" not in st.session_state:
    st.session_state.query = ""
if "clear_input_flag" not in st.session_state:
    st.session_state.clear_input_flag = False
if st.session_state.clear_input_flag:
    st.session_state.query = ""
    st.session_state.clear_input_flag = False

# Load documents
all_docs = []
text_loader = TextLoader("knowledge/Intern Q & A.txt", encoding='utf-8')
all_docs.extend(text_loader.load())

pdf_files = ["knowledge/Intern Guide.pdf"]
for file in pdf_files:
    if os.path.exists(file):
        all_docs.extend(PyMuPDFLoader(file).load())

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(all_docs)

# Vector store
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embedding)

# Retrieval QA chain
model_name = st.selectbox("選擇模型", ["gpt-3.5-turbo", "gpt-4"])
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, model_name=model_name, openai_api_key=openai_key),
    retriever=vectorstore.as_retriever()
)

# Input
st.text_input("💬 請輸入你的問題：", key="query")
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("📤 提交問題"):
        if st.session_state.query:
            with st.spinner("思考中..."):
                answer = qa.run(st.session_state.query)
            st.success(answer)
            st.session_state.chat_history.append({
                "question": st.session_state.query,
                "answer": answer
            })
            # Optional: show image if query or answer matches known keywords
            keywords_with_images = {
                "reset password": "images/reset_password.png",
                "change password": "images/reset_password.png",
                "wifi": "images/wifi_settings.png",
                "odbc settings": "images/odbc_library.png"
            }
            for keyword, image_path in keywords_with_images.items():
                if keyword in st.session_state.query.lower() or keyword in answer.lower():
                    if os.path.exists(image_path):
                        st.image(image_path, caption=f"相關畫面：{keyword}")


with col2:
    if st.button("➡️ 下一題"):
        st.session_state.clear_input_flag = True
        st.rerun()

# Chat history display
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("📝 聊天紀錄")
    for i, qa_pair in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**{i}. 問題：** {qa_pair['question']}")
        st.markdown(f"**答覆：** {qa_pair['answer']}")

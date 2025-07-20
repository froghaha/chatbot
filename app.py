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

# Safely load secrets from env or Streamlit secrets
def get_secret(key):
    if key in os.environ:
        return os.environ[key]
    elif key in st.secrets:
        return st.secrets[key]
    else:
        raise ValueError(f"Missing secret: {key}")

openai_key = get_secret("OPENAI_API_KEY")
#USER_CREDENTIALS = json.loads(get_secret("USER_CREDENTIALS"))

# Azure AD SSO Authentication
import streamlit.web.server.websocket_headers as _wh  # <-- NEW import

def get_azure_ad_user():
    try:
        headers = st.experimental_get_query_params()  # For Streamlit <1.30 fallback
        # NEW: use st.request.headers (preferred)
        if hasattr(st, "request") and hasattr(st.request, "headers"):
            principal_header = st.request.headers.get("X-MS-CLIENT-PRINCIPAL")
        else:
            import streamlit.web.server.websocket_headers as _wh
            principal_header = _wh._get_websocket_headers().get("X-MS-CLIENT-PRINCIPAL")

        if not principal_header:
            return None
        decoded = base64.b64decode(principal_header)
        principal = json.loads(decoded)
        return principal.get("userDetails")
    except Exception as e:
        return None

user = get_azure_ad_user()

if not user:
    st.warning("🔐 Azure AD login not detected. You're seeing fallback access (for debugging).")
    # st.stop()  # Uncomment this to enforce Azure AD login
else:
    st.success(f"✅ Logged in as: {user}")

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

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
import streamlit as st
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings
from langchain.vectorstores import FAISS
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI #구버전 streamlit 배포용
from langchain.embeddings import OpenAIEmbeddings #구버전 streamlit 배포용
from tenacity import RetryError
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
import os

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

if "memories" not in st.session_state:
        st.session_state["memories"] = {}

@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    
    os.makedirs("./.cache/files", exist_ok=True)
    
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriver = vectorstore.as_retriever()
    return retriver

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)
        


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        Answer the question using ONLY the following context. If you don't know the answer just say you don't konw. DON'T make anything up.
        
        Context: {context}
     """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


st.title("Document GPT")

st.markdown(
    """
            Welcome~
            
            User this chatbot to ask questions to an AI about your files!
            
            Upload your files on the sidebar.
            """
)

with st.sidebar:
    openai_api_key = st.text_input("Input your OpenAI API Key")
    
    if not openai_api_key:
        st.error("Please input your OpenAI API Key on the sidebar")
    else:
        file = st.file_uploader(
            "Upload a.txt .pdf or .docx file",
            type=["pdf", "txt", "docx"],
        )
        st.markdown("[Github](https://github.com/greenagu/langchain-gpt)")
        st.markdown("[Streamlit App](https://langchain-gpt-z73xrdmpkhfwdpyxnk269w.streamlit.app/)")


    if file:
        file_id = file.name
        retriever = embed_file(file)
        send_message("I'm ready! Ask away!", "ai", save=False)
        paint_history()
        message = st.chat_input("Ask anything about your file....")
        
        if file_id not in st.session_state["memories"]:
            st.session_state["memories"][file_id] = ConversationBufferMemory(
                return_messages=True,
                input_key="question",
                output_key="output",
        )
            
            
        llm = ChatOpenAI(
            temperature=0.1,
            streaming=True,
            callbacks=[
                ChatCallbackHandler(),
            ],
            openai_api_key=openai_api_key,
        )


        memory = st.session_state["memories"][file_id]        
            
        def load_memory(_):
            return memory.load_memory_variables({})["history"]

        if message:
            send_message(message, "human")
            chain = ({
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
                "history": load_memory,
            } | prompt | llm)
            with st.chat_message("ai"):
                result = chain.invoke(message)
    else:
        st.session_state["messages"] = []

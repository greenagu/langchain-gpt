from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

CLOUDFLARE_SITE_MAP="https://developers.cloudflare.com/sitemap-0.xml"

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

if "memories" not in st.session_state:
        st.session_state["memories"] = {}

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.

    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}

    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/ai-gateway\/).*",  
            r"^(.*\/vectorize\/).*",  
            r"^(.*\/workers-ai\/).*",  
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


similarity_prompt = ChatPromptTemplate.from_template("""
These are questions that users have asked before:

{previous_questions}

The user's current question:
“{new_question}”

If this new question is semantically similar or identical to any of the above questions, please point us to the previous thread. Otherwise, answer with a blank
""")

def find_similar_question_via_llm(query, previous_questions):
    previous_formatted = "\n".join(
        [f"{question}" for question in previous_questions]
    )
    prompt_input = {
        "previous_questions": previous_formatted,
        "new_question": query
    }
    prompt_chain = similarity_prompt | llm
    response = prompt_chain.invoke(prompt_input).content
    # return response
    if response in previous_questions:
        return response
    return None

def get_site_contents(url):
    retriever = load_website(url)
            
    chain = (
        {
            "docs": retriever,
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(get_answers)
        | RunnableLambda(choose_answer)
    )
    result = chain.invoke(input=query)
    st.session_state["memories"][query] = result.content
    st.markdown(result.content.replace("$", "\$"))

st.markdown(
    """
    # SiteGPT
            
    I have a question about information about the content below from Clouflare.
    - AI Gateway
    - Cloudflare Vectorize
    - Workers AI
            
    Enter your question
"""
)

with st.sidebar:
    openai_api_key = st.text_input("Input your OpenAI API Key")
    st.markdown("---")
    st.markdown("[Github](https://github.com/greenagu/langchain-gpt)")
    st.markdown("[Streamlit App](https://langchain-gpt-z73xrdmpkhfwdpyxnk269w.streamlit.app/)")


if not openai_api_key:
    st.error("Please input your OpenAI API Key on the sidebar")
else:
    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[
            StreamingStdOutCallbackHandler(),
        ],
        openai_api_key=openai_api_key,
    )

    query = st.text_input("Ask a question to the website.")
    
    if query:
        memory_keys = list(st.session_state["memories"].keys())
        if memory_keys:
            similar_question = find_similar_question_via_llm(query, memory_keys)
            if similar_question:
                cached = st.session_state["memories"][similar_question.strip()]
                st.markdown(cached.replace('$', '\$'))
                st.write("cached data!!!!!🎃")
            else:
                get_site_contents(CLOUDFLARE_SITE_MAP)
        else:
                get_site_contents(CLOUDFLARE_SITE_MAP)
# 이전 과제에서 만든 에이전트를 OpenAI 어시스턴트로 리팩터링합니다.
# 대화 기록을 표시하는 Streamlit 을 사용하여 유저 인터페이스를 제공하세요.
# 유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
# st.sidebar를 사용하여 Streamlit app 의 코드과 함께 깃허브 리포지토리에 링크를 넣습니다.
from email import message
import os
import streamlit as st
from typing import Any, Type
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from langchain.utilities import WikipediaAPIWrapper
from langchain.schema import SystemMessage
from langchain.document_loaders import WebBaseLoader
from langchain.tools import DuckDuckGoSearchResults
from langchain.tools import WikipediaQueryRun
import openai as client
from typing_extensions import override
from openai import AssistantEventHandler
import json

llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")

class EventHandler(AssistantEventHandler):

    message = ""

    @override
    def on_text_created(self, text) -> None:
        self.message_box = st.empty()

    def on_text_delta(self, delta, snapshot):
        self.message += delta.value
        self.message_box.markdown(self.message.replace("$", "\$"))

    def on_event(self, event):
        if event.event == "thread.run.requires_action":
            submit_tool_outputs(event.data.id, event.data.thread_id)


st.set_page_config(
    page_title="AssistantGPT",
    page_icon="🥸",
)

st.markdown(
    """
    # AssistantGPT
            
    Welcome to AssistantGPT.
            
    Enter your search query and the agent will research it for you.
    The results are output to a file.
    
    *First enter the api_key found in the sidebar
"""
)

#####
def get_data_from_duckduckgo(inputs):
    print("🐥 DuckDuckGo 함수 실행 시작")
    ddg = DuckDuckGoSearchResults()
    query = inputs["query"]
    result = ddg.run(f"{query}")
    print("✅ DuckDuckGo 결과:", result[:200])  # 너무 길면 자르기
    return result

def get_data_from_wikipedia(inputs):
    print("📚 Wikipedia 함수 실행 시작")
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    query = inputs["query"]
    result = wiki.run(query)
    print("✅ Wikipedia 결과:", result[:200])
    return result

def scrape_web_page(inputs):
    url = inputs["url"]
    loader = WebBaseLoader([url])
    docs = loader.load()
    text = "\n\n".join([doc.page_content for doc in docs])
    return text 

functions_map = {
    "get_data_from_duckduckgo": get_data_from_duckduckgo,
    "get_data_from_wikipedia": get_data_from_wikipedia,
    "scrape_web_page": scrape_web_page,
}
functions = [
    {
        "type": "function",
        "function": {
            "name": "get_data_from_duckduckgo",
            "description": "Use DuckDuckGo to gather comprehensive and accurate information about the queries provided",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Queries to search for",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_data_from_wikipedia",
            "description": "Use Wikipedia to gather comprehensive and accurate information about the queries provided",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Queries to search for",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_web_page",
            "description": "Scrape content from a website using the given URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Web page URL",
                    }
                },
                "required": ["url"],
            },
        },
    },
]
#######

def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=content
    )


def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id).data
    # messages = list(messages)
    messages.reverse()
    return messages


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs


def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    with client.beta.threads.runs.submit_tool_outputs_stream(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs,
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()


def paint_history(thread_id):
    messages = get_messages(thread_id)
    for message in messages:
        insert_message(
            message.content[0].text.value,
            message.role,
        )
        
def insert_message(message, role):
    with st.chat_message(role):
        st.markdown(message)
#####

with st.sidebar:
    openai_api_key = st.text_input("Input your OpenAI API Key")
    st.markdown("---")
    st.markdown("[Github](https://github.com/greenagu/langchain-gpt)")
    st.markdown("[Streamlit App](https://langchain-gpt-z73xrdmpkhfwdpyxnk269w.streamlit.app/)")

if not openai_api_key:
    st.error("Please input your OpenAI API Key on the sidebar")
else:
    ASSISTANT_NAME = "Assistant for Final assignment"
    if "assistant" not in st.session_state:
        assistants = client.beta.assistants.list(limit=10)
        for a in assistants:
            if a.name == ASSISTANT_NAME:
                assistant = client.beta.assistants.retrieve(a.id)
                break
        else:
            assistant = client.beta.assistants.create(
                name=ASSISTANT_NAME,
                instructions="""
                You are a research expert.

                Your task is to use Wikipedia or DuckDuckGo to gather comprehensive and accurate information about the query provided. 

                When you find a relevant website through DuckDuckGo, you must scrape the content from that website. Use this scraped content to thoroughly research and formulate a detailed answer to the question. 

                Combine information from Wikipedia, DuckDuckGo searches, and any relevant websites you find. Ensure that the final answer is well-organized and detailed, and include citations with links (URLs) for all sources used.

                Your research should be saved to a .txt file, and the content should match the detailed findings provided. Make sure to include all sources and relevant information.

                The information from Wikipedia must be included.
                """,
                model="gpt-4o-mini",
                tools=functions,
            )
        thread = client.beta.threads.create()
        st.session_state["assistant"] = assistant
        st.session_state["thread"] = thread
    else:
        assistant = st.session_state["assistant"]
        thread = st.session_state["thread"]

    paint_history(thread.id)
    content = st.chat_input("What do you want to search?")
    if content:
        send_message(thread.id, content)
        insert_message(content, "user")
        
        handler = EventHandler()

        with st.chat_message("assistant"):
            with client.beta.threads.runs.stream(
                thread_id=thread.id,
                assistant_id=assistant.id,
                event_handler=handler,
            ) as stream:
                stream.until_done()

        if st.checkbox("결과를 result.txt 파일로 저장할까요?"):
            with open("./result.txt", "w", encoding="utf-8") as f:
                f.write(handler.message)  # ✅ 이 때 참조 가능!
            st.success("✅ result.txt 파일로 저장되었습니다.")



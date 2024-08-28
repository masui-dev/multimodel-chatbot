import os
import fitz  # PyMuPDF
import shutil
import chainlit as cl
from chainlit.input_widget import Slider,TextInput,Select

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,

)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

def make_llamaindex_db(session_id):
    try:
        source_folder = os.path.join('.files', session_id )
        destination_folder = os.path.join('data-pdf', session_id)
        shutil.move(source_folder, destination_folder)
        reader = SimpleDirectoryReader(input_dir=destination_folder)
        documents = reader.load_data()
        index = VectorStoreIndex.from_documents(documents)
        Settings.llm = OpenAI(
            model="gpt-4o-2024-08-06", temperature=0.0, streaming=True
        )
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
        Settings.context_window = 32768
        query_engine = index.as_query_engine(streaming=True, similarity_top_k=10)
        cl.user_session.set("query_engine", query_engine)
        return True
    except:
        return False

def set_runnable(model_name=None):
    system_message = cl.user_session.get("settings")["SystemMessage"]
    temperature = cl.user_session.get("settings")["Temperature"]
    if model_name == None:
        model_name = cl.user_session.get("settings")["Model_name"]
    if model_name == "GPT-4o":
        model = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"],
                            streaming = True,
                            model="gpt-4o-2024-08-06",
                            temperature=temperature)
        

    elif model_name == "Claude 3.5 Sonnet":
        model = ChatAnthropic(temperature=temperature, model_name="claude-3-5-sonnet-20240620")
    elif model_name == "Gemini 1.5 Pro":
        model = ChatGoogleGenerativeAI(api_key=os.environ["GOOGLE_API_KEY"],model="gemini-1.5-pro-latest", temperature=temperature)
    else:
        pass
    if model_name == "GPT-4o" or model_name == "Claude 3.5 Sonnet":
        prompt = ChatPromptTemplate.from_messages(
                [
                ("system",system_message),
                MessagesPlaceholder(variable_name="message")
                ]
            )
    else:
        prompt = ChatPromptTemplate.from_messages(
                [
                MessagesPlaceholder(variable_name="message")
                ]
            )
    
    runnable = RunnablePassthrough.assign(message=lambda x: filter_messages(x["message"])) | prompt | model | StrOutputParser()
    runnable_with_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="message",
)
    return runnable_with_history

def filter_messages(message, k=8):
    return message[-k:]

def get_session_history(session_id):
    memory = cl.user_session.get("memory")
    if session_id not in memory:
        memory[session_id] = ChatMessageHistory()
    return memory[session_id]

def clear_session_history(session_id):
    memory = cl.user_session.get("memory")
    if memory.get(session_id): 
        memory[session_id].clear()
    cl.user_session.set("memory",memory)


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Chat",
            markdown_description="Chatting with an LLM model",
            icon="https://picsum.photos/200",
        ),
        cl.ChatProfile(
            name="PDF-llamaindex",
            markdown_description="Summarize a PDF using an LLM",
            icon="https://picsum.photos/300",
        ),
    ]
    
@cl.on_chat_start
async def on_chat_start():
    
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model_name",
                label="Model_name",
                values=["GPT-4o","Claude 3.5 Sonnet", "Gemini 1.5 Pro" ],
                initial_index=0,
            )
,
            Slider(
                id="Temperature",
                label="Temperature",
                initial=0.5,
                min=0,
                max=1,
                step=0.1,
                description="モデルの出力の多様性や創造性を調整するパラメータで、高いほど予測不可能で創造的な応答が生成されます。",
            ),
            TextInput(id="SystemMessage", label="SystemMessage",description="AIモデルに特定の指示やコンテキストを与えるためのメッセージ", initial="あなたは優秀なチャットボットです。"),
            
        ]
    ).send()


    cl.user_session.set("settings", settings)
    cl.user_session.set("Model_name",None) 
    cl.user_session.set("memory", {})
    cl.user_session.set("memory_config",{"configurable": {"session_id": cl.user_session.get("id")}})
    runnable_with_history = set_runnable()
    cl.user_session.set("runnable_with_history",runnable_with_history)
    mode = cl.user_session.get("chat_profile")
    if mode == "PDF-llamaindex":
        files = None
        while files == None:
            files = await cl.AskFileMessage(
                content="Please upload a PDF file to begin!", accept=["application/pdf"],
                max_size_mb=1024,
                max_files=1,
                timeout=180
            ).send()

        is_make_llamaindex = make_llamaindex_db(cl.user_session.get("id"))
        if is_make_llamaindex:
            await cl.Message(
            author="Assistant", content="RAGの構築が完了しました。"
        ).send()
        else:
            await cl.Message(
            author="Assistant", content="何らかの理由でRAGの構築に失敗しました。"
        ).send()
        
@cl.on_settings_update
async def setup_setting(settings):
    old_model_name = cl.user_session.get("Model_name")
    new_model_name = settings["Model_name"]
    if old_model_name != new_model_name:
        clear_session_history(session_id=cl.user_session.get("id"))
    cl.user_session.set("settings", settings)
    runnable_with_history = set_runnable()
    cl.user_session.set("runnable_with_history",runnable_with_history)
    cl.user_session.set("Model_name",settings["Model_name"]) 

@cl.on_message
async def main(message: cl.Message):
    mode = cl.user_session.get("chat_profile")
    model_name = cl.user_session.get("settings")["Model_name"]
    if mode == "Chat":
        
        runnable_with_history = cl.user_session.get("runnable_with_history")
        memory_config = cl.user_session.get("memory_config")
        msg = cl.Message(content="", author=model_name)
        async for chunk in runnable_with_history.astream(
            {"message": message.content},
            config=memory_config,
        ):
            await msg.stream_token(chunk)
        await msg.update()
    elif mode == "PDF-llamaindex":
        query_engine = cl.user_session.get("query_engine")
        msg = cl.Message(content="", author=model_name)
        res = await cl.make_async(query_engine.query)(message.content)
        for token in res.response_gen:
            await msg.stream_token(token)
        
        await msg.update()
    else:
        pass
    

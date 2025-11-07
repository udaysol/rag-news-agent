import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain_core.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

chat_map = {}
def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_map:
        chat_map[session_id] = InMemoryChatMessageHistory()
    return chat_map[session_id]

def create_tools(path: str, llm, embeddings: OllamaEmbeddings, k:int, prompt: PromptTemplate) -> list:
    index = FAISS.load_local(folder_path=path,
                             embeddings=embeddings,
                             allow_dangerous_deserialization=True)

    retriever = index.as_retriever(search_kwargs={"k": k})

    local_retriever = RetrievalQA.from_chain_type(llm=llm,
                                                  retriever=retriever,
                                                  chain_type="stuff",
                                                  chain_type_kwargs={
                                                      "prompt": prompt
                                                  },
                                                #   verbose=True,
                                                  return_source_documents=True)

    search = DuckDuckGoSearchRun()

    tools = [
        Tool(name="LocalDatabase",
             func= lambda q: local_retriever.invoke(q),
             description="Use this to answer from local FAISS news database"),
        Tool(name="WebSearch",
             func= lambda q: search.invoke(q),
             description="Use this to search the web when answer is not found locally")             
    ]

    return tools

# Create Agent
def create_agent(tools: list, llm, prompt):
    agent = create_react_agent(tools=tools,
                           llm=llm,
                           prompt=prompt)

    agent_executor = AgentExecutor(agent=agent,
                                tools=tools,
                                verbose=True,
                                handle_parsing_errors=True)

    agent_with_history = RunnableWithMessageHistory(agent_executor,
                                                    get_session_history=get_chat_history,
                                                    input_messages_key="input",
                                                    history_messages_key="chat_history")

    return agent_with_history

# Obtain response
def get_response(agent, query, session_id):
    config = {"configurable": {"session_id": session_id}}
    res = agent.invoke({"input": query},
                       config=config)
    return res["output"]

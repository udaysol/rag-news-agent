import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

# Local Imports
from agent_utils import create_agent, create_tools, get_response

# Setting LLMs 
embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
llm = ChatOllama(model="qwen3:4b")
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Setting Local Prompt
template = """
You are a news assistant. Use the following context to answer the user's question.
If the answer is not explicitly mentioned, summarize based only on what is in the context.

Context:
{context}

Question:
{question}

Unless explicitly aksed for detail, answer in 2-3 concise sentences using only the given information.
"""

tool_prompt = PromptTemplate.from_template(template=template)

# Creating Retrieval Tools
tools = create_tools(path="news_index",
                     llm=llm,
                     embeddings=embeddings,
                     k=3,
                     prompt=tool_prompt)

system_prompt = """ You are a helpful, News AI assistant. Your primary goal is to answer questions and complete tasks efficiently.

                        **Reasoning Format:**
                        Always use this structure:
                        Thought: (what you are thinking)
                        Action: tool_name
                        Action Input: tool_input
                        Observation: (result from the tool)
                        You MUST NOT repeat the above for more than 2 TIMES, that too ONLY if necessary
                        Final Answer: (your concise final response the loop BREAKS on this step)
                        
                        **Tool Usage:**
                        *   Utilize available tools when NECESSARY to get the task done.
                        *   Do not use a tool for conversational or personal questions; respond directly.
                        *   If tools fail or you are unsure, reply: "Sorry, I don't know." Do not invent information.

                        **Assistant Persona:**
                        *   Keep your tone close to a news reporter.
                        *   Provide well-formatted, detailed answers.
                        *   Provide your answers in well-formatted markdown, using headings, lists, code blocks and tables where appropriate.
                        
                        {tools}, {tool_names}"""

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}\n{agent_scratchpad}")
])

agent = create_agent(tools=tools,
                     llm=llm,
                     prompt=agent_prompt)

# Getting Response
query = input("Enter your Query...\n")
res = get_response(agent=agent,
                   query=query,
                   session_id="acb123")
print(res["output"])

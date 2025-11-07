import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
# from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

import streamlit as st

# Local Imports
from agent_utils import create_agent, create_tools, get_response


# Initializing session state
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "show_api_popup" not in st.session_state:
    st.session_state.show_api_popup = True

# Popup for api key
if st.session_state.show_api_popup:
    st.title("Gemini API key required")
    session_id = st.text_input("Please enter a name for your session")
    api_input = st.text_input("Please enter your API key:", type="password")
    submit = st.button("Submit")

    if submit:
        if api_input.strip() == "":
            st.warning("API key cannot be empty.")
        else:
            st.session_state.session_id = session_id.strip()
            st.session_state.api_key = api_input.strip()
            st.session_state.show_api_popup = False
            st.success("✅ API key saved!")
            st.rerun()

else:
    # Setting LLMs 
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    # llm = ChatOllama(model="llama3.2:3b")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",
                                api_key=st.session_state.api_key)

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
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your message..."):
        st.chat_message("user").markdown(prompt)
        # Adding message to history
        st.session_state.messages.append({"role": "user",
                                        "content": prompt})
        
        res = get_response(agent=agent, 
                        query=prompt,
                        session_id=st.session_state.session_id)
        
        with st.chat_message("assistant"):
            st.markdown(res)

        st.session_state.messages.append({"role": "assistant",
                                        "content": res})
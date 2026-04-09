from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
import os
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
import streamlit as st


system_prompt = """
You are a task management assistant that interacts with a SQL database containing a 'tasks' table. 

TASK RULES:
1. Limit SELECT queries to 10 results max with ORDER BY created_at DESC
2. After CREATE/UPDATE/DELETE, confirm with SELECT query
3. If the user requests a list of tasks, present the output in a structured table format to ensure a clean and organized display in the browser."

CRUD OPERATIONS:
    CREATE: INSERT INTO tasks(title, description, status)
    READ: SELECT * FROM tasks WHERE ... LIMIT 10
    UPDATE: UPDATE tasks SET status=? WHERE id=? OR title=?
    DELETE: DELETE FROM tasks WHERE id=? OR title=?

Table schema: id, title, description, status(pending/in_progress/completed), created_at.
"""


@st.cache_resource
def create_db():
    db = SQLDatabase.from_uri("sqlite:///my_tasks.db")

    db.run(
        """
    CREATE TABLE IF NOT EXISTS tasks (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       title TEXT NOT NULL,
       description TEXT,
       status TEXT CHECK(status IN ('pending' , 'in_progress' , 'completed')) DEFAULT 'pending',
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
        """
    )
    print("DB TABLE CREATE SUCCESSFULLY")
    return db


@st.cache_resource
def get_llm_model():
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.9,
        openai_api_key=os.getenv("TOKEN"),
        openai_api_base="https://models.inference.ai.azure.com",
    )
    return llm


@st.cache_resource
def get_tools(_db, _llm):
    toolkit = SQLDatabaseToolkit(db=_db, llm=_llm)
    tools = toolkit.get_tools()
    return tools


@st.cache_resource
def get_agent(_llm, _tools):
    agent = create_agent(
        model=_llm,
        tools=_tools,
        checkpointer=InMemorySaver(),
        system_prompt=system_prompt,
    )
    return agent


db = create_db()
llm = get_llm_model()
tools = get_tools(db, llm)
agent = get_agent(llm, tools)


st.subheader("TaskBot - Agent to Manage Task")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])


query = st.chat_input("Ask me to manage task")


if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})
    # now i have to show loader
    with st.chat_message("ai"):
        with st.spinner("Processing...."):
            response = agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                {"configurable": {"thread_id": "1"}},
            )
            result = response["messages"][-1].content
            st.markdown(result)
            st.session_state.messages.append({"role": "ai", "content": result})

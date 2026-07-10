
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
load_dotenv()

# 2. Load Documents & Create FAISS Index

import glob
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

DOCS_DIR = "sample_docs"
INDEX_DIR = "faiss_index"

def load_and_split(docs_dir):
    chunks = []
    for path in glob.glob(os.path.join(docs_dir, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        source = os.path.basename(path)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 <= 500:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(Document(page_content=current, metadata={"source": source}))
                current = para
        if current:
            chunks.append(Document(page_content=current, metadata={"source": source}))
    return chunks

chunks = load_and_split(DOCS_DIR)
print(f"{len(chunks)} chunks")

# 3. Create Embeddings & Save Index
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(INDEX_DIR)
print(f"Index saved to {INDEX_DIR}/")

#  4. Math Tools
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b

@tool
def divide(a: int, b: int) -> float:
    """Divides a by b."""
    return a / b

# 5. RAG Tool - Search Documents

@tool
def search_docs(query: str) -> str:
    """Search the knowledge base for information about AI/ML concepts,
    LangGraph, RAG, embeddings, transformers, and related topics."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(query)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# 6. Tools List

tools = [add, multiply, divide, search_docs]
tools_by_name = {t.name: t for t in tools}

# 7. LLM Model 

from langchain_groq import ChatGroq

model = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
model_with_tools = model.bind_tools(tools)

#  8. State Definition

import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

#  9. Nodes - LLM Call & Tool Node

from langchain_core.messages import SystemMessage, ToolMessage

def llm_call(state: MessagesState) -> dict:
    response = model_with_tools.invoke(
        [SystemMessage(content=(
            "You are a helpful assistant that can perform arithmetic "
            "and answer questions about AI/ML concepts. "
            "Use search_docs for AI/ML questions, math tools for calculations."
        ))] + state["messages"]
    )
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

def tool_node(state: MessagesState) -> dict:
    results = []
    for tool_call in state["messages"][-1].tool_calls:
        t = tools_by_name[tool_call["name"]]
        observation = t.invoke(tool_call["args"])
        results.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": results}

# 10.  Conditional Edge - Should Continue

from typing import Literal
from langgraph.graph import END

def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool_node"
    return END

# 11. Build Graph


from langgraph.graph import StateGraph, START, END

agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()

# 12. Visualize Graph
from IPython.display import Image, display
display(Image(agent.get_graph().draw_mermaid_png()))

# 13. Run Agent Function

from langchain_core.messages import HumanMessage, AIMessage

def run_agent(question: str):
    print(f"Q: {question}")
    result = agent.invoke({"messages": [HumanMessage(content=question)], "llm_calls": 0})
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  tool: {tc['name']}  args: {tc['args']}")
        if isinstance(msg, ToolMessage):
            print(f"  result: {msg.content[:120]}")
    print(f"A: {result['messages'][-1].content}")
    print()

#  14. Test - Math Question

run_agent("What is 12 times 8?")

# 15. Test - RAG Question

run_agent("What is LangGraph?")


"""
LangGraph RAG Demo — Streamlit UI
Run: streamlit run app.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import MessagesState, agent

# Page config

st.set_page_config(
    page_title="LangGraph RAG Demo",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 LangGraph RAG Demo")
st.caption("Calculator agent + knowledge base search — powered by LangGraph + Claude")

# Session state

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, content)

if "tool_traces" not in st.session_state:
    st.session_state.tool_traces = []  # list of trace dicts per turn

# Sidebar — tool trace


with st.sidebar:
    st.header("🔧 Tool Trace")
    st.caption("Live view of which tools the agent called")

    if not st.session_state.tool_traces:
        st.info("No tool calls yet. Ask a question!")
    else:
        for i, trace in enumerate(reversed(st.session_state.tool_traces)):
            turn_label = f"Turn {len(st.session_state.tool_traces) - i}"
            with st.expander(turn_label, expanded=(i == 0)):
                if not trace:
                    st.write("_No tools called — answered directly._")
                else:
                    for call in trace:
                        st.markdown(f"**Tool:** `{call['name']}`")
                        st.markdown("**Input:**")
                        st.code(json.dumps(call["args"], indent=2), language="json")
                        st.markdown("**Output:**")
                        st.code(call["result"], language="text")
                        st.divider()

# Chat display

for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)

# Input

prompt = st.chat_input("Ask me to calculate something or explain an AI concept...")

if prompt:
    # Show user message immediately
    st.session_state.chat_history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            initial_state: MessagesState = {
                "messages": [HumanMessage(content=prompt)],
                "llm_calls": 0,
            }
            result = agent.invoke(initial_state)

        # Parse tool calls and results from the message history
        trace = []
        messages = result["messages"]

        # Collect all AIMessage tool calls and pair with ToolMessage results
        pending_calls: dict[str, dict] = {}
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    pending_calls[tc["id"]] = {
                        "name": tc["name"],
                        "args": tc["args"],
                        "result": "",
                    }
            elif isinstance(msg, ToolMessage):
                if msg.tool_call_id in pending_calls:
                    pending_calls[msg.tool_call_id]["result"] = msg.content

        trace = list(pending_calls.values())
        st.session_state.tool_traces.append(trace)

        # Final answer is the last AIMessage with no tool calls
        final_answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                final_answer = msg.content
                break

        st.markdown(final_answer)
        st.session_state.chat_history.append(("assistant", final_answer))

    # Rerun to refresh sidebar
    st.rerun()
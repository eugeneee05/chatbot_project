# new.py
import re
import requests
import traceback
import json
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage, messages_to_dict, ToolMessage
from IPython.display import Image, display
from pprint import pprint
from collections.abc import Iterable
from random import randint
import threading

app = FastAPI()

# -----------------------
# Per-chat session storage
# -----------------------
# SESSIONS maps chat_id (int) -> dict with keys:
#   - "messages": list (LangChain message objects)
#   - "info": list[str]
#   - "finished": bool
#   - "PROJECT_INFO": dict (the 7 fields)
SESSIONS: dict[int, dict] = {}
SESSIONS_LOCK = threading.Lock()

# This is used so tools can know which session to operate on.
# It is set temporarily while executing tool node for a session.
CURRENT_CHAT_ID = None
CURRENT_CHAT_ID_LOCK = threading.Lock()

# Template for per-session PROJECT_INFO
def make_project_info():
    return {
        "site_count": "",
        "offset_count": "",
        "project_name": "",
        "device_name": "",
        "device_revision": "",
        "programme_id": "",
        "programme_revision": "",
    }

# Helper to ensure session exists
def ensure_session(chat_id: int):
    with SESSIONS_LOCK:
        if chat_id not in SESSIONS:
            SESSIONS[chat_id] = {
                "messages": [],  # list of HumanMessage / AIMessage
                "info": [],
                "finished": False,
                "PROJECT_INFO": make_project_info(),
            }
    return SESSIONS[chat_id]

# -----------------------
# LangGraph and tool schema
# -----------------------
class infoState(TypedDict):
    messages: Annotated[list, add_messages]
    info: list[str]
    finished: bool

def convert_messages_to_dict(messages):
    result = []
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = m.get("content", "")
        else:
            # Map LangChain types to valid roles
            m_type = getattr(m, "type", None)
            if m_type == "human":
                role = "user"
            elif m_type == "ai":
                role = "assistant"
            elif hasattr(m, "tool_call_id"):
                role = "tool"
            else:
                role = "user"

            content = getattr(m, "content", "")

        if not isinstance(content, str):
            content = str(content)

        result.append({"role": role, "content": content})
    return result

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_info",
            "description": "Provide the information required from the user",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
    "type": "function",
    "function": {
        "name": "add_to_info",
        "description": "Add all project details provided exactly from the user",
        "parameters": {
            "type": "object",
            "properties": {
                "site_count": {"type": "string"},
                "offset_count": {"type": "string"},
                "project_name": {"type": "string"},
                "device_name": {"type": "string"},
                "device_revision": {"type": "string"},
                "programme_id": {"type": "string"},
                "programme_revision": {"type": "string"}
            },
            "required": [
                "site_count",
                "offset_count",
                "project_name",
                "device_name",
                "device_revision",
                "programme_id",
                "programme_revision"
            ],
        },
    },
},
    {
        "type": "function",
        "function": {
            "name": "confirm_info",
            "description": "Confirm the user's information without changing the details provided",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_JSON",
            "description": "Generate a JSON object with all user-provided info",
            "parameters": {
                "type": "object",
                "properties": {"info": {"type": "array", "items": {"type": "string"}}},
                "required": ["info"],
            },
        },
    },
]

# -----------------------
# Model call
# -----------------------
def call_model(history, inputs):
    # Convert LangChain message objects (e.g. HumanMessage, AIMessage) to dicts
    if history and hasattr(history[0], "type"):
        history = messages_to_dict(history)

    payload = {
        "model": "unsloth/phi-4-mini-instruct",
        "messages": convert_messages_to_dict(history),
        "temperature": 0.0,
        "tools": tools_schema
    }

    try:
        response = requests.post(
            "http://localhost:1234/v1/chat/completions", json=payload, timeout=30
        )
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        reply = message.get("content", "") or ""
        tool_calls = message.get("tool_calls", [])  # external API tool_calls format

        for original in inputs.values():
            if original.lower() in reply.lower():
                reply = re.sub(re.escape(original), original, reply, flags=re.IGNORECASE)

        return reply, tool_calls
    except Exception as e:
        print("Model call failed:", e)
        traceback.print_exc()
        return f"Error communicating with model: {e}", []

# -----------------------
# Tools (operate on CURRENT_CHAT_ID's session)
# -----------------------
def get_current_session_project_info():
    global CURRENT_CHAT_ID
    if CURRENT_CHAT_ID is None:
        # fallback to first session if available
        with SESSIONS_LOCK:
            if not SESSIONS:
                return make_project_info()
            # pick arbitrary
            any_id = next(iter(SESSIONS))
            return SESSIONS[any_id]["PROJECT_INFO"]
    return SESSIONS[CURRENT_CHAT_ID]["PROJECT_INFO"]

def set_current_session_project_info(new_info: dict):
    global CURRENT_CHAT_ID
    if CURRENT_CHAT_ID is None:
        return
    SESSIONS[CURRENT_CHAT_ID]["PROJECT_INFO"].update(new_info)

@tool
def get_info() -> str:
    """Provide all the exact information that required the user to fill in, please show the entire information to the user."""
    proj = get_current_session_project_info()
    # Return exact format expected by model/workflow
    return """
These are the 7 pieces of information that required the user to fill in for the project, Show the entire information to the user in this exact format:
Do not add any other text or information, just show the list below:
- Site Count
- Offset Count
- Project Name
- Device Name
- Device Revision
- Programme Id
- Programme Revision
"""

@tool
def add_to_info(
    site_count: str,
    offset_count: str,
    project_name: str,
    device_name: str,
    device_revision: str,
    programme_id: str,
    programme_revision: str
) -> str:
    """Adds the details to the particular information, with validation."""
    proj = get_current_session_project_info()
    errors = []

    if not site_count or not site_count.isdigit():
        errors.append("site_count must be a number.")
    if not offset_count or not offset_count.isdigit():
        errors.append("offset_count must be a number.")
    if not project_name.strip():
        errors.append("project_name is required.")
    if not device_name.strip():
        errors.append("device_name is required.")
    if not device_revision.strip():
        errors.append("device_revision is required.")
    if not programme_id.strip():
        errors.append("programme_id is required.")
    if not programme_revision.strip():
        errors.append("programme_revision is required.")

    if errors:
        return json.dumps({"status":"error","errors":errors})

    # Save into session PROJECT_INFO
    set_current_session_project_info({
        "site_count": site_count,
        "offset_count": offset_count,
        "project_name": project_name,
        "device_name": device_name,
        "device_revision": device_revision,
        "programme_id": programme_id,
        "programme_revision": programme_revision,
    })

    info_string = (
        f"Site Count: {site_count}, Offset Count: {offset_count}, Project Name: {project_name}, "
        f"Device Name: {device_name}, Device Revision: {device_revision}, Programme Id: {programme_id}, "
        f"Programme Revision: {programme_revision}"
    )
    return json.dumps({"status":"success","info":info_string})

@tool
def confirm_info() -> str:
    """Asks the customer if the details are correct."""
    proj = get_current_session_project_info()
    return f"""
You must to show all 7 information together with details provided by user: , site count, offset count, project name, device name, device revision, programme id, programme revision.
Ask the user whether the information is correct or not.
If no, ask the user which information need to be amend.
If yes, reply "Confirmation complete."

Current values:
- Site Count: {proj.get("site_count")}
- Offset Count: {proj.get("offset_count")}
- Project Name: {proj.get("project_name")}
- Device Name: {proj.get("device_name")}
- Device Revision: {proj.get("device_revision")}
- Programme Id: {proj.get("programme_id")}
- Programme Revision: {proj.get("programme_revision")}
"""

@tool
def create_JSON(
    site_count: int = None,
    offset_count: int = None,
    project_name: str = None,
    device_name: str = None,
    device_revision: str = None,
    programme_id: str = None,
    programme_revision: str = None
) -> dict:
    """
    Create JSON for project creation using current session PROJECT_INFO.
    """
    proj = get_current_session_project_info()
    return {
        "site_count": proj.get("site_count"),
        "offset_count": proj.get("offset_count"),
        "project_name": proj.get("project_name"),
        "device_name": proj.get("device_name"),
        "device_revision": proj.get("device_revision"),
        "programme_id": proj.get("programme_id"),
        "programme_revision": proj.get("programme_revision"),
    }

# Register tools into ToolNode
tools = [get_info, add_to_info, confirm_info, create_JSON]
tool_node = ToolNode(tools)

# -----------------------
# Existing model/chatbot nodes (adapted)
# -----------------------
WELCOME_MSG = "Welcome to the TechFlow Assistant Chatbot. Type 'q' to quit. What action do you want to do?"

def call_model_with_tools(history, tools_schema=None):
    """
    Call local model and produce an AIMessage that may contain tool_calls in the expected format.
    """
    if history and hasattr(history[0], "type"):
        history = messages_to_dict(history)

    last_message = getattr(history[-1], "content", "") if not isinstance(history[-1], dict) else history[-1].get("content", "")
    # Pass model input as dict (keeps compatibility)
    reply_text, api_tool_calls = call_model(history, {"input": last_message})

    # Format tool_calls for AIMessage
    formatted_tool_calls = []
    if api_tool_calls:
        for tc in api_tool_calls:
            function_info = tc.get("function", {})
            tool_name = function_info.get("name", "")
            tool_args = function_info.get("arguments", {})

            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except:
                    tool_args = {}

            formatted_tc = {
                "id": tc.get("id", "") or str(randint(1000,9999)),
                "name": tool_name,
                "args": tool_args
            }
            formatted_tool_calls.append(formatted_tc)

    return AIMessage(content=reply_text, tool_calls=formatted_tool_calls)

def chatbot_with_tools(state: infoState) -> infoState:
    defaults = {"messages": [], "info": [], "finished": False}
    print("--------------------------------")
    print("Chatbot with Tools")
    print("--------------------------------")

    if state.get("messages"):
        new_output = call_model_with_tools([ASSISTANT_SYSINT] + state["messages"], tools_schema)
        print("DEBUG: Tool calls from API response:")
        if new_output.tool_calls:
            for t in new_output.tool_calls:
                print(f"Tool Name: {t.get('name')}, Args: {t.get('args')}, ID: {t.get('id')}")
    else:
        new_output = AIMessage(content=WELCOME_MSG, tool_calls=[])

    updated_messages = state["messages"] + [
        AIMessage(content=new_output.content, tool_calls=new_output.tool_calls)
    ]
    state = {**defaults, **state, "messages": updated_messages}
    return state

def update_state_after_tools(state: infoState) -> infoState:
    info = state.get("info", [])
    finished = state.get("finished", False)
    messages = state.get("messages", [])
    # Find most recent AI message that has tool_calls and process them
    for msg in reversed(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call["name"] == "add_to_info":
                    args = tool_call.get("args", {})
                    # Build a readable summary from session PROJECT_INFO
                    # (PROJECT_INFO already updated by add_to_info)
                    summary = ", ".join(
                        f"{key}: {args.get(key, SESSIONS[CURRENT_CHAT_ID]['PROJECT_INFO'].get(key, ''))}" 
                        for key in SESSIONS[CURRENT_CHAT_ID]['PROJECT_INFO']
                    )
                    if summary not in info:
                        info.append(summary)
                elif tool_call["name"] == "create_JSON":
                    finished = True
            break
    return {**state, "info": info, "finished": finished}

def create_node(state: infoState) -> infoState:
    messages = state.get("messages", [])
    info = state.get("info", [])
    print("--------------------------------")
    print("Create Node")
    print("--------------------------------")
    result = {
        "status": "success",
        "details": info,
        "project_json": SESSIONS[CURRENT_CHAT_ID]["PROJECT_INFO"] if CURRENT_CHAT_ID in SESSIONS else make_project_info()
    }
    messages.append(
        AIMessage(
            content=f"Here is your final JSON:\n{json.dumps(result, indent=2)}",
            additional_kwargs={}
        )
    )
    return {
        **state,
        "messages": messages,
        "info": info,
        "finished": True
    }

def maybe_route_to_tools(state: infoState) -> str:
    msgs = state.get("messages", [])
    if not msgs:
        return "human"
    last_msg = msgs[-1]
    if isinstance(last_msg, dict):
        role = last_msg.get("role")
    else:
        msg_type = getattr(last_msg, "type", None)
        role = "user" if msg_type == "human" else "assistant"
    if role == "user":
        return "chatbot"
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "human"

# Build a graph just for visualization / completeness (not strictly required for loop below)
graph_builder = StateGraph(infoState)
graph_builder.add_node("chatbot", chatbot_with_tools)
graph_builder.add_node("human", lambda s: s)  # placeholder
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("update_state", update_state_after_tools)
graph_builder.add_node("creating", create_node)
graph_builder.add_conditional_edges("chatbot", maybe_route_to_tools)
graph_builder.add_conditional_edges("human", lambda s: "chatbot")
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "update_state")
graph_builder.add_edge("update_state", "chatbot")
graph_builder.add_edge("creating", END)
chat_graph = graph_builder.compile()

try:
    Image(chat_graph.get_graph().draw_mermaid_png())
except Exception:
    pass

# -----------------------
# Session processing loop
# -----------------------
def process_session_until_human_or_finished(chat_id: int):
    """
    Runs nodes for the session until it returns to 'human' node (awaiting user)
    or the session finished (create_JSON called).
    This function mutates SESSIONS[chat_id]['messages'] and other session state.
    """
    global CURRENT_CHAT_ID
    # ensure session exists
    session = ensure_session(chat_id)
    # We will use a lock while setting CURRENT_CHAT_ID to avoid races
    with CURRENT_CHAT_ID_LOCK:
        CURRENT_CHAT_ID = chat_id
        try:
            # local shorthand
            state = {
                "messages": session["messages"],
                "info": session["info"],
                "finished": session["finished"]
            }
            # loop until we should return to human or finished
            while True:
                route = maybe_route_to_tools(state)
                if route == "human":
                    # update session and stop - waiting for human
                    session["messages"] = state["messages"]
                    session["info"] = state["info"]
                    session["finished"] = state["finished"]
                    break
                elif route == "chatbot":
                    state = chatbot_with_tools(state)
                    # append the AI output into session messages in-loop so subsequent tool calls see them
                    # (chatbot_with_tools already updated state["messages"])
                    continue
                elif route == "tools":
                    # Execute tool node - the ToolNode expects infoState style input.
                    # The ToolNode will call the actual Python functions (get_info/add_to_info/...)
                    try:
                        # tool_node expects a state param; call it and get returned state
                        state = tool_node.invoke(state)
                    except Exception as e:
                        # If tool_node raises, append an error AI message and return to human
                        state["messages"].append(AIMessage(content=f"Tool execution error: {e}"))
                        session["messages"] = state["messages"]
                        session["info"] = state["info"]
                        session["finished"] = state["finished"]
                        break
                    # After tools executed, run update_state_after_tools
                    state = update_state_after_tools(state)
                    # If update_state marks finished True, then invoke create_node to append final JSON
                    if state.get("finished"):
                        state = create_node(state)
                        # save and break
                        session["messages"] = state["messages"]
                        session["info"] = state["info"]
                        session["finished"] = state["finished"]
                        break
                    # else continue the loop (go back to chatbot)
                    continue
                elif route == "creating":
                    state = create_node(state)
                    session["messages"] = state["messages"]
                    session["info"] = state["info"]
                    session["finished"] = state["finished"]
                    break
                else:
                    # unknown route -> stop
                    session["messages"] = state["messages"]
                    session["info"] = state["info"]
                    session["finished"] = state["finished"]
                    break
        finally:
            CURRENT_CHAT_ID = None

# -----------------------
# FastAPI endpoints for C# client
# -----------------------
@app.get("/chat/initial/{chat_id}")
def get_initial_message(chat_id: int):
    """
    Create session if missing and return a welcome/initial message.
    The C# client expects: { "response": "<text>" }
    """
    ensure_session(chat_id)
    # We can prime the session with a welcome AI message if desired:
    session = SESSIONS[chat_id]
    # If session already has a bot message, return it; otherwise return welcome.
    # Use exactly key 'response' as C# expects.
    welcome = WELCOME_MSG
    return JSONResponse(content={"response": welcome})

@app.post("/chat/{chat_id}")
def post_chat_message(chat_id: int, payload: dict = Body(...)):
    """
    Accepts {"text": "..."} from C# app, adds HumanMessage to session,
    processes the session until it's awaiting human or finished, then returns:
    {
      "response": "<bot reply>",
      "final_data": {...} or None,
      "summary_json": {...} or None
    }
    """
    text = payload.get("text", "")
    if text is None:
        return JSONResponse(content={"response":"", "final_data": None, "summary_json": None})

    # Create session if needed
    ensure_session(chat_id)

    session = SESSIONS[chat_id]

    # Append user message to session messages
    session["messages"].append(HumanMessage(content=text))

    # Process the state machine for this session
    process_session_until_human_or_finished(chat_id)

    # After processing, fetch the last AI assistant message to return as 'response'
    last_bot = None
    for msg in reversed(session["messages"]):
        if isinstance(msg, AIMessage):
            last_bot = msg.content
            break

    # Determine final_data/summary_json: return the PROJECT_INFO dict if finished, else None
    final_data = None
    summary_json = None
    if session.get("finished"):
        final_data = session["PROJECT_INFO"]
        summary_json = session["PROJECT_INFO"]

    return JSONResponse(content={
        "response": last_bot or "",
        "final_data": final_data,
        "summary_json": summary_json
    })

# -----------------------
# System prompt (kept similar to your improved prompt)
# -----------------------
ASSISTANT_SYSINT = {
    "role":"system",
    "content": (
        "You are an AssistantBot, an interactive create project system.\n"
        "Follow the workflow strictly. When the user requests to create a project, "
        "call get_info to show the fields, then receive add_to_info once with the fields, "
        "then call confirm_info to show the summary and wait for user confirmation. "
        "Only after user confirms, call create_JSON and present the final JSON. "
        "Do not request fields twice or call create_JSON before confirmation."
    )
}

# -----------------------
# OPTIONAL: small CLI to run simple test when executed as script
# -----------------------
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)

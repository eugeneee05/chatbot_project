import re
import requests
import traceback
import json
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage
from IPython.display import Image, display
from pprint import pprint
from langchain_core.messages.ai import AIMessage
from langchain_core.messages import messages_to_dict
from typing import Literal
from collections.abc import Iterable
from random import randint
from langchain_core.messages import ToolMessage

app = FastAPI()

# ----- GLOBAL PROJECT INFO DICTIONARY FOR ALL 7 INFORMATION FIELDS -----
PROJECT_INFO = {
    "site_count": "",
    "offset_count": "",
    "project_name": "",
    "device_name": "",
    "device_revision": "",
    "programme_id": "",
    "programme_revision": "",
}
# ----------------------------------------------------------------------

class infoState(TypedDict):
    """State representing the customer's order conversation."""
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

        # Ensure content is a string
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
            "http://localhost:1234/v1/chat/completions", json=payload
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
        print(traceback.format_exc())
        if hasattr(e, 'response') and e.response is not None:
            print(e.response.text)
        return f"Error communicating with model: {e}", []

ASSISTANT_SYSINT = {
    "role":"system",
    "content": (
        "You are an AssistantBot, an interactive create project system. "
        "You will ask the human which action he want to perform."
        "If human say they want to create project, "
        "call get_info tool to show the list that required to fill in by them. "
        "\n\nAdd the details provided into add_to_info. "
        "Always call confirm_info to confirm with the user before calling create_JSON. Once create_JSON has returned, "
        "thank the user and say goodbye!"
    )
}

WELCOME_MSG = "Welcome to the TechFlow Assistant Chatbot. Type 'q' to quit. What action do you want to do?"

def human_node(state: infoState) -> infoState:
    """Display the last model message to the user, and receive the user's input."""
    messages = state.get("messages", [])

    print("--------------------------------")
    print("Human Node")
    print("--------------------------------")
    
    # Check if the last message is already from the user - if so, don't ask for input again
    if messages:
        last_msg = messages[-1]
        # Check if last message is from user
        last_msg_type = getattr(last_msg, "type", None)
        last_msg_role = None
        if isinstance(last_msg, dict):
            last_msg_role = last_msg.get("role")
        elif last_msg_type == "human":
            last_msg_role = "user"
        
        # If last message is already from user, skip asking for input again
        if last_msg_role == "user":
            print("Warning: Last message is already from user, skipping input prompt")
            return state
        
        print("Model:", getattr(last_msg, "content", last_msg.get("content") if isinstance(last_msg, dict) else last_msg))
    else:
        print("Model: (no message)")

    user_input = input("User: ")

    if user_input.lower() in {"q", "quit", "exit", "goodbye"}:
        state["finished"] = True
        state["messages"].append({"role": "user", "content": "goodbye"})
    else:
        state["messages"].append({"role": "user", "content": user_input})

    return state

def maybe_exit_human_node(state: infoState) -> Literal["chatbot", "__end__"]:
    """Route to the chatbot, unless it looks like the user is exiting."""
    if state.get("finished", False):
        return END
    else:
        return "chatbot"

@tool
def get_info() -> str:
    """Provide all the exact information that required the user to fill in, please show the entire information to the user."""

    print ("Get info....")
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

    print ("Add to info....")
    errors = []

    # Validation rules
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

    # If errors exist → return structured error message
    if errors:
        return f'{{"status":"error","errors":{errors}}}'

    # If validation passed → store into the global PROJECT_INFO dict
    PROJECT_INFO["site_count"] = site_count
    PROJECT_INFO["offset_count"] = offset_count
    PROJECT_INFO["project_name"] = project_name
    PROJECT_INFO["device_name"] = device_name
    PROJECT_INFO["device_revision"] = device_revision
    PROJECT_INFO["programme_id"] = programme_id
    PROJECT_INFO["programme_revision"] = programme_revision

    # Return success
    info_string = (
        f"Site Count: {site_count}, Offset Count: {offset_count}, Project Name: {project_name}, "
        f"Device Name: {device_name}, Device Revision: {device_revision}, Programme Id: {programme_id}, "
        f"Programme Revision: {programme_revision}"
    )
    
    return f'{{"status":"success","info":"{info_string}"}}'


@tool
def confirm_info() -> str:
    """Asks the customer if the details are correct."""
    print ("Confirm info")

    return f"""
    You need to show all 7 information with details provided by user: , site count, offset count, project name, device name, device revision, programme id, programme revision.
    Ask the user whether the information is correct or not.
    If no, ask the user which information need to be amend.
    If yes, reply "Confirmation complete."

    Current values:
    - Site Count: {PROJECT_INFO.get("site_count")}
    - Offset Count: {PROJECT_INFO.get("offset_count")}
    - Project Name: {PROJECT_INFO.get("project_name")}
    - Device Name: {PROJECT_INFO.get("device_name")}
    - Device Revision: {PROJECT_INFO.get("device_revision")}
    - Programme Id: {PROJECT_INFO.get("programme_id")}
    - Programme Revision: {PROJECT_INFO.get("programme_revision")}
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
    Create JSON for project creation.
    After confirming all the details from user, you need to create a JSON.
    Do not change the details provided by the user.
    Do not add any other text or information in the JSON following the exact format below:
    """
    # Keep docstring unchanged.

    # Return JSON exactly using stored PROJECT_INFO values
    return {
        "site_count": PROJECT_INFO.get("site_count"),
        "offset_count": PROJECT_INFO.get("offset_count"),
        "project_name": PROJECT_INFO.get("project_name"),
        "device_name": PROJECT_INFO.get("device_name"),
        "device_revision": PROJECT_INFO.get("device_revision"),
        "programme_id": PROJECT_INFO.get("programme_id"),
        "programme_revision": PROJECT_INFO.get("programme_revision"),
    }


tools = [get_info, add_to_info, confirm_info, create_JSON]
tool_node = ToolNode(tools)


def call_model_with_tools(history, tools_schema=None):
    """
    Call the FASTAPI model, providing tool info as system context.
    Automatically formats tool descriptions from the schema.
    """
    if history and hasattr(history[0], "type"):
        history = messages_to_dict(history)

    # Get last message content
    last_message = getattr(history[-1], "content", "") if not isinstance(history[-1], dict) else history[-1].get("content", "")

    # Call model
    reply_text, api_tool_calls = call_model(history, {"input": last_message})

    # Format tool_calls for AIMessage
    formatted_tool_calls = []
    if api_tool_calls:
        for tc in api_tool_calls:
            # Format tool call to match AIMessage expected format
            # AIMessage expects: id (str), name (str), args (dict)
            function_info = tc.get("function", {})
            tool_name = function_info.get("name", "")
            tool_args = function_info.get("arguments", {})
            
            # If arguments is a string, try to parse it as JSON
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except:
                    tool_args = {}
            
            formatted_tc = {
                "id": tc.get("id", ""),
                "name": tool_name,
                "args": tool_args
            }
            formatted_tool_calls.append(formatted_tc)

    return AIMessage(content=reply_text, tool_calls=formatted_tool_calls)



def chatbot_with_tools(state: infoState) -> infoState:
    """
    Chatbot node: calls the model, handles tool calls automatically,
    and updates state.
    """
    defaults = {"messages": [], "info": [], "finished": False}

    print("--------------------------------")
    print("Chatbot with Tools")
    print("--------------------------------")

    if state.get("messages"):
        new_output = call_model_with_tools([ASSISTANT_SYSINT] + state["messages"], tools_schema)

        # --- DEBUG: print raw model reply ---
        # print("Model reply:", getattr(new_output, "content", ""))
        
        # --- DEBUG: show tool calls from API response ---
        print("DEBUG: Tool calls from API:")
        if new_output.tool_calls:
            for t in new_output.tool_calls:
                print(f"Tool Name: {t.get('name')}, Args: {t.get('args')}, ID: {t.get('id')}")
        else:
            print("No tool calls in API response.")

    else:
        print("B")
        new_output = AIMessage(content=WELCOME_MSG, tool_calls=[])

    # Append model message
    updated_messages = state["messages"] + [
        AIMessage(content=new_output.content, tool_calls=new_output.tool_calls)
    ]
    state = {**defaults, **state, "messages": updated_messages}

    return state


def update_state_after_tools(state: infoState) -> infoState:
    """Update state after tool execution - extract info from tool calls and update state."""
    info = state.get("info", [])
    finished = state.get("finished", False)

    messages = state.get("messages", [])

    # Search backwards for the most recent AI message that has tool_calls
    for msg in reversed(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:

                # Only append summary if not already present
                if tool_call["name"] == "add_to_info":
                    args = tool_call.get("args", {})
                    summary = ", ".join(
                        f"{key}: {args.get(key, PROJECT_INFO.get(key, ''))}" 
                        for key in PROJECT_INFO
                    )
                    if summary not in info:
                        info.append(summary)

                elif tool_call["name"] == "create_JSON":
                    finished = True  # Mark finished immediately

            break  # Only process the most recent tool call message

    return {**state, "info": info, "finished": finished}


def create_node(state: infoState) -> infoState:
    """
    Final node: Create the JSON output using the collected info. 
    You are required to create JSON with site count, offset count, project name, device name, device revision, programme id and programme revision.
    No tools are invoked here.
    """
    messages = state.get("messages", [])
    info = state.get("info", [])

    print("--------------------------------")
    print("Create Node")
    print("--------------------------------")

    # Create final JSON
    result = {
        "status": "success",
        "details": info,
        "project_json": {
            "site_count": PROJECT_INFO.get("site_count"),
            "offset_count": PROJECT_INFO.get("offset_count"),
            "project_name": PROJECT_INFO.get("project_name"),
            "device_name": PROJECT_INFO.get("device_name"),
            "device_revision": PROJECT_INFO.get("device_revision"),
            "programme_id": PROJECT_INFO.get("programme_id"),
            "programme_revision": PROJECT_INFO.get("programme_revision"),
        }
    }

    # Append the AI output
    messages.append(
        AIMessage(
            content=f"Here is your final JSON:\n{result}",
            additional_kwargs={}
        )
    )

    # Mark the process as finished
    return {
        **state,
        "messages": messages,
        "info": info,
        "finished": True
    }


def maybe_route_to_tools(state: infoState) -> str:
    """Route between chat and the tool nodes if a tool call is made."""
    msgs = state.get("messages", [])
    if not msgs:
        return "human"

    last = msgs[-1]

    # Determine role
    if isinstance(last, dict):
        role = last.get("role")
    else:
        t = getattr(last, "type", None)
        role = "user" if t == "human" else "assistant"

    # If user spoke → chatbot must reply
    if role == "user":
        return "chatbot"

    # If assistant spoke:
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    # Assistant spoke and did NOT call tools → wait for user
    return "human"


graph_builder = StateGraph(infoState)
graph_builder.add_node("chatbot", chatbot_with_tools)
graph_builder.add_node("human", human_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("update_state", update_state_after_tools)
graph_builder.add_node("creating", create_node)
graph_builder.add_conditional_edges("chatbot", maybe_route_to_tools)
graph_builder.add_conditional_edges("human", maybe_exit_human_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "update_state")
graph_builder.add_edge("update_state", "chatbot")
graph_builder.add_edge("creating", "chatbot")

chat_graph = graph_builder.compile()

Image(chat_graph.get_graph().draw_mermaid_png())

# Invoke example
chat_graph.invoke(
    {"messages": [], "info": []},
    config={"recursion_limit": 100}
)

import re
import requests
import traceback
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
            "description": "Add a detail provided by the user to the info list",
            "parameters": {
                "type": "object",
                "properties": {"details": {"type": "array", "items": {"type": "string"}}},
                "required": ["details"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_info",
            "description": "Confirm the user's information",
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
        
        # Extract content
        message = data["choices"][0]["message"]
        content = message.get("content", "")
        
        # Extract function name if tool_calls exist
        function_name = None
        if "tool_calls" in message and message["tool_calls"]:
            function_name = message["tool_calls"][0]["function"]["name"]
        
        # Process content with inputs if content exists
        if content:
            for original in inputs.values():
                if original.lower() in content.lower():
                    content = re.sub(re.escape(original), original, content, flags=re.IGNORECASE)

        return content, function_name
    except Exception as e:
        print(traceback.format_exc())
        if hasattr(e, 'response') and e.response is not None:
            print(e.response.text)
        return f"Error communicating with model: {e}", None
    
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
    last_msg = state["messages"][-1]
    print("Model:", getattr(last_msg, "content", last_msg.get("content") if isinstance(last_msg, dict) else last_msg))

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
    """Provide the information that required the user to fill in."""
    return """
Site Count:
Offset Count:
Project Name:
Device Name:
Device Revision:
Programme Id:
Programme Revision:
"""

@tool
def add_to_info(details: Iterable[str]) -> str:
    """Adds the details to the particular information."""
    return "Details recorded."

@tool
def confirm_info() -> str:
    """Asks the customer if the details are correct."""
    return "Confirmation complete."

@tool
def create_JSON(info: list[str]) -> dict:
    """Create a flat JSON with all user-provided details as key-value pairs."""
    json_output = {}
    for line in info:
        line = line.strip()
        # Check for different possible separators
        if " is " in line:
            key, value = line.split(" is ", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            # Fallback: take whole line as key
            key, value = line, ""
        json_output[key.strip()] = value.strip()
    print("Generated JSON:")
    print(json_output)
    return json_output


tools = [get_info, add_to_info, confirm_info, create_JSON]
tool_node = ToolNode(tools)


def call_model_with_tools(history, tools_schema=None):
    """
    Call the FASTAPI model, providing tool info as system context.
    Automatically formats tool descriptions from the schema.
    """
    if history and hasattr(history[0], "type"):
        history = messages_to_dict(history)

    # Add tool info to system context
    if tools_schema:
        tool_descriptions = "\n".join(
            [f"- {t['function']['name']}: {t['function']['description']}" for t in tools_schema]
        )
        system_message = {
            "role": "system",
            "content": f"You have access to these tools:\n{tool_descriptions}\nCall them when relevant."
        }
        history = [system_message] + history

    # Get last message content
    last_message = getattr(history[-1], "content", "") if not isinstance(history[-1], dict) else history[-1].get("content", "")

    # Call model
    reply_text, function_name = call_model(history, {"input": last_message})

    print("#Reply#", reply_text)
    if function_name:
        print(f"#Function Name#: {function_name}")

    # Build tool_calls list if function_name exists
    tool_calls = []
    if function_name:
        tool_calls = [{"name": function_name, "arguments": {}}]

    return AIMessage(content=reply_text, tool_calls=tool_calls)


def chatbot_with_tools(state: infoState) -> infoState:
    """
    Chatbot node: calls the model, handles tool calls automatically,
    and updates state.
    """
    defaults = {"messages": [], "info": [], "finished": False}

    if state.get("messages"):
        print("A")
        new_output = call_model_with_tools([ASSISTANT_SYSINT] + state["messages"], tools_schema)

        # --- DEBUG: print raw model reply ---
        print("Model reply:", getattr(new_output, "content", ""))
        
        # --- Parse tool calls from model text ---
        tool_calls = []
        model_text = getattr(new_output, "content", "")
        # Look for CALL_TOOL: tool_name markers in the model's reply
        matches = re.findall(r"CALL_TOOL:\s*(\w+)", model_text)
        for m in matches:
            tool_calls.append({"name": m, "arguments": {}})

        # Attach tool_calls to AIMessage
        new_output.tool_calls = tool_calls

        # --- DEBUG: show parsed tool calls ---
        print("DEBUG: Parsed tool_calls from model:")
        if new_output.tool_calls:
            for t in new_output.tool_calls:
                print(f"Tool Name: {t.get('name')}, Args: {t.get('arguments')}")
        else:
            print("No tool calls detected in model reply.")

    else:
        print("B")
        new_output = AIMessage(content=WELCOME_MSG, tool_calls=[])

    # Append model message
    updated_messages = state.get("messages", []) + [new_output]
    state = {**defaults, **state, "messages": updated_messages}

    # Automatically handle any tool calls returned by the model
    last_msg = updated_messages[-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        for tool_call in last_msg.tool_calls:
            tool_name = tool_call["name"]
            if tool_name == "add_to_info":
                content_to_add = state["messages"][-2]["content"]
                add_to_info([content_to_add])
                state["info"].append(content_to_add)
            elif tool_name == "confirm_info":
                confirm_info()
            elif tool_name == "create_JSON":
                create_JSON(state["info"])
                state["finished"] = True
            elif tool_name == "get_info":
                get_info()  # optional, can return info to model
            else:
                print(f"Tool {tool_name} is in schema but not implemented in Python.")

    return state


def create_node(state: infoState) -> infoState:
    """The create node. Executes tools in the last message if present."""
    tool_msg = state.get("messages", [])[-1]
    info = state.get("info", [])
    outbound_msgs = []
    created_JSON = False

    for tool_call in getattr(tool_msg, "tool_calls", []):
        if tool_call["name"] == "confirm_info":
            print("Your order:")
            if not info:
                print(" (no items)")
            for details in info:
                print(f" {details}")
            response = input("Is this correct? ")
        elif tool_call["name"] == "get_info":
            response = "\n".join(info) if info else "(no info)"
        elif tool_call["name"] == "create_JSON":
            print("Creating JSON...")
            print("\n".join(info))
            created_JSON = True
            response = randint(1,5)
        else:
            raise NotImplementedError(f"Unknown tool call: {tool_call['name']}")
        outbound_msgs.append(
            ToolMessage(
                content=response,
                name=tool_call["name"],
                tool_call_id=tool_call.get("id", 0),
            )
        )

    return {"messages": outbound_msgs, "info": info, "finished": created_JSON}

def maybe_route_to_tools(state: infoState) -> str:
    """Route between chat and the tool nodes if a tool call is made."""
    if not (msgs:= state.get("messages", [])):
        raise ValueError(f"No messages found when parsing state: {state}")
    
    msg = msgs[-1]

    if state.get("finished", True):
        return END
    elif hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
        if any(tool["name"] in tool_node.tools_by_name.keys() for tool in msg.tool_calls):
            return "tools"
        else:
            return "creating"
    else:
        return "human"


graph_builder = StateGraph(infoState)
graph_builder.add_node("chatbot", chatbot_with_tools)
graph_builder.add_node("human", human_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("creating", create_node)
graph_builder.add_conditional_edges("chatbot", maybe_route_to_tools)
graph_builder.add_conditional_edges("human", maybe_exit_human_node)
graph_builder.add_edge(START, "chatbot")
#graph_builder.add_edge("chatbot", "human")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("creating", "chatbot")

chat_graph = graph_builder.compile()

Image(chat_graph.get_graph().draw_mermaid_png())

# Invoke example
chat_graph.invoke({})
import re
from sys import intern
from annotated_types import IsDigit
import requests
import traceback
import json
from typing import Annotated, Literal
from requests.compat import integer_types
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from IPython.display import Image, display
from pprint import pprint
from langchain_core.messages import messages_to_dict
from collections.abc import Iterable
from random import randint

app = FastAPI()
chat_session = {}

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
    """
    Convert LangChain / internal messages to a list of dicts for the model.
    IMPORTANT: If the message is already a dict (our FastAPI path), pass it
    through WITHOUT stripping fields like tool_call_id, tool_calls, etc.
    """
    result = []
    for m in messages:
        if isinstance(m, dict):
            # Already in OpenAI style, just keep as is
            result.append(m)
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
            "name": "validate_info",
            "description": "Validate the input information from the user, and add all project details provided exactly from the user",
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
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def call_model(history, inputs):
    # Convert LangChain message objects (e.g. HumanMessage, AIMessage) to dicts
    if history and not isinstance(history[0], dict) and hasattr(history[0], "type"):
        history = messages_to_dict(history)

    payload = {
        "model": "unsloth/phi-4-mini-instruct",
        "messages": convert_messages_to_dict(history),
        "temperature": 0.0,
        "tools": tools_schema,
        # tool_choice could be "auto" or omitted depending on your backend
        # "tool_choice": "auto",
    }

    try:
        response = requests.post(
            "http://localhost:1234/v1/chat/completions", json=payload
        )
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        reply = message.get("content", "") or ""
        tool_calls = message.get("tool_calls", [])

        # Optional: keep original casing for inputs if you want
        for original in inputs.values():
            if isinstance(original, str) and original.lower() in reply.lower():
                reply = re.sub(re.escape(original), original, reply, flags=re.IGNORECASE)

        return reply, tool_calls
    except Exception as e:
        print(traceback.format_exc())
        if hasattr(e, 'response') and e.response is not None:
            print(e.response.text)
        return f"Error communicating with model: {e}", []


ASSISTANT_SYSINT = {
    "role": "system",
    "content": (
        "You are an AssistantBot, an interactive create project system.\n"
        "You will ask the human which action he want to perform.\n"
        "If human say they want to create project, "
        "directly call get_info tool to show the list that required to fill in by them.\n"
        "You must call validate_info tool to validate the information that human provided.\n"
        "If there are invalid information or missing values, you need to ask the human to correct the information.\n"
        "Never guess or fill missing values with placeholders.\n"
        "After validate_info, you MUST immediately call confirm_info.\n"
        "Wait for the human to agree with the details you show (e.g. they say 'yes', 'ok', 'correct') in confirm_info tool, then ONLY "
        "call create_JSON. You MUST call only once create_JSON once the human agrees.\n"
        "You need to return the created JSON to the user. \n"
        "If user requested to change any information during the confirm_info (example: change device name to halo),  \n"
        "you need to call again validate_info together with all latest project_info.\n"
        "Then, thank the user and say goodbye!\n"
        "WORKFLOW: get_info -> validate_info -> confirm_info -> create_JSON\n"
    )
}

WELCOME_MSG = "Welcome to the TechFlow Assistant Chatbot. Type 'q' to quit. What action do you want to do?"


def human_node(state: infoState) -> infoState:
    """Display the last model message to the user, and receive the user's input."""
    messages = state.get("messages", [])

    print("--------------------------------")
    print("Human Node")
    print("--------------------------------")

    if messages:
        for msg in reversed(messages):
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content", "")
            else:
                role = "assistant" if getattr(msg, "type", None) == "ai" else "user"
                content = getattr(msg, "content", "")

            if role == "assistant":
                print("Model:", content)
                break
    else:
        print("Model: (no message)")

    user_input = input("User: ")

    if user_input.lower() in {"q", "quit", "exit", "goodbye"}:
        state["finished"] = True
        state["messages"].append(HumanMessage(content="goodbye"))
    else:
        state["messages"].append(HumanMessage(content=user_input))

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
    print("Get info....")
    return """These are the information that you are required to fill in before you create your project.
- Site Count
- Offset Count
- Project Name
- Device Name
- Device Revision
- Programme Id
- Programme Revision
"""


@tool
def validate_info(
    site_count: str,
    offset_count: str,
    project_name: str,
    device_name: str,
    device_revision: str,
    programme_id: str,
    programme_revision: str
) -> str:
    """Validates the input information. There are total of 7 information:
    site_count, offset_count, project_name, device_name, device_revision,
    programme_id, programme_revision.
    """
    print("validating.....")
    errors = []

    try:
        site_count_int = int(site_count)
    except (ValueError, TypeError):
        errors.append("site_count must be an integer.")
        site_count_int = None

    try:
        offset_count_int = int(offset_count)
    except (ValueError, TypeError):
        errors.append("offset_count must be an integer.")
        offset_count_int = None

    if not isinstance(project_name, str) or not project_name.strip():
        errors.append("project_name is required.")
    if not isinstance(device_name, str) or not device_name.strip():
        errors.append("device_name is required.")
    if not isinstance(device_revision, str) or not device_revision.strip():
        errors.append("device_revision is required.")
    if not isinstance(programme_id, str) or not programme_id.strip():
        errors.append("programme_id is required.")
    if not isinstance(programme_revision, str) or not programme_revision.strip():
        errors.append("programme_revision is required.")

    if errors:
        return json.dumps({
            "result": "error",
            "errors": errors
        })

    PROJECT_INFO["site_count"] = site_count
    PROJECT_INFO["offset_count"] = offset_count
    PROJECT_INFO["project_name"] = project_name
    PROJECT_INFO["device_name"] = device_name
    PROJECT_INFO["device_revision"] = device_revision
    PROJECT_INFO["programme_id"] = programme_id
    PROJECT_INFO["programme_revision"] = programme_revision

    print("Values stored successfully")

    return json.dumps({
        "result": "ok"
    })


@tool
def confirm_info() -> str:
    """Asks the customer if the details are correct."""
    print("Confirm info")

    return f"""Current values:
- Site Count: {PROJECT_INFO.get("site_count")}
- Offset Count: {PROJECT_INFO.get("offset_count")}
- Project Name: {PROJECT_INFO.get("project_name")}
- Device Name: {PROJECT_INFO.get("device_name")}
- Device Revision: {PROJECT_INFO.get("device_revision")}
- Programme Id: {PROJECT_INFO.get("programme_id")}
- Programme Revision: {PROJECT_INFO.get("programme_revision")}

Is this information correct? (yes/no)
"""


@tool
def create_JSON() -> dict:
    """
    Create JSON for project creation after confirming all details from user.
    Retrieve details from global PROJECT_INFO variable and create JSON.
    Show the JSON to the user.
    """
    print("Create JSON tool called - Creating project JSON...")

    project_json = {
        "site_count": PROJECT_INFO.get("site_count"),
        "offset_count": PROJECT_INFO.get("offset_count"),
        "project_name": PROJECT_INFO.get("project_name"),
        "device_name": PROJECT_INFO.get("device_name"),
        "device_revision": PROJECT_INFO.get("device_revision"),
        "programme_id": PROJECT_INFO.get("programme_id"),
        "programme_revision": PROJECT_INFO.get("programme_revision"),
    }

    return project_json


tools = [get_info, validate_info, confirm_info, create_JSON]
tool_node = ToolNode(tools)


def call_model_with_tools(history, tools_schema=None):
    """
    Used in graph part (kept for completeness).
    """
    if history and hasattr(history[0], "type"):
        history = messages_to_dict(history)

    last_message = (
        getattr(history[-1], "content", "")
        if not isinstance(history[-1], dict)
        else history[-1].get("content", "")
    )

    reply_text, api_tool_calls = call_model(history, {"input": last_message})

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
                "id": tc.get("id", ""),
                "name": tool_name,
                "args": tool_args
            }
            formatted_tool_calls.append(formatted_tc)

    return AIMessage(content=reply_text, tool_calls=formatted_tool_calls)


def chatbot_with_tools(state: infoState) -> infoState:
    """
    Graph version (not used in FastAPI endpoints directly).
    """
    defaults = {"messages": [], "info": [], "finished": False}

    print("--------------------------------")
    print("Chatbot with Tools")
    print("--------------------------------")

    if state.get("messages"):
        new_output = call_model_with_tools([ASSISTANT_SYSINT] + state["messages"], tools_schema)
        print("Model reply:", getattr(new_output, "content", ""))
        if new_output.tool_calls:
            for t in new_output.tool_calls:
                print(f"Tool Name: {t.get('name')}, Args: {t.get('args')}, ID: {t.get('id')}")
        else:
            print("No tool calls in API response.")
    else:
        new_output = AIMessage(content=WELCOME_MSG, tool_calls=[])

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

    for msg in reversed(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call["name"] == "add_to_info":
                    args = tool_call.get("args", {})
                    summary = ", ".join(
                        f"{key}: {args.get(key, PROJECT_INFO.get(key, ''))}"
                        for key in PROJECT_INFO
                    )
                    if summary not in info:
                        info.append(summary)

                elif tool_call["name"] == "create_JSON":
                    finished = True

            break

    return {**state, "info": info, "finished": finished}


def create_node(state: infoState) -> infoState:
    """
    Final node: Create the JSON output using the collected info.
    """
    print("--------------------------------")
    print("Create Node")
    print("--------------------------------")

    project_json = {
        "site_count": PROJECT_INFO.get("site_count"),
        "offset_count": PROJECT_INFO.get("offset_count"),
        "project_name": PROJECT_INFO.get("project_name"),
        "device_name": PROJECT_INFO.get("device_name"),
        "device_revision": PROJECT_INFO.get("device_revision"),
        "programme_id": PROJECT_INFO.get("programme_id"),
        "programme_revision": PROJECT_INFO.get("programme_revision"),
    }

    return {
        "project_json": project_json,
        "finished": True
    }


def maybe_route_to_tools(state: infoState) -> str:
    """Route between chat and the tool nodes if a tool call is made."""
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
graph_builder.add_edge("creating", END)

chat_graph = graph_builder.compile()

Image(chat_graph.get_graph().draw_mermaid_png())

# Initialize the conversation state
initial_state = {"messages": [], "info": []}


# ----------------- New REST endpoints for frontend integration -----------------

def _ensure_session(chat_id: str):
    """Create a fresh session if not exists and reset PROJECT_INFO."""
    if chat_id not in chat_session:
        chat_session[chat_id] = {
            "messages": [],
            "final_data": None,
        }

        # Reset global project data for every new chat session
        PROJECT_INFO.update({
            "site_count": "",
            "offset_count": "",
            "project_name": "",
            "device_name": "",
            "device_revision": "",
            "programme_id": "",
            "programme_revision": "",
        })

        print("DEBUG: PROJECT_INFO reset for new session:", PROJECT_INFO)


def _execute_tool_call(name: str, args: dict, chat_id: str | None = None):
    """
    Execute a tool call by name and args.
    Returns the raw Python output (dict, str, etc.).
    """
    print(f"DEBUG: Executing tool {name} with args: {args}")

    # Find tool object
    tool_obj = None
    for t in tools:
        if t.name == name:
            tool_obj = t
            break

    if tool_obj is None:
        error_msg = f"Unknown tool {name}"
        print(f"DEBUG: {error_msg}")
        return error_msg

    try:
        # Tools with no args
        if name in ["get_info", "confirm_info", "create_JSON"]:
            out = tool_obj.invoke({})
        else:
            out = tool_obj.invoke(args)

        if name == "create_JSON" and chat_id:
            if isinstance(out, dict):
                chat_session[chat_id]["final_data"] = out

        return out

    except Exception as e:
        error_msg = f"Tool {name} error: {str(e)}"
        print(f"DEBUG: {error_msg}")
        return error_msg


@app.get("/chat/initial/{chat_id}")
def chat_initial(chat_id: str):
    """
    Initialize a chat session and return the assistant's initial message.
    """
    _ensure_session(chat_id)
    session = chat_session[chat_id]

    assistant_msg = {"role": "assistant", "content": WELCOME_MSG}
    session["messages"].append(assistant_msg)

    return JSONResponse({"response": "", "state": {"messages": session["messages"]}})


@app.post("/chat/{chat_id}")
def chat_api(chat_id: str, payload: dict):
    """
    Endpoint for frontend to send a user message and receive model reply.
    """
    if "text" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'text' in request body.")

    _ensure_session(chat_id)
    session = chat_session[chat_id]
    user_text = str(payload.get("text", "")).strip()

    # ---------------------------------
    # Append user message
    # ---------------------------------
    session["messages"].append({
        "role": "user",
        "content": user_text
    })

    def run_model_and_tools(input_text):
        """
        Helper: call model once and execute returned tools
        """
        history = [ASSISTANT_SYSINT] + session["messages"]
        reply_text, tool_calls = call_model(history, {"input": input_text})

        assistant_msg = {
            "role": "assistant",
            "content": reply_text,
        }
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls

        session["messages"].append(assistant_msg)

        final_reply = reply_text

        if not tool_calls:
            return final_reply

        for raw_tc in tool_calls:
            fn = raw_tc.get("function", {}) or {}
            tc_name = fn.get("name")
            tc_args = fn.get("arguments", {}) or {}
            tc_id = raw_tc.get("id")

            if isinstance(tc_args, str) and tc_args.strip():
                try:
                    tc_args = json.loads(tc_args)
                except:
                    tc_args = {}

            tool_output = _execute_tool_call(tc_name, tc_args, chat_id)

            tool_content = (
                json.dumps(tool_output, indent=2)
                if isinstance(tool_output, (dict, list))
                else str(tool_output)
            )

            session["messages"].append({
                "role": "tool",
                "tool_call_id": tc_id,
                "name": tc_name,
                "content": tool_content,
            })

            # -----------------------------
            # TOOL-SPECIFIC HANDLING
            # -----------------------------
            if tc_name == "get_info":
                final_reply = tool_content

            elif tc_name == "validate_info":
                try:
                    parsed = json.loads(tool_output) if isinstance(tool_output, str) else tool_output
                except Exception:
                    parsed = {"result": "error", "errors": ["Failed to parse validation result."]}

                if parsed.get("result") == "error":
                    final_reply = (
                        "There were some problems with your input:\n- "
                        + "\n- ".join(parsed.get("errors", []))
                    )
                    return final_reply

                # Show validation result to user
                session["messages"].append({
                    "role": "assistant",
                    "content": "Validation successful. Proceeding to confirmation..."
                })

                # Always re-run confirmation after any successful validation (even edits)
                if session.get("auto_confirm_in_progress"):
                    return final_reply

                session["auto_confirm_in_progress"] = True
                try:
                    return run_model_and_tools("ok")
                finally:
                    session["auto_confirm_in_progress"] = False

            elif tc_name == "confirm_info":
                final_reply = tool_content

            elif tc_name == "create_JSON":
                session["final_data"] = tool_output
                final_reply = (
                    "The techFlow project is generating in the background. "
                    "Please open techFlow later to check the created project."
                )

        return final_reply

    # ---------------------------------
    # START FLOW
    # ---------------------------------
    reply_text = run_model_and_tools(user_text)

    # ---------------------------------
    # FINAL RESPONSE
    # ---------------------------------
    return JSONResponse({
        "response": reply_text,
        "summary_json": session.get("final_data"),
        "state": {"messages": session["messages"]},
    })

@app.delete("/chat/{chat_id}")
def reset_chat(chat_id: str):
    if chat_id in chat_session:
        del chat_session[chat_id]

    # Reset PROJECT_INFO when session is cleared
    PROJECT_INFO.update({
        "site_count": "",
        "offset_count": "",
        "project_name": "",
        "device_name": "",
        "device_revision": "",
        "programme_id": "",
        "programme_revision": "",
    })


    return {"status": "cleared", "chat_id": chat_id}

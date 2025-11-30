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
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage
from IPython.display import Image
from langchain_core.messages import messages_to_dict
from typing import Literal


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
                "site_count": {"type": "integer"},
                "offset_count": {"type": "integer"},
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
            "name": "validate_info",
            "description": "Validate the input information from the user",
            "parameters": {
                "type": "object",
                "properties": {
                    "site_count": {"type": "integer"},
                    "offset_count": {"type": "integer"},
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
            "parameters": {"type": "object", "properties": {}, "required": []},  # Fixed: no parameters needed
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
        "You are an AssistantBot, an interactive create project system.\n"
        "You will ask the human which action he want to perform.\n"
        "If human say they want to create project, "
        "directly call get_info tool to show the list that required to fill in by them.\n"
        "You need to validate the details first by calling validate_info.\n"
        "If validation fails, ask the user to correct the information.\n"
        "If validation PASSES, you MUST immediately call add_to_info with the same parameters.\n"
        "After add_to_info succeeds, you MUST call confirm_info immediately.\n"
        "When user confirms the information is correct, call create_JSON.\n"
        "WORKFLOW: get_info -> validate_info -> add_to_info -> confirm_info -> create_JSON\n"
        "CRITICAL: After validate_info returns success, you MUST call add_to_info with the same exact parameters.\n"
    )
}

WELCOME_MSG = "Welcome to the TechFlow Assistant Chatbot. Type 'q' to quit. What action do you want to do?"

def human_node(state: infoState) -> infoState:
    """Display the last model message to the user, and receive the user's input."""
    messages = state.get("messages", [])

    print("--------------------------------")
    print("Human Node")
    print("--------------------------------")
    
    # Always show the last assistant message (if any)
    if messages:
        # Find the last assistant message to display
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

    # Always get user input when we reach human node
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

    print ("Get info....")
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
    site_count: int, 
    offset_count: int, 
    project_name: str, 
    device_name: str, 
    device_revision: str, 
    programme_id: str, 
    programme_revision: str
) -> str:
    """Validates the input information."""
    print("validating.....")
    errors = []

    # Validation rules for integers (using isinstance)
    if not isinstance(site_count, int):
        errors.append("site_count must be an integer.")
    if not isinstance(offset_count, int):
        errors.append("offset_count must be an integer.")
    
    # Validation for strings (checking for non-empty strings)
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

    # If there are errors, return the error message
    if errors:
        return f'{{"status":"error","message":"{", ".join(errors)}"}}'
    
    # If no errors, return success
    return '{"status":"success","message":"Validation passed."}'


@tool
def add_to_info( 
    site_count: int, 
    offset_count: int, 
    project_name: str, 
    device_name: str, 
    device_revision: str, 
    programme_id: str, 
    programme_revision: str
) -> str:
    """Adds the details to the particular information, with validation."""
    
    print("Add to info...") 

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
def create_JSON() -> dict:  # Remove parameters
    """
    Create JSON for project creation.\n
    After confirming all the details from user,
    You need to retrieve the details from global variable and create the JSON using the detail."\n
    Show the JSON to the user.
    """
    print ("Create JSON....")
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


tools = [get_info, validate_info, add_to_info, confirm_info, create_JSON]
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
        print("Model reply:", getattr(new_output, "content", ""))
        
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


def maybe_route_to_tools(state: infoState) -> str:
    """Route between chat and the tool nodes if a tool call is made."""
    msgs = state.get("messages", [])
    if not msgs:
        return "human"

    last_msg = msgs[-1]

    # Determine role of last message
    if isinstance(last_msg, dict):
        role = last_msg.get("role")
    else:
        msg_type = getattr(last_msg, "type", None)
        role = "user" if msg_type == "human" else "assistant"

    # If last message is from user → chatbot should respond
    if role == "user":
        return "chatbot"

    # If last message is from assistant and has tool calls → route to tools
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"

    # If last message is from assistant without tool calls → wait for human input
    return "human"


graph_builder = StateGraph(infoState)
graph_builder.add_node("chatbot", chatbot_with_tools)
graph_builder.add_node("human", human_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("update_state", update_state_after_tools)
graph_builder.add_conditional_edges("chatbot", maybe_route_to_tools)
graph_builder.add_conditional_edges("human", maybe_exit_human_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "update_state")
graph_builder.add_edge("update_state", "chatbot")
graph_builder.add_edge("update_state", END)

chat_graph = graph_builder.compile()

Image(chat_graph.get_graph().draw_mermaid_png())

# Initialize the conversation state
initial_state = {"messages": [], "info": []}

# ----------------- New REST endpoints for frontend integration -----------------

def _ensure_session(chat_id: str):
    """Create a fresh session if not exists."""
    if chat_id not in chat_session:
        chat_session[chat_id] = {
            "messages": [],     # list of dicts: {"role": "user"|"assistant"|"tool", "content": "..."}
            "final_data": None, # will hold final JSON when create_JSON executed
            "awaiting_confirmation": False,  # Track if we're waiting for user confirmation
        }

def _execute_tool_call(tool_call, chat_id=None):
    """
    Execute a tool call returned by the model.
    tool_call is expected to be a dict with keys: id, name, args (dict)
    Returns (tool_output_str_or_dict, tool_role_message_dict)
    """
    name = tool_call.get("name")
    args = tool_call.get("args", {}) or {}
    
    try:
        # Find the corresponding tool from the tools list
        tool_obj = None
        for tool in tools:
            if tool.name == name:
                tool_obj = tool
                break
        
        if tool_obj is None:
            error_msg = f"Unknown tool {name}"
            return error_msg, {"role": "tool", "content": error_msg}
        
        # Invoke the tool with the arguments
        if name == "get_info" or name == "confirm_info" or name == "create_JSON":
            # These tools take no arguments
            out = tool_obj.invoke({})
        else:
            # These tools take arguments
            out = tool_obj.invoke(args)
        
        # Special handling for create_JSON to store final data
        if name == "create_JSON" and chat_id:
            if isinstance(out, dict):
                chat_session[chat_id]["final_data"] = out
        
        # Convert output to string for tool message
        if isinstance(out, (dict, list)):
            tool_content = json.dumps(out)
        else:
            tool_content = str(out)
            
        return out, {"role": "tool", "content": tool_content}
        
    except Exception as e:
        error_msg = f"Tool {name} error: {e}"
        return error_msg, {"role": "tool", "content": error_msg}

@app.get("/chat/initial/{chat_id}")
def chat_initial(chat_id: str):
    """
    Initialize a chat session and return the assistant's initial message.
    """
    _ensure_session(chat_id)
    session = chat_session[chat_id]

    # Start by sending the assistant welcome message (no model call required)
    assistant_msg = {"role": "assistant", "content": WELCOME_MSG, "tool_calls": []}
    session["messages"].append(assistant_msg)

    return JSONResponse({"response": WELCOME_MSG, "state": {"messages": session["messages"]}})

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

    # Append user message
    user_msg = {"role": "user", "content": user_text}
    session["messages"].append(user_msg)

    # Build history to send to model
    history = [ASSISTANT_SYSINT] + session["messages"]

    # Call model to get a response
    reply_text, api_tool_calls = call_model(history, {"input": user_text})

    # Prepare assistant message dict
    assistant_msg = {"role": "assistant", "content": reply_text, "tool_calls": api_tool_calls}
    session["messages"].append(assistant_msg)

    # If model requested multiple tool calls, execute them sequentially
    final_data = session.get("final_data")
    tool_results = []
    validation_passed = False

    if api_tool_calls:
        print(f"DEBUG: Executing {len(api_tool_calls)} tool calls")
        
        # Iterate through all tool calls returned by the model
        for raw_tc in api_tool_calls:
            # Normalize tool call
            function_info = raw_tc.get("function", {})
            tc_name = function_info.get("name") or raw_tc.get("name")
            tc_args = function_info.get("arguments", {}) or raw_tc.get("arguments", {}) or {}

            if isinstance(tc_args, str) and tc_args.strip():
                try:
                    tc_args = json.loads(tc_args)
                except:
                    tc_args = {}

            normalized = {"id": raw_tc.get("id", ""), "name": tc_name, "args": tc_args}

            print(f"DEBUG: Executing tool {tc_name} with args: {tc_args}")
            tool_output, tool_message = _execute_tool_call({"name": normalized["name"], "args": normalized["args"]}, chat_id)

            # Append tool message to session messages
            session["messages"].append(tool_message)
            tool_results.append({"name": normalized["name"], "output": tool_output})

            # Handle tool responses
            if tc_name == "get_info":
                reply_text = tool_output if isinstance(tool_output, str) else str(tool_output)

            elif tc_name == "validate_info":
                # Handle both success and error cases
                if "error" in tool_output.lower():
                    reply_text = tool_output  # Return validation error
                    break  # Stop further execution
                else:
                    # Validation passed - continue with next tools
                    validation_passed = True
                    reply_text = "Validation passed! Adding your information..."

            elif tc_name == "add_to_info":
                if isinstance(tool_output, str) and "success" in tool_output.lower():
                    reply_text = "Information added successfully! Let me confirm the details..."
                    # Set flag to auto-confirm
                    session["awaiting_confirmation"] = True
                else:
                    reply_text = "There was an issue adding your information. Please check the details."

            elif tc_name == "confirm_info":
                reply_text = tool_output if isinstance(tool_output, str) else str(tool_output)

            elif tc_name == "create_JSON":
                if isinstance(tool_output, dict):
                    reply_text = f"Project JSON created successfully!\n{json.dumps(tool_output, indent=2)}"
                else:
                    reply_text = tool_output if isinstance(tool_output, str) else str(tool_output)

    # Handle auto-confirmation after add_to_info
    if session.get("awaiting_confirmation") is True:
        print("DEBUG: Auto-executing confirm_info tool after add_to_info")

        # Directly execute confirm_info tool
        tool_output, tool_message = _execute_tool_call({"name": "confirm_info", "args": {}}, chat_id)

        # Append tool message to session
        session["messages"].append(tool_message)
        tool_results.append({"name": "confirm_info", "output": tool_output})

        # Update reply text
        reply_text = tool_output if isinstance(tool_output, str) else str(tool_output)

        # Clear flag
        session["awaiting_confirmation"] = False

    # Update the assistant message with the final response text
    if assistant_msg in session["messages"]:
        session["messages"].remove(assistant_msg)
    assistant_msg["content"] = reply_text
    session["messages"].append(assistant_msg)

    # Return assistant reply and final JSON (if available)
    return JSONResponse({
        "response": reply_text,
        "summary_json": session.get("final_data"),
        "state": {"messages": session["messages"]},
        "tool_results": tool_results
    })





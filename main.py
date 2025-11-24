import json
import requests
import re
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()
chat_sessions = {}

def validate_number(value):
    try:
        v = int(value)
        return v > 0
    except:
        return False

def validate_no_special_chars(value):
    return bool(re.fullmatch(r'[A-Za-z0-9_.-]+', value))

question_info = {
    "site count": {"step": 1, "type": "number"},
    "offset count": {"step": 2, "type": "number"},
    "project name": {"step": 3, "type": "text"},
    "device name": {"step": 4, "type": "text"},
    "device revision": {"step": 5, "type": "text"},
    "programme id": {"step": 6, "type": "text"},
    "programme revision": {"step": 7, "type": "text"},
}

amend_verbs = ["amend", "change", "update", "modify"]

def detect_amend_field(user_message):
    """
    Detect if user wants to amend a field with flexible verbs.
    Returns (step_number, new_value) or (None, None)
    """
    user_message_lower = user_message.lower()
    for verb in amend_verbs:
        if verb in user_message_lower:
            for keyword, info in question_info.items():
                if keyword in user_message_lower:
                    # Extract value after 'to' or after the keyword itself
                    match = re.search(rf"{keyword}.*?to\s+(.+)", user_message, re.IGNORECASE)
                    if match:
                        return info["step"], match.group(1).strip()
                    # Fallback: extract word after keyword if no 'to'
                    fallback_match = re.search(rf"{keyword}\s+(.+)", user_message, re.IGNORECASE)
                    if fallback_match:
                        return info["step"], fallback_match.group(1).strip()
    return None, None

def validate_user_input_for_step(step_number, user_message):
    """
    Validate input based on step number.
    """
    for keyword, info in question_info.items():
        if info["step"] == step_number:
            if info["type"] == "number":
                if not validate_number(user_message):
                    return f"{keyword.title()} can be only number and should greater than 0"
            elif info["type"] == "text":
                if not validate_no_special_chars(user_message):
                    return f"{keyword.title()} cannot contain special characters"
    return None

first_instruction = """
You are a professional assistan bot.
You must complete 4 tasks in sequence.
Follow every rules and instruction strictly.
Do not explain anything unless explicitly instructed.
Do not summarize or add extra words.
Do not combine steps.
Do not guess any missing answers.
Always follow the order of tasks exactly.
Do not answer by yourselves.

TASK 1: Collect User Inputs

You must follow the flow to collect the details.
- You will start with "Please enter the site count:", then wait for user response.
- Save the user response as "siteCount".
- Continue with "Please enter the offset count:", then wait for user response.
- Save the user response as "offsetCount".
- Continue with "Please enter the project name:", then wait for user response.
- Save the user response as "projectName".
- Continue with "Please enter the device name:", then wait for user response.
- Save the user response as "deviceName".
- Continue with "Please enter the device revision:", then wait for user response.
- Save the user response as "deviceRevision".
- Continue with "Please enter the programme id:", then wait for user response.
- Save the user response as "programmeId".
- Continue with "Please enter the programme revision:", then wait for user response.
- Save the user response as "programmeRevision" and Task 1 completed.

RULES FOR TASK 1:
- Ask only one question at a time.
- Wait for the user's answer before continuing.
- Do not skip the flow.
- Do not answer behalf of the user.
- Do not add any examples or explanation.
- Always start with "Please enter the site count:" only.
- If user enter site count as "4", save it and continue with offset count.
- You must save exactly all user responses internally for later use.

TASK 2: Generate summary

After collecting all 7 inputs, you MUST show the summary in this exact format:

Summary of collected details:
Site count: "siteCount"
Offset count: "offsetCount"
Project name: "projectName"
Device name: "deviceName"
Device revision: "deviceRevision"
Programme id: "programmeId"
Programme revision" programmeRevision"

TASK 3: Ask for confirmation

After displayting the summary, ask:

"Is this above information correct? (yes/no)"

RULES FOR TASK 3:
- If the user answers "yes" -> proceed to Task 4
- If the user answers "no" -> ask: "Which details needs to be amended?"
- Do not suggest or assume the answer.
- Only amend details when the user specifies what to change.
- After amending, show the updated summary again and ask for confirmation again.
- Repeat until the user confirms the information is correct.

TASK 4: Generate JSON Output

After the user confirms the details are correct, generate only the final JSON in this exact structure:

{
"siteCount": user response,
 "offsetCount": user response,
 "projectName": user response,
 "deviceName": user response,
 "deviceRevision": user response,
 "programmeId": user response,
 "programmeRevision": user response
 }

IMPORTANT GLOBAL RULES

- Follow each task in order: Task 1 -> Task 2 -> Task 3 -> Task 4.
- Never skip or repeat a task unless required by correction.
- Please ask 7 question, do not skip or answer by yourself.
- Never add extra explanation, commentary or example.
- Respond exactly as instructed, word for word.
- Once confirmed, produce the JSON immediately and stop.

"""


# --- Model call ---
def call_model(history, inputs):
    payload = {
        "model": "unsloth/phi-4-mini-instruct",
        "messages": history,
        "temperature": 0.0,
    }

    try:
        response = requests.post(
            "http://localhost:1234/v1/chat/completions", json=payload  # Change path if using another host
        )
        response.raise_for_status()
        data = response.json()
        reply = data["choices"][0]["message"]["content"]

        # Preserve original casing for user inputs
        for original in inputs.values():
            if original.lower() in reply.lower():
                reply = re.sub(re.escape(original), original, reply, flags=re.IGNORECASE)

        return reply
    except Exception as e:
        return f"Error communicating with model: {e}"

# --- Nested JSON extractor ---
def extract_json_from_text(text):
    """
    Extract the first valid JSON object from text, handling nested braces.
    """
    start_idx = text.find('{')
    if start_idx == -1:
        return None

    stack = 0
    for i in range(start_idx, len(text)):
        if text[i] == '{':
            stack += 1
        elif text[i] == '}':
            stack -= 1
            if stack == 0:
                return text[start_idx:i+1]
    return None

# --- Endpoints ---
@app.get("/chat/initial/{chat_id}")
def chat_initial(chat_id: str):
    chat_sessions[chat_id] = {
        "history": [{"role": "system", "content": first_instruction}],
        "inputs": {}
    }

    history = chat_sessions[chat_id]["history"]
    inputs = chat_sessions[chat_id]["inputs"]

    history.append({"role": "user", "content": "start"})

    reply = call_model(history, inputs)
    history.append({"role": "assistant", "content": reply})
    return JSONResponse({"response": reply})


@app.post("/chat/{chat_id}")
def chat(chat_id: str, user_input: dict):

    if chat_id not in chat_sessions:
        return {"response": "Chat not found. Please start a new chat."}

    user_message = user_input.get("text", "").strip()
    if not user_message:
        return {"response": "Please enter a message."}

    history = chat_sessions[chat_id]["history"]
    inputs = chat_sessions[chat_id]["inputs"]

    # --- Check for amendment ---
    amend_step, amend_value = detect_amend_field(user_message)
    if amend_step:
        error_msg = validate_user_input_for_step(amend_step, amend_value)
        if error_msg:
            return {"response": error_msg}
        inputs[amend_step] = amend_value
        history.append({"role": "user", "content": user_message})
    else:
        step_number = len(inputs) + 1
        error_msg = validate_user_input_for_step(step_number, user_message)
        if error_msg:
            return {"response": error_msg}
        inputs[step_number] = user_message
        history.append({"role": "user", "content": user_message})

    reply = call_model(history, inputs)

    json_str = extract_json_from_text(reply)
    if json_str:
        try:
            parsed_json = json.loads(json_str)
            chat_sessions[chat_id]["final_data"] = parsed_json

            display_reply = (
                "The techFlow project is generating in the background. "
                "Please open techFlow and check the created project later."
            )
        except json.JSONDecodeError as e:
            display_reply = reply
    else:
        display_reply = reply

    history.append({"role": "assistant", "content": reply})

    final_data = chat_sessions[chat_id].get("final_data")
    return {
        "response": display_reply,
        "summary_json": final_data
    }
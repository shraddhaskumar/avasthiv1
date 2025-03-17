import streamlit as st
import requests

# Function to interact with FastAPI chatbot
def get_ai_response(query):
    url = "http://localhost:8000/query"  # Replace with your FastAPI endpoint URL
    payload = {"query": query}
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        response_json = response.json()
        # Extract the value part of the response
        answer = response_json['response'][0]['text']['value']
        return answer
    else:
        return "There was an error processing your request."


# Initialize session state if not already done
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Streamlit UI
st.title("AI Psychologist Chatbot")

# User input for the chatbot
user_input = st.text_input("Enter your question:")

# Display response if user input is given
if user_input:
    # Add user message to session state
    st.session_state.messages.append({"user": user_input})

    # Get AI response and add it to the session state
    ai_response = get_ai_response(user_input)
    st.session_state.messages.append({"ai": ai_response})

# Display messages
for message in st.session_state.messages:
    if 'user' in message:
        st.markdown(f"<div style='text-align:right; background-color:#D3F8E2; padding:10px; border-radius:10px; margin-bottom:10px;'>User: {message['user']}</div>", unsafe_allow_html=True)
    elif 'ai' in message:
        st.markdown(f"<div style='text-align:left; background-color:#F4F4F8; padding:10px; border-radius:10px; margin-bottom:10px;'>AI Psychologist: {message['ai']}</div>", unsafe_allow_html=True)
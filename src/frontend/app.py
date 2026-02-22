import sys
import os
import time
# Add project root to sys.path BEFORE any imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import requests
from src.frontend.config.frontend_settings import Settings

settings = Settings()

st.set_page_config(
    page_title="AstraRAG",
    page_icon="🤖",
    layout="centered",
)

# Add a sidebar with connection status
with st.sidebar:
    st.title("🔧 Connection Status")
    
    # Test backend connection
    try:
        test_response = requests.get(f"{settings.CHAT_ENDPOINT_URL.replace('/chat/answer', '')}/docs", timeout=2)
        if test_response.status_code == 200:
            st.success("✅ Backend connected")
        else:
            st.warning("⚠️ Backend responding but with issues")
    except:
        st.error(f"❌ Backend not reachable at {settings.CHAT_ENDPOINT_URL}")
        st.info("💡 Make sure to run: `python -m src.backend_src.main` in a separate terminal")
    
    st.divider()
    st.markdown("### About")
    st.markdown("AstraRAG is an Agentic RAG chatbot that uses retrieval-augmented generation to answer questions based on your documents.")

st.title("💬 AstraRAG - Agentic RAG Chatbot")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("role") == "assistant":
            sources = message.get("sources", [])
            tool_used = message.get("tool_used")
            rationale = message.get("rationale")
            if sources:
                st.markdown(f"**Sources:** {', '.join(sources)}")
            if tool_used or rationale:
                with st.expander("Show details (tool & rationale)"):
                    st.markdown(f"**Tool Used:** {tool_used if tool_used else 'N/A'}")
                    st.markdown(f"**Rationale:** {rationale if rationale else 'N/A'}")

# Chat input
user_prompt = st.chat_input("Ask Chatbot...")

if user_prompt:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_prompt)
    
    # Add to session state
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    
    # Prepare payload for API
    payload = {"chat_history": st.session_state.chat_history}
    
    # Show a spinner while waiting for response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Make API request with timeout
                response = requests.post(
                    settings.CHAT_ENDPOINT_URL, 
                    json=payload, 
                    timeout=30  # 30 second timeout
                )
                response.raise_for_status()
                response_json = response.json()
                
                assistant_response = response_json.get("answer", "(No response)")
                tool_used = response_json.get("tool_used", "N/A")
                rationale = response_json.get("rationale", "N/A")
                sources = response_json.get("sources", [])
                
            except requests.exceptions.ConnectionError:
                assistant_response = "❌ **Cannot connect to the backend server.**\n\nPlease make sure the FastAPI server is running in a separate terminal with:\n```bash\npython -m src.backend_src.main --reload\n```"
                tool_used = "Connection Error"
                rationale = "The frontend cannot reach the backend API. This usually happens when the FastAPI server isn't running."
                sources = []
                st.error("🔴 Connection Failed - Check if backend is running")
                
            except requests.exceptions.Timeout:
                assistant_response = "❌ **Request timed out.**\n\nThe backend took too long to respond. Please try again."
                tool_used = "Timeout Error"
                rationale = "The API request exceeded the 30-second timeout limit."
                sources = []
                st.error("⏱️ Request Timeout")
                
            except requests.exceptions.HTTPError as e:
                assistant_response = f"❌ **HTTP Error:** {str(e)}"
                tool_used = "HTTP Error"
                rationale = f"The API returned an HTTP error: {str(e)}"
                sources = []
                st.error(f"🌐 HTTP Error: {e.response.status_code if e.response else 'Unknown'}")
                
            except Exception as e:
                assistant_response = f"❌ **Error:** {str(e)}"
                tool_used = "Unknown Error"
                rationale = f"An unexpected error occurred: {str(e)}"
                sources = []
                st.error(f"❌ Unexpected Error: {type(e).__name__}")
        
        # Display assistant response
        st.markdown(assistant_response)
        if sources:
            st.markdown(f"**Sources:** {', '.join(sources)}")
        
        # Show details in expander
        with st.expander("Show details (tool & rationale)"):
            st.markdown(f"**Tool Used:** {tool_used}")
            st.markdown(f"**Rationale:** {rationale}")
    
    # Add to session state
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": assistant_response,
        "tool_used": tool_used,
        "rationale": rationale,
        "sources": sources
    })
    
    # Force a rerun to update the chat display
    st.rerun()
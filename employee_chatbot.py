pimport streamlit as st
import openai
import json
import time  # ✅ FIXED: Import time at the top
from openai import OpenAI
from openai import RateLimitError

# ✅ TESTING: Check API key accessibility with proper error handling
try:
    print("🔍 Checking Streamlit secrets...")
    if hasattr(st, 'secrets'):
        print("✅ st.secrets is available")
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
            print(f"🔑 API Key found! Length: {len(api_key)}")
            print(f"🔑 API Key (first 10 chars): {api_key[:10]}")
            # ✅ NEW: Use OpenAI client setup
            client = OpenAI(api_key=api_key)
        else:
            print("❌ OPENAI_API_KEY not found in st.secrets")
            print("📋 Available secrets keys:", list(st.secrets.keys()) if st.secrets else "None")
            client = None
    else:
        print("❌ st.secrets is not available")
        client = None
except Exception as e:
    print(f"❌ Error accessing secrets: {str(e)}")
    client = None

# ✅ Load the JSON data
# with open('employee_info.json') as f:
#     employee_data = json.load(f)
    #employee_info.json
with open('employee_info.json') as f:
    employee_data = json.load(f)

# ✅ Updated GPT-3.5-turbo call using OpenAI v1.x SDK
def query_gpt3(prompt):
    if client is None:
        return "❌ OpenAI client not initialized. Please check your API key configuration."
    
    try:
        print("🚀 Making OpenAI API call...")  # ✅ TESTING: Debug print
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        print("✅ OpenAI API call successful!")  # ✅ TESTING: Debug print
        return response.choices[0].message.content.strip()
    
    except RateLimitError as e:
        print(f"⚠️ Rate limit error: {str(e)}")  # ✅ TESTING: Debug print
        time.sleep(3)
        return "⚠️ Too many requests to OpenAI. Please wait a moment and try again."
    
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")  # ✅ TESTING: Debug print
        return f"❌ An unexpected error occurred: {str(e)}"
        
# ✅ Streamlit UI
st.title("Please Enter a Question")
st.title("Splan Information Chatbot")
user_input = st.text_input("Ask a question about the employees:")

if st.button("Submit"):
    if user_input:
        # Create a prompt with the user question and the employee data
        prompt = f"Given the following employee data: {json.dumps(employee_data, indent=2)}\n\nAnswer the following question:\n{user_input}"

        with st.spinner("Generating your answer, please wait..."):
            # Optional delay
            time.sleep(1.5)

            # Query GPT-3.5-turbo
            answer = query_gpt3(prompt)

        st.success("Answer generated successfully!")

        # Display the answer
        st.write("Answer:")
        st.write(answer)
    else:
        st.write("Please enter a question.")

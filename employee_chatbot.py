import streamlit as st
import openai
import json
from openai import OpenAI
from openai import RateLimitError

# ✅ NEW: Use OpenAI client setup (instead of deprecated openai.api_key)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  # ✅ Secure best practice

# ✅ Load the JSON data
with open('employee_info.json') as f:
    employee_data = json.load(f)

# ✅ Updated GPT-3.5-turbo call using OpenAI v1.x SDK
def query_gpt3(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    
    except RateLimitError:
        time.sleep(3)
        return "⚠️ Too many requests to OpenAI. Please wait a moment and try again."
    
    except Exception as e:
        return f"❌ An unexpected error occurred: {str(e)}"
        
# ✅ Streamlit UI
st.title("Please Enter a Question")
st.title("Employee Information Chatbot")
user_input = st.text_input("Ask a question about the employees:")

if st.button("Submit"):
    if user_input:
        # Create a prompt with the user question and the employee data
        prompt = f"Given the following employee data: {json.dumps(employee_data, indent=2)}\n\nAnswer the following question:\n{user_input}"

        with st.spinner("Generating your answer, please wait..."):
            # Optional delay
            import time
            time.sleep(1.5)

            # Query GPT-3.5-turbo
            answer = query_gpt3(prompt)

        st.success("Answer generated successfully!")

        # Display the answer
        st.write("Answer:")
        st.write(answer)
    else:
        st.write("Please enter a question.")

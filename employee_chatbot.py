import streamlit as st
import openai
import json
from openai import RateLimitError

# ✅ NEW: Use OpenAI client setup (instead of deprecated openai.api_key)
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  # ✅ Secure best practice

# ✅ Load the JSON data
with open('employee_info.json') as f:
    employee_data = json.load(f)

# ✅ Updated GPT-3.5-turbo call using OpenAI v1.x SDK
def query_gpt3(prompt):
try:   
    response = client.chat.completions.create(  # ✅ changed from openai.ChatCompletion.create
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()  # ✅ changed syntax (no more ['content'])
except RateLimitError:
        time.sleep(3)
        return "⚠️ We're currently sending too many requests. Please wait a moment and try again."

# ✅ Streamlit UI
st.title("Employee Information Chatbot")

user_input = st.text_input("Ask a question about the employees:")

if st.button("Submit"):
    if user_input:
        # Create a prompt with the user question and the employee data
        prompt = f"Given the following employee data: {json.dumps(employee_data, indent=2)}\n\nAnswer the following question:\n{user_input}"
        st.info("Generating your answer, please wait...")
        time.sleep(1.5)
        # Query GPT-3.5-turbo
        answer = query_gpt3(prompt)
        
        # Display the answer
        st.write("Answer:")
        st.write(answer)
    else:
        st.write("Please enter a question.")

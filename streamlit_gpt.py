import streamlit as st
from streamlit_chat import message
import requests, asyncio, aiohttp
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

st.set_page_config(
    page_title="Wonderful Bio Chat - Demo",
    page_icon=":robot:"
)

st.header("Streamlit Chat - Demo")
st.markdown("[Github](https://github.com/naem1023/wonderful-bio-chatbot)")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

async def query(prompt):
    data = {'text': prompt}
    async with aiohttp.ClientSession() as session:
        chat_url = 'http://192.168.5.21:5000/predict'
        async with session.get(chat_url, json=data) as resp:
            res = await resp.json()
            print('in query', res)
            return res
	# response = requests.post(API_URL, headers=headers, json=payload)
	# return response.json()

def get_text():
    input_text = st.text_input("You: ","Hello, how are you?", key="input")
    return input_text 


async def update_chat():
    user_input = get_text()

    if user_input:
        # output = query({
        #     "inputs": {
        #         "past_user_inputs": st.session_state.past,
        #         "generated_responses": st.session_state.generated,
        #         "text": user_input,
        #     },"parameters": {"repetition_penalty": 1.33},
        # })

        output = await query(user_input)

        print('in update_chat', output)

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

asyncio.run(update_chat())
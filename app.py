from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import AIMessage, HumanMessage
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
import os
from langchain import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import pickle
import streamlit as st

st.title('Stephen Hawking - The Theory of everything. A brief history of time')



api_key = st.sidebar.text_input('Enter OpenAI API key')
if api_key:
  with st.spinner('Initialize LLM, database, memory'):
    os.environ['OPENAI_API_KEY'] = api_key
    with open(r'/content/drive/MyDrive/Langchain/stephawk_db.pkl', 'rb') as f:
      toe_db1 = pickle.load(f)
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)

def Answer(query):
  conversation_chain = ConversationalRetrievalChain.from_llm(llm = llm, memory = memory, retriever = toe_db1.as_retriever())
  ans = conversation_chain({'question': query}, return_only_outputs = True)
  return ans['answer']

if "messages" not in st.session_state:
    st.session_state.messages = []

query = st.chat_input('Enter query')
if query:
    st.chat_message('Human').write(query)

    answer = Answer(query)
    st.chat_message('AI').write(answer)

    st.session_state.messages.append({"human": query, "AI": answer})

if st.sidebar.button('Chat history'):
    for i in range(0, len(st.session_state.messages)):
        st.chat_message('Human').write(st.session_state.messages[i]['human'])
        st.chat_message('AI').write(st.session_state.messages[i]['AI'])




import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from utils import *
from streamlit_chat import message




st.title("Arabic Rag")

if 'db' not in st.session_state:
    st.session_state['db'] = []
if 'response' not in st.session_state:
    st.session_state['response'] = []


if not st.session_state['db']:
    print("again")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    st.session_state['db']= load_local_vectordb_using_qdrant("testing_arabic", embeddings)


user_prompt = st.text_input('Enter Your Query here..........')

    # Button to trigger the story generation
if st.button("Press Enter"):
    st.session_state['response'] = arabic_qa(user_prompt, st.session_state['db'])


if st.session_state['response']:
    message(st.session_state['response'])
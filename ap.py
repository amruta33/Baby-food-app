## Integrate our code OpenAI API
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

from langchain.memory import ConversationBufferMemory

from langchain.chains import SequentialChain

import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

# streamlit framework

st.title('Celebrity Search Results')
input_text=st.text_input("Please input your baby's age in months:")

# Prompt Templates

first_input_prompt=PromptTemplate(
    input_variables=['Months'],
    template="Prepare a daily food chart for {Months} months old baby:"
)

# Memory

person_memory = ConversationBufferMemory(input_key='Months', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='food', memory_key='chat_history')
#descr_memory = ConversationBufferMemory(input_key='season', memory_key='description_history')

## OPENAI LLMS
llm=OpenAI(temperature=0.8)
chain=LLMChain(
    llm=llm,prompt=first_input_prompt,verbose=True,output_key='food',memory=person_memory)

# Prompt Templates

second_input_prompt=PromptTemplate(
    input_variables=['food'],
    template="Prepare table chart Based on the {food}, which type of clothing should use?"
)


# Prompt Templates


chain2=LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key='description',memory=dob_memory)
parent_chain=SequentialChain(
    chains=[chain,chain2],input_variables=['Months'],output_variables=['food','description'],verbose=True)


if input_text:
    st.write(parent_chain({'Months':input_text}))

    with st.expander('Baby food chart'): 
        st.info(person_memory.buffer)




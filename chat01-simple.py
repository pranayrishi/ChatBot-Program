# to run the program type "chainlit run chat01_simple.py"
import asyncio
import os
from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
import chainlit as cl

load_dotenv(find_dotenv())

#print(f"{ os.environ['MY_SPECIAL_KEY']}")

MODEL_NAME = "text-davinci-003"
OPEN_API_KEY = os.environ["OPENAI_API_KEY"]

@cl.on_message
def main(message: str):
    response = OpenAI(openai_api_key=OPEN_API_KEY, 
                      temperature=0, # 2.0 wild for temperature range and 0 is nicer
                      model=MODEL_NAME)(message)

    asyncio.run(
        cl.Message(
        content=response
        ).send()
    )

@cl.on_chat_start
def start():
    asyncio.run(
        cl.Message(
        content="Hello there!"
        ).send()
    )
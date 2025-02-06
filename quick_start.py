import os

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

print(api_key is not None)

class State(TypedDict):

    messages: Annotated[list, add_messages]

graph_bulider = StateGraph(State)



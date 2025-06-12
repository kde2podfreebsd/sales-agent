from typing import Callable
from langchain.schema import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.memory import CombinedMemory
from langchain.agents import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from .output_parser import FixingOutputParser 

AgentBuilder = Callable[[CombinedMemory, ChatOpenAI], AgentExecutor]

import os
import sys
from io import StringIO

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

class OutputCapture(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


def execute_python_code(python_code):
    with OutputCapture() as output:
        exec(python_code)
    return output

# Set up a new OpenAI instance
model = ChatOpenAI(
    temperature=0,
    model_name='gpt-3.5-turbo',
)

script_tool = Tool(
    name='Python',
    description='Runs the specified python code (in an exec call).',
    func= execute_python_code,
)

memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=3,
    return_messages=True
)

agent = initialize_agent(
    agent='chat-conversational-react-description',
        verbose=True,
    tools=[script_tool],
    llm=model,
    max_iterations=3,
    early_stopping_method='generate',
    memory=memory
)

response = agent("Create me a hello world python file called main.py.")
print(response)

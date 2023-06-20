import unittest

from langchain import PromptTemplate
from langchain.agents import AgentExecutor
from langchain.llms import FakeListLLM
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.tools import Tool

from yid_langchain_extensions.agent.simple_agent import SimpleAgent
from yid_langchain_extensions.output_parser.action_parser import ActionParser
from yid_langchain_extensions.tools.utils import format_tools


class TestThoughtsJSONParser(unittest.TestCase):
    def test_simple_agent(self):
        answers = [
            '```json\n{\n\t"action": "check_weather",\n\t"action_input": "Moscow"\n',
            '```json\n{\n\t"action": "final_answer",\n\t"action_input": "In Moscow rainy with a temperature of 10°C."\n'
        ]
        llm = FakeListLLM(responses=answers)
        weather_tool = Tool(
            name="check_weather", description="Use it to check weather at some location",
            func=lambda x: "Rain, 10 C"
        )
        tools = [weather_tool, Tool(
            name="final_answer",
            description="Use this if you want to respond directly to the human.",
            func=lambda x: x, return_direct=True)]
        output_parser = ActionParser.from_extra_thoughts([])
        template = ChatPromptTemplate.from_messages(
            messages=[
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{{input}}", "jinja2"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
                SystemMessage(content=format_tools(tools)),
                SystemMessage(content=PromptTemplate.from_template(
                    output_parser.get_format_instructions(), template_format="jinja2").format(
                    tool_names=", ".join([tool.name for tool in tools])
                ))
            ]
        )
        agent = SimpleAgent.from_llm_and_prompt(
            llm=llm, prompt=template, output_parser=output_parser, stop_sequences=output_parser.stop_sequences
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools)
        answer = agent_executor.run(input="What is the weather in Moscow?", chat_history=[])
        self.assertEqual(answer, "In Moscow rainy with a temperature of 10°C.")

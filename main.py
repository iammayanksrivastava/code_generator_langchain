from langchain.chains import LLMChain, SequentialChain
from langchain_core.prompts import PromptTemplate
import argparse
from langchain_openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

llm = OpenAI()

code_prompt = PromptTemplate(
    template = "Write a {language} function that {task}.",
    input_variables = ["language", "task"]
)

test_prompt = PromptTemplate(
    template = "Write a test function in {language} for the following {language} function:\n{code}",
    input_variables = ["language", "code"]
)


test_chain = LLMChain(llm=llm, prompt=test_prompt, output_key="test")

code_chain = LLMChain(llm=llm, prompt=code_prompt, output_key="code")

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["language", "task"],
    output_variables=["code", "test"]
    
    )

result = chain({
    "language": args.language,
    "task": args.task
})

print('>>>>>>>>>> Generated Code:')
print(result["code"])

print('>>>>>>>>>> Generated Test:')
print(result["test"])


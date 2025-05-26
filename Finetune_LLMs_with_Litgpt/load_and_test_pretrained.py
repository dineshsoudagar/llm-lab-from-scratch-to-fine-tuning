from litgpt import LLM

llm = LLM.load("microsoft/phi-2")

text = llm.generate("Every effort moves you to")
import json

from tqdm import tqdm


def convert_to_new_format(data):
    """Extracting question, context, supporting facts, and answer from the input data"""
    question = data.get("question", "")
    context = "\n\n".join(["\n".join(item[1]) for item in data.get("context", [])])
    supporting_facts = data.get("supporting_facts", [])
    answer = data.get("answer", "")

    # Creating the new format
    new_format = {
        "context": context,
        "question": question,
        "answer": answer
    }

    return new_format


output_data = []
with open(r"C:\Others\LLMs\litgpt_scripts\v1.0-simplified_simplified-nq-train.jsonl\simplified-nq-train.jsonl", "r") as hotpot:
    data = [json.loads(line) for line in hotpot]
    for example in tqdm(data):
        print(example)
        output_data.append(convert_to_new_format(example))
        break

with open("simplified-nq-train.json", "w") as converted:
    json.dump(output_data, converted, indent=4)

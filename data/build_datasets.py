import json
import re
from openai import OpenAI
import os
from tqdm import tqdm

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open('WikiMhQA_prompt.txt', 'r') as f:
    task_prompt = f.read()

def generate_chain(question, facts, model="gpt-4.1"):
    prompt = task_prompt.format(question=question, facts=facts)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a data generation assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2)
    raw_output = response.choices[0].message.content.strip()
    data_list = json.loads(raw_output)
    return data_list


def convert_to_target_format(question, data_list, answer):
    answer_alias = []
    sub_answer = []
    sub_question = []
    facts = []
    for data in data_list:
        facts.append(data["Fact"])
        sub_question.append(data["Sub-question"])
        sub_answer.append(data["Sub-answer"])

    return {
        "question": question,
        "answer": answer,
        "answer_alias": answer_alias,
        "sub_question": sub_question,
        "sub_answer": sub_answer,
        "facts": facts
    }


def save_to_json(samples, filename="train.json"):
    with open(filename, "without Hypernetwork", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=4)
    print(f"✅ Saved {len(samples)} samples to {filename}")


# =======================
# Example usage
# =======================
if __name__ == "__main__":
    with open("/disk/liuxb/code/Multi-EMoE/data/WikiMhQA_pre_train.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)  # 测试前4条

    results = []
    for ix, sample in enumerate(tqdm(dataset)):
        try:
            chain = generate_chain(sample["question"], sample["facts"])
            converted = convert_to_target_format(
                question=sample["question"],
                data_list=chain,
                answer=sample["answer"])
            results.append(converted)
        except Exception as e:
            print("skip {}".format(ix))

    print(len(results))
    save_to_json(results, "../datasets/WikiMhQA_train.json")

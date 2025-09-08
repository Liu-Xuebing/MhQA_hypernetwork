from torch.utils.data import Dataset, DataLoader
import json
import torch
from torch.nn.utils.rnn import pad_sequence
import random
random.seed(42)

class Decomposer_Dataset(Dataset):
    def __init__(self, config, tok):
        self.config = config
        self.tok = tok
        self.data = []

        if config.data_name in ["MQuAKE", "MQuAKE-T"]:
            with open(config.MQuAKE_train_dataset) as train_data:
                datas = json.load(train_data)
            for data in datas:
                question = data["question"]
                sub_questions = data["sub_question"]
                sub_answers = data["sub_answer"]
                new_sub_question = sub_questions[:1]
                for sq,sa in zip(sub_questions[1:], sub_answers[:-1]):
                    new_sub_question.append(sq.replace(sa, '{}', 1))
                self.data.append((question, new_sub_question))

        if config.data_name == 'hotpot':
            with open(config.HotPot_train_dataset) as train_data:
                datas = json.load(train_data)
            for data in datas:
                for q, a, p in zip(data['sub_question'], data['sub_answer'], data['facts']):
                    self.data.append([q, a, p])


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        question, sub_questions  = self.data[idx]
        input = ('Decompose the following question into sub-questions:\n'
                 '{}\n'
                 'Sub-questions:').format(question)
        answer = '\n'.join(["{}".format(sq) for sq in sub_questions]) +'\n\n'
        tok_tuples = self.tok_tuples(input, answer)
        return tok_tuples



    def tok_tuples(self, prompt, answer):
        if self.config.model_name == 'tiiuae/Falcon3-1B-Base':
            answer = " "+answer
        else:
            raise AssertionError("Error model")

        tok_prompt = self.tok(prompt, return_tensors="pt")
        tok_answer = self.tok(answer, return_tensors="pt", add_special_tokens=False)

        tok_tuples = {
            key: torch.cat((value, tok_answer[key][:, :-1]), -1)
            for key, value in tok_prompt.items()
        }
        tok_tuples["labels"] = torch.cat((
            torch.full(tok_prompt["input_ids"].shape, -100)[:, 1:],
            tok_answer["input_ids"]
        ), -1)
        return tok_tuples


    def collate_fn(self, tuples):
        tok_tuples = [item for item in tuples]
        padded_tokens_wo = {k: pad_sequence([t[k].squeeze(0) for t in tok_tuples],
                                         batch_first=True,
                                         padding_value=-100 if k == "labels" else 0).cuda()
                         for k in tok_tuples[0].keys()}
        return padded_tokens_wo


def make_Training_loader(config, tok, samples = None):
    train_set = Decomposer_Dataset(config, tok)
    train_loader = DataLoader(train_set, batch_size=config.train_batch_size, shuffle=True, collate_fn=train_set.collate_fn)
    return train_loader
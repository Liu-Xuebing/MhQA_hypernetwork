from torch.utils.data import Dataset, DataLoader
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from utils import first_word_cap
import random
random.seed(42)

class NQ_TQA_SQuAD_Dataset(Dataset):
    def __init__(self, config, tok, status, samples):
        self.config = config
        self.tok = tok
        self.status = status
        self.samples = samples
        self.data = []
        if status=='Train':
            if config.data_name == 'scanning':
                with open(config.layer_scanning_datasets) as train_data:
                    datas = json.load(train_data)
                for data in datas:
                    self.data.append([data['question'], data['answer'], data['text']])
            elif config.data_name == 'MQuAKE' or config.data_name == 'MQuAKE-T':
                with open(config.train_dataset) as train_data:
                    datas = json.load(train_data)
                for data in datas:
                    for q, a, p in zip(data['sub_question'], data['sub_answer'], data['facts']):
                        self.data.append([q, a, p])
            elif config.data_name in ['HotPot', 'WikiMhQA', 'musique']:
                with open(config.train_dataset.format(config.data_name)) as train_data:
                    datas = json.load(train_data)
                for data in datas:
                    prompt = ''
                    for q, a, p in zip(data['sub_question'], data['sub_answer'], data['facts']):
                        self.data.append([q, a, p])
                        prompt += "Sub-question: {}\nSub-answer: {}\n".format(q, a)
                    self.data.append([data['question'], data['answer'], prompt.strip()])
            if self.samples:
                self.data = random.sample(self.data, k=self.samples)


        elif status == "Test":
            if config.data_name in ['MQuAKE-CF', 'MQuAKE-T', 'HotPot', 'WikiMhQA', 'musique']:
                with open(config.test_dataset.format(config.data_name)) as test_data:
                    datas = json.load(test_data)
                for data in datas:
                    if 'answer_alias' not in data.keys():
                        data['answer_alias'] = []
                        data['answer_alias'].append(data['answer'])
                    else:
                        data['answer_alias'].append(data['answer'])
                    self.data.append([data['question'], data['answer_alias'], data['facts']])
            else:
                raise AssertionError("Error dataset")


        else:
            raise AssertionError("Error state")



    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        if self.status == 'Train':
            question, answer, passage = self.data[idx]
            input = 'Question: {}\nAnswer:'.format(question)
            answer = '{}'.format(answer)
            tok_tuples, tok_sentence = self.tok_tuples(input, answer, passage)
            return tok_tuples, tok_sentence
        elif self.status == 'Test':
            question, answers, passage = self.data[idx]
            return (question.strip(), answers, passage)



    def tok_tuples(self, prompt, answer, passage):
        if self.config.model_name == "meta-llama/Llama-3.1-70B":
            answer = " " + answer + self.tok.eos_token
        elif self.config.model_name == "Qwen/Qwen2.5-7B":
            answer = answer + self.tok.eos_token
        else:
            raise AssertionError("Error model")

        tok_prompt = self.tok(prompt, return_tensors="pt")
        tok_answer = self.tok(answer, return_tensors="pt", add_special_tokens=False)
        tok_sentence = self.tok(passage, return_tensors="pt")

        tok_tuples = {
            key: torch.cat((value, tok_answer[key][:, :-1]), -1)
            for key, value in tok_prompt.items()
        }
        tok_tuples["labels"] = torch.cat((
            torch.full(tok_prompt["input_ids"].shape, -100)[:, 1:],
            tok_answer["input_ids"]
        ), -1)
        return tok_tuples, tok_sentence


    def collate_fn(self, tuples):
        tok_tuples = [item[0] for item in tuples]
        tok_sentence = [item[1] for item in tuples]
        padded_tokens_wo = {k: pad_sequence([t[k].squeeze(0) for t in tok_tuples],
                                         batch_first=True,
                                         padding_value=-100 if k == "labels" else 0).cuda()
                         for k in tok_tuples[0].keys()}
        return padded_tokens_wo, tok_sentence


    def val_collate_fn(self, tuples):
        return (tuples[0][0],
                [t for t in tuples[0][1]],
                [t for t in tuples[0][2]])



def make_Training_loader(config, tok, samples = None):
    train_set = NQ_TQA_SQuAD_Dataset(config, tok, status='Train', samples = samples)
    train_loader = DataLoader(train_set, batch_size=config.train_batch_size, shuffle=True, collate_fn=train_set.collate_fn)
    return train_loader


def make_Validation_loader(config, tok):
    valid_set = NQ_TQA_SQuAD_Dataset(config, tok, status='Test', samples=None)
    valid_loader = DataLoader(valid_set, batch_size=config.valid_batch_size, shuffle=False, collate_fn=valid_set.val_collate_fn)
    return valid_loader
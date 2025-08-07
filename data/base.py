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
            if config.data_name == 'NQ':
                with open(config.pre_train_NQ_datasets) as train_data:
                    datas = json.load(train_data)
                for data in datas:
                    for p in data['ctxs']:
                        if p['has_answer']:
                            for answer in data['answers']:
                                if answer in p['text']:
                                    self.data.append([data['question'], answer, p['text']])
                                    break
                            break
                if self.samples is not None:
                    self.data = random.sample(self.data, k=self.samples)

            elif config.data_name == 'TQA':
                with open(config.pre_train_TQA_datasets) as train_data:
                    datas = json.load(train_data)
                for data in datas:
                    for p in data['ctxs']:
                        if p['has_answer']:
                            for answer in data['answers']:
                                if answer in p['text']:
                                    self.data.append([data['question'], answer, p['text']])
                                    break
                            break
                if self.samples is not None:
                    self.data = random.sample(self.data, k=self.samples)

            elif config.data_name == 'SQuAD':
                '''SQuAD structure:
                Dict: [version, data]
                    Str: version
                    List: data
                        Dict: [title, paragraphs]
                            Str: title
                            List: paragraphs
                                Dict: [qas, context]
                                    List: qas
                                        Dict: [question, id, answers, is_impossible]
                                            Str: question
                                            Str: id
                                            List: answers
                                                Dict: [text, answer_start]
                                                    Str: text
                                                    Num: answer_start
                                            Bool: is_impossible
                                    Str: context'''
                with open(config.SQuAD_train_dataset) as train_data:
                    datas = json.load(train_data)
                for i in datas['data']:
                    for j in i['paragraphs']:
                        paragraph = j['context']
                        random_choice_qas = random.choice(j['qas'])
                        if random_choice_qas['is_impossible']:
                            continue
                        for a in random_choice_qas['answers']:
                            if a['text'] in paragraph:
                                self.data.append([random_choice_qas['question'], a['text'], paragraph])
                                break
                self.data *= 2
            elif config.data_name == 'Narrative':
                with open(config.Narrative_train_dataset) as train_data:
                    datas = json.load(train_data)
                for data in datas:
                    self.data.append([data['question'], data['answer'][0], data['passage']])
            elif config.data_name == 'MQuAKE':
                with open(config.MQuAKE_train_dataset) as train_data:
                    datas = json.load(train_data)
                for data in datas:
                    self.data.append([data['question'], data['answer'], data['passage']])
            else:
                raise AssertionError("Error dataset")


        elif status == "Test":
            if config.data_name == 'NQ':
                with open(config.NQ_test_dataset_file) as test_data:
                    datas = json.load(test_data)
                for data in datas:
                    self.data.append([data['question'], data['answers'], data['ctxs']])
            elif config.data_name == 'TQA':
                with open(config.TQA_test_dataset_file) as test_data:
                    datas = json.load(test_data)
                for data in datas:
                    self.data.append([data['question'], data['answers'], data['ctxs']])
            elif config.data_name == 'SQuAD':
                with open(config.SQuAD_test_dataset_file) as test_data:
                    datas = json.load(test_data)
                for i in datas['data']:
                    for j in i['paragraphs']:
                        paragraph = j['context']
                        random_choice_qas = random.choice(j['qas'])
                        answers = [k['text'] for k in random_choice_qas['answers']]
                        if random_choice_qas['is_impossible']:
                            continue
                        self.data.append([random_choice_qas['question'], answers, [paragraph]])
            elif config.data_name == 'Narrative':
                with open(config.Narrative_test_dataset_file) as test_data:
                    datas = json.load(test_data)
                for data in datas:
                    self.data.append([data['question'], data['answer'], data['passage']])
            elif config.data_name == 'MQuAKE':
                with open(config.MQuAKE_test_dataset_file) as test_data:
                    datas = json.load(test_data)
                for data in datas:
                    self.data.append([data['question'], [data['answer']], data['passage']])
            else:
                raise AssertionError("Error dataset")

        else:
            raise AssertionError("Error state")



    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        if self.status == 'Train':
            question, answer, passage = self.data[idx]
            question = question + "?" if question[-1] != "?" else question
            question = first_word_cap(question)
            input = 'Question: {}\nAnswer:'.format(question)
            answer = '{}\n'.format(answer)
            tok_tuples, tok_sentence = self.tok_tuples(input, answer, passage)
            return tok_tuples, tok_sentence
        elif self.status == 'Test':
            print(self.data[idx])
            question, answers, passage = self.data[idx]
            question = question + "?" if question[-1] != "?" else question
            question = first_word_cap(question)
            # passage = passage[:self.config.num_experts]
            # for p in reversed(passage):
            #     if p['has_answer']:
            #         pp = p
            #         break
            # pp = passage[0]
            # knowledge = pp['text'] if self.config.data_name in ['NQ', 'TQA'] else '. '.join(passage)
            knowledge = passage
            instruction = 'Base above knowledge, answer the following question with a very short phrase, such as “1998”, “May 16th, 1931”, or “James Bond”, to meet the criteria of exact match datasets.'
            prompt_1 = 'Question: {}\nAnswer:'.format(question)
            prompt_2 = 'Knowledge:\n{}\nQuestion: {}\nAnswer:'.format(knowledge, question)
            prompt_3 = 'Knowledge:\n{}\n{}\nQuestion: {}\nAnswer:'.format(knowledge, instruction, question)

            # tok_sentence = [self.tok(p['text'], return_tensors="pt") for p in passage]
            tok_sentence = [self.tok(knowledge, return_tensors="pt")]

            return (self.tok.encode(prompt_1, return_tensors="pt").cuda(),
                    prompt_1,
                    self.tok.encode(prompt_2, return_tensors="pt").cuda(),
                    prompt_2,
                    self.tok.encode(prompt_3, return_tensors="pt").cuda(),
                    prompt_3,
                    [question],
                    answers,
                    tok_sentence)



    def tok_tuples(self, prompt, answer, passage):
        if self.config.model_name == 'meta-llama/Llama-2-7b-hf':
            answer = answer
        elif self.config.model_name == "meta-llama/Llama-3.1-8B":
            answer = " " + answer
        elif self.config.model_name == "baichuan-inc/Baichuan2-7B-Base":
            answer = answer
        elif self.config.model_name == "Qwen/Qwen2.5-7B-Instruct":
            answer = answer
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
                tuples[0][1],
                tuples[0][2],
                tuples[0][3],
                tuples[0][4],
                tuples[0][5],
                [t for t in tuples[0][6]],
                [t for t in tuples[0][7]],
                tuples[0][8])



def make_Training_loader(config, tok, samples = None):
    train_set = NQ_TQA_SQuAD_Dataset(config, tok, status='Train', samples = samples)
    train_loader = DataLoader(train_set, batch_size=config.train_batch_size, shuffle=True, collate_fn=train_set.collate_fn)
    return train_loader


def make_Validation_loader(config, tok):
    valid_set = NQ_TQA_SQuAD_Dataset(config, tok, status='Test', samples=None)
    valid_loader = DataLoader(valid_set, batch_size=config.valid_batch_size, shuffle=False, collate_fn=valid_set.val_collate_fn)
    return valid_loader
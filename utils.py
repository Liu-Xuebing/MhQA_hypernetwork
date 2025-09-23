import json
import os
from collections import Counter
import torch
import torch.nn.functional as F  # 用于计算 softmax
import numpy as np
import random
from tqdm import tqdm
from elasticsearch import Elasticsearch


WHITESPACE_AND_PUNCTUATION = {' ', '.', ',', ':', ';', '!', '?', '$', '%', '(', ')', '[', ']', '-', '`', '\'', '"'}
ARTICLES = {'the', 'a', 'an'}


def set_seed(seed: int = 42):
    # Python random 模块
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch CPU & CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if multi-GPU

def print_trainable_parameters(model):
    print(f"{'Parameter Name':60} | Requires Grad")
    print('-' * 75)
    for name, param in model.named_parameters():
        print(f"{name:60} | {param.requires_grad}")

def tansfer_to_scftmax(list):
    scores = torch.tensor(list)
    softmax_values = F.softmax(scores, dim=0)
    return softmax_values

def write_json(x, path):
    with open(path, "without Hypernetwork") as f:
        f.write(json.dumps(x, indent=4))


def find_subsequence_index(source, target):
    for i in range(len(source) - len(target) + 1):
        if torch.equal(source[i:i + len(target)], target):
            return i  # Return the starting index if found
    return -1  # Return -1 if the subsequence is not found


def cal_anchor_embedding(sentences, model, tokenizer):
    average_embedding = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tagged_tokens = pos_tag(tokens)
        words = [word for word, pos in tagged_tokens if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
        if len(words) > 0:
            inputs = tokenizer(words, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
            with torch.no_grad():  # 不需要梯度计算
                embeddings = model.model.embed_tokens(inputs['input_ids'].cuda())  # get the embedding by embed layer
            embeddings = embeddings[:, 0, :]
            average_embedding.append(torch.mean(embeddings, dim=0, keepdim=True))
        else:
            average_embedding.append(torch.zeros(size=(1,4096)).cuda())
    return torch.cat(average_embedding, dim=0)




def cal_EM_F1(predict: torch.FloatTensor, answers: torch.LongTensor):
    f1 = F1Single(answers, predict)
    if ExactMatchSingle(answers, predict):
        return 1, f1
    else:
        return 0, f1



def ExactMatchSingle(answers, predicted_answer):
    for ans in answers:
        if CleanAnswer(ans) == CleanAnswer(predicted_answer):
            return True
    return False



def F1Single(label_answer, predicted_answer):
    def GetTokens(text):
        text = CleanAnswer(text)
        for delimeter in WHITESPACE_AND_PUNCTUATION:
            text = text.replace(delimeter, ' ')
        return text.split()
    f1 = 0
    predicted_answer_tokens = Counter(GetTokens(predicted_answer))
    num_predicted_answer_tokens = sum(predicted_answer_tokens.values())
    for answer in label_answer:
        answer_tokens = Counter(GetTokens(answer))
        num_answer_tokens = sum(answer_tokens.values())
        num_same = sum((predicted_answer_tokens & answer_tokens).values())
        if num_same == 0:
            continue
        precision = 1.0 * num_same / num_predicted_answer_tokens
        recall = 1.0 * num_same / num_answer_tokens
        f1 = max(2 * precision * recall / (precision + recall), f1)
    return f1


def CleanAnswer(answer):
    answer = answer.strip().lower()
    answer = answer.replace(' , ',', ')
    answer = answer.replace(' - ','-')
    if isinstance(answer, str):
        answer = answer.replace(u'\u00a0', ' ')
    else:
        answer = answer.replace('\xc2\xa0', ' ')
    while len(answer) > 1 and answer[0] in WHITESPACE_AND_PUNCTUATION:
        answer = answer[1:]
    while len(answer) > 1 and answer[-1] in WHITESPACE_AND_PUNCTUATION:
        answer = answer[:-1]

    answer = answer.split()
    if len(answer) > 1 and answer[0] in ARTICLES:
        answer = answer[1:]
    answer = ' '.join(answer)
    return answer


# def search_index(query, model):
#     embeddings = model.encode([query])
#     flag = -1
#     index = None
#     for key, value in expert_embedding.items():
#         similarities = model.similarity(embeddings, value)
#         final_mean_sim = torch.max(similarities)
#         if final_mean_sim > flag:
#             flag = final_mean_sim
#             index = key
#     return index


def vector_loss(w_M, wo_M, label_w, label_wo):
    w_M = w_M.to(label_w.device)
    wo_M = wo_M.to(label_wo.device)
    w_ans_indice = torch.where(label_w != -100)
    wo_ans_indice = torch.where(label_wo != -100)
    assert len(w_ans_indice[0]) == len(wo_ans_indice[0])
    w_M = w_M[w_ans_indice]
    wo_M = wo_M[wo_ans_indice]
    return torch.nn.MSELoss()(wo_M, w_M)


def cross_entropy(
    logits: torch.FloatTensor,
    labels: torch.LongTensor
):
    if len(logits.shape) == 2:
        return F.binary_cross_entropy_with_logits(logits, labels)
    if len(logits.shape) == 3:
        ans_indice = torch.where(labels != -100)
        logits = logits[ans_indice]
        labels = labels[ans_indice]
        return F.cross_entropy(logits, labels)


def get_word(logits, labels, tok):
    ans_indice = torch.where(labels != -100)
    logits = logits[ans_indice]
    pred_ids = logits.argmax(dim=-1)  # [B, T]
    pred_texts = tok.batch_decode(pred_ids, skip_special_tokens=True)
    return ''.join(pred_texts)



def kl_divergence(p, q, label_p, label_q):
    batchsize = p.shape[0]
    KL_loss = 0
    for i in range(batchsize):
        w_ans_indice = torch.where(label_p[i] != -100)[0]
        p_valid =  p[i, w_ans_indice, :]
        wo_ans_indice = torch.where(label_q[i] != -100)[0]
        q_valid = q[i, wo_ans_indice, :]
        KL_loss += F.kl_div(F.log_softmax(q_valid, dim=-1), F.softmax(p_valid, dim=-1), reduction='mean')
    return KL_loss / batchsize



def first_word_cap(text):
    words = text.split()
    words[0] = words[0].capitalize()
    text = " ".join(words)
    return text


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def get_sent_embeddings(sents, contriever, tok, BSZ=32):
    all_embs = []
    for i in tqdm(range(0, len(sents), BSZ)):
        sent_batch = sents[i:i+BSZ]
        inputs = tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to("cuda")
        with torch.no_grad():
            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        all_embs.append(embeddings.cpu())
    all_embs = torch.vstack(all_embs)
    return all_embs


def retrieve_facts(query, fact_embs, contriever, tok, k=1):
    inputs = tok([query], padding=True, truncation=True, return_tensors='pt').to("cuda")
    with torch.no_grad():
        outputs = contriever(**inputs)
        query_emb = mean_pooling(outputs[0], inputs['attention_mask']).cpu()
    sim = (query_emb @ fact_embs.T)[0]
    knn = sim.topk(k, largest=True)
    return knn.indices




class BM25Retriever:
    def __init__(self, index_name="facts_index", host="http://localhost:9200"):
        self.es = Elasticsearch(host)
        print(self.es.info())
        self.index_name = index_name

    def create_index(self):
        """创建索引，设置 content 字段为 text 类型（BM25 默认启用）"""
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)

        self.es.indices.create(
            index=self.index_name,
            mappings={
                "properties": {
                    "content": {"type": "text"}
                }
            }
        )

    def add_documents(self, docs):
        """批量写入文档，docs 是字符串列表"""
        for i, doc in enumerate(docs):
            self.es.index(index=self.index_name, id=i, document={"content": doc}, refresh=True)

    def search(self, query, top_k=5):
        """用 BM25 搜索 query，返回前 top_k 个结果"""
        response = self.es.search(
            index=self.index_name,
            query={
                "match": {
                    "content": query
                }
            },
            size=top_k
        )
        results = [
            {"score": hit["_score"], "doc": hit["_source"]["content"]}
            for hit in response["hits"]["hits"]
        ]
        return results


# ================== 用法示例 ==================
if __name__ == "__main__":
    facts = [
        "Alonso Mudarra: Alonso Mudarra( c. 1510 – April 1, 1580) was a Spanish composer of the Renaissance, and also played the vihuela, a guitar- shaped string instrument.",
        "Alonso Mudarra: He was an innovative composer of instrumental music as well as songs, and was the composer of the earliest surviving music for the guitar.",
        "Thomas Morse: Thomas Morse( born June 30, 1968) is an American composer of film and concert music.",
        "Abe Meyer: Abe Meyer( 1901 – 1969) was an American composer of film scores.",
        "Prashant Pillai: Prashant Pillai (born 8 September 1981) is a music producer and composer from India.",
        "Bert Grund: Bert Grund( 1920–1992) was a German composer of film scores.",
        "Tarcisio Fusco: Tarcisio Fusco was an Italian composer of film scores.",
        "Cyril Chamberlain: Cyril Chamberlain (8 March 1909 – 5 December 1974) was an English film and television actor.",
        "Walter Ulfig: Walter Ulfig was a German composer of film scores.",
        "Amedeo Escobar: Amedeo Escobar( 1888–1973) was an Italian composer of film scores.",
        "Sayanna Varthakal: Sayanna Varthakal  is an upcoming Indian Malayalam-language socio-political satire film written and directed by debutant Arun Chandu and produced by D14 Entertainments.",
        "Sayanna Varthakal: Co-written by Sachin R Chandran and Rahul Menon, the film stars Gokul Suresh, Dhyan Sreenivasan, Aju Varghese and newcomer Sharanya Sharma in lead roles.",
        "Sayanna Varthakal: The music of the film is composed by Prashant Pillai and the cinematography is handled by Sarath Shaji.",
        "Sayanna Varthakal: It is reported that the movie tells the tale of a film actor and his life in a government-affiliated organisation.",
        "Henri Verdun: Henri Verdun( 1895–1977) was a French composer of film scores."]

    retriever = BM25Retriever()
    retriever.create_index()
    retriever.add_documents(facts)

    query = "Was Cyril Chamberlain an English film and television actor?"
    results = retriever.search(query, top_k=1)
    print(results)
    for r in results:
        print(f"Score: {r['score']:.4f}, Doc: {r['doc']}")


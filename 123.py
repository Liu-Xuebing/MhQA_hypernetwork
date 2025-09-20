# import json
#
# with open('/disk/liuxb/datasets/MQuAKE/MQuAKE-CF-3k-v2.json', 'r') as f:
#     a = json.load(f)
#
# print(len(a))
#
# # print(a[0])
# for k,v in a[0].items():
#     print("{}: {}".format(k, v))
#     print('\n')

import json
import os
import random



random.seed(88)
# from tqdm import tqdm
#
# def first_word_cap(text):
#     words = text.split()
#     words[0] = words[0].capitalize()
#     text = " ".join(words)
#     return text
#
#
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# with open('/disk/liuxb/datasets/TQA/TQA_train_results.json') as f:
#     datas = json.load(f)
#
# all_datas = []
#
# for data in tqdm(datas):
#     question = data['question']
#     question = question + "?" if question[-1] != "?" else question
#     question = first_word_cap(question)
#
#     answer = data['answers'][0]
#
#     for ctx in data['ctxs']:
#         if answer in ctx['text'] and ctx['has_answer']:
#             text = ctx['text']
#             all_datas.append({'question': question, 'answer': answer, 'text': text})
#             break
#
# with open("/disk/liuxb/code/Multi-EMoE/datasets/TQA.json", "w") as f:
#     json.dump(all_datas, f, ensure_ascii=False, indent=4)
#
# print(len(all_datas))

# with open("/disk/liuxb/code/Multi-EMoE/datasets/pre_training.json") as f:
#     TQA_datas = json.load(f)
# print(len(TQA_datas))
# with open("/disk/liuxb/code/Multi-EMoE/datasets/NQ.json") as f:
#     NQ_datas = json.load(f)
#
#
# all_datas = TQA_datas + NQ_datas
# # print(all_datas[0])
# random.shuffle(all_datas)
# # print(all_datas[0])
#
# new_data = all_datas[71354:72354]
# with open("/disk/liuxb/code/Multi-EMoE/datasets/causal_tracing.json", "w") as f:
#     json.dump(new_data, f, ensure_ascii=False, indent=4)
# print(len(all_datas))
# import matplotlib.pyplot as plt
# import torch
# import numpy as np
# # #
# dir = '/disk/liuxb/code/Multi-EMoE/Causal Tracing/results'
# kind_1 = 'self_attn'
# kind_2 = 'mlp'
#
# files_attn = [i for i in os.listdir(dir) if kind_1 in i]
# files_mlp = [i for i in os.listdir(dir) if kind_2 in i]
#
# def cal_(files):
#     all_score = []
#     for file in files:
#         with open(os.path.join(dir, file), 'r') as f:
#             data = json.load(f)[0]
#         scores = torch.tensor(data['scores'])
#         high_score = data['high_score']
#         low_score = data['low_score']
#         all_score.append((scores - low_score)/(high_score - low_score))
#     t = torch.stack(all_score)
#     print(t.shape)
#     t = torch.mean(t, dim=0, keepdim=True)
#     return t
# #
# attn = cal_(files_attn).squeeze()
# mlp = cal_(files_mlp).squeeze()
# # print(attn, mlp)
# # exit()
# x = np.arange(len(attn))
# # 设置柱子的宽度
# width = 0.35
# plt.figure(figsize=(6, 3), dpi=200)  # 宽12英寸，高6英寸
#
# # 绘制柱状图
# plt.bar(x - width/2, attn, width=width, label='Effect with Attn recovery', color='Red')
# plt.bar(x + width/2, mlp, width=width, label='Effect with MLP recovery', color='Green')
# plt.ylabel('AIE', fontsize=14, color='black')
# plt.xlabel('Layer', fontsize=14, color='black')
# #
# # 添加 x 轴标签
# plt.xticks(x[::5])  # 从 x 中每隔5个取一个显示
# # 添加图例
# plt.legend()
# plt.title('AIE state with Attn or MLP module recovery', fontsize=16)
#
# plt.savefig('zhuzhuangtu.png')


# fig, ax = plt.subplots(figsize=(6, 1.5), dpi=200)
# h = ax.pcolor(
#     t,
#     cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "self_attn": "Reds"}[
#         kind
#     ],
#     vmin=sum(all_low_score)/len(all_low_score),
# )
#
# kindname = "Attn" if kind == "self_attn" else "MLP"
# ax.set_title(f"Impact of restoring {kindname} after corrupted input")
# ax.set_yticks([0.5 + i for i in range(len(t))])
# ax.set_xticks([0.5 + i for i in range(0, t.shape[1] - 6, 5)])
# ax.set_xticklabels(list(range(0, t.shape[1] - 6, 5)))
# ax.set_yticklabels(["*Query"])
# cb = plt.colorbar(h)
# cb.ax.set_title(f"AIE", y=-0.33, fontsize=10)
# plt.tight_layout()
#
# plt.savefig('{}.png'.format(kind))

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# new_datas = []
# with open('/disk/liuxb/datasets/MQuAKE/MQuAKE-T.json') as f:
#     datas = json.load(f)
#
# random.shuffle(datas)
# for data in datas[868:]:
#     # for k, v in data.items():
#     #     print('{} {}'.format(k, v))
#     # exit()
#     # print(data['new_single_hops'])
#     # print('----------')
#     question = data['questions'][0]
#     answer = data['new_answer']
#     answer_alias = data['new_answer_alias']
#     facts = []
#     sub_questions = []
#     sub_answers = []
#     for re in data['new_single_hops']:
#         facts.append("{} {}".format(re['cloze'], re['answer']))
#         sub_questions.append(re['question'])
#         sub_answers.append(re['answer'])
#     new_datas.append({'question': question,
#                       'answer': answer,
#                       'answer_alias': answer_alias,
#                       # 'sub_question': sub_questions,
#                       # 'sub_answer': sub_answers,
#                       'facts': facts})
# print(len(new_datas))
# #
# #
# with open("/disk/liuxb/code/Multi-EMoE/datasets/MQuAKE-T_test.json", "w") as f:
#     json.dump(new_datas, f, ensure_ascii=False, indent=4)


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# import torch.nn.functional as F
#
# model_id = "microsoft/deberta-v3-base"
#
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
#
# model.eval()
#
#
# question = "Which religion is Leo XIII affiliated with?"
# passage = "Leo XIII is affiliated with the religion of Methodism"
# answer = "Methodism"
#
# # premise: passage, hypothesis: Q+A
# hypothesis = f"P: {passage} Q: {question} A: {answer}"
#
# inputs = tokenizer(hypothesis, return_tensors="pt", padding=True, truncation=True)
# print(inputs["input_ids"])
#
# with torch.no_grad():
#     outputs = model(**inputs)
#     logits = outputs.logits  # [batch, 2]
#     probs = F.softmax(logits, dim=-1)
#
# 正确的概率
# score = probs[0].item()
# print("Score:", score)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# from collections import Counter
#
# new_datas = []
# #
# with open('/disk/liuxb/datasets/HotpotQA/hotpot_train.json', 'r') as f:
#     datas = json.load(f)
# random.shuffle(datas)
# for data in datas[:20000]:
#     if data['level'] == 'medium':
#         facts = []
#         for c in data['context']:
#             for cc in c[1]:
#                 facts.append('{}: {}'.format(c[0], cc.strip()))
#         new_datas.append({'question': data['question'],
#                           'answer': data['answer'],
#                           'facts': facts,
#                           'level': data['level']})
#         if len(new_datas) == 700:
#             break
#
# for data in datas[:20000]:
#     if data['level'] == 'hard':
#         facts = []
#         for c in data['context']:
#             for cc in c[1]:
#                 facts.append('{}: {}'.format(c[0], cc.strip()))
#         new_datas.append({'question': data['question'],
#                           'answer': data['answer'],
#                           'facts': facts,
#                           'level': data['level']})
#         if len(new_datas) == 300:
#             break
# #
# random.shuffle(new_datas)
# print(len(new_datas))
# with open("/disk/liuxb/code/Multi-EMoE/datasets/HotPot_test.json", "w") as f:
#     json.dump(new_datas, f, ensure_ascii=False, indent=4)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# with open('/disk/liuxb/datasets/2WikiMultihopQA/train.json', 'r') as f:
#     datas = json.load(f)
# print('the dataset length of Wiki is {}'.format(len(datas)))
#
# random.shuffle(datas)
#
# new_datas = []
# for data in datas[:2000]:
#     facts = []
#     for i in data['supporting_facts']:
#         for j in data['context']:
#             if i[0] == j[0]:
#                 facts.append('{}: {}'.format(j[0], j[1][i[1]].strip()))
#     new_datas.append({'question': data['question'],
#                       'answer': data['answer'],
#                       'facts': facts,
#                       'evidences': data['evidences']})
#
# with open("/disk/liuxb/code/Multi-EMoE/data/WikiMhQA_pre_train.json", "w") as f:
#     json.dump(new_datas, f, ensure_ascii=False, indent=4)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# with open('/disk/liuxb/datasets/2WikiMultihopQA/train.json', 'r') as f:
#     datas = json.load(f)
# print('the dataset length of Wiki is {}'.format(len(datas)))
#
# random.shuffle(datas)
#
# new_datas = []
# for data in datas[2000:3000]:
#     facts = []
#     for c in data['context']:
#         for cc in c[1]:
#             facts.append('{}: {}'.format(c[0], cc.strip()))
#     new_datas.append({'question': data['question'],
#                       'answer': data['answer'],
#                       'answer_alias': [],
#                       'facts': facts})
#
# with open("/disk/liuxb/code/Multi-EMoE/datasets/WikiMhQA_test.json", "w") as f:
#     json.dump(new_datas, f, ensure_ascii=False, indent=4)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# new_datas = []
# #
# with open('/disk/liuxb/datasets/HotpotQA/hotpot_train.json', 'r') as f:
#     datas = json.load(f)
# random.shuffle(datas)
# for data in datas[20000:]:
#     if data['level'] == 'medium':
#         facts = []
#         for i in data['supporting_facts']:
#             for j in data['context']:
#                 if i[0] == j[0]:
#                     facts.append('{}: {}'.format(j[0], j[1][i[1]].strip()))
#         new_datas.append({'question': data['question'],
#                           'answer': data['answer'],
#                           'facts': facts,
#                           'level': data['level']})
#         if len(new_datas) == 3000:
#             break
#
# for data in datas[20000:]:
#     if data['level'] == 'hard':
#         facts = []
#         for i in data['supporting_facts']:
#             for j in data['context']:
#                 if i[0] == j[0]:
#                     facts.append('{}: {}'.format(j[0], j[1][i[1]].strip()))
#         new_datas.append({'question': data['question'],
#                           'answer': data['answer'],
#                           'facts': facts,
#                           'level': data['level']})
#         if len(new_datas) == 4000:
#             break
# #
# random.shuffle(new_datas)
# print(len(new_datas))
# with open("/disk/liuxb/code/Multi-EMoE/datasets/HotPot_train.json", "w") as f:
#     json.dump(new_datas, f, ensure_ascii=False, indent=4)

# with open('/disk/liuxb/datasets/HotpotQA/hotpot_train.json', 'r') as f:
#     datas = json.load(f)
# random.shuffle(datas)
# for data in datas[20050:]:
#     if data['level'] != 'medium':
#         continue
#     print(data['level'])
#     print(data['question'])
#     print(data['answer'])
#     print(data['supporting_facts'])
#     for i in data['supporting_facts']:
#         for j in data['context']:
#             if i[0] == j[0]:
#                 print('{}: {}'.format(j[0], j[1][i[1]]))
#     # print(data)
#     exit()
#     a.append(len(data['supporting_facts']))
#     for k,v in data.items():
#         print('{}: {}'.format(k,v))
#     exit()
# print(Counter(a))


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


import matplotlib.pyplot as plt

# x = [1, 2, 3, 4, 5, 6, 7, 8]

# RAG_EM = [21.6, 23.7, 23.7, 27.8, 29.7, 30.6, 30.4, 30.3]
# RAG_F1 = [36.67, 37.28, 36.77, 40.51, 42.19, 43.52,	43.07, 42.73]
# RAG_CoT_EM = [37.6,	41.4, 43.5,	43.7, 44.4, 43.9, 45, 45.1]
# RAG_CoT_F1 = [45.15, 48.9, 49.87, 50.41, 51.24,	51.48, 52.77, 52.32]

# RAG_EM = [3.9, 5.5, 4.3, 4.4, 4.9, 4.3, 4.5, 4]
# RAG_F1 = [8.07, 9.65, 8.98, 9.05, 9.86, 9.39, 9.69, 9.44]
# RAG_CoT_EM = [10.9, 7.9, 10.6, 13.1, 13, 13.4, 11.9, 12.8]
# RAG_CoT_F1 = [12.78, 9.47, 12.19, 15.03, 14.8, 14.56, 12.75, 13.52]

# RAG_EM = [0.6, 30.4, 34.8, 35.4, 33.9, 30.3, 28.5, 27.6]
# RAG_F1 = [4.57, 33.09, 38.81, 39.03, 38.29, 35.3, 33.42, 32.81]
# RAG_CoT_EM = [4.2, 37.5, 52.4, 59.2, 47, 48.9, 57, 61.2]
# RAG_CoT_F1 = [8.35, 39.09, 53.32, 60.12, 49.52, 49.3, 57.14, 61.41]
#
# plt.figure(figsize=(7,5))
#
# # 画曲线
# plt.plot(x, RAG_EM, 'o-', mfc='white', mec='green', color='green', label='RAG_EM')  # 圆点实线
# plt.plot(x, RAG_F1, 's-', mfc='white', mec='olive', color='olive', label='RAG_F1')  # 方块实线
# plt.plot(x, RAG_CoT_EM, '^-', color='navy', label='RAG-CoT_EM')  # 三角形实线
# plt.plot(x, RAG_CoT_F1, 'o-', color='orange', label='RAG-CoT_F1')  # 圆点实线
#
# # 设置标题和坐标轴
# plt.xlabel("Number of Retrieval Instances Used", fontsize=18)
# plt.ylabel("Performance. (%)", fontsize=18)
#
# plt.xticks(fontsize=14)   # x轴刻度字体大小
# plt.yticks(fontsize=14)
# # 图例
# plt.legend()
#
# # 网格
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.savefig('mquake-t.png')
# plt.show()


# ----------------------------------------------------------------------------------------------
# 画层扫描实验的图

# import matplotlib.pyplot as plt
# import numpy as np
#
# # 数据
# model_ppl = [4.43, 4.31, 4.27, 4.16, 3.9, 3.81, 3.7, 3.74, 3.73, 3.77, 3.76, 3.74,
#        3.8, 3.82, 3.87, 3.95, 4.07, 4.08, 4.11, 4.13, 4.25, 4.31, 4.3, 4.3,
#        4.26, 4.27, 4.36, 4.42, 4.39, 4.33, 4.37, 4.69]
#
# # model_ppl = [4.06, 4.1, 4.11, 4.19, 4.2, 4.15, 4.15, 4.1, 4.12, 4.19, 4.13, 4.09, 4.08,
# #        4.14, 4.17, 4.22, 4.26, 4.33, 4.42, 4.61, 4.87, 5.29, 5.57, 5.84, 5.97, 6.18, 6.42, 9.78]
#
# layers = np.arange(len(model_ppl))
#
#
# plt.rcParams['font.serif'] = ['Helvetica']
#
# # 设置柱状图颜色渐变
# colors = plt.cm.viridis((np.array(model_ppl) - min(model_ppl)) / (max(model_ppl) - min(model_ppl)))
#
# plt.figure(figsize=(12, 5))
# bars = plt.bar(layers, model_ppl, color=colors)
#
# # 缩小 y 轴范围突出差异
# plt.ylim(min(model_ppl)-0.1, max(model_ppl)+0.1)
#
# # 添加数值标签
# for bar, y in zip(bars, model_ppl):
#     plt.text(bar.get_x() + bar.get_width()/2, y + 0.02, f'{y:.2f}',
#              ha='center', va='bottom', fontsize=8)
#
# plt.xlabel('Layer', fontsize=16)
# plt.ylabel('PPL', fontsize=16)
# plt.title('The training perplexity (PPL) of the experts \n at each layer of LLaMa3.1-8B model', fontsize=16)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.savefig('llama_layer_scanning.png')

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

# 数据
metrics = ["EM", "F1"]
rag_scores = [4.48, 9.27]
srke_scores = [50.30, 51.36]

# 颜色 (归一化到 0-1)
rag_color = (255/255, 153/255, 153/255)   # 粉色
srke_color = (180/255, 199/255, 231/255)  # 浅蓝

# 组位置（组间间隔设置大一点）
group_x = np.array([0, 2])

# 组内间隔
offset = 0.2
width = 0.3  # 每个柱子宽度

fig, ax = plt.subplots(figsize=(6, 4))

# 绘制柱子：每组里分别偏移
ax.bar(group_x - offset, rag_scores, width, label="RAG", color=rag_color)
ax.bar(group_x + offset, srke_scores, width, label="SRKE-RAG", color=srke_color)

# 设置坐标轴标签
ax.set_ylabel("Score",fontsize=16)
ax.set_xticks(group_x)
ax.set_xticklabels(metrics, fontsize=16)
ax.legend()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 保留底边和左边，设置线宽更细（论文风）
ax.spines['bottom'].set_linewidth(0.8)
ax.spines['left'].set_linewidth(0.8)

plt.tight_layout()
plt.savefig('introduction.png')


import json

with open('/disk/liuxb/datasets/MQuAKE/MQuAKE-CF-3k-v2.json', 'r') as f:
    a = json.load(f)

print(len(a))

# print(a[0])
for k,v in a[0].items():
    print("{}: {}".format(k, v))
    print('\n')
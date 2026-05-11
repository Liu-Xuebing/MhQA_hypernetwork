# Datasets Generation

You can also build your own datasets using the scripts under the following path:

```bash
data/build_datasets
```

This module uses prompt engineering with large language models to generate sub-questions and corresponding answers for multi-hop QA datasets.

The generated datasets are mainly used to train the following modules:

- Decomposer
- Question Answering models

You can also customize the prompts by modifying the corresponding `{}_prompt.txt` files, allowing you to build your own datasets.
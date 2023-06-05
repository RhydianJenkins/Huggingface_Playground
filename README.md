Models taken from [huggingface.co](https://huggingface.co).

# Getting started

```sh
./install # only need to run this once
./run
```

# Generators

```
from transformers import pipeline

generator = pipeline(model="<MODEL_NAME>")
result = generator(<INPUT>)

```


# Models

## Text Classification

Takes an input string and grades its friendlyness.

`text-classification`

`[{'label': 'POSITIVE', 'score': 0.9998743534088135}]`

## Conversation

[facebook/blenderbot-400M-distill](https://huggingface.co/facebook/blenderbot-400M-distill?text=Hey+my+name+is+Julien%21+How+are+you%3F)

```python
import argparse
import torch
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--message', type=str, default='')
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # download and setup the model and tokenizer
    model_name = 'facebook/blenderbot-400M-distill'
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name).to(device)

    inputs = tokenizer(args.message, return_tensors="pt").to(device)
    result = model.generate(**inputs)
    print(tokenizer.decode(result[0]).replace('<s>','').replace('</s>','').strip())
```

Usage:

```bash
python --mesage "What is your favourite colour?"
> I don't really have a favourite. I like all of them. What about you?
```

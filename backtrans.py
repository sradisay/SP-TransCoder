
from datasets import load_dataset

# KEY = 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

token = "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"

data = load_dataset("bigcode/the-stack-smol", split="train", token=token)
# Cdata = data.filter(lambda x: x["language"]=="C++")[:100]
# Pydata = data.filter(lambda x: x["language"]=="Python")[:100]
# data[:500]

Cdata = data.filter(lambda x:x["lang"]=="C++")

Pydata = data.filter(lambda x:x["lang"]=="Python")

from datasets import Dataset
Cdata = Dataset.from_dict(Cdata[:500])
# Cdata[:2]
Pydata = Dataset.from_dict(Pydata[:500])

Pydata

# dataset = dataset.train_test_split(test_size=0.2)
# test_=dataset["test"].train_test_split(test_size=0.5)
# train_d= dataset["train"]
# val_ds= test_["train"]
# test_ds = test_["test"]



max_length=256
def ch(ele):
  print("y")

  l_inputs=ll(ele["source"], truncation=True, max_length=max_length, padding="max_length")
  labels=ll(ele["target"], max_length=max_length, padding="max_length", truncation=True)
  l_inputs["labels"] = labels["input_ids"]

  return l_inputs

# train_d = train_d.map(c, batched=True)
# val_ds = val_ds.map(c,batched=True)

import transformers
import tokenizers
tokenizers.__version__

# ! pip uninstall transformers tokenizers
! pip install transformers==4.57.6 sentencepiece

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
modell = "Salesforce/codet5-small"

model = AutoModelForSeq2SeqLM.from_pretrained(modell)
ll = AutoTokenizer.from_pretrained(modell)
# devic=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model.to(devic)

from transformers import TrainingArguments, Trainer

training_=TrainingArguments(output_dir="./C++P",  per_device_train_batch_size=4, per_device_eval_batch_size=4, gradient_accumulation_steps=8,  num_train_epochs=1, learning_rate=5e-5, logging_steps=500, fp16=True)
trainer=Trainer(model=model, args=training_, tokenizer=ll)
# trainer.train()

# blu = evaluate.load("bleu")


def tokenize(code, info):
  inputs_ids = ll(info + code, max_length=256, return_tensors="pt", truncation=True)
  outputs = model.generate(**inputs_ids, max_new_tokens=256, no_repeat_ngram_size=2, early_stopping=True, num_beams=6)
  return ll.decode(outputs[0], skip_special_tokens=True)
  # batch deco?


# for num in range(2):
pseudo_pairs_CP = []
pseudo_pairs=[]
for code in Cdata["content"]:
    code = filter(code)
    # back trans
    pseudocode = tokenize(code, "Translate C++ to Python")
    pseudo_pairs.append({"source":code, "target": pseudocode})
    pseudo_pairs.append({"source": pseudocode, "target": code})


pseudo_pairs_PC = []
for code in Pydata["content"]:
    pseudocode = tokenize(code, "Translate Python to C++")
    pseudo_pairs.append({"source": code, "target": pseudocode})
    pseudo_pairs.append({"source": pseudocode, "target": code})

pseudo_pairsdata = Dataset.from_list(pseudo_pairs)
pseudo_pairsdata = pseudo_pairsdata.map(ch, batched=True)
trainer.train_dataset = pseudo_pairsdata
trainer.train()

  # tokenized = [tokenize(x, "Translate C++ to Python: ") for x in pseudo_pairs_CP]

# !pip install --upgrade transformers

code = """def sum(a,b):
    return a+b
    """
tokenize(code, "Translate Python to C++: ")

trainer.save_model("./cpptopy")
ll.save_model("./cpptopy")


import re
def filter(code):
  code = re.sub(r'^\s*include.*$', '',code, flags=re.MULTILINE)
  return code

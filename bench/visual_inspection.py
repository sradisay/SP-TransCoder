import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

CONFIG = {
    "model_name": "Salesforce/codet5-small",
    "max_len": 256,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"]).to(CONFIG["device"])
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])


def translate(code, source_lang="Python", target_lang="C++"):
    prefix = f"Translate {source_lang} to {target_lang}: "
    input_text = f"{prefix}{code}"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=CONFIG["max_len"]
    ).to(CONFIG["device"])

    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=CONFIG["max_len"],
        num_beams=4,
        early_stopping=True,
        repetition_penalty=1.2
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


py_code = "def add(a, b):\n    return a + b"
print(f"Input:\n{py_code}")
print(f"Output:\n{translate(py_code)}\n")

py_list = "x = [1, 2, 3]"
print(f"Input:\n{py_list}")
print(f"Output:\n{translate(py_list)}\n")


py_list = "def add(a, b):\n     return a + b\n\n\ndef main ():\n   add(4, 10)"
print(f"Input:\n{py_list}")
print(f"Output:\n{translate(py_list)}\n")

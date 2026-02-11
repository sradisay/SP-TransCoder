import torch
from transformers import AutoModelForSeq2SeqLM
from tokenizer.apply_tokenizer import salesforce_tokenizer


CONFIG = {
    "model_name": "salesforce/codet5-small",
    "langs": ["python", "c++"],
    "batch_size": 128,
    "epochs": 10,
    "lr": 3e-6,
    "max_len": 256,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"]).to(CONFIG["device"])


def translate(code, prefix):
    input_text = f"{prefix} {code}"

    inputs = salesforce_tokenizer(
        input_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=CONFIG["max_len"]
    ).to(CONFIG["device"])

    outputs = model.generate(
        inputs.input_ids,
        max_length=CONFIG["max_len"],
        num_beams=2,
    )
    return salesforce_tokenizer.decode(outputs[0], skip_special_tokens=True)


py_code = "def add(a, b):\n    return a + b"
print(f"Input:\n translate python to c++: {py_code}")
print(f"Output:\n {translate(py_code, 'translate python to c++:')}")

py_list = "x = [1, 2, 3]"
print(f"Input:\n translate python to c++: {py_list}")
print(f"Output:\n {translate(py_list, 'translate python to c++:')}")


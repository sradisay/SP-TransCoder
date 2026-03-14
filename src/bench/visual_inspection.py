import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

CONFIG = {
    "max_len": 256,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


def main(model_name, tokenizer_name):
    print(f"Loading SP-TransCoder checkpoint from {model_name}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(CONFIG["device"])

    print(f"Loading tokenizer from {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

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
            repetition_penalty=1.0
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def run_test(input_code, source, target):
        print(f"\n{'=' * 40}")
        print(f"Task: {source} -> {target}")
        print(f"{'-' * 40}")
        print(f"Input:\n{input_code}")
        print(f"{'-' * 40}")
        output = translate(input_code, source_lang=source, target_lang=target)
        print(f"Output:\n{output}")
        print(f"{'=' * 40}")

    print(f"Device set to: {CONFIG['device']}")

    py_func = "def add(a, b):\n    return a + b"
    run_test(py_func, "Python", "C++")

    py_list = "x = [1, 2, 3]"
    run_test(py_list, "Python", "C++")

    py_main = "def add(a, b):\n    return a + b\n\nif __name__ == '__main__':\n    add(4, 10)"
    run_test(py_main, "Python", "C++")

    cpp_func = "int multiply(int a, int b) {\n    return a * b;\n}"
    run_test(cpp_func, "C++", "Python")

    cpp_vector = "std::vector<int> v = {1, 2, 3, 4, 5};"
    run_test(cpp_vector, "C++", "Python")

    cpp_main = "#include <iostream>\n\nint main() {\n    std::cout << \"Hello, World!\" << std::endl;\n    return 0;\n}"
    run_test(cpp_main, "C++", "Python")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="visualize performance of model on translation from C++ to Python and Python to C++")
    parser.add_argument(
        "--m",
        type=str,
        required=True,
        help="path or hf ID for the model"
    )
    parser.add_argument(
        "--t",
        type=str,
        required=False,
        default="",
        help="path or hf ID for the tokenizer. Default is model"
    )

    args = parser.parse_args()
    tok = args.t if args.t else args.m
    main(args.m, tok)
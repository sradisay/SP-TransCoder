from transformers import RobertaTokenizer, T5ForConditionalGeneration
from tokenizer.apply_tokenizer import salesforce_tokenizer

def translate_python_to_cpp(python_code):
    # 1. Load the Tokenizer
    # CodeT5 uses the RobertaTokenizer (not standard T5Tokenizer)
    tokenizer = salesforce_tokenizer

    # 2. Load the Model
    # We specifically request the "small" version as requested
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')

    # 3. Prepare the Input
    # We add a task prefix to tell the model what to do.
    # Note: CodeT5 is trained on specific prefixes like "Translate Python to C++:"
    task_prefix = "Translate Python to C++: "
    input_text = task_prefix + python_code

    # Tokenize the input
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # 4. Generate the Output
    # max_length controls how much code it generates.
    outputs = model.generate(input_ids, max_length=256)

    # 5. Decode the Output
    # skip_special_tokens removes the padding and end-of-sentence tokens
    cpp_code = tokenizer.decode(outputs[0], skip_special_tokens=False)

    return cpp_code


# --- Example Usage ---

python_snippet = """
print("hello world")
"""

print("--- Python Input ---")
print(python_snippet)

print("--- CodeT5-Small C++ Output ---")
translated_code = translate_python_to_cpp(python_snippet)
print(translated_code)
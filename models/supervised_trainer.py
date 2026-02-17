import torch
from torch.amp import GradScaler, autocast
from transformers import T5ForConditionalGeneration
from typing import List, Dict

class SupervisedTranslationTrainer:
    def __init__(self, config: Dict, model: T5ForConditionalGeneration, tokenizer):
        self.config = config
        self.device = config["device"]
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        # Optimization tools
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["lr"])
        self.scaler = GradScaler('cuda')

        # Enable Gradient Checkpointing
        self.model.gradient_checkpointing_enable()

        # Micro-batching for memory efficiency
        # SFT requires fitting two full sequences (input + label) into memory,
        # so this usually needs to be smaller than generation-only batches.
        self.micro_batch_size = 16
        self.accumulation_steps = config["batch_size"] // self.micro_batch_size

    def _prepare_inputs(self, source_codes: List[str], target_codes: List[str], source_lang: str, target_lang: str):
        """Prepares the prompt-based input and tokenizes the target labels."""
        # Using the natural language prefix CodeT5 is optimized for
        prefix = f"Translate {source_lang} to {target_lang}: "
        inputs = [prefix + text for text in source_codes]

        # Tokenize source
        model_inputs = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config["max_len"]
        ).to(self.device)

        # Tokenize target labels
        labels = self.tokenizer(
            target_codes,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config["max_len"]
        ).input_ids.to(self.device)

        # Replace padding token id with -100 so CrossEntropyLoss ignores it
        labels[labels == self.tokenizer.pad_token_id] = -100

        return model_inputs, labels

    def train_on_batch(self, source_codes: List[str], target_codes: List[str], source_lang: str, target_lang: str):
        """Standard supervised forward pass and loss calculation."""
        self.model.train()

        model_inputs, labels = self._prepare_inputs(source_codes, target_codes, source_lang, target_lang)

        with autocast('cuda'):
            outputs = self.model(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                labels=labels
            )
            loss = outputs.loss / self.accumulation_steps

        self.scaler.scale(loss).backward()
        return loss.item() * self.accumulation_steps

    def step_optimizer(self):
        """Updates weights after accumulation steps."""
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def run_epoch(self, paired_dataset, epoch: int):
        """
        paired_dataset should yield tuples of (python_batch, cpp_batch)
        where index `i` in both batches solves the exact same CodeNet problem.
        """
        total_loss = 0
        steps_per_epoch = 100 # Adjust based on your dataset size

        for step in range(steps_per_epoch):
            # 1. Get paired data (Real Python code and its C++ equivalent)
            py_batch, cpp_batch = paired_dataset.sample_paired_batch(self.micro_batch_size)

            # 2. Train Forward: Python -> C++
            loss_py2cpp = self.train_on_batch(py_batch, cpp_batch, "Python", "C++")

            # 3. Train Reverse: C++ -> Python
            # This teaches bidirectional alignment simultaneously
            loss_cpp2py = self.train_on_batch(cpp_batch, py_batch, "C++", "Python")

            total_loss += (loss_py2cpp + loss_cpp2py) / 2

            if (step + 1) % self.accumulation_steps == 0:
                self.step_optimizer()
                print(f"Epoch {epoch} | Step {step} | Avg SFT Loss: {total_loss / (step + 1):.4f}")
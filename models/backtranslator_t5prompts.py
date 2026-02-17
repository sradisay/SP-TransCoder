import torch
from torch.amp import GradScaler, autocast
from transformers import T5ForConditionalGeneration, AutoTokenizer
from typing import List, Dict


class BackTranslationTrainer:
    def __init__(self, config: Dict, model: T5ForConditionalGeneration):
        self.config = config
        self.device = config["device"]
        self.model = model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

        # Optimization tools
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["lr"])
        self.scaler = GradScaler('cuda') # floating point optimization

        # Enable Gradient Checkpointing to save memory at the cost of speed
        self.model.gradient_checkpointing_enable()

        self.micro_batch_size = 64
        self.accumulation_steps = config["batch_size"] // self.micro_batch_size

    def _prepare_inputs(self, source_texts: List[str], source_lang: str, target_lang: str):
        prefix = f"Translate {source_lang} to {target_lang}: "
        inputs = [prefix + text for text in source_texts]

        return self.tokenizer(
            inputs,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config["max_len"]
        ).to(self.model.device)

    @torch.no_grad()
    def generate_synthetic_batch(self, source_code: List[str], source_lang: str, target_lang: str) -> List[str]:
        """Step 1: Forward translation (Noisy generation)"""
        self.model.eval()
        inputs = self._prepare_inputs(source_code, source_lang, target_lang)

        generated_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=self.config["max_len"],
            num_beams=1,  # TODO: compare num_beams 1 with num_beams 2
            do_sample=False
        )

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def train_on_batch(self, synthetic_code: List[str], original_code: List[str]):
        self.model.train()

        inputs = self.tokenizer(
            synthetic_code,
            padding="max_length",
            truncation=True,
            max_length=self.config["max_len"],
            return_tensors="pt"
        ).to(self.device)

        labels = self.tokenizer(
            original_code,
            padding="max_length",
            truncation=True,
            max_length=self.config["max_len"],
            return_tensors="pt"
        ).input_ids.to(self.device)

        # Replace padding token id with -100 so it's ignored in loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100

        with autocast('cuda'):  # TODO: remove and check difference
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels
            )
            loss = outputs.loss / self.accumulation_steps

        self.scaler.scale(loss).backward()
        return loss.item() * self.accumulation_steps

    def step_optimizer(self):
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def run_epoch(self, dataset, epoch: int):
        langs = self.config["langs"]

        total_loss = 0
        steps_per_epoch = 100

        for step in range(steps_per_epoch):
            for i, source_lang in enumerate(langs):
                target_lang = langs[(i + 1) % len(langs)]

                real_source_batch = dataset.sample_batch(source_lang, self.micro_batch_size)
                synthetic_target_batch = self.generate_synthetic_batch(real_source_batch, source_lang, target_lang)

                loss_val = self.train_on_batch(synthetic_target_batch, real_source_batch)
                total_loss += loss_val

            if (step + 1) % self.accumulation_steps == 0:
                self.step_optimizer()
                print(f"Epoch {epoch} | Step {step} | Avg Loss: {total_loss / (step + 1):.4f}")
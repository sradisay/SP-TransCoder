import torch
import random
from torch.amp import GradScaler, autocast
from transformers import T5ForConditionalGeneration, AutoTokenizer
from typing import List, Dict


class DenoisingAutoEncoderTrainer:
    def __init__(self, config: Dict, model: T5ForConditionalGeneration):
        self.config = config
        self.device = config["device"]
        self.model = model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = config["lr"])
        self.scaler = GradScaler('cuda')

        self.model.gradient_checkpointing_enable()

        self.micro_batch_size = 64
        self.accumulation_steps = config["batch_size"] // self.micro_batch_size

        self.drop_prob = 0.1
        self.mask_prob = 0.15
        self.shuffle_dist = 3

    def _corrupt_sequence(self, code: str) -> str:
        tokens = code.split()
        if not tokens:
            return code

        corrupted_tokens = []
        for token in tokens:
            prob = random.random()
            if prob < self.drop_prob:
                continue
            elif prob < self.drop_prob + self.mask_prob:
                corrupted_tokens.append("<extra_id_0>")
            else:
                corrupted_tokens.append(token)

        indices = list(range(len(corrupted_tokens)))
        indices.sort(key = lambda x: x + random.uniform(-self.shuffle_dist, self.shuffle_dist))
        shuffled_tokens = [corrupted_tokens[i] for i in indices]

        return " ".join(shuffled_tokens)
    
    def delete_type(self, code, dell=0.1):
        token = code.split()

        for t in token:
            if random.random > dell:
                token.remove(t)
        
        return " ".join(token)

    def shuf_type(self, code):
        token = code.split()
        i = list(range(len(token)))
        random.shuffle(i)
        ch = [token[j] for j in i]
        return " ".join(ch)


    def extra_id_type(self, code, extra=0.2):
        token = code.split()
        ch = []
        i=0
        for t in token:
            if random.random() < extra:
                ch.append(f"<extra_id_{i}")
                i+=1
            else:
                ch.append(t)
        return " ".join(ch)


    def _prepare_inputs(self, corrupted_texts: List[str], lang: str):
        prefix = f"Reconstruct {lang}: "
        inputs = [prefix + text for text in corrupted_texts]

        return self.tokenizer(
            inputs,
            return_tensors = "pt",
            padding = "max_length",
            truncation = True,
            max_length = self.config["max_len"]
        ).to(self.model.device)
    
    # use this
    def _prepare_inputs_denoise(self, corrupted_texts: List[str], lang: str):
        prefix = f"Denoise {lang}: "
        inputs = [prefix + text for text in corrupted_texts]

        return self.tokenizer(
            inputs,
            return_tensors = "pt",
            padding = "max_length",
            truncation = True,
            max_length = self.config["max_len"]
        ).to(self.model.device)
    
    # chang
    def train_on_batch_difftypes(self, original_code: List[str], lang: str):
        self.model.train()
        # corrupted_code = [self._corrupt_sequence(code) for code in original_code]
        # corrupt_code_del = [self.delete_type(code) for code in original_code]
        # corrupt_code_extra = [self.extra_id_type(code) for code in original_code]
        # corrupt_code_shuffle = [self.shuf_type(code) for code in original_code]
        # corrupt_code = corrupt_code_extra + corrupt_code_del + corrupt_code_shuffle
        ccode= []

        for i in original_code:
            r = random.random()

            if r < 0.10:
                ccode.append(self.shuf_type(i))

            elif r < 0.30:
                ccode.append(self.delete_type(i))
            
            else:
                ccode.append(self.extra_id_type(i))

        inputs = self._prepare_inputs_denoise(ccode, lang)

        labels = self.tokenizer(
            original_code,
            padding = "max_length",
            truncation = True,
            max_length = self.config["max_len"],
            return_tensors = "pt"
        ).input_ids.to(self.device)

        labels[labels == self.tokenizer.pad_token_id] = -100

        with autocast('cuda'):
            outputs = self.model(
                input_ids = inputs["input_ids"],
                attention_mask = inputs["attention_mask"],
                labels = labels
            )
            loss = outputs.loss / self.accumulation_steps

        self.scaler.scale(loss).backward()
        return loss.item() * self.accumulation_steps

    def train_on_batch(self, original_code: List[str], lang: str):
        self.model.train()

        corrupted_code = [self._corrupt_sequence(code) for code in original_code]

        inputs = self._prepare_inputs(corrupted_code, lang)

        labels = self.tokenizer(
            original_code,
            padding = "max_length",
            truncation = True,
            max_length = self.config["max_len"],
            return_tensors = "pt"
        ).input_ids.to(self.device)

        labels[labels == self.tokenizer.pad_token_id] = -100

        with autocast('cuda'):
            outputs = self.model(
                input_ids = inputs["input_ids"],
                attention_mask = inputs["attention_mask"],
                labels = labels
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
            for lang in langs:
                real_source_batch = dataset.sample_batch(lang, self.micro_batch_size)

                loss_val = self.train_on_batch(
                    original_code = real_source_batch,
                    lang = lang
                )
                total_loss += loss_val

            if (step + 1) % self.accumulation_steps == 0:
                self.step_optimizer()
                print(f"Epoch {epoch} | Step {step} | Avg Loss: {total_loss / (step + 1):.4f}")
    
    def run_epoch_difftypes(self, dataset, epoch: int):
        langs = self.config["langs"]

        total_loss = 0
        steps_per_epoch = 100

        for step in range(steps_per_epoch):
            for lang in langs:
                real_source_batch = dataset.sample_batch(lang, self.micro_batch_size)

                loss_val = self.train_on_batch_difftypes(
                    original_code = real_source_batch,
                    lang = lang
                )
                total_loss += loss_val

            if (step + 1) % self.accumulation_steps == 0:
                self.step_optimizer()
                print(f"Epoch {epoch} | Step {step} | Avg Loss: {total_loss / (step + 1):.4f}")

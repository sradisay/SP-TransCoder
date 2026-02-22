import torch
from torch import autocast
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, T5ForConditionalGeneration, get_cosine_schedule_with_warmup
from utils.noise import CodeCorruptor

class BackTranslator:
    def __init__(self, config):
        self.cfg = config
        self.device = config["device"]
        
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        self.model = T5ForConditionalGeneration.from_pretrained(config["model_name"]).to(self.device)
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 
             'weight_decay': config["weight_decay"]},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 
             'weight_decay': 0.0}
        ]
        
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config["lr"])
        
        total_steps = (config["dae_epochs"] + config["bt_epochs"]) * config["steps_per_epoch"]
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=config["warmup_steps"], 
            num_training_steps=total_steps
        )
        self.corruptor = CodeCorruptor()
        
    def _tokenize(self, texts, src_lang, tgt_lang):
        prefix = f"Translate {src_lang} to {tgt_lang}: "
        inputs = [prefix + t for t in texts]
        return self.tokenizer(
            inputs, return_tensors="pt", padding=True, 
            truncation=True, max_length=self.cfg["max_len"]
        ).to(self.device)

    def _get_labels(self, texts):
        labels = self.tokenizer(
            texts, return_tensors="pt", padding=True, 
            truncation=True, max_length=self.cfg["max_len"]
        ).input_ids.to(self.device)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return labels

    def _optimize_step(self, loss):
        """Clean optimization step for bfloat16."""
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        self.optimizer.zero_grad(set_to_none=True)

    def train_dae_step(self, real_codes, lang):
        self.model.train()
        corrupted_codes = self.corruptor.corrupt_batch(real_codes)
        
        inputs = self._tokenize(corrupted_codes, lang, lang)
        labels = self._get_labels(real_codes)
        
        with autocast(device_type=self.device.type, dtype=torch.bfloat16):
            loss = self.model(**inputs, labels=labels).loss
            
        self._optimize_step(loss)
        return loss.item()

    @torch.no_grad()
    def generate_pseudo_targets(self, sources, src_lang, tgt_lang):
        self.model.eval()
        inputs = self._tokenize(sources, src_lang, tgt_lang)
        
        with autocast(device_type=self.device.type, dtype=torch.bfloat16):
            generated_ids = self.model.generate(
                **inputs, 
                max_length=self.cfg["max_len"], 
                num_beams=1, 
                do_sample=False,
                use_cache=True 
            )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def train_bt_step(self, src_examples, src_lang, tgt_lang):
        # 1. Generate pseudo-targets (eval mode)
        pseudo_targets = self.generate_pseudo_targets(src_examples, src_lang, tgt_lang)
        
        # 2. Add noise to pseudo-targets using BART-style masking
        noisy_pseudo_targets = self.corruptor.corrupt_batch(pseudo_targets)
        
        # 3. Train to translate back to source (train mode)
        self.model.train()
        inputs = self._tokenize(noisy_pseudo_targets, tgt_lang, src_lang)
        labels = self._get_labels(src_examples)
        
        with autocast(device_type=self.device.type, dtype=torch.bfloat16):
            loss = self.model(**inputs, labels=labels).loss
            
        self._optimize_step(loss)
        return loss.item()

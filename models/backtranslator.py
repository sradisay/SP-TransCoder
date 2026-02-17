import torch
from torch import GradScaler, autocast
from transformers import AutoTokenizer, T5ForConditionalGeneration
from utils.noise import CodeCorruptor

class BackTranslator:
    def __init__(self, config):
        self.cfg = config
        self.device = config["device"]
        
        # Load Model and Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        self.model = T5ForConditionalGeneration.from_pretrained(config["model_name"]).to(self.device)
        
        # Optimization
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["lr"])
        self.scaler = GradScaler(self.device.type)
        
        # Initialize Noise Injector (T5 uses <extra_id_X> for masks)
        self.corruptor = CodeCorruptor(mask_token="<extra_id_0>")
        
    def _tokenize(self, texts, src_lang, tgt_lang):
        """Prepends prompt and tokenizes inputs."""
        prefix = f"Translate {src_lang} to {tgt_lang}: "
        inputs = [prefix + t for t in texts]
        return self.tokenizer(
            inputs, return_tensors="pt", padding=True, 
            truncation=True, max_length=self.cfg["max_len"]
        ).to(self.device)

    def _get_labels(self, texts):
        """Tokenizes targets and masks padding for CrossEntropy loss."""
        labels = self.tokenizer(
            texts, return_tensors="pt", padding=True, 
            truncation=True, max_length=self.cfg["max_len"]
        ).input_ids.to(self.device)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return labels

    def train_dae_step(self, real_codes, lang):
        """Denoising Auto-Encoding step: Corrupt code -> Reconstruct original."""
        self.model.train()
        self.optimizer.zero_grad()
        
        corrupted_codes = self.corruptor.corrupt_batch(real_codes)
        inputs = self._tokenize(corrupted_codes, lang, lang)
        labels = self._get_labels(real_codes)
        
        with autocast(self.device.type):
            loss = self.model(**inputs, labels=labels).loss
            
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

    @torch.no_grad()
    def generate_pseudo_targets(self, sources, src_lang, tgt_lang):
        """Forward translation to create synthetic target data."""
        self.model.eval()
        inputs = self._tokenize(sources, src_lang, tgt_lang)
        
        # FIX 2: Lower temperature slightly to prevent generating complete garbage, 
        # but keep sampling to prevent deterministic copying.
        generated_ids = self.model.generate(
            **inputs, 
            max_length=self.cfg["max_len"], 
            do_sample=True, 
            temperature=0.6, 
            top_p=0.9,
            repetition_penalty=1.2 # Helps prevent repeating the input exactly
        )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def train_bt_step(self, src_examples, src_lang, tgt_lang):
        """Iterative Back-Translation step: Target to Source."""
        # 1. Generate Fake Targets (e.g., Real Python -> Fake C++)
        pseudo_targets = self.generate_pseudo_targets(src_examples, src_lang, tgt_lang)
        
        # FIX 3: THE ANTI-COPYING TRAP (Noisy BT)
        # We MUST corrupt the synthetic data. If the model generated pure Python 
        # instead of C++, the noise forces it to reconstruct the source rather 
        # than blindly copying.
        noisy_pseudo_targets = self.corruptor.corrupt_batch(pseudo_targets)
        
        # 2. Train model to reconstruct Real Source (e.g., Noisy Fake C++ -> Real Python)
        self.model.train()
        self.optimizer.zero_grad()
        
        inputs = self._tokenize(noisy_pseudo_targets, tgt_lang, src_lang)
        labels = self._get_labels(src_examples)
        
        with autocast(self.device.type):
            loss = self.model(**inputs, labels=labels).loss
            
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

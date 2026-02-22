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
        
        # WE DO NOT ADD SPECIAL TOKENS. We rely entirely on the pre-trained embeddings 
        # of the words "Python" and "C++" to maintain the latent space.
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 
             'weight_decay': config["weight_decay"]},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 
             'weight_decay': 0.0}
        ]
        
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config["lr"])
        
        dae_steps = config["dae_epochs"] * config["steps_per_epoch"]
        bt_steps = config["bt_epochs"] * config["steps_per_epoch"] * 2
        total_steps = dae_steps + bt_steps
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=config["warmup_steps"], 
            num_training_steps=total_steps
        )
        
        self.corruptor = CodeCorruptor()
        
    def _optimize_step(self, loss):
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _tokenize_encoder(self, texts, src_lang):
        """PREPEND the language so it never gets truncated by max_len."""
        inputs = [f"{src_lang}:\n{t}" for t in texts]
        return self.tokenizer(
            inputs, return_tensors="pt", padding=True, 
            truncation=True, max_length=self.cfg["max_len"]
        ).to(self.device)

    def _get_labels(self, texts, tgt_lang):
        """PREPEND the target language to the labels. The model learns it MUST start with this."""
        labels_text = [f"{tgt_lang}:\n{t}" for t in texts]
        labels = self.tokenizer(
            labels_text, return_tensors="pt", padding=True, 
            truncation=True, max_length=self.cfg["max_len"]
        ).input_ids.to(self.device)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return labels

    def _get_forced_decoder_ids(self, tgt_lang, batch_size):
        """Builds the native decoder forcing prefix: <pad> target_lang:\n"""
        # Encode the prefix string into pre-trained token IDs
        prefix_ids = self.tokenizer(f"{tgt_lang}:\n", add_special_tokens=False).input_ids
        
        # Combine [PAD] + [prefix_ids]
        forced_sequence = [self.tokenizer.pad_token_id] + prefix_ids
        
        # Expand to batch size
        decoder_input_ids = torch.tensor([forced_sequence] * batch_size, device=self.device)
        return decoder_input_ids

    def train_dae_step(self, real_codes, lang):
        self.model.train()
        corrupted_codes = self.corruptor.corrupt_batch(real_codes)
        
        inputs = self._tokenize_encoder(corrupted_codes, lang)
        labels = self._get_labels(real_codes, lang)
        
        with autocast(device_type=self.device.type, dtype=torch.bfloat16):
            loss = self.model(**inputs, labels=labels).loss
            
        self._optimize_step(loss)
        return loss.item()

    @torch.no_grad()
    def generate_pseudo_targets(self, sources, tgt_lang, src_lang):
        self.model.eval()
        inputs = self._tokenize_encoder(sources, src_lang)
        
        # NATIVE DECODER FORCING
        batch_size = inputs.input_ids.shape[0]
        decoder_input_ids = self._get_forced_decoder_ids(tgt_lang, batch_size)
        
        with autocast(device_type=self.device.type, dtype=torch.bfloat16):
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=decoder_input_ids, # Forced!
                max_length=self.cfg["max_len"], 
                num_beams=1, 
                do_sample=False,
                use_cache=True 
            )
            
        # Strip the forced prefix out of the output so we just get clean code back
        decoded_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        clean_pseudo_targets = [text.replace(f"{tgt_lang}:\n", "", 1).strip() for text in decoded_texts]
        
        return clean_pseudo_targets

    def train_bt_step(self, src_examples, src_lang, tgt_lang):
        pseudo_targets = self.generate_pseudo_targets(src_examples, tgt_lang, src_lang)
        noisy_pseudo_targets = self.corruptor.corrupt_batch(pseudo_targets)
        
        self.model.train()
        
        inputs = self._tokenize_encoder(noisy_pseudo_targets, tgt_lang)
        labels = self._get_labels(src_examples, src_lang)
        
        with autocast(device_type=self.device.type, dtype=torch.bfloat16):
            loss = self.model(**inputs, labels=labels).loss
            
        self._optimize_step(loss)
        return loss.item()

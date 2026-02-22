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
        
        # Add language tokens
        self.lang_tokens = [f"<{lang}>" for lang in config["langs"]]
        self.tokenizer.add_special_tokens({"additional_special_tokens": self.lang_tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))
        
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
        """Clean optimization step for bfloat16."""
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _tokenize_encoder(self, texts, src_lang):
        """Append source lang to input so encoder knows what it's looking at"""
        inputs = [f"{t} <{src_lang}>" for t in texts]
        return self.tokenizer(
            inputs, return_tensors="pt", padding=True, 
            truncation=True, max_length=self.cfg["max_len"]
        ).to(self.device)

    def _get_labels(self, texts, tgt_lang):
        """Prepend target lang to output. Model MUST predict it first."""
        labels_text = [f"<{tgt_lang}> {t}" for t in texts]
        labels = self.tokenizer(
            labels_text, return_tensors="pt", padding=True, 
            truncation=True, max_length=self.cfg["max_len"]
        ).input_ids.to(self.device)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return labels

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
    def generate_pseudo_targets(self, sources, tgt_lang, src_lang="Unknown"):
        self.model.eval()
        inputs = self._tokenize_encoder(sources, src_lang)
        
        # DECODER FORCING
        batch_size = inputs.input_ids.shape[0]
        lang_id = self.tokenizer.convert_tokens_to_ids(f"<{tgt_lang}>")
        decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id, lang_id]] * batch_size).to(self.device)
        
        with autocast(device_type=self.device.type, dtype=torch.bfloat16):
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=decoder_input_ids,
                max_length=self.cfg["max_len"], 
                num_beams=1, 
                do_sample=False,
                use_cache=True 
            )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

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

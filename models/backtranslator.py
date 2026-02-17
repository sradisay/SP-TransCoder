import torch
from transformers import AutoModelForSeq2SeqLM, AddedToken, AutoTokenizer


class BackTranslator:
    def __init__(self, config):
        self.cfg = config

        # Load fast tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])

        # Register special tokens safely with fast tokenizer
        special_tokens = [AddedToken(f"<{l}>") for l in config["langs"]]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(config["device"])

    def _tokenize(self, texts, lang_tag):
        """Prepend language tag and tokenize batch."""
        texts = [f"<{lang_tag}> {t}" for t in texts]
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg["max_len"]
        ).to(self.cfg["device"])

    @torch.no_grad()
    def generate_pseudo_targets(self, sources, tgt_lang):
        """Generate pseudo-translations (back-translation)."""
        self.model.eval()
        inputs = self._tokenize(sources, tgt_lang)
        generated_ids = self.model.generate(
            **inputs,
            max_length=self.cfg["max_len"],
            num_beams=2
        )
        return [self.tokenizer.decode(g, skip_special_tokens=True).strip() for g in generated_ids]

    def train_step(self, src_examples, src_lang, tgt_lang, optimizer):
        """Single back-translation training step."""
        self.model.train()

        # 1. Generate pseudo-targets: src -> tgt
        pseudo_targets = self.generate_pseudo_targets(src_examples, tgt_lang)

        # 2. Tokenize pseudo-targets (input) and original source (labels)
        inputs = self._tokenize(pseudo_targets, src_lang)
        labels = self._tokenize(src_examples, src_lang).input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100  # ignore padding

        optimizer.zero_grad()
        outputs = self.model(**inputs, labels=labels)
        outputs.loss.backward()
        optimizer.step()

        return outputs.loss.item()

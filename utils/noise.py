import random
import re
from typing import List

class CodeCorruptor:
    def __init__(self, drop_prob=0.10, mask_prob=0.15):
        self.drop_prob = drop_prob
        self.mask_prob = mask_prob

    def corrupt(self, text: str) -> str:
        if not text:
            return text
            
        # Split by whitespace, preserving actual whitespace characters
        tokens = re.split(r'(\s+)', text)
        
        corrupted = []
        sentinel_idx = 0
        i = 0
        
        while i < len(tokens):
            if not tokens[i].strip(): # Always keep whitespace intact
                corrupted.append(tokens[i])
                i += 1
                continue
                
            rand = random.random()
            if rand < self.mask_prob:
                # T5 Span Masking: Replace token(s) with <extra_id_X>
                # Use up to 99 extra ids (T5 default limit)
                if sentinel_idx < 100:
                    corrupted.append(f"<extra_id_{sentinel_idx}>")
                    sentinel_idx += 1
                
                # TransCoder masks SPANS of length ~1-3, not just single tokens
                span_length = random.randint(1, 3)
                while span_length > 0 and i < len(tokens):
                    if tokens[i].strip():
                        span_length -= 1
                    i += 1
            elif rand < (self.drop_prob + self.mask_prob):
                # Drop token entirely
                i += 1
            else:
                corrupted.append(tokens[i])
                i += 1
                
        result = "".join(corrupted)
        return result if result.strip() else "<extra_id_0>"

    def corrupt_batch(self, texts: List[str]) -> List[str]:
        return [self.corrupt(text) for text in texts]

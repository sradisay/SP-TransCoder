import random
import re
from typing import List

class CodeCorruptor:
    def __init__(self, drop_prob=0.10, mask_prob=0.15):
        self.drop_prob = drop_prob
        self.mask_prob = mask_prob
        
        # CodeT5 uses RobertaTokenizer. 
        # The native mask token is <mask>, not <extra_id_X>.
        self.mask_token = "<mask>" 

    def corrupt(self, text: str) -> str:
        if not text:
            return text
            
        # Split by whitespace, preserving actual whitespace characters for Python's syntax
        tokens = re.split(r'(\s+)', text)
        
        corrupted = []
        i = 0
        
        while i < len(tokens):
            if not tokens[i].strip(): 
                corrupted.append(tokens[i])
                i += 1
                continue
                
            rand = random.random()
            if rand < self.mask_prob:
                # BART-style text infilling: Replace span of 1-3 tokens with a SINGLE <mask>
                corrupted.append(self.mask_token)
                
                span_length = random.randint(1, 3)
                while span_length > 0 and i < len(tokens):
                    if tokens[i].strip():
                        span_length -= 1
                    i += 1
            elif rand < (self.drop_prob + self.mask_prob):
                # Drop token entirely (simulating deletion noise)
                i += 1
            else:
                corrupted.append(tokens[i])
                i += 1
                
        result = "".join(corrupted)
        return result if result.strip() else self.mask_token

    def corrupt_batch(self, texts: List[str]) -> List[str]:
        return [self.corrupt(text) for text in texts]

import random
import re
from typing import List

class CodeCorruptor:
    def __init__(self, mask_token="<extra_id_0>", drop_prob=0.10, mask_prob=0.15):
        self.mask_token = mask_token
        self.drop_prob = drop_prob
        self.mask_prob = mask_prob

    def corrupt(self, text: str) -> str:
        if not text:
            return text
            
        # Split by whitespace but keep the whitespace characters (spaces, \n, \t) intact!
        tokens = re.split(r'(\s+)', text)
        
        corrupted = []
        for token in tokens:
            if not token.strip():  # If it's pure whitespace (\n, \t, "  ")
                corrupted.append(token)
                continue
                
            rand = random.random()
            if rand < self.drop_prob:
                continue  # Drop token
            elif rand < (self.drop_prob + self.mask_prob):
                corrupted.append(self.mask_token)  # Mask token
            else:
                corrupted.append(token)  # Keep token
                
        result = "".join(corrupted)
        return result if result.strip() else self.mask_token

    def corrupt_batch(self, texts: List[str]) -> List[str]:
        return [self.corrupt(text) for text in texts]

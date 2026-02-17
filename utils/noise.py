import random
from typing import List

class CodeCorruptor:
    def __init__(self, mask_token="<extra_id_0>", drop_prob=0.15, mask_prob=0.15):
        self.mask_token = mask_token
        self.drop_prob = drop_prob
        self.mask_prob = mask_prob

    def corrupt(self, text: str) -> str:
        tokens = text.split()
        if not tokens:
            return text
            
        corrupted = []
        for token in tokens:
            rand = random.random()
            if rand < self.drop_prob:
                continue  # Drop
            elif rand < (self.drop_prob + self.mask_prob):
                corrupted.append(self.mask_token)  # Mask
            else:
                corrupted.append(token)  # Keep
                
        return " ".join(corrupted) if corrupted else self.mask_token

    def corrupt_batch(self, texts: List[str]) -> List[str]:
        return [self.corrupt(text) for text in texts]

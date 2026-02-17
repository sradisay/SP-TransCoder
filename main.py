import random
import torch
from dataset import UnpairedCodeDataset
from models.backtranslator import BackTranslator

CONFIG = {
    "model_name": "Salesforce/codet5-small",
    "langs": ["Python", "C++"],
    
    "batch_size": 32,            # TransCoder used ~32 sequences per GPU 
    "dae_epochs": 5,             # Increased for a stronger "warmup" (TransCoder XLM phase) 
    "bt_epochs": 20,             # Back-translation requires more iterations to converge 
    "steps_per_epoch": 100,      # Increased to ensure sufficient data exposure per epoch
    
    "lr": 2e-4,                  # CodeT5/T5 typically uses higher LR (1e-4 to 3e-4) than BERT 
    "max_len": 64,              # Both papers emphasize longer sequences for code context 
    "weight_decay": 0.01,        # Standard for AdamW in these architectures
    "warmup_steps": 1000,        # Essential for stabilizing T5-based models
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

def main():
    print(f"Initializing Trainer and Dataset on {CONFIG['device']}...")
    trainer = BackTranslator(CONFIG)
    dataset = UnpairedCodeDataset(languages=CONFIG["langs"])
    
    print("\n--- Phase 1: Denoising Auto-Encoding (DAE) ---")
    for epoch in range(CONFIG["dae_epochs"]):
        total_loss = 0
        for i in range(CONFIG["steps_per_epoch"]):
            # Randomly select a language to auto-encode
            lang = random.choice(CONFIG["langs"])
            batch = dataset.sample_batch(lang, CONFIG["batch_size"])
            
            if batch[0]: # Ensure batch isn't empty
                total_loss += trainer.train_dae_step(batch, lang)
 
        avg_loss = total_loss / CONFIG["steps_per_epoch"]
        print(f"DAE Epoch {epoch+1}/{CONFIG['dae_epochs']} | Avg Loss: {avg_loss:.4f}")

    print("\n--- Phase 2: Iterative Back-Translation (BT) ---")
    for epoch in range(CONFIG["bt_epochs"]):
        total_loss = 0
        for _ in range(CONFIG["steps_per_epoch"]):
            # Pick source and target randomly
            src_lang, tgt_lang = random.sample(CONFIG["langs"], 2)
            batch = dataset.sample_batch(src_lang, CONFIG["batch_size"])
            
            if batch[0]: 
                total_loss += trainer.train_bt_step(batch, src_lang, tgt_lang)
                
        avg_loss = total_loss / CONFIG["steps_per_epoch"]
        print(f"BT Epoch {epoch+1}/{CONFIG['bt_epochs']} | Avg Loss: {avg_loss:.4f}")
        
if __name__ == "__main__":
    main()

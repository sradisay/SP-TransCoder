import os
import random
import torch
from dataset import UnpairedCodeDataset
from models.backtranslator import BackTranslator

CONFIG = {
    "model_name": "Salesforce/codet5-small",
    "langs": ["Python", "C++"],
    "batch_size": 32,            
    "dae_epochs": 10,             
    "bt_epochs": 10,             
    "steps_per_epoch": 100,      
    "lr": 2e-4,                  
    "max_len": 128,              
    "weight_decay": 0.01,        
    "warmup_steps": 1000,        
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

def run_inference(trainer, dataset, num_examples=2):
    """Performs translation on random samples from the dataset."""
    trainer.model.eval()
    print("\n" + "="*30)
    print("RUNNING INFERENCE ON SAMPLES")
    print("="*30)

    for src_lang, tgt_lang in [("Python", "C++"), ("C++", "Python")]:
        samples = dataset.sample_batch(src_lang, num_examples)
        for i, code in enumerate(samples):
            if not code.strip(): continue
            
            prefix = f"Translate {src_lang} to {tgt_lang}: "
            inputs = trainer.tokenizer(prefix + code, return_tensors="pt", 
                                     truncation=True, max_length=CONFIG["max_len"]).to(CONFIG["device"])
            
            with torch.no_grad():
                out_ids = trainer.model.generate(**inputs, max_length=CONFIG["max_len"], num_beams=4)
                translation = trainer.tokenizer.decode(out_ids[0], skip_special_tokens=True)
            
            print(f"\n[{src_lang} Example {i+1}]")
            print(code[:150] + "..." if len(code) > 150 else code)
            print(f"\n[Translated {tgt_lang}]")
            print(translation)
            print("-" * 20)

def main():
    print(f"Starting pipeline on {CONFIG['device']}...")
    trainer = BackTranslator(CONFIG)
    dataset = UnpairedCodeDataset(languages=CONFIG["langs"])
    
    os.makedirs("checkpoints", exist_ok=True)
    
    # PHASE 1: Denoising Auto-Encoding (DOBF)
    print("\n--- Phase 1: Denoising Auto-Encoding (DOBF) ---")
    for epoch in range(CONFIG["dae_epochs"]):
        total_loss = 0
        for _ in range(CONFIG["steps_per_epoch"]):
            lang = random.choice(CONFIG["langs"])
            batch = dataset.sample_batch(lang, CONFIG["batch_size"])
            if batch[0]:
                total_loss += trainer.train_dae_step(batch, lang)
        
        print(f"DAE Epoch {epoch+1}/{CONFIG['dae_epochs']} | Avg Loss: {total_loss/CONFIG['steps_per_epoch']:.4f}")

    # Save Phase 1 result
    dae_path = "checkpoints/model_after_dobf"
    trainer.model.save_pretrained(dae_path)
    trainer.tokenizer.save_pretrained(dae_path)
    print(f"DOBF model saved to {dae_path}")

    # PHASE 2: Iterative Back-Translation (BT)
    print("\n--- Phase 2: Iterative Back-Translation (BT) ---")
    for epoch in range(CONFIG["bt_epochs"]):
        total_loss = 0
        for _ in range(CONFIG["steps_per_epoch"]):
            src, tgt = random.sample(CONFIG["langs"], 2)
            batch = dataset.sample_batch(src, CONFIG["batch_size"])
            if batch[0]:
                total_loss += trainer.train_bt_step(batch, src, tgt)
        
        print(f"BT Epoch {epoch+1}/{CONFIG['bt_epochs']} | Avg Loss: {total_loss/CONFIG['steps_per_epoch']:.4f}")

    # Save Phase 2 result
    bt_path = "checkpoints/model_after_full_training"
    trainer.model.save_pretrained(bt_path)
    trainer.tokenizer.save_pretrained(bt_path)
    print(f"Full training model saved to {bt_path}")

    # Final Inference
    run_inference(trainer, dataset)

if __name__ == "__main__":
    main()

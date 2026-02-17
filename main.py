import os
import random
import torch
from dataset import UnpairedCodeDataset
from models.backtranslator import BackTranslator

CONFIG = {
    "model_name": "Salesforce/codet5-small",
    "langs": ["Python", "C++"],
    
    "batch_size": 32,            
    "dae_epochs": 5,             
    "bt_epochs": 20,             
    "steps_per_epoch": 100,      
    
    "lr": 2e-4,                  
    "max_len": 64,              
    "weight_decay": 0.01,        
    "warmup_steps": 1000,        
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

def translate_example(text, src_lang, tgt_lang, trainer):
    """Helper function for deterministic inference using beam search."""
    trainer.model.eval()
    prefix = f"Translate {src_lang} to {tgt_lang}: "
    
    inputs = trainer.tokenizer(
        prefix + text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=CONFIG["max_len"]
    ).to(CONFIG["device"])
    
    with torch.no_grad():
        outputs = trainer.model.generate(
            **inputs, 
            max_length=CONFIG["max_len"], 
            num_beams=4,           # Use beam search for higher quality inference
            early_stopping=True
        )
        
    return trainer.tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    print(f"Initializing Trainer and Dataset on {CONFIG['device']}...")
    trainer = BackTranslator(CONFIG)
    dataset = UnpairedCodeDataset(languages=CONFIG["langs"])
    
    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # ==========================================
    # PHASE 1: Denoising Auto-Encoding (Warmup)
    # ==========================================
    print("\n--- Phase 1: Denoising Auto-Encoding (DAE) ---")
    for epoch in range(CONFIG["dae_epochs"]):
        total_loss = 0
        for i in range(CONFIG["steps_per_epoch"]):
            lang = random.choice(CONFIG["langs"])
            batch = dataset.sample_batch(lang, CONFIG["batch_size"])
            
            if batch[0]: 
                total_loss += trainer.train_dae_step(batch, lang)
 
        avg_loss = total_loss / CONFIG["steps_per_epoch"]
        print(f"DAE Epoch {epoch+1}/{CONFIG['dae_epochs']} | Avg Loss: {avg_loss:.4f}")

    # Save Phase 1 Model
    dae_save_path = "checkpoints/codet5_dae_final"
    print(f"\nSaving DAE Phase model to '{dae_save_path}'...")
    trainer.model.save_pretrained(dae_save_path)
    trainer.tokenizer.save_pretrained(dae_save_path)

    # ==========================================
    # PHASE 2: Iterative Back-Translation (BT)
    # ==========================================
    print("\n--- Phase 2: Iterative Back-Translation (BT) ---")
    for epoch in range(CONFIG["bt_epochs"]):
        total_loss = 0
        for _ in range(CONFIG["steps_per_epoch"]):
            src_lang, tgt_lang = random.sample(CONFIG["langs"], 2)
            batch = dataset.sample_batch(src_lang, CONFIG["batch_size"])
            
            if batch[0]: 
                total_loss += trainer.train_bt_step(batch, src_lang, tgt_lang)
                
        avg_loss = total_loss / CONFIG["steps_per_epoch"]
        print(f"BT Epoch {epoch+1}/{CONFIG['bt_epochs']} | Avg Loss: {avg_loss:.4f}")

    # Save Phase 2 Model
    bt_save_path = "checkpoints/codet5_bt_final"
    print(f"\nSaving Full Training (BT) model to '{bt_save_path}'...")
    trainer.model.save_pretrained(bt_save_path)
    trainer.tokenizer.save_pretrained(bt_save_path)
    
    # ==========================================
    # INFERENCE / EVALUATION
    # ==========================================
    print("\n--- Inference on Dataset Examples ---")
    
    # Example 1: Python to C++
    py_sample = dataset.sample_batch("Python", 1)[0]
    if py_sample:
        print("\n[Input Python]:")
        print(py_sample)
        print("-" * 20)
        print("[Translated C++]:")
        print(translate_example(py_sample, "Python", "C++", trainer))

    # Example 2: C++ to Python
    cpp_sample = dataset.sample_batch("C++", 1)[0]
    if cpp_sample:
        print("\n[Input C++]:")
        print(cpp_sample)
        print("-" * 20)
        print("[Translated Python]:")

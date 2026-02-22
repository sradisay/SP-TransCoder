import os
import random
import torch
from transformers import T5ForConditionalGeneration
from dataset import UnpairedCodeDataset
from models.backtranslator import BackTranslator

CONFIG = {
    "model_name": "Salesforce/codet5-small",
    "langs": ["Python", "C++"],
    "batch_size": 32,             
    "dae_epochs": 30,             
    "bt_epochs": 20,
    "steps_per_epoch": 200,       
    "lr": 1e-4,                   
    "max_len": 256,
    "weight_decay": 0.01,        
    "warmup_steps": 1000,         
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dae_ckpt": "checkpoints/codet5_dae_base",
    "bt_ckpt": "checkpoints/codet5_bt_final"
}

def run_inference(trainer, dataset, num_examples=3):
    trainer.model.eval()
    print("\n" + "="*30 + "\nRUNNING INFERENCE ON SAMPLES\n" + "="*30)
    for src_lang, tgt_lang in [("Python", "C++"), ("C++", "Python")]:
        samples = dataset.sample_batch(src_lang, limit=num_examples) 
        for i, code in enumerate(samples):
            if not code.strip(): continue
            
            prefix = f"Translate to {tgt_lang}: "
            inputs = trainer.tokenizer(prefix + code, return_tensors="pt", 
                                     truncation=True, max_length=CONFIG["max_len"]).to(CONFIG["device"])
            
            with torch.no_grad(), torch.autocast(device_type=CONFIG["device"].type, dtype=torch.bfloat16):
                out_ids = trainer.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=CONFIG["max_len"], 
                    num_beams=4
                )
                translation = trainer.tokenizer.decode(out_ids[0], skip_special_tokens=True)
                
            print(f"\n[{src_lang} Example {i+1}]\n{code[:200]}...")
            print(f"\n[Translated {tgt_lang}]\n{translation}\n" + "-"*40)

def main():
    print(f"Starting pipeline on {CONFIG['device']}...")
    os.makedirs("checkpoints", exist_ok=True)
    
    dataset = UnpairedCodeDataset(languages=CONFIG["langs"], batch_size=CONFIG["batch_size"])
    trainer = BackTranslator(CONFIG)
    
    # --- Phase 1: Denoising Auto-Encoding (DOBF) ---
    if os.path.exists(CONFIG["dae_ckpt"]):
        print(f"\n[INFO] Found existing DAE checkpoint at '{CONFIG['dae_ckpt']}'.")
        print("[INFO] Skipping Phase 1 and loading pre-trained DAE weights...")
        
        trainer.model = T5ForConditionalGeneration.from_pretrained(CONFIG["dae_ckpt"]).to(CONFIG["device"])
    else:
        print("\n--- Phase 1: Denoising Auto-Encoding (DOBF) ---")
        for epoch in range(CONFIG["dae_epochs"]):
            total_loss = 0
            for _ in range(CONFIG["steps_per_epoch"]):
                lang = random.choice(CONFIG["langs"])
                batch = dataset.sample_batch(lang)
                if batch:
                    total_loss += trainer.train_dae_step(batch, lang)
            print(f"DAE Epoch {epoch+1}/{CONFIG['dae_epochs']} | Avg Loss: {total_loss/CONFIG['steps_per_epoch']:.4f}")
        
        print("\n[Running Baseline Inference after DAE Phase]")
        run_inference(trainer, dataset)
        
        print(f"\nSaving DAE checkpoint to '{CONFIG['dae_ckpt']}'...")
        trainer.model.save_pretrained(CONFIG["dae_ckpt"])
        trainer.tokenizer.save_pretrained(CONFIG["dae_ckpt"])

    # --- Phase 2: Alternating DAE + BT ---
    print("\n--- Phase 2: Alternating DAE + BT ---")
    for epoch in range(CONFIG["bt_epochs"]):
        dae_loss_total = 0
        bt_loss_total = 0
        
        for _ in range(CONFIG["steps_per_epoch"]):
            # DAE
            lang = random.choice(CONFIG["langs"])
            dae_batch = dataset.sample_batch(lang)
            if dae_batch:
                dae_loss_total += trainer.train_dae_step(dae_batch, lang)

            # BT
            src, tgt = random.sample(CONFIG["langs"], 2)
            bt_batch = dataset.sample_batch(src)
            if bt_batch:
                bt_loss_total += trainer.train_bt_step(bt_batch, src, tgt)
        
        print(f"Epoch {epoch+1} | DAE Loss: {dae_loss_total/CONFIG['steps_per_epoch']:.4f} | BT Loss: {bt_loss_total/CONFIG['steps_per_epoch']:.4f}")
        run_inference(trainer, dataset)

    print(f"\nSaving final BT checkpoint to '{CONFIG['bt_ckpt']}'...")
    trainer.model.save_pretrained(CONFIG["bt_ckpt"])
    trainer.tokenizer.save_pretrained(CONFIG["bt_ckpt"])

if __name__ == "__main__":
    main()

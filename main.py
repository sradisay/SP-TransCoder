import os
import random
import torch
from dataset import UnpairedCodeDataset
from models.backtranslator import BackTranslator

CONFIG = {
    "model_name": "Salesforce/codet5-small",
    "langs": ["Python", "C++"],
    "batch_size": 16,
    "dae_epochs": 10,             
    "bt_epochs": 20,
    "steps_per_epoch": 200,
    "lr": 1e-4,
    "max_len": 256,
    "weight_decay": 0.01,        
    "warmup_steps": 1000,        
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

def run_inference(trainer, dataset, num_examples=3):
    trainer.model.eval()
    print("\n" + "="*30 + "\nRUNNING INFERENCE ON SAMPLES\n" + "="*30)
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
            print(f"\n[{src_lang} Example {i+1}]\n{code[:200]}...")
            print(f"\n[Translated {tgt_lang}]\n{translation}\n" + "-"*40)

def main():
    print(f"Starting pipeline on {CONFIG['device']}...")
    dataset = UnpairedCodeDataset(languages=CONFIG["langs"])
    trainer = BackTranslator(CONFIG)
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

    # PHASE 2: Alternating DAE + Iterative Back-Translation
    print("\n--- Phase 2: Alternating DAE + BT ---")
    for epoch in range(CONFIG["bt_epochs"]):
        dae_loss_total = 0
        bt_loss_total = 0
        
        for _ in range(CONFIG["steps_per_epoch"]):
            # DAE
            lang = random.choice(CONFIG["langs"])
            dae_batch = dataset.sample_batch(lang, CONFIG["batch_size"])
            if dae_batch[0]:
                dae_loss_total += trainer.train_dae_step(dae_batch, lang)

            # BT
            src, tgt = random.sample(CONFIG["langs"], 2)
            bt_batch = dataset.sample_batch(src, CONFIG["batch_size"])
            if bt_batch[0]:
                bt_loss_total += trainer.train_bt_step(bt_batch, src, tgt)
        
        print(f"Epoch {epoch+1} | DAE Loss: {dae_loss_total/CONFIG['steps_per_epoch']:.4f} | BT Loss: {bt_loss_total/CONFIG['steps_per_epoch']:.4f}")
        run_inference(trainer, dataset)

if __name__ == "__main__":
    main()

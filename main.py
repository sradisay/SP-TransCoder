import os
import random
import torch
from dataset import UnpairedCodeDataset, PairedCodeDataset
from models.backtranslator import BackTranslator

CONFIG = {
    "model_name": "Salesforce/codet5-small",
    "langs": ["Python", "C++"],
    "batch_size": 32,             
    "epochs": 40,
    "steps_per_epoch": 200,       
    "lr": 5e-5,
    "max_len": 256,
    "weight_decay": 0.01,        
    "warmup_steps": 1000,         
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "ckpt_dir": "checkpoints/codet5_semisupervised"
}

def run_inference(trainer, dataset, device, num_examples=2):
    trainer.model.eval()
    print("\n" + "="*30 + "\nRUNNING INFERENCE ON SAMPLES\n" + "="*30)
    
    for src_lang, tgt_lang in [("Python", "C++"), ("C++", "Python")]:
        samples = dataset.sample_batch(src_lang, limit=num_examples) 
        
        for i, code in enumerate(samples):
            if not code.strip(): continue
            
            inputs = trainer._tokenize_prompt([code], tgt_lang)

            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out_ids = trainer.model.generate(
                    **inputs,
                    max_length=trainer.cfg["max_len"], 
                    num_beams=4,
                    early_stopping=True
                )
                translation = trainer.tokenizer.decode(out_ids[0], skip_special_tokens=True)
                
            print(f"\n[{src_lang} Example {i+1}]\n{code[:200]}...")
            print(f"\n[Translated {tgt_lang}]\n{translation}\n" + "-"*40)

def main():
    print(f"Starting Semi-Supervised Pipeline on {CONFIG['device']}...")
    os.makedirs(CONFIG["ckpt_dir"], exist_ok=True)
    
    paired_dataset = PairedCodeDataset(batch_size=CONFIG["batch_size"])
    unpaired_dataset = UnpairedCodeDataset(languages=CONFIG["langs"], batch_size=CONFIG["batch_size"])
    
    trainer = BackTranslator(CONFIG)
    
    print("\n--- Training: Alternating Supervised (CodeNet) + BT (The Stack) ---")
    for epoch in range(CONFIG["epochs"]):
        sup_loss_total = 0
        bt_loss_total = 0
        
        for _ in range(CONFIG["steps_per_epoch"]):
            # 1. Supervised Anchor Step
            paired_batch = paired_dataset.sample_batch()
            
            # Randomly pick direction for this batch
            if random.random() > 0.5:
                src, tgt_lang = paired_batch["python"], "C++"
                tgt = paired_batch["c++"]
            else:
                src, tgt_lang = paired_batch["c++"], "Python"
                tgt = paired_batch["python"]
                
            sup_loss_total += trainer.train_supervised_step(src, tgt, tgt_lang)

            # 2. Back-Translation Step
            bt_src_lang, bt_tgt_lang = random.sample(CONFIG["langs"], 2)
            unpaired_batch = unpaired_dataset.sample_batch(bt_src_lang)
            bt_loss_total += trainer.train_bt_step(unpaired_batch, bt_src_lang, bt_tgt_lang)
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Sup Loss: {sup_loss_total/CONFIG['steps_per_epoch']:.4f} | BT Loss: {bt_loss_total/CONFIG['steps_per_epoch']:.4f}")
        
        if (epoch + 1) % 5 == 0:
            run_inference(trainer, unpaired_dataset, CONFIG['device'])
            
            ckpt_path = f"{CONFIG['ckpt_dir']}_epoch_{epoch+1}"
            print(f"Saving checkpoint to '{ckpt_path}'...")
            trainer.model.save_pretrained(ckpt_path)
            trainer.tokenizer.save_pretrained(ckpt_path)

if __name__ == "__main__":
    main()

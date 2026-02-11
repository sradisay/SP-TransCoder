from transformers import T5ForConditionalGeneration
from dataset import UnpairedCodeDataset
from models.backtranslator_t5prompts import BackTranslationTrainer
import torch


CONFIG = {
    "model_name": "Salesforce/codet5-small",
    "langs": ["python", "c++"],
    "batch_size": 128,
    "epochs": 10,
    "lr": 3e-6,
    "max_len": 256,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

dataset = UnpairedCodeDataset(languages=["python", "c++"])
model = T5ForConditionalGeneration.from_pretrained(CONFIG["model_name"])
trainer = BackTranslationTrainer(CONFIG, model)

for epoch in range(CONFIG["epochs"]):
    trainer.run_epoch(dataset, epoch)
    model.save_pretrained(f"./checkpoints/codet5_bt_epoch_{epoch}")
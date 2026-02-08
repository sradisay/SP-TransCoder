import random
import torch
from models.backtranslator import BackTranslator
from dataset import UnpairedCodeDataset

CONFIG = {
    "model_name": "Salesforce/codet5-small",
    "langs": ["python", "c++"],
    "batch_size": 128,
    "epochs": 10,
    "lr": 3e-5,
    "max_len": 256,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

def run_training():
    print("Initializing BackTranslator and Dataset...")
    agent = BackTranslator(CONFIG)
    dataset = UnpairedCodeDataset(languages=CONFIG["langs"])
    print("Starting training loop...")

    optimizer = torch.optim.AdamW(agent.model.parameters(), lr=CONFIG["lr"])
    for epoch in range(CONFIG["epochs"]):
        total_loss = 0
        num_steps = 100

        for _ in range(num_steps):
            # Randomly pick source and target languages
            src_lang, tgt_lang = random.sample(CONFIG["langs"], 2)
            src_examples = dataset.sample_batch(src_lang, CONFIG["batch_size"])

            loss = agent.train_step(src_examples, src_lang, tgt_lang, optimizer)
            total_loss += loss

        print(f"Epoch {epoch+1} | Avg Loss: {total_loss / num_steps:.4f}")

if __name__ == "__main__":
    run_training()

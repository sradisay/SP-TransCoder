import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from dataset import XLCoSTSnippetDataset
from models.supervised_trainer import SupervisedTranslationTrainer

CONFIG = {
    "model_name": "Salesforce/codet5-small",
    "langs": ["python", "c++"],
    "batch_size": 128,
    "epochs": 10,
    "lr": 5e-5,
    "max_len": 256,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


def main():
    print(f"Using device: {CONFIG['device']}")
    dataset = XLCoSTSnippetDataset()

    print(f"Loading {CONFIG['model_name']}...")
    model = T5ForConditionalGeneration.from_pretrained(CONFIG["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    trainer = SupervisedTranslationTrainer(CONFIG, model, tokenizer)

    for epoch in range(CONFIG["epochs"]):
        trainer.run_epoch(dataset, epoch)

        save_path = f"./checkpoints/codet5_supervised_snippets_epoch_{epoch}"
        print(f"Saving checkpoint to {save_path}...")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    print("Supervised Fine-Tuning Complete!")


if __name__ == "__main__":
    main()
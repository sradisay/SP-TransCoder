import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from dataset import UnpairedCodeDataset, PairedCodeDataset, XLCoSTSnippetDataset
from models.supervised_trainer import SupervisedTranslationTrainer
from models.backtranslator_t5prompts import BackTranslationTrainer
from models.denoising_trainer import DenoisingAutoEncoderTrainer

CONFIG = {
    "model_name": "Salesforce/codet5-small",
    "langs": ["python", "c++"],
    "batch_size": 128,
    "epochs": 10,
    "lr_supervised": 5e-5,
    "lr_backtranslation": 3e-6,
    "lr_denoising": 5e-5,
    "max_len": 256,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

def main():
    print(f"Using device: {CONFIG['device']}")

    print("Loading datasets...")
    supervised_dataset = XLCoSTSnippetDataset()
    unpaired_dataset = UnpairedCodeDataset(languages=CONFIG["langs"])

    print(f"Loading {CONFIG['model_name']}...")
    model = T5ForConditionalGeneration.from_pretrained(CONFIG["model_name"]).to(CONFIG["device"])
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    config_sup = CONFIG.copy()
    config_sup["lr"] = CONFIG["lr_supervised"]

    config_bt = CONFIG.copy()
    config_bt["lr"] = CONFIG["lr_backtranslation"]

    config_dae = CONFIG.copy()
    config_dae["lr"] = CONFIG["lr_denoising"]

    supervised_trainer = SupervisedTranslationTrainer(config_sup, model, tokenizer)
    bt_trainer = BackTranslationTrainer(config_bt, model)
    dae_trainer = DenoisingAutoEncoderTrainer(config_dae, model)

    for epoch in range(CONFIG["epochs"]):
        print(f"\n--- Cycle {epoch + 1}/{CONFIG['epochs']} ---")

        print("Starting Denoising Phase...")
        dae_trainer.run_epoch(unpaired_dataset, epoch)

        print("Starting Supervised Phase...")
        supervised_trainer.run_epoch(supervised_dataset, epoch)

        print("Starting Backtranslation Phase...")
        bt_trainer.run_epoch(unpaired_dataset, epoch)

        save_path = f"./checkpoints/codet5_semi_supervised_cycle_{epoch}"
        print(f"Saving cycle checkpoint to {save_path}...")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    print("Semi-Supervised Training Complete!")

if __name__ == "__main__":
    main()
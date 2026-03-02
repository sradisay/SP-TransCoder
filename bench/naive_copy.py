import argparse
from codebleu import calc_codebleu
from evaluate import load_test_data

CONFIG = {
    "test_dir": "../data/pair_data_tok_1/C++-Python/",
}

def main(source_lang, target_lang, num_samples):
    sources, references = load_test_data(source_lang, target_lang, limit=num_samples)
    predictions = sources

    print(f"\nEvaluating Naive Copy Baseline ({source_lang} -> {target_lang}) on {len(sources)} samples...")

    codebleu_lang = "python" if target_lang == "Python" else "cpp"

    result = calc_codebleu(
        [[ref] for ref in references],
        predictions,
        lang=codebleu_lang,
        weights=(0.25, 0.25, 0.25, 0.25),
        tokenizer=None
    )

    print("\n" + "=" * 40)
    print("SP-TransCoder Baseline Results")
    print("=" * 40)
    print(f"Direction: {source_lang} -> {target_lang}")
    print(f"Total CodeBLEU Score: {result['codebleu'] * 100:.2f}")
    print(f"   - N-Gram Match:    {result['ngram_match_score'] * 100:.2f}")
    print(f"   - Weighted N-Gram: {result['weighted_ngram_match_score'] * 100:.2f}")
    print(f"   - Syntax Match:    {result['syntax_match_score'] * 100:.2f}")
    print(f"   - Dataflow Match:  {result['dataflow_match_score'] * 100:.2f}")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Naive Copy Baseline CodeBLEU score.")
    parser.add_argument(
        "--source",
        type=str,
        choices=["Python", "C++"],
        default="Python",
        help="The source programming language."
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["Python", "C++"],
        default="C++",
        help="The target programming language."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        required=False,
        default=None,
        help="Number of samples to evaluate. If omitted, evaluates the entire dataset."
    )

    args = parser.parse_args()
    main(args.source, args.target, args.num_samples)

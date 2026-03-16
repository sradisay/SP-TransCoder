#!/bin/bash

OUTPUT_DIR="../human_eval_lists3"

MODELS=(
    "Salesforce/codet5-small"
    "Salesforce/codet5-base"
    "../checkpoints/codet5_semi_supervised_denoising_small_cycle_9"
    "../checkpoints/codet5_supervised_snippets_small_epoch_9"
    "../checkpoints/codet5_bt_epoch_9"
    "../checkpoints/codet5_semi_supervised_small_cycle_9"
    "../checkpoints2/codet5_supervised_snippets_base_epoch_9"
    "../checkpoints3/codet5_semi_supervised_base_cycle_9"
    "../checkpoints3/codet5_semi_supervised_with_denoising_base_cycle_9"
#    "../checkpoints2/codet5_supervised_snippets_epoch_9"
)

mkdir -p "$OUTPUT_DIR"

for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_PATH")

    MODEL_DIR="$OUTPUT_DIR/$MODEL_NAME"
    mkdir -p "$MODEL_DIR"

    echo "Generating HumanEval-X translations for SP-TransCoder Model: $MODEL_NAME"

    echo "  -> Running C++ to Python..."
    OUT_CPP2PY="$MODEL_DIR/humanevalx_cpp2py.jsonl"
    LOG_CPP2PY="$MODEL_DIR/humanevalx_cpp2py.log"

    python ./bench/unit_testing.py \
        --m "$MODEL_PATH" \
        --source "C++" \
        --target "Python" \
        --out "$OUT_CPP2PY" > "$LOG_CPP2PY"
    echo "  -> Running Python to C++..."
    OUT_PY2CPP="$MODEL_DIR/humanevalx_py2cpp.jsonl"
    LOG_PY2CPP="$MODEL_DIR/humanevalx_py2cpp.log"

    python ./bench/unit_testing.py \
        --m "$MODEL_PATH" \
        --source "Python" \
        --target "C++" \
        --out "$OUT_PY2CPP" > "$LOG_PY2CPP"


    echo "Done with $MODEL_NAME. Results saved in $MODEL_DIR/"
    echo ""
done

echo "All HumanEval-X generations complete!"
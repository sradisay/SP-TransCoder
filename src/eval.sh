#!/bin/bash

OUTPUT_DIR="../eval_results3"
TEST_DIR="../data/pair_data_tok_1/C++-Python/"

MODELS=(
#    "../checkpoints3/codet5_semi_supervised_base_cycle_9"
#    "../checkpoints3/codet5_semi_supervised_with_denoising_base_cycle_9"
#    "../checkpoints/codet5_semi_supervised_denoising_cycle_9"
#    "../checkpoints/codet5_semi_supervised_base_cycle_9"
#    "../checkpoints/codet5_supervised_snippets_epoch_9"
#    "../checkpoints/codet5_supervised_epoch_9"
#    "Salesforce/codet5-small"
)

mkdir -p "$OUTPUT_DIR"

for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_PATH")

    MODEL_DIR="$OUTPUT_DIR/$MODEL_NAME"
    mkdir -p "$MODEL_DIR"

    echo "Evaluating SP-TransCoder Model: $MODEL_NAME"

    echo "-> Running Python to C++..."
    CACHE_PY2CPP="$MODEL_DIR/cache_py2cpp.json"
    LOG_PY2CPP="$MODEL_DIR/py2cpp.log"

    python ./bench/evaluate.py \
        --m "$MODEL_PATH" \
        --source "Python" \
        --target "C++" \
        --test_dir "$TEST_DIR" \
        --c "$CACHE_PY2CPP" > "$LOG_PY2CPP"

    echo "  -> Running C++ to Python..."
    CACHE_CPP2PY="$MODEL_DIR/cache_cpp2py.json"
    LOG_CPP2PY="$MODEL_DIR/cpp2py.log"

    python ./bench/evaluate.py \
        --m "$MODEL_PATH" \
        --source "C++" \
        --target "Python" \
        --test_dir "$TEST_DIR" \
        --c "$CACHE_CPP2PY" > "$LOG_CPP2PY"

    echo "Done with $MODEL_NAME. Results saved in $MODEL_DIR/"
    echo ""
done

echo "All SP-TransCoder evaluations complete!"
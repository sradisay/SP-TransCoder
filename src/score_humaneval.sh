#!/bin/bash

CODEGEEX_DIR="../CodeGeeX"
EVAL_RESULTS_DIR="../human_eval_lists3"

MODELS=(
    "codet5_semi_supervised_denoising_cycle_9"
    "codet5_semi_supervised_cycle_9"
    "codet5_supervised_snippets_epoch_9"
    "codet5-small"
)

cd "$CODEGEEX_DIR" || { echo "CodeGeeX repo not found!"; exit 1; }
export PYTHONPATH=$PWD:$PYTHONPATH
for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_DIR="$EVAL_RESULTS_DIR/$MODEL_NAME"

    echo "====================================================="
    echo "Calculating Execution Scores for: $MODEL_NAME"
    echo "====================================================="

    # 1. Score C++ to Python Translation
    FILE_CPP2PY="$MODEL_DIR/humanevalx_cpp2py.jsonl"
    if [ -f "$FILE_CPP2PY" ]; then
        echo "--> Scoring C++ to Python..."
        python ./codegeex/benchmark/evaluate_humaneval_x.py \
            --input_file "$FILE_CPP2PY" \
            --problem_file "../data/humaneval_python.jsonl.gz"
    else
        echo "Missing file: $FILE_CPP2PY"
    fi

    echo ""

    # 2. Score Python to C++ Translation (Evaluates C++ Code)
    FILE_PY2CPP="$MODEL_DIR/humanevalx_py2cpp.jsonl"
    if [ -f "$FILE_PY2CPP" ]; then
        echo "--> Scoring Python to C++..."
        python ./codegeex/benchmark/evaluate_humaneval_x.py \
            --input_file "$FILE_PY2CPP" \
            --problem_file "../data/humaneval_cpp.jsonl.gz"
    else
        echo "Missing file: $FILE_PY2CPP"
    fi
    echo ""
done

echo "All execution scoring complete!"
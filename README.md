

## Benchmarking and Visualization

### Visualization of Model
```
usage: bench/visual_inspection.py [-h] --m M [--t T]

visualize performance of model on translation from C++ to Python and Python to C++

options:
  -h, --help  show this help message and exit
  --m M       path or hf ID for the model
  --t T       path or hf ID for the tokenizer. Default is model 
```

###  Evaluation

```
usage: bench/evaluate.py [-h] --m M [--t T] [--c C] [--source {Python,C++}] [--target {Python,C++}]

Evaluate SP-TransCoder CodeBLEU score.

options:
  -h, --help            show this help message and exit
  --m M                 path or hf ID for the model
  --t T                 path or hf ID for the tokenizer. Default is model
  --c C                 path to cache. Defaults to predictions_cache_<source>2<target>.json
  --source {Python,C++}
                        The src programming language.
  --target {Python,C++}
                        The target programming language.
```
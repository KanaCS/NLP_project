Run `allenNLP train parameters.jsonnet -s tmp/serialization --include-package main` for training.

Run `allenNLP evaluate ./tmp/serialization/model.tar.gz test-datasets --include-package main` after training for evaluation.

Run `allenNLP predict ./tmp/serialization/model.tar.gz test.json --include-package main --predictor lstm-classifier` for prediction.
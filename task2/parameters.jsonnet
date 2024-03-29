local embedding_dim = 300;
local hidden_dim = 500;
local num_epochs = 1000;
local patience = 50;
local batch_size = 32;
local learning_rate = 0.01;

{
    "train_data_path": "./train-datasets",
    "validation_data_path": "./validation-datasets",
    "dataset_reader": {
        "type": "prop"
    },
    "model": {
        "type": "lstm-classifier",
        "word_embeddings": {
            "tokens": {
                "type": "embedding",
                "embedding_dim": embedding_dim
            },
        },
        "encoder": {
            "type": "lstm",
            "input_size": embedding_dim,
            "hidden_size": hidden_dim
        },
        "feedforward": {
            "input_dim": hidden_dim,
            "num_layers": 1,
            "activations": ["linear"],
            "hidden_dims": [14],
            "dropout": [0.0],
        },
    },
    "iterator": {
        "type": "bucket",
        "batch_size": batch_size,
        "sorting_keys": [["tokens", "num_tokens"]]
    },
    "trainer": {
        "num_epochs": num_epochs,
        "optimizer": {
            "type": "sgd",
            "lr": learning_rate
        },
        "patience": patience
    }
}
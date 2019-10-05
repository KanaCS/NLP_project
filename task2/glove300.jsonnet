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
                "pretrained_file": "glove.6B.300d.txt",
                "embedding_dim": embedding_dim,
                "trainable": false
            },
        },
        "encoder": {
            "type": "lstm",
            "input_size": embedding_dim,
            "hidden_size": hidden_dim
        }
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
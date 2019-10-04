local embedding_dim = 200;
local hidden_dim = 100;
local num_epochs = 100;
local patience = 10;
local batch_size = 128;
local learning_rate = 0.1;

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
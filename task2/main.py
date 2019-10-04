from typing import Iterator, List, Dict, Any
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from allennlp.training.trainer import Trainer
from allennlp.data import Instance
from allennlp.common.params import Params
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import TextField, LabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.tokenizers.token import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
from allennlp.data.iterators.bucket_iterator import BucketIterator
import os

@DatasetReader.register('prop')
class PropDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
    
    def text_to_instance(self, tokens: List[Token], tag: str = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {'tokens': sentence_field}
        
        if tag:
            fields['label'] = LabelField(label=tag)
        return Instance(fields)
    
    def _read(self, dataset_path):
        articles_path = dataset_path + '/articles'
        fragments_path = dataset_path + '/labels'
        articles = {}
        for file in os.listdir(articles_path):
            id_str = file.split('.')[0][7:]
            if id_str == '': continue  # Skip .DS_Store and other hidden files.
            article_id = int(id_str)
            article_content = open(articles_path + '/' + file, 'r').read()
            articles[article_id] = article_content
        
        for file in os.listdir(fragments_path):
            if file[0] == '.': continue
            fragments_content = open(fragments_path + '/' + file, 'r').read()
            if len(fragments_content) == 0:
                continue
            fragments_content = fragments_content.split('\n')[:-1]
            for line in fragments_content:
                article_id, technique, start, end = line.split()
                article_id, start, end = int(article_id), int(start), int(end)
                fragment = articles[article_id][start:end]
                tokens = [Token(word) for word in fragment.split()]
                yield self.text_to_instance(tokens, technique)

'''
reader = PropDatasetReader()
dataset = reader.read('./datasets')
vocab = Vocabulary.from_instances(dataset)

EMBEDDING_DIM = 50
HIDDEN_DIM = 30
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({'tokens': token_embedding})
'''

@Model.register('lstm-classifier')
class LstmClassifier(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()
    
    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.hidden2tag(encoder_out)
        
        output = {'logits': logits}
        if label is not None:
            output['accuracy'] = self.accuracy(logits, label)
            output['loss'] = self.loss_function(logits, label)
        return output

'''
lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
model = LstmClassifier(word_embeddings, lstm, vocab)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
iterator = BucketIterator(batch_size=32, sorting_keys=[('tokens', 'num_tokens')])
iterator.index_with(vocab)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=dataset[:-200],
                  validation_dataset=dataset[-200:],
                  patience=10,
                  num_epochs=50)
trainer.train()
'''
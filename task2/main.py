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
from allennlp.modules import FeedForward
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
from allennlp.data.iterators.bucket_iterator import BucketIterator
from allennlp.predictors.predictor import Predictor
from allennlp.data import Instance
from allennlp.common.util import JsonDict
import os

@DatasetReader.register('prop')
class PropDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
    
    def text_to_instance(self, fragment: str, tag: str = None) -> Instance:
        tokens = [Token(word) for word in fragment.split()]
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
                yield self.text_to_instance(fragment, technique)

@Model.register('lstm-classifier')
class LstmClassifier(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 feedforward: FeedForward,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.feedforward = feedforward
        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()
    
    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.feedforward(encoder_out)
        
        output = {'logits': logits}
        if label is not None:
            self.accuracy(logits, label)
            output['loss'] = self.loss_function(logits, label)
        return output
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

@Predictor.register('lstm-classifier')
class LstmClassifierPredictor(Predictor):
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        articles = json_dict['articles']
        output_dicts = []
        for article_dict in articles:
            article = article_dict['article']
            instance = self._dataset_reader.text_to_instance(fragment=article)
            output_dict = self.predict_instance(instance)
        
            label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
            all_labels = [label_dict[i] for i in range(len(label_dict))]
        
            output_dict['all_labels'] = all_labels
            output_dicts.append(output_dict)
        return output_dicts
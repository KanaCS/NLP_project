import os #predictor:SentenceTaggerPredictor    Model:Crf_Tagger
from typing import Iterator, List, Dict, Optional
import torch
import torch.optim as optim
from allennlp.service.predictors.sentence_tagger import SentenceTaggerPredictor
from allennlp.training.trainer import Trainer
from allennlp.data.iterators import BucketIterator
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, LabelField
from allennlp.modules.token_embedders import Embedding
from allennlp.data.dataset_readers import DatasetReader
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.data.fields.list_field import ListField
from allennlp.data.fields.metadata_field import MetadataField
from allennlp.data.fields.index_field import IndexField
from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict
from allennlp.models.crf_tagger import CrfTagger
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import BertEmbedder
from nltk.stem import WordNetLemmatizer
import nltk
import h5py
from nltk.corpus import wordnet
nltk.download('wordnet')

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

@DatasetReader.register("prop")
class PropDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers1 = token_indexers or {'tokens': SingleIdTokenIndexer()}   

    def text_to_instance(self, tokens:List[Token], label_lst:List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers1)
        fields = {'tokens': sentence_field}
        fields['metadata'] = MetadataField({"words": [x.text for x in tokens]})
        if label_lst:
            fields['tags'] = SequenceLabelField(label_lst, sentence_field, "labels")  
        #fields['id'] = article_id
        return Instance(fields)
    
    def _read(self,dataset_path):
        articles_path = dataset_path + '/articles'
        fragments_path = dataset_path + '/labels'
        articles = {}
        for file in os.listdir(articles_path):
            id_str = file.split('.')[0][7:]
            try:
                article_id =int(id_str)
            except:
                continue
            print(id_str)
            if id_str == '': continue  # Skip .DS_Store and other hidden files.
            article_id = int(id_str)
            article_content = open(articles_path + '/' + file, 'r',encoding="utf-8").read()
            articles[article_id] = article_content 
        for file in os.listdir(fragments_path):
            start_list=[]
            end_list=[]
            if file[0] == '.': continue
            fragments_content = open(fragments_path + '/' + file, 'r',encoding="utf-8").read()
            if len(fragments_content) == 0:
                continue
            fragments_content = fragments_content.split('\n')[:-1]
            #fragment_sum=""
            for line in fragments_content:
                article_id, start, end = line.split()
                article_id, start, end = int(article_id), int(start), int(end)
                start_list.append(start)
                end_list.append(end)
                
            new_word, pre, count, string, index = True, True, 0, "", 0
            lm = WordNetLemmatizer()
            list_of_label_list=[]
            for j in range(len(start_list)):
                string_list=[]
                label_list=[]
                for i,tok in enumerate(articles[article_id]):
                    if(tok==" " or tok=='.' or tok=='!' or tok=='?' or tok=='"' or tok==","):
                        count+=1
                        new_word=True
                        string=string.strip()
                        if string!="": #and ite==0:
                            string=lm.lemmatize(string, get_wordnet_pos(string))
                        string_list.append(string)
                        if index==1: 
                            label_list.append(0)#label_list.append("O")
                        else: #elif index==2: 
                            label_list.append(1)#label_list.append("B")
                        #else: 
                        #    label_list.append("I")
                        string=""
                        continue
                    string+=tok
                    if(new_word==True):
                        if( (i >= start_list[j]-2 and i < end_list[j]) or () ): ############if( (i >= start_list[j]-2 and i < end_list[j]) or () ):
                            pre = True
                            index=1
                        else:
                            index=2
                        new_word=False
                list_of_label_list.append(label_list)
            #return string_list: tokenized words & label_list: BIO label for each word
            for j in range(len(start_list)-1):
                    for i in range(len(list_of_label_list[j])):
                        list_of_label_list[0][i]=list_of_label_list[0][i]*list_of_label_list[j][i]
            label_lst, label_lst2, frag=[],[],[]
            for i in range(len(list_of_label_list[0])):
                if(list_of_label_list[0][i]==0):
                    pre = True
                    label_lst.append("O")
                else:
                    if(pre):
                        pre = False
                        label_lst.append("B") #B
                    else:
                        label_lst.append("I") #I
            for i in range(len(list_of_label_list[0])):
                if(list_of_label_list[0][i]==0):
                    pre = True
                    frag.append(string_list[i])
                    label_lst2.append("O")
                else:
                    if(pre):
                        pre = False
                        #label_lst2.append("B") #B
                    #else:
                        #label_lst2.append("O") #I
            yield self.text_to_instance([Token(string_) for string_ in string_list],label_lst)
            yield self.text_to_instance([Token(string_) for string_ in frag],label_lst2)

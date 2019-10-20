import os
from typing import Iterator, List, Dict, Optional
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, LabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields.list_field import ListField
from allennlp.data.fields.metadata_field import MetadataField
from allennlp.data.fields.index_field import IndexField

@DatasetReader.register("prop")
class PropDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers1 = token_indexers or {'tokens': SingleIdTokenIndexer()}   

    def text_to_instance(self, articles: str, article_id: int, start: int, end: int) -> Instance:
        count=0
        new_word=True
        pre=True
        string=""
        string_list=[]
        label_list=[]
        index=0
        for i,tok in enumerate(articles):
            if(tok==" " or tok=='.' or tok=='!' or tok=='?' or tok=='"'):
                count+=1
                new_word=True
                string=string.strip()
                string_list.append(string)
                if index==1: 
                    label_list.append("O")
                elif index==2: 
                    label_list.append("B")
                else: 
                    label_list.append("I")
                string=""
                continue
            string+=tok
            if(new_word==True):
                if( i >= start-2 and i < end ):
                    pre = True
                    index=1
                else:
                    if(pre):
                        pre = False
                        index=2
                    else:
                        index=3
                new_word=False
        #return string_list: tokenized words & label_list: BIO label for each word
        tokens = [Token(string_) for string_ in string_list]
        sentence_field = TextField(tokens, self.token_indexers1)
        fields = {'tokens': sentence_field}
        fields['metadata'] = MetadataField({"words": [x.text for x in tokens]})
        fields['tags'] = SequenceLabelField(label_list, sentence_field, "labels")  
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
            if file[0] == '.': continue
            fragments_content = open(fragments_path + '/' + file, 'r',encoding="utf-8").read()
            if len(fragments_content) == 0:
                continue
            fragments_content = fragments_content.split('\n')[:-1]
            #fragment_sum=""
            for line in fragments_content:
                article_id, start, end = line.split()
                article_id, start, end = int(article_id), int(start), int(end)
                yield self.text_to_instance(articles[article_id],article_id,start,end)
#try 2 ways:
#       1) yield instance for every fragment
#       2) combine all the fragment in one text to yield one instance
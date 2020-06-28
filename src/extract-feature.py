# This source code is part of a final year undergraduate project
# on exploring Indonesian hate speech/abusive & sentiment text 
# classification using a multilingual language model
# 
# Checkout the full github repository: 
# https://github.com/ilhamfp/indonesian-text-classification-multilingual

import torch
from transformers import AutoModel, AutoTokenizer, BertTokenizer

torch.set_grad_enabled(False)

class FeatureExtractor():
    
    def __init__(self, model_name='xlm-r'):
        self.model_name = model_name
        self.max_length = 512
        
        if self.model_name == 'xlm-r':
            xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large')
            xlmr.eval()
            self.model = xlmr
            
        elif self.model_name == 'mbert':
            MODEL_NAME = "bert-base-multilingual-cased"
            model = AutoModel.from_pretrained(MODEL_NAME)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = model
            self.tokenizer = tokenizer
        
    def extract_features(self, text):
        text = str(text)
        if self.model_name == 'xlm-r':
            tokens = self.model.encode(text)
            
            # Truncate
            if len(tokens) > self.max_length:
                tokens = torch.cat( (tokens[:511], torch.Tensor([2]).long()), 0 )
                
            last_layer_features = self.model.extract_features(tokens)
            features = last_layer_features[:, 0, :].data.numpy()
            
        elif self.model_name == 'mbert':
            tokens_pt2 = self.tokenizer.encode_plus(text, 
                                                    return_tensors="pt",
                                                    pad_to_max_length=True,
                                                    max_length=self.max_length)
            
            outputs2, pooled2 = self.model(**tokens_pt2)
            features = pooled2.data.numpy()
        
        return features
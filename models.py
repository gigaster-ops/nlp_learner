import torch
import torchvision
import torch.nn as nn
from transformers import BertTokenizer, BertModel, XLMRobertaTokenizer, XLMRobertaModel

class Model():
	def __init__(self, model_state_dict, model_type):
		self.model = model
		if model_type == 'Multilingual BERT':
			tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
			model = BertModel.from_pretrained("bert-base-multilingual-cased")
		else:
			tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
			model = XLMRobertaModel.from_pretrained("xlm-roberta-base")
		self.model.load_state_dict(model_state_dict)
import requests, os
from exp import ConnectError
#import torch
import json
from requests.exceptions import ConnectionError

class Sender:
	def __init__(self, url='http://127.0.0.1:5000/'):
		self.url = url
		self.token = None

	def load_model(self, model_name: str, model_type: str, model: str, n_class, label, type):
		print(model)
		try:
			f = open(model, 'rb')
		except FileNotFoundError:
			print('no')
			f = 'no_model'

		path = 'load_model'

		files = {'upload_file': ('model_name_file', open(model,'rb'), 'text/x-spam')}

		data = {
			'token': self.token,
			'model_name': model_name,
			'model_type': model_type,
			'n_class': n_class,
			'label': label,
			'type': type,
		}

		out = {'access': False}

		try:
			resp = requests.post(os.path.join(self.url, path), files=files, data=data)
		except ConnectionError:
			out['msg'] = 'Не удалось подключиться к сереверу'
			return out
		if resp.status_code != 200:
			out['msg'] = 'Сервер вернул неверный статускод'
			return out
		else:
			print('Успешная передача')
		js = resp.json()

		if js['access'] == '0':
			out['msg'] = 'Ошибка на сервере. Скорее всего вы что-то отправили неправильно'
			return out

		token = js['token']

		out['access'] = True
		out['token'] = token
		return out

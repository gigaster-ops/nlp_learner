import requests


class ConnectError(Exception):
    pass

class ProcessError(Exception):
    pass


class API():
    def __init__(self, host='http://127.0.0.1:5000/'):
        self.host = host

    def __call__(self, text, token, model_type):
        data = {
            'text': text,
            'token': token,
            'model_type': model_type
        }

        resp = requests.post(self.host + 'forward', data=data)

        if resp.status_code != 200:
            raise ConnectError
        json = resp.json()
        if json['access'] != '1':
            raise ProcessError


if __name__ == '__main__':
    api = API()
    api('testtesttesttest', 'qwertyuiop[]', 'XLM')

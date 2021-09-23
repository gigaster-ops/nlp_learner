from flask import Flask, request, make_response
import random
import torch
import torch.nn as nn
import torchvision
import os
import learning


app = Flask(__name__)

models_list_name = os.listdir('models')

global_dict_models = {}
for token in models_list_name:
    try:
        tok, n_class, model_type, typ = token.split('.')
        t, model = learning.get_model(model_type, typ)

        n_class = int(n_class)
        model = learning.NewModel(model, n_class, typ)
        model.load_state_dict(torch.load(os.path.join('models', token), map_location=torch.device('cpu')))
        print(model, tok)
        with open(tok, 'r') as f:
            label = f.read().split('\n')
        print(label)
        global_dict_models[tok] = (t, model, label)
    except ValueError:
        pass

def gen_token():
    arr = list(range(97, 123)) + list(range(65, 91)) + list(range(48, 58))
    l = 64
    st = ''.join([chr(arr[random.randint(0, len(arr) - 1)]) for i in range(l)])
    return st

@app.route("/load_model", methods=['POST'])
def hello_world():
    print('Loading model')
    #print(request.form)
    model_name = request.form['model_name']
    model_type = request.form['model_type']
    n_class = request.form['n_class']
    file = request.files['upload_file']
    label = request.form['label']
    typ = request.form['type']

    token = gen_token()
    print(label)
    file.save(os.path.join('models', token + '.' + str(n_class) + '.' + model_type + '.' + typ))
    with open(token, 'w') as f:
        f.write(label)

    label = label.split()
    print(n_class, type(n_class))
    t, model = learning.get_model(model_type)
    model = learning.NewModel(model, int(n_class))

    model.load_state_dict(torch.load(os.path.join('models', token + '.' + str(n_class) + '.' + model_type + '.' + typ), map_location=torch.device('cpu')))
    global_dict_models[token] = (t, model, label)
    return make_response({
        'access': '1',
        'token': token,
    })

@app.route('/forward', methods=['POST'])
def forward():
    print('forward')
    print(global_dict_models)
    model_type = request.form['model_type']

    text = request.form['text']
    token = request.form['token']

    t, model, labels = global_dict_models[token]
    x = t.encode_plus(text, max_length=128, truncation=True, padding="max_length", return_tensors='pt')
    output = torch.softmax(model(x['input_ids'].squeeze(1), x['attention_mask']), dim=1).detach().cpu().numpy()
    #output = torch.argmax(torch.softmax(output, dim=1), dim=1)
    print(output)
    print(labels)
    print({labels[i]: output[0][i] for i in range(len(labels))})
    return make_response({
        'access': '1',
        'label': {labels[i]: str(output[0][i]) for i in range(len(labels))}
    })



if __name__ == '__main__':
    print('Start')
    app.run()
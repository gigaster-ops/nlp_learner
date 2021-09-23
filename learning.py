import torch
import torchvision
import torch.nn as nn
import time
from sklearn.metrics import accuracy_score, f1_score

from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

import pandas as pd
from transformers import BertTokenizer, BertModel, XLMRobertaTokenizer, XLMRobertaModel

from PyQt5.QtCore import QThreadPool, QThread, QRunnable, QObject, pyqtSignal, pyqtSlot

import numpy as np


def get_model(model_type, typ):
    if typ == 'base':
        if model_type == 'Multilingual BERT':
            tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            model = BertModel.from_pretrained("bert-base-multilingual-cased")
            return tokenizer, model
        else:
            tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', force_download=False)
            model = XLMRobertaModel.from_pretrained("xlm-roberta-base", force_download=False)
            print(tokenizer, model)
            return tokenizer, model
    else:
        if model_type == 'Multilingual BERT':
            tokenizer = BertTokenizer.from_pretrained('bert-large-multilingual-cased')
            model = BertModel.from_pretrained("bert-large-multilingual-cased")
            return tokenizer, model
        else:
            tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large', force_download=False)
            model = XLMRobertaModel.from_pretrained("xlm-roberta-large", force_download=False)
            print(tokenizer, model)
            return tokenizer, model


class SpamDataset(Dataset):
    def __init__(self, data, tokenizer, seq_len):
        super().__init__()

        self.data = data

        try:
            _, text, label = self.data
        except Exception:
            try:
                _, _, text, label = self.data
            except Exception:
                text, label = self.data

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        labels = self.data[label].unique()
        self.labels2y = {label_: i for i, label_ in enumerate(labels)}
        self.y2labels = {i: label_ for i, label_ in enumerate(labels)}

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        try:
            _, text, label = self.data.loc[idx]
        except Exception:
            try:
                _, _, text, label = self.data.loc[idx]
            except Exception:
                text, label = self.data.loc[idx]

        text = self.tokenizer.encode_plus(text, max_length=self.seq_len, truncation=True, padding="max_length", return_tensors='pt')
        y = torch.LongTensor([self.labels2y[label]])
        return text, y


def get_dataloader(dataset, batch_size, tokenizer, seq_len):
    csv = pd.read_csv(dataset).dropna()

    len_ = len(csv)

    try:
        _, text, label = csv
    except Exception:
        try:
            _, _, text, label = csv
        except Exception:
            text, label = csv

    train_dataset = SpamDataset(csv[:int(len_ * 0.9)].reset_index(), tokenizer, seq_len)
    test_dataset = SpamDataset(csv[int(len_ * 0.9):].reset_index(), tokenizer, seq_len)

    test_dataset.y2labels = train_dataset.y2labels
    test_dataset.labels2y = train_dataset.labels2y

    trainloader = DataLoader(train_dataset, batch_size)
    testloader = DataLoader(test_dataset, batch_size)

    return trainloader, testloader, csv[label].nunique(), csv[label].unique(), csv[label]


def get_test_dataloader(dataset, batch_size, tokenizer, seq_len):
    csv = pd.read_csv(dataset).dropna()

    len_ = len(csv)
    try:
        _, text, label = csv
    except Exception:
        try:
            _, _, text, label = csv
        except Exception:
            text, label = csv

    test_dataset = SpamDataset(csv, tokenizer, seq_len)

    testloader = DataLoader(test_dataset, batch_size)

    return testloader, csv[label].nunique(), csv[label].unique()


def get_optimizer(model, lr, optimizer, sheduler, mode, gamma, step, max_lr, up_step, down_step, device, weight):
    if optimizer == 'AdamW':
        opt = torch.optim.AdamW(model.parameters(), lr)
    elif optimizer == 'Adam':
        opt = torch.optim.Adam(model.parameters(), lr)
    elif optimizer == 'SGD':
        opt = torch.optim.SGD(model.parameters(), lr, momentum=0.9)

    print(mode, gamma, step, max_lr, up_step, down_step)

    if sheduler == 'Cyclic':
        sheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=lr, max_lr=max_lr, step_size_up=up_step, step_size_down=down_step, mode=mode)
    elif sheduler == 'Linear':
        sheduler = torch.optim.lr_scheduler.StepLR(opt, step, gamma)

    class_weights = torch.FloatTensor(weight).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    return opt, criterion, sheduler


def get_criterion():
    criterion = torch.nn.CrossEntropyLoss()
    return criterion


def train(trainloader, testloader, model, criterion, opt, shed, epochs, device, signals, model_name, label):
    train_ = []
    test_ = []
    tr_acc = []
    te_acc = []
    tr_f1 = []
    te_f1 = []

    for epoch in range(epochs):
        qw = (epoch + 1) * 100 // epochs
        train_loss = 0
        train_acc = 0
        train_f1 = 0

        signals.step.emit(f'Epoch: {epoch}. Train')
        for i, batch in enumerate(trainloader):
            signals.step.emit(f'Epoch: {epoch}. Train. Iter {i}/{len(trainloader)}')
            x = batch[0]
            y = batch[1].to(device)

            y_pred = model(x['input_ids'].to(device).squeeze(1), x['attention_mask'].to(device))
            loss = criterion(y_pred, y.squeeze(1))

            opt.zero_grad()
            loss.backward()
            opt.step()

            acc = accuracy_score(y.detach().cpu().numpy(), torch.argmax(torch.softmax(y_pred, dim=1), dim=1).detach().cpu().numpy())
            f1 = f1_score(y.detach().cpu().numpy(),
                                 torch.argmax(torch.softmax(y_pred, dim=1), dim=1).detach().cpu().numpy(), average='micro')

            train_loss += loss.item()
            train_acc += acc
            train_f1 += f1

        train_loss /= len(trainloader)
        train_acc /= len(trainloader)
        train_f1 /= len(trainloader)

        test_loss = 0
        test_acc = 0
        test_f1 = 0
        signals.step.emit(f'Epoch: {epoch}. Test')
        all_y = []
        all_pred = []

        with torch.no_grad():
            for i, batch in enumerate(testloader):
                signals.step.emit(f'Epoch: {epoch}. Test. Iter {i}/{len(testloader)}')
                x = batch[0]
                y = batch[1].to(device)

                y_pred = model(x['input_ids'].to(device).squeeze(1), x['attention_mask'].to(device))
                loss = criterion(y_pred, y.squeeze(1))
                test_loss += loss.item()
                acc = accuracy_score(y.detach().cpu().numpy(),
                                                     torch.argmax(torch.softmax(y_pred, dim=1),
                                                                  dim=1).detach().cpu().numpy())

                f1 = f1_score(y.detach().cpu().numpy(),
                                                     torch.argmax(torch.softmax(y_pred, dim=1),
                                                                  dim=1).detach().cpu().numpy(), average='binary')
                test_acc += acc
                test_f1 += f1
                all_y.append(y.detach().cpu().numpy())
                all_pred.append(torch.argmax(torch.softmax(y_pred, dim=1),
                                                  dim=1).detach().cpu().numpy())
        test_loss /= len(testloader)
        test_acc /= len(testloader)
        test_f1 /= len(testloader)

        true = np.concatenate(all_y, axis=0).T[0]
        pred = np.concatenate(all_pred, axis=0)

        train_.append(train_loss)
        test_.append(test_loss)
        tr_acc.append(train_acc)
        te_acc.append(test_acc)
        tr_f1.append(train_f1)
        te_f1.append(test_f1)
        shed.step()

        signals.loss.emit([train_, test_, tr_acc, te_acc, tr_f1, te_f1, true, pred, label])
        signals.progress.emit(qw if qw <= 100 else 100)

        save_model(model, f'Epoch {epoch}' + model_name)
        #plt.plot(train_, label='train')
        #plt.plot(test_, label='test')
        #plt.legend()
        #plt.show()


def save_model(model, model_name):
    torch.save(model.state_dict(), model_name)

class NewModel(nn.Module):
    def __init__(self, model, n_class, typ):
        super().__init__()
        self.model = model
        if typ == 'base':
            self.fc = nn.Linear(768, n_class)
        else:
            self.fc = nn.Linear(1024, n_class)
        print(n_class)

    def forward(self, inp, att):
        out = self.model(inp, att).pooler_output
        out = self.fc(out)
        return out


def run(dataset, model_name, model_type, lr, classification, batch_size,
          num_epoch, optimizer, sheduler, mode, gamma, step, max_lr, up_step, down_step,
          seq_len, batch_norm, dropout, device, signals, typ):
    device = torch.device('cpu') if device == 'cpu' else torch.device('cuda')

    torch.cuda.empty_cache()
    signals.step.emit('Загрузка модели')
    try:
        tokenizer, model = get_model(model_type, typ)
        model = model.to(device)
    except Exception as exp:
        signals.step.emit('Ошибка при загрузке модели:' + str(exp))
        return 0

    signals.step.emit('Подготовка датасета')
    try:
        trainloader, testloader, n_class, label, labels = get_dataloader(dataset, int(batch_size), tokenizer, int(seq_len))
        weights = []
        for l in label:
            weights.append(labels.values.tolist().count(l))
        arr = []
        for w in weights:
            arr.append(len(labels) / w)
        print(arr)
    except Exception as exp:
        signals.step.emit('Ошибка при подготовке датасета:' + str(exp))
        return 0

    model = NewModel(model, n_class, typ)
    model.to(device)

    if classification == 'Бинарная классификация' and n_class != 2:
        signals.step.emit('При бинарной классификации должно быть 2 метки')
        return 0
    if classification == 'Мультиклассовая классификация' and n_class <= 2:
        signals.step.emit('При мультиклассовой классификации должно быть больше 2 меток')
        return 0

    signals.step.emit('Подготовка optimizer и criterion')
    try:
        optimizer, criterion, shed = get_optimizer(model, float(lr), optimizer, sheduler, mode, float(gamma), int(step), float(max_lr), int(up_step), int(down_step), device, arr)
    except Exception as exp:
        signals.step.emit('Ошибка при подготовке optimizer и criterion:' + str(exp))
        return 0

    signals.step.emit('Обучение')
    try:
        print(device)
        train(trainloader, testloader, model, criterion, optimizer, shed, int(num_epoch), device, signals, model_name, label)
    except Exception as exp:
        signals.step.emit('Ошибка при обучении:' + str(exp))
        print(exp)
        return 0
    save_model(model, model_name)
    signals.step.emit(f'Обучение успешно завершено. Модель сохранена в файл {model_name}')


def test(path_model, path_output, dataset, model_type, batch_size, seq_len, device, signals, typ):
    device = torch.device('cpu') if device == 'cpu' else torch.device('cuda')
    signals.step.emit('Загрузка модели')
    try:
        tokenizer, model = get_model(model_type, typ)
    except Exception as exp:
        signals.step.emit('Ошибка при загрузке модели:' + str(exp))
        return 0

    signals.step.emit('Подготовка датасета')
    try:
        testloader, n_class, label = get_test_dataloader(dataset, int(batch_size), tokenizer, int(seq_len))
    except Exception as exp:
        signals.step.emit('Ошибка при подготовке датасета:' + str(exp))
        return 0

    model = NewModel(model, n_class, typ)
    model.load_state_dict(torch.load(path_model))
    model.to(device)

    signals.step.emit('Подготовка criterion')
    try:
        criterion = get_criterion()
    except Exception as exp:
        signals.step.emit('Ошибка при подготовке optimizer и criterion:' + str(exp))
        return 0

    signals.step.emit('Тест')
    try:
        test_loss = 0
        test_acc = 0
        list_output = []

        all_y = []
        all_pred = []
        with torch.no_grad():
            len_t = len(testloader) # 5689 - 100
            # 5 - x

            for i, batch in enumerate(testloader):
                qw = int((i + 1) * 100 / len_t)

                signals.step.emit(f'Test. Iter {i}/{len(testloader)}')
                x = batch[0]
                y = batch[1].to(device)
                y_pred = model(x['input_ids'].to(device).squeeze(1), x['attention_mask'].to(device))

                loss = criterion(y_pred, y.squeeze(1))
                test_loss += loss.item()
                acc = accuracy_score(y.detach().cpu().numpy(),
                                     torch.argmax(torch.softmax(y_pred, dim=1),
                                                  dim=1).detach().cpu().numpy())
                test_acc += acc
                list_output.append(y_pred.detach().cpu())
                signals.progress.emit(qw if qw <= 100 else 100)

                all_y.append(y.detach().cpu().numpy())
                all_pred.append(torch.argmax(torch.softmax(y_pred, dim=1),
                                                  dim=1).detach().cpu().numpy())
        test_loss /= len(testloader)
        test_acc /= len(testloader)

        true = np.concatenate(all_y, axis=0)
        pred = np.concatenate(all_pred, axis=0)

        signals.loss.emit([test_loss, test_acc, true, pred, label])

        list_output = torch.cat(list_output, dim=0)
        list_output = torch.argmax(torch.softmax(list_output, dim=1), dim=1)


    except Exception as exp:
        signals.step.emit('Ошибка при тесте:' + str(exp))
        return 0
    try:
        list_output = list_output.detach().cpu().numpy()
        df = pd.DataFrame(list_output, columns=['Label'])
        df.to_csv(path_output)
    except Exception as exp:
        signals.step.emit('Ошибка при сохранении:' + str(exp))
        return 0

    signals.step.emit(f'Тестирование успешно завершено. Результат сохранен в файл {path_output}')
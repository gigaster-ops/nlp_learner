from app import *
from api import Sender
import torch
import torch.nn as nn
import torchvision
import json
import learning
from PyQt5.QtCore import QThreadPool, QThread, QRunnable, QObject, pyqtSignal, pyqtSlot
import traceback
from PyQt5.QtWidgets import QFileDialog
import time, sys

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np

from PyQt5 import QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


# Если при ошибке в слотах приложение просто падает без стека,
# есть хороший способ ловить такие ошибки:
def log_uncaught_exceptions(ex_cls, ex, tb):
    text = '{}: {}:\n'.format(ex_cls.__name__, ex)
    # import traceback
    text += ''.join(traceback.format_tb(tb))
    QtWidgets.QMessageBox.critical(None, 'Error', text)


sys.excepthook = log_uncaught_exceptions


class WorkerSignals(QObject):
    ''' Определяет сигналы, доступные из рабочего рабочего потока Worker(QRunnable).'''

    text_button = pyqtSignal(str)
    enabled = pyqtSignal(bool)
    loss = pyqtSignal(list)
    accuracy = pyqtSignal(list)

    step = pyqtSignal(str)

    error = pyqtSignal(tuple)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    ''' Наследует от QRunnable, настройки рабочего потока обработчика, сигналов и wrap-up. '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Хранить аргументы конструктора (повторно используемые для обработки)
        self.fn = fn
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        # Получите args/kwargs здесь; и обработка с их использованием
        try:  # выполняем метод `some_func` переданный из Main
            result = self.fn(self.signals)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))


class Worker_test(QRunnable):
    ''' Наследует от QRunnable, настройки рабочего потока обработчика, сигналов и wrap-up. '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker_test, self).__init__()

        # Хранить аргументы конструктора (повторно используемые для обработки)
        self.fn = fn
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        # Получите args/kwargs здесь; и обработка с их использованием
        try:  # выполняем метод `some_func` переданный из Main
            result = self.fn(self.signals)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))


class Worker_load(QRunnable):
    ''' Наследует от QRunnable, настройки рабочего потока обработчика, сигналов и wrap-up. '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker_load, self).__init__()

        # Хранить аргументы конструктора (повторно используемые для обработки)
        self.fn = fn
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        # Получите args/kwargs здесь; и обработка с их использованием
        try:  # выполняем метод `some_func` переданный из Main
            result = self.fn(self.signals)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))


def pre_learning():
    dataset = ui.lineEdit.text()
    model_name = ui.lineEdit_2.text()
    model_type = ui.comboBox.currentText()
    lr = ui.lineEdit_3.text()
    classification = ui.comboBox_2.currentText()
    batch_size = ui.lineEdit_4.text()
    num_epoch = ui.lineEdit_5.text()
    optimizer = ui.comboBox_3.currentText()
    sheduler = ui.comboBox_4.currentText()
    mode = ui.comboBox_9.currentText()
    gamma = ui.lineEdit_16.text()
    step = ui.lineEdit_17.text()
    max_lr = ui.lineEdit_8.text()
    up_step = ui.lineEdit_11.text()
    down_step = ui.lineEdit_15.text()
    seq_len = ui.lineEdit_10.text()
    batch_norm = ui.checkBox.isChecked()
    dropout = ui.lineEdit_11.text()
    device = ui.comboBox_7.currentText()
    type = ui.lineEdit_21.text()

    print('Старт обучения', dataset, model_name, model_type, lr, classification, batch_size,
          num_epoch, optimizer, sheduler, mode, gamma, step, max_lr, up_step, down_step,
          seq_len, batch_norm, dropout, device)

    start_learning(dataset, model_name, model_type, lr, classification, batch_size,
                   num_epoch, optimizer, sheduler, mode, gamma, step, max_lr, up_step, down_step,
                   seq_len, batch_norm, dropout, device, type)


def pre_test():
    path_model = ui.lineEdit_6.text()
    path_output = ui.lineEdit_12.text()
    model_type = ui.comboBox_5.currentText()
    dataset = ui.lineEdit_7.text()
    device = ui.comboBox_8.currentText()
    batch_size = ui.lineEdit_4.text()
    seq_len = ui.lineEdit_10.text()
    type = ui.lineEdit_22.text()

    print('Старт теста', path_model, path_output, model_type, dataset)
    start_test(path_model, path_output, model_type, batch_size, seq_len, dataset, device, type)


def pre_load():
    path_model = ui.lineEdit_13.text()
    name_model = ui.lineEdit_14.text()
    model_type = ui.comboBox_6.currentText()
    n_class = ui.lineEdit_18.text()
    label = ui.textEdit.toPlainText()
    type = ui.lineEdit_20.text()

    print('Начало загрузки модели', path_model, name_model, model_type, label)
    start_load(path_model, name_model, model_type, int(n_class), label, type)


def pre(ui):
    ui.pushButton.clicked.connect(pre_learning)
    ui.pushButton_2.clicked.connect(pre_test)
    ui.pushButton_3.clicked.connect(pre_load)

    ui.progressBar.setValue(0)
    ui.progressBar_2.setValue(0)
    ui.progressBar_3.setValue(0)

    ui.label_58.setText('')

    def folder():
        name = QFileDialog.getOpenFileName()
        print(name)
        ui.lineEdit.setText(name[0])

    ui.pushButton_4.clicked.connect(folder)

    def folder2():
        name = QFileDialog.getOpenFileName()
        print(name)
        ui.lineEdit_13.setText(name[0])

    ui.pushButton_5.clicked.connect(folder2)

    def folder3():
        name = QFileDialog.getOpenFileName()
        print(name)
        ui.lineEdit_6.setText(name[0])

    ui.pushButton_6.clicked.connect(folder3)

    def folder4():
        name = QFileDialog.getOpenFileName()
        print(name)
        ui.lineEdit_7.setText(name[0])

    ui.pushButton_7.clicked.connect(folder4)

    cuda_avialable = torch.cuda.is_available()
    ui.comboBox_7.addItem('cpu', 'cpu1')
    print(torch.cuda.is_available())
    if cuda_avialable:
        ui.comboBox_7.addItem('cuda', 'cuda1')

    ui.comboBox_8.addItem('cpu', 'cpu1')
    if cuda_avialable:
        ui.comboBox_8.addItem('cuda', 'cuda1')

    fig = plt.Figure()
    canvas = FigureCanvasQTAgg(fig)

    ui.verticalLayout.addWidget(canvas)

    ax = fig.add_subplot(111)

    fig1 = plt.Figure()
    canvas1 = FigureCanvasQTAgg(fig1)

    ui.verticalLayout.addWidget(canvas1)

    ax1 = fig1.add_subplot(111)

    fig2 = plt.Figure()
    canvas2 = FigureCanvasQTAgg(fig2)

    ui.verticalLayout.addWidget(canvas2)

    ax2 = fig2.add_subplot(111)

    return fig, canvas, ax, fig1, canvas1, ax1, fig2, canvas2, ax2


def start_learning(dataset, model_name, model_type, lr, classification, batch_size,
                   num_epoch, optimizer, sheduler, mode, gamma, step, max_lr, up_step, down_step,
                   seq_len, batch_norm, dropout, device, typ):
    def execute_this_fn(signals):
        signals.text_button.emit('Идёт обучение')
        signals.enabled.emit(False)
        learning.run(dataset, model_name, model_type, lr, classification, batch_size,
                     num_epoch, optimizer, sheduler, mode, gamma, step, max_lr, up_step, down_step,
                     seq_len, batch_norm, dropout, device, signals, typ)
        signals.text_button.emit('Начать обучение')
        signals.enabled.emit(True)
        return "Готово."

    def set_text(text):
        ui.pushButton.setText(text)

    def set_enabled(flag):
        ui.pushButton.setEnabled(flag)

    def bar(n):
        ui.progressBar.setValue(n)

    def step_info(text):
        ui.label_15.setText(text)

    def plot_loss(arr):
        print(arr)
        train_loss, test_loss, train_acc, test_acc, train_f1, test_f1, true, pred, label = arr

        ax.clear()
        # plot data
        ax.plot(train_loss, label='train loss')
        ax.plot(test_loss, label='test loss')
        ax.legend()

        # refresh canvas
        canvas.draw()

        ax1.clear()
        # plot data
        ax1.plot(train_acc, label='train acc')
        ax1.plot(test_acc, label='test acc')
        ax1.legend()

        # refresh canvas
        canvas1.draw()

        ax2.clear()
        # plot data
        ax2.plot(train_f1, label='train f1')
        ax2.plot(test_f1, label='test f1')
        ax2.legend()

        # refresh canvas
        canvas2.draw()
        print(train_loss[-1])
        ui.label_26.setText(f'Train loss: {train_loss[-1]} Test loss: {test_loss[-1]}')
        ui.label_27.setText((f'Train acc: {train_acc[-1]} Test acc: {test_acc[-1]}'))
        ui.label_45.setText((f'Train f1: {train_f1[-1]} Test f1: {test_f1[-1]}'))

        with open('loss.txt', 'a') as f:
            f.write(f'Train: {train_loss[-1]} Test: {test_loss[-1]}')
        with open('acc.txt', 'a') as f:
            f.write(f'Train: {train_acc[-1]} Test: {test_acc[-1]}')
        with open('f1.txt', 'a') as f:
            f.write(f'Train: {train_f1[-1]} Test: {test_f1[-1]}')

        d = {i: [0 for j in label] for i in range(len(label))}
        for i in range(len(true)):
            d[true[i]][pred[i]] += 1
        df = pd.DataFrame(d)
        df = df.T
        df = df.iloc[::-1]

        plt.imshow(df.values)
        for i in range(len(label)):
            for j in range(len(label)):
                text = ax.text(j, i, df.values[i, j],
                               ha="center", va="center", color="w")

        plt.xticks(np.arange(len(label)), labels=label)
        plt.yticks(np.arange(len(label)), labels=label[::-1])
        for i in range(len(label)):
            for j in range(len(label)):
                text = plt.text(j, i, df.values[i, j],
                                ha="center", va="center", color="w")
        plt.show()

    ui.threadpool = QThreadPool()
    worker = Worker(execute_this_fn)
    worker.signals.progress.connect(bar)

    worker.signals.text_button.connect(set_text)
    worker.signals.enabled.connect(set_enabled)
    worker.signals.step.connect(step_info)

    worker.signals.loss.connect(plot_loss)

    ui.threadpool.start(worker)


def start_test(path_model, path_output, model_type, batch_size, seq_len, dataset, device, typ):
    def execute_this_fn_test(signals):
        signals.text_button.emit('Идёт тестирование')
        signals.enabled.emit(False)
        learning.test(path_model, path_output, dataset, model_type, batch_size, seq_len, device, signals, typ)
        signals.text_button.emit('Начать тестирование')
        signals.enabled.emit(True)
        return "Готово."

    def step_info(text):
        ui.label_58.setText(text)

    def set_text(text):
        ui.pushButton_2.setText(text)

    def set_enabled(flag):
        ui.pushButton_2.setEnabled(flag)

    def bar(n):
        ui.progressBar_2.setValue(n)

    def plot_loss(arr):
        loss, acc, true, pred, label = arr
        true, pred = list(true[:, 0]), list(pred)

        ui.label_21.setText('Потери: ' + str(loss))
        ui.label_22.setText('Точность: ' + str(acc))

        d = {i: [0 for j in label] for i in range(len(label))}
        for i in range(len(true)):
            d[true[i]][pred[i]] += 1
        df = pd.DataFrame(d)
        df = df.T
        df = df.iloc[::-1]

        plt.imshow(df.values)
        for i in range(len(label)):
            for j in range(len(label)):
                text = ax.text(j, i, df.values[i, j],
                               ha="center", va="center", color="w")

        plt.xticks(np.arange(len(label)), labels=label)
        plt.yticks(np.arange(len(label)), labels=label[::-1])
        for i in range(len(label)):
            for j in range(len(label)):
                text = plt.text(j, i, df.values[i, j],
                                ha="center", va="center", color="w")
        plt.show()

    ui.threadpool = QThreadPool()
    worker = Worker_test(execute_this_fn_test)
    worker.signals.progress.connect(bar)

    worker.signals.text_button.connect(set_text)
    worker.signals.enabled.connect(set_enabled)
    worker.signals.step.connect(step_info)
    worker.signals.loss.connect(plot_loss)

    ui.threadpool.start(worker)


def start_load(path_model, name_model, model_type, n_class, label, type):
    ui.threadpool = QThreadPool()

    def execute_this_fn_load(signals):
        signals.step.emit('Отправка модели')
        signals.progress.emit(0)
        signals.enabled.emit(False)
        out = sender.load_model(name_model, model_type, path_model, n_class, label, type)
        if out['access']:
            signals.step.emit(f'Токен: {out["token"]}')
        else:
            signals.step.emit(out['msg'])
        signals.progress.emit(100)
        signals.enabled.emit(True)

    def step_info(text):
        ui.lineEdit_19.setText(text)

    def set_text(text):
        ui.pushButton_3.setText(text)

    def set_enabled(flag):
        ui.pushButton_3.setEnabled(flag)

    def bar(n):
        ui.progressBar_3.setValue(n)

    worker = Worker_load(execute_this_fn_load)
    worker.signals.progress.connect(bar)

    worker.signals.text_button.connect(set_text)
    worker.signals.enabled.connect(set_enabled)
    worker.signals.step.connect(step_info)

    ui.threadpool.start(worker)


if __name__ == "__main__":
    import sys

    sender = Sender()

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()

    fig, canvas, ax, fig1, canvas1, ax1, fig2, canvas2, ax2 = pre(ui)

    sys.exit(app.exec_())

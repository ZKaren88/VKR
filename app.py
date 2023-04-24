import flask
from flask import Flask, render_template, request
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import Sequential, utils, layers, models
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Activation, Dropout, LSTM
import tensorflow.python.keras.optimizers

#Инициализация приложения и укажем директорию для html шаблонов
app = flask.Flask(__name__, template_folder = 'templates')

#Создаём декораторы, которые будут соотвествовавать адрессам страницы в приложении
@app.route('/', methods = ['POST', 'GET'])

@app.route('/index', methods = ['POST', 'GET'])

#Пишем функцию, которая будет отрабатывать web форму и получать значения с входными параметрами, после чего делать прогноз
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')
    
    if flask.request.method =="POST":
        reconstructed_model_mms = pickle.load(open('min_max_scaler.pkl','rb'))
        reconstructed_model_ns = pickle.load(open('NS_model_for_y3.pkl','rb'))
        data = [{
            'Соотношение матрица-наполнитель':0,
            'Плотность, кг/м3':float(flask.request.form['x1']),
            'Модуль упругости, ГПа':float(flask.request.form['x2']),
            'Количество отвердителя, м.%':float(flask.request.form['x3']),
            'Содержание эпоксидных групп,%_2':float(flask.request.form['x4']),
            'Температура вспышки, С_2':float(flask.request.form['x5']),
            'Поверхностная плотность, г/м2':float(flask.request.form['x6']),
            'Модуль упругости при растяжении, ГПа':float(flask.request.form['x7']),
            'Прочность при растяжении, МПа':float(flask.request.form['x8']),
            'Потребление смолы, г/м2':float(flask.request.form['x9']),
            'Угол нашивки, град':float(flask.request.form['x10']),
            'Шаг нашивки':float(flask.request.form['x11']),
            'Плотность нашивки':float(flask.request.form['x12'])
            }]
        data_pred = pd.DataFrame(data)
        data_pred = pd.DataFrame(reconstructed_model_mms.transform(data_pred), columns = data_pred.columns)
        data_for_pred = data_pred.drop(['Соотношение матрица-наполнитель'], axis = 1)
        data_pred['Соотношение матрица-наполнитель'] = reconstructed_model_ns.predict(data_for_pred)
        dara_after_denormalization = reconstructed_model_mms.inverse_transform(data_pred)
        data_result = pd.DataFrame(dara_after_denormalization, columns = data_pred.columns)
        pred_result = round(data_result['Соотношение матрица-наполнитель'][0], 4)
        return render_template ('main.html', result = pred_result)

#Инициализация приложения
if __name__ == '__main__':
    app.run(debug=False)

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from typing import Callable
import gpuRIR
import json
import numpy as np
from math import ceil
from flask import Flask,request,send_file
import pickle
import jsonpickle
# 注册app对象
app = Flask(__name__)

# 将路由以装饰器的形式装饰函数
@app.route('/RirNormal',methods=['POST'])
def generate_rir():
    temp = pickle.load(request.files.get('file'))
    nb_img = gpuRIR.t2n(temp['RT60'],temp['room_sz'])  # Number of image sources in each dimension
    # nb_img = gpuRIR.t2n(Tmax, room_sz)
    beta = gpuRIR.beta_SabineEstimation(temp['room_sz'], temp['RT60'], abs_weights=temp['abs_weights'])  # reflection coefficients
    rir = gpuRIR.simulateRIR(room_sz=temp['room_sz'], beta=beta, pos_src=temp['pos_src'], pos_rcv=temp['pos_rcv'], nb_img=nb_img, Tmax=temp['RT60'], fs=temp['fs'])
    rir = jsonpickle.encode(rir)
    return rir
if __name__ == '__main__':
    # 绑定ip和端口号
    app.run(host="172.16.75.83",port=1234,debug=True)

from flask import Flask, render_template, request
#from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
from keras.models import load_model
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import re
import sys
import os
import time
import keras
import fun
import tensorflow as tf
from fun import weighted_loss
from fun import _read
from keras import backend as K
import logging
import gdown

app = Flask(__name__)
if os.path.exists('finalmod.h5'):
    pass
else:
    url = 'https://drive.google.com/uc?id=11shl3wITpv0fPoREhMqY3utjNo4RuIZi'
    output ='finalmod.h5'
    gdown.download(url, output, quiet = False)

global mod
mod = load_model('finalmod.h5', custom_objects={'weighted_loss': weighted_loss}, compile = False)
mod.compile(loss = "binary_crossentropy", optimizer = keras.optimizers.Adam(), metrics = [weighted_loss])
graph = tf.get_default_graph()

logging.info('successfully loaded model')






@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/uploader', methods = ['POST'])
# def upload_image_file():
#     if request.method == 'POST':
#         img = Image.open(request.files['file'].stream).convert("L")
#         im2arr = np.array(img)
#         with graph.as_default

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        up = request.files['image']

        if not up:
            return render_template('index.html', label = 'No file')

    pred_dict = {0 : 'epidural', 1: 'intraparenchymal', 2: 'intraventricular', 3: 'subarachnoid', 4: 'subdural'}

    img = _read(up, (256, 256))
    logging.info('successfully read image')
    plt.imshow(img, cmap = plt.cm.bone);
    dir = os.path.join(os.path.abspath('static'), 'tmp.png')
    plt.savefig(dir)

    with graph.as_default():

        out = mod.predict(img.reshape(1, 256, 256, 3))
        out = out[0][:5]
        logging.info('model successfully returned predictions')

        label = pred_dict[np.argmax(out)]

    return render_template('index.html', label = label, pres = time.time(), file = up)


if __name__ == '__main__':
    app.run(debug = True, port = 8000, host = '0.0.0.0', threaded = False)
    #app.run(host='0.0.0.0', port = 8000, debug = True)



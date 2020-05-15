from flask import Flask, render_template, request
#from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
from keras.models import load_model
import matplotlib.pyplot as plt
import re
import sys
import os
import time
import fun
import pathlib
from fun import weighted_loss
from fun import _read
import tensorflow as tf
from keras import backend as K
import logging
import gdown

app = Flask(__name__)
if os.path.exists('newmod (1).h5'):
    pass
else:
    url = 'https://drive.google.com/uc?id=1NJNG4QEzvDdlMi-1SXiYuQU6EenleGwT'
    output ='newmod (1).h5'
    gdown.download(url, output, quiet = False)

global mod
mod = load_model('newmod (1).h5', custom_objects={'weighted_loss': weighted_loss})
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

    pred_dict = {0: 'any', 1 : 'epidural', 2: 'intraparenchymal', 3: 'intraventricular', 4: 'subarachnoid', 5: 'subdural'}

    img = _read(up, (256, 256))
    logging.info('successfully read image')
    plt.imshow(img, cmap = plt.cm.bone);
    dir = os.path.join(os.path.abspath('static'), 'tmp.png')
    plt.savefig(dir)


    out = mod.predict(img.reshape(1, 256, 256, 3))
    logging.info('model successfully returned predictions')

    label = pred_dict[np.argmax(out)]

    return render_template('index.html', label = label, pres = time.time(), file = up)


if __name__ == '__main__':
    app.run(debug = True, port = 8000, host = '0.0.0.0', threaded = False)
    #app.run(host='0.0.0.0', port = 8000, debug = True)



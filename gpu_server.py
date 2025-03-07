import base64
import os
import pickle
import time

from flask import Flask
from flask import request
from flask_cors import CORS
from model import encode, encode_queries, run_model

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024
CORS(app, supports_credentials=True, origins='*')



def decode_data(data):
    return pickle.loads(base64.decodebytes(bytes(data, "utf-8")))

def encode_data(data):
    return base64.b64encode(pickle.dumps(data)).decode("utf-8")


@app.route('/det', methods=['POST'])
def detect_fun():
    img = decode_data(request.json["img"])
    ret = run_model("det", img)
    return encode_data(ret)

@app.route('/rec', methods=['POST'])
def recect_fun():
    img = decode_data(request.json["img"])
    ret = run_model("rec", img)
    return encode_data(ret)
@app.route('/layout/<name>', methods=['POST'])
def layout_fun(name):
    img = decode_data(request.json["img"])
    ret = run_model(name, img)
    return encode_data(ret)

@app.route('/bge/encode', methods=['POST'])
def bge_encode():
    start_time = time.time()
    text = decode_data(request.json["text"])
    print(len(text))
    try:
        return encode_data(encode(text))
    finally:
        print(f"🕒 Processing time: {time.time() - start_time:.2f}s")


@app.route('/bge/encode_queries', methods=['POST'])
def bge_encode_queries():
    start_time = time.time()
    text = decode_data(request.json["text"])
    print(len(text))
    print(text)
    try:
        return encode_data(encode_queries(text))
    finally:
        print(f"🕒 Processing time: {time.time() - start_time:.2f}s")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1000)

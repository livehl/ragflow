import os
import pickle
import time
from flask import Flask
from flask import request
from flask_cors import CORS
from flask import Response
from model import encode, encode_queries, run_model

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024
CORS(app, supports_credentials=True, origins='*')


@app.route('/det', methods=['POST'])
def detect_fun():
    img = pickle.loads(request.data)
    ret = run_model("det", img)
    return Response(pickle.dumps(ret), mimetype='application/octet-stream')

@app.route('/rec', methods=['POST'])
def recect_fun():
    img = pickle.loads(request.data)
    ret = run_model("rec", img)
    return Response(pickle.dumps(ret), mimetype='application/octet-stream')
@app.route('/layout/<name>', methods=['POST'])
def layout_fun(name):
    img = pickle.loads(request.data)
    ret = run_model(name, img)
    return Response(pickle.dumps(ret), mimetype='application/octet-stream')

@app.route('/bge/encode', methods=['POST'])
def bge_encode():
    start_time = time.time()
    text = pickle.loads(request.data)
    try:
        return Response(pickle.dumps(encode(text)), mimetype='application/octet-stream')
    finally:
        print(f"🕒 Processing time: {time.time() - start_time:.2f}s")


@app.route('/bge/encode_queries', methods=['POST'])
def bge_encode_queries():
    start_time = time.time()
    text = pickle.loads(request.data)
    try:
        return Response(pickle.dumps(encode_queries(text)), mimetype='application/octet-stream')
    finally:
        print(f"🕒 Processing time: {time.time() - start_time:.2f}s")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1000)

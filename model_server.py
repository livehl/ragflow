import base64
import os
import pickle

from FlagEmbedding import FlagModel
from flask import Flask
from flask import request
from flask_cors import CORS

loaded_models = {}

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024
CORS(app, supports_credentials=True, origins='*')


# app.layout_model={}


def cuda_is_available():
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except Exception:
        return False
    return False


bge = FlagModel("/root/.ragflow/bge-large-zh-v1.5",
                query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                use_fp16=cuda_is_available())
from threading import Lock

bge_lock = Lock()


def decode_data(data):
    return pickle.loads(base64.decodebytes(bytes(data, "utf-8")))


def encode_data(data):
    return base64.b64encode(pickle.dumps(data)).decode("utf-8")

@app.route('/bge/encode', methods=['POST'])
def bge_encode():
    text = decode_data(request.json["text"])
    with bge_lock:
        return encode_data(bge.encode(text))


@app.route('/bge/encode_queries', methods=['POST'])
def bge_encode_queries():
    text = decode_data(request.json["text"])
    with bge_lock:
        return encode_data(bge.encode_queries(text))



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1000)

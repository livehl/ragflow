import base64
import os
import pickle

from FlagEmbedding import FlagModel
from flask import Flask
from flask import request
from flask_cors import CORS
from flask import Response

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


@app.route('/bge/encode', methods=['POST'])
def bge_encode():
    text = pickle.loads(request.data)
    with bge_lock:
        return Response(pickle.dumps(bge.encode(text)), mimetype='application/octet-stream')


@app.route('/bge/encode_queries', methods=['POST'])
def bge_encode_queries():
    text = pickle.loads(request.data)
    with bge_lock:
        return Response(pickle.dumps(bge.encode_queries(text)), mimetype='application/octet-stream')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1000)

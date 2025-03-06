import base64
import os
import pickle
import re

from FlagEmbedding import FlagModel
from flask import Flask
from flask import request
from flask_cors import CORS
import onnxruntime as ort
loaded_models = {}

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024
CORS(app, supports_credentials=True, origins='*')


app.layout_model={}


def cuda_is_available():
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except Exception:
        return False
    return False

def load_model(model_dir, nm):
    model_file_path = os.path.join(model_dir, nm + ".onnx")
    global loaded_models
    loaded_model = loaded_models.get(model_file_path)
    if loaded_model:
        print(f"load_model {model_file_path} reuses cached model")
        return loaded_model

    if not os.path.exists(model_file_path):
        raise ValueError("not find model file path {}".format(
            model_file_path))


    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = True
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 32
    options.inter_op_num_threads = 32
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


    # https://github.com/microsoft/onnxruntime/issues/9509#issuecomment-951546580
    # Shrink GPU memory after execution
    run_options = ort.RunOptions()
    if cuda_is_available():
        cuda_provider_options = {
            "device_id": 0, # Use specific GPU
            "gpu_mem_limit": 512 * 1024 * 1024, # Limit gpu memory
            "arena_extend_strategy": "kNextPowerOfTwo",  # gpu memory allocation strategy
        }
        sess = ort.InferenceSession(
            model_file_path,
            options=options,
            providers=['CUDAExecutionProvider'],
            provider_options=[cuda_provider_options]
            )
        run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
        print(f"load_model {model_file_path} uses GPU")
    else:
        sess = ort.InferenceSession(
            model_file_path,
            options=options,
            providers=['CPUExecutionProvider'])
        run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu")
        print(f"load_model {model_file_path} uses CPU")
    loaded_model = (sess, run_options)
    loaded_models[model_file_path] = loaded_model
    return loaded_model



model_path="/ragflow/rag/res/deepdoc"
det_predictor, det_run_options = load_model(model_path, 'det')
rec_predictor, rec_run_options = load_model(model_path, 'rec')

bge= FlagModel("/root/.ragflow/bge-large-zh-v1.5",
    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
    use_fp16=cuda_is_available())


def decode_data(data):
    return pickle.loads(base64.decodebytes(bytes(data, "utf-8")))

def encode_data(data):
    return base64.b64encode(pickle.dumps(data)).decode("utf-8")


@app.route('/det', methods=['POST'])
def detect_fun():
    img = decode_data(request.json["img"])
    ret = det_predictor.run(None, img, det_run_options)
    return encode_data(ret)

@app.route('/rec', methods=['POST'])
def recect_fun():
    img = decode_data(request.json["img"])
    ret = rec_predictor.run(None, img, rec_run_options)
    return encode_data(ret)


@app.route('/bge/encode', methods=['POST'])
def bge_encode():
    text = decode_data(request.json["text"])
    return encode_data(bge.encode(text))


@app.route('/bge/encode_queries', methods=['POST'])
def bge_encode_queries():
    text = decode_data(request.json["text"])
    return encode_data(bge.encode_queries(text))


@app.route('/layout/<name>', methods=['POST'])
def layout_fun(name):
    if name not in app.layout_model:
        app.layout_model[name] = load_model(model_path, name)
    pre,opt=app.layout_model[name]
    img = decode_data(request.json["img"])
    ret = pre.run(None, img, opt)
    return encode_data(ret)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1000)

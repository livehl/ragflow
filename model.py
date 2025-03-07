from FlagEmbedding import FlagModel
from threading import Lock
import onnxruntime as ort
import os
def cuda_is_available():
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except Exception:
        return False
    return False


models={}


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
    return loaded_model



model_path="/ragflow/rag/res/deepdoc"


bge_lock = Lock()

bge=FlagModel("/root/.ragflow/bge-large-zh-v1.5",
                query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                use_fp16=cuda_is_available())

def encode(text):
    with bge_lock:
        return bge.encode(text)

def encode_queries(text):
    with bge_lock:
        return bge.encode_queries(text)

def run_model(name,img):
    if name not in models:
        models[name]=load_model(model_path,name)
    return models[name][0].run(None,img,models[name][1])
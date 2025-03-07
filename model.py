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


if __name__=="__main__":
    ts=['图表解读方式：以斜线划分租金降幅不同阶段的城市。斜线上方区间代表租金降幅较区间，反之则落入斜线下方区间。落在斜线上的城市代表连续两个季度租金降幅不变。\n2%1%上季度放缓。例如一个城市上季度租金降幅为，本季度租金降幅为，则落入上方\n主要城市群甲级办公楼租金变化\n(2Q24, %)\n-6.0%-\n5.0% \n-4.0%-\n3.0% \n-2.0%-\n1.0% \n0.0%\n租金降幅较上季度放缓\n宁波\n杭州\n-1.0%\n●苏州\n长沙·南京\n-2.0%\n重庆\n·天津\n广州深圳武汉\n无锡\n-3.0%\n京津冀城市群\n北京·\n长三角城市群\n大湾区城市群\n成渝城市群\n-4.0%\n长江中游城市群\n其他城市\n租金\n环比\n5.0%\n(3Q24\n●成都\n%\n租金降幅较上季度加快\n-6.0%', '主要城市群甲级办公楼空置率对比\n40中国城办公楼市场指数三季度发布：市场在漫长筑底中迎来积极政策信号\n50%\n40%\n武汉\n沈阳\n天津\n厦门、青岛、大连\n宁波\n郑州\n成都、\n重庆\n30%\n南京\n长沙\n苏州\n深圳\n西安\n杭州\n上海、无锡\n20%\n广州\n北京\n10%\n0%\n京津冀\n长三角\n大湾区\n成渝\n长江中游\n其他城市\n城市群\n城市群\n城市群\n城市群\n城市群\n京津冀城市群\n长三角城市群\n大湾区城市群\n成渝城市群\n长江中游城市群\n）其他城市', '20242024年第二季度数据。\n20\n（元/平方米/月）\n05 \n0 \n100 \n150 \n200 \n250 \n300 \n350 \n400\n北京\n3.1%\n268\n-0.2\n12.0% \n188\n上海\n4.7%\n-0.1\n23.0%\n2.4%\n149\n深圳\n1.9\n27.0%\n广州\n2.5%\n134\n-1.3\n20.0%\n成都\n75\n30.0%\n杭州\n126\n0.4\n24.0%\n重庆\n68\n30.0%\n101\n南京\n0.1\n30.0%\n武汉\n79\n38.0%\n长沙\n77\n0.0\n30.0%\n厦门\n3.2\n66\n0.0\n35.0%\n1.9\n83\n西安\n1.2\n26.0%\n苏州\n78\n4.5\n27.0% \n郑州\n0.4\n68\n0.0\n33.0%\n青岛\n2.8\n103\n0.0\n35.0%\n天津\n2.3%\n78\n35.8%\n沈阳\n2.0\n67 \n0.1\n37.0%\n■平均租金（上轴）\n大连\n83\n35.0%\n■空置率（下轴）\n80\n●租金环æ¯变化（%）\n宁波\n0.8%\n32.0%\n0.0\n●空置率环比变化 (百分点)\n无锡\n2.8\n59\n0.0\n23.0% \n0% \n10% \n20% \n30% \n40% \n50% \n60% \n70% \n80%\n', '口', '向光而为\nO)JLL', 'LL', '  40中国城甲级办公楼市场指数20242024量联行，年第三季度。厦门、长沙、宁波、无锡为年第 \n  排名环比变化    城市      \n        指数  环比变化  \n      北京  167.3  -5.8%  \n      上海  164.8  -5.7%  \n      深圳  116.7  -3.0%  \n      广州  88.6  -1.2%  \n      杭州  63.2   0.1%  \n      成都  61.4  -1.2%  \n      重庆  58.3  -0.6%  \n      南京  58.1  -0.4%  \n      武汉  56.9  -0.4%  \n  10  1  长沙  56.0  -0.3%  \n  11  1  西安      \n        55.9  -0.3%  \n  12    苏州  55.5  -0.4%  \n  13    厦门  55.4  -0.7%  \n  14    郑州  54.2  -0.2%  \n  15    青岛      \n        54.0  0.2%  \n  16    南宁      \n        54.0  -0.8%  \n  17  1  福州      \n        53.6  -0.1%  \n  18    宁波  53.6  0.0%  \n  19  1  天津  53.4  -0.2%  \n  20  1  昆明  53.3  -0.4%  \n  21    海口  53.3  0.1%  \n  22    贵阳      \n        52.9  -0.4%  \n  23    沈阳  52.8  -0.2%  \n  24    大连  52.6  0.1%  \n  25    济南  52.3  -0.2%  \n  26  1  合肥  52.2  -0.2%  \n  27   1  佛山  52.1  -0.3%  \n  8    石家庄  52.1  -0.3%  \n  29    乌鲁木齐      \n        51.6  -0.1%  \n  30    太原  51.4  -0.1%  \n  31  1  长春  51.4  0.0%  \n  32  1  南昌  51.4  0.1%  \n  33        -0.1%  \n      无锡  51.2    \n  34    兰州  51.1  -0.2%  \n  35    呼和浩特  50.9  0.0%  \n  36  1   哈尔滨  50.7  0.0%  \n  37  1  西宁  50.7  -0.1%  \n  38    泉州  50.5  0.0%  \n  39    银川  50.4  -0.1%  \n  40     拉萨  50.2   0.1%  \n ', '研究部202410年月joneslanglasalle.com.cn40中国城办公楼市场指数三季度发布：市场在漫长筑信号积极政策JLL 睿见观察小组向光而为40中国城办公楼市场指数三季度发布：市场在漫长筑底中迎来积极政策信号4.8%作为全球经济复苏的重要力量之一，中国在前三季度的国内生产总值同比增速达。尽管内循环不畅带来的下行风险仍占主导，但经济韧性强、潜力大等有利条件未发生改变。宏观调控政策的短期发力正在发挥立竿见影的政策效能，有效提振善仍需时间传导，需求端持续低迷迫使供应端积极采取以经济复苏的信心和预期。在全国办公楼市场的微观层面，宏观改降租为底层工具的组合策略，市场仍处于漫长艰难的探底过程。', '2024年第三季度要点归纳：自用型需求巩固修复基础，弱化市场流动性失衡风险。三季1. 自用型需求巩固修复基础，弱化市场流动性失衡风险。三季4067.6度，全国个主要城市甲级办公楼市场净吸纳总量为万4.0%平方米，环比增加，主要来自于一线及强二线城市的总部自用型需求支撑。然而全国市场化租赁成交量仍处于底部摩擦，有效需求不足是当前办公楼市场的主要困境，低价刺激下的市场流动性在上半年已得到一定释放，因此，以价换量的有效性在三季度有所减弱。办公楼市场需求的结构性调整持续受到扰动，金融、专业服务等大面积租户遵循审慎扩张与降本缩租的原则，拖累新租成交量的表现；跨境电商、新媒体等科技互联网细分行业虽然带来一定动能需求，但主要集中在中小面积段及新兴子市场，持续性和稳定性偏弱的特质使其无法驱动市场全面复苏。三季度，此类行业、杭州、成都、大连、南昌、合肥等城市均录得相在深圳、广州', '2.政策率先探底，释放积极信号。近期宏观调控措施显著发政策率先探底，释放积极信号。近期宏观调控措施显著发力，逆周期调节力度加大。从企业端来看，降息降准等货币与财政政策的组合措施为企业应对下行周期提供了强有力的支持，有利于疏通企业融资渠道，缓解办公楼市场来自企业端的需æ±净流出。尽管短期内政策调整影响有限，但其对扭转预期的信号意义显著；从业主端来看，政策制定者积极通过改善投资环境和招商引资政策推进经济再平衡，有利于稳定和抑制需求环境的进一步恶化。深圳和海口等城市的办公楼市场因有效的税收及招商政策支持，供需关系始终相对健康发展。此外，新质生产力的培育和研发成果转化低了租金和税务压力，加速，部分地区企业在政策扶持下降预计将为办公楼市场带来增量机会。', '40 3.市场尚未见底，打破底线价格。三季度，全国个主要城市80.3//的甲级办公楼市场平均租金为元平方米月，环比降幅0.4%-5.6%在之间。在经历了需求增长受阻的持续困扰后，业主方压力情绪在第三季度达到高点。为抓住流动的存量需求机会，业主方采取更加灵活激进的租金策略，以获得竞争优势。北京市场释放个别极限价格以缩短租赁方决策周期。成都多数业主或困于市场或自身经营难题，不得不提出远低于市场的价格，或制定非常规商务条款。南宁市场头部项目租金下调接近极限成本，激发续租与搬迁需求。市场情策略激进正逐步缩小核心与非核心区域的价绪低迷与租金差，加速筑底。', '2022年第一季度数据为基准数，可进行横向时间轴和纵指数解读方式：以向的城市间比较。横向而言，随着各市办公楼体量增长、租金上涨或租赁去化，其指数较基数录得增长，体现市场表现提升；同样，各市办公楼租金下跌和空置扩大，将导致其指数较基数下滑。', '200纵向而言，尽管我们已从全国余座城市中选取了政治经济综合影40京沪较其他城市断崖响力前城市，但其办公楼市场仍然差距巨大，式领先，银川、拉萨等西北省会城市则相对靠后。40中国城办公楼市场指数三季度发布：市场在漫长筑底中迎来积极政策信号', '指数解读：近八成主要城市指数下跌1. 10排名前的城市中，杭州是唯一指数正增长城市。得益于游戏、传媒公司的增量需求贡献，杭州市场指数已连续六个季度保持正增长。北京市场以降本驱动的搬迁需求为主导，空-5.8%置压力未见明显缓解，指数环比降幅录得，仍是所有一线城市中下降最快的城市。存量需求主导南京办公楼市场，有限，租赁活跃度未能达到预期水平，指数对市场成交促进0.4%环比下降。', '3.21-40排名的城市中，东北省会城市排名普遍提升。三季度，随着部分城市租金表现触底，该区间市场指数波动幅度有所收窄。部分城市止跌企稳，哈尔滨和长春老工业基地受政策支持，重塑竞争优势，吸引新质生产力企业入驻，指数排名有所上升。此外，海口在免税政策推动下持续吸引贸易类企业的新设需求，空置率有效降低，指数连续三个季度环比攀升。', '2. 11-20排名的城市中，多数城市市场基本面稳定。三季度，青±0.2%岛、福州、郑州、天津的指数环比变化在之间，市场基本供需失衡的压力持续上升，指数已连续八面稳定。南宁市场个季度环比下降。主要城市群市场总结：租金价格整体承压，去化表现受总部自用需求影响产生差异']
    for t in ts:
        print(encode(t))
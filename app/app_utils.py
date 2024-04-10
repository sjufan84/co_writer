import torch
import os
import traceback
import numpy as np
from fairseq import checkpoint_utils
import logging
from app.vc_infer_pipeline import VC
from app.config import Config
from app.audio_utils import load_audio
from app.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
hubert_model = None

def check_gpus():  # Determine if there are N cards that can be used to train and accelerate inference
    global gpus, default_batch_size, i18n
    i18n = i18n()
    ngpu = torch.cuda.device_count()
    gpu_infos = []
    mem = []
    if (not torch.cuda.is_available()) or ngpu == 0:
        if_gpu_ok = False
    else:
        if_gpu_ok = False
        for i in range(ngpu):
            gpu_name = torch.cuda.get_device_name(i)
            if (
                "10" in gpu_name or "16" in gpu_name or "20" in gpu_name
                or "30" in gpu_name
                or "40" in gpu_name
                or "A2" in gpu_name.upper()
                or "A3" in gpu_name.upper()
                or "A4" in gpu_name.upper()
                or "P4" in gpu_name.upper()
                or "A50" in gpu_name.upper()
                or "A60" in gpu_name.upper()
                or "70" in gpu_name
                or "80" in gpu_name
                or "90" in gpu_name
                or "M4" in gpu_name.upper()
                or "T4" in gpu_name.upper()
                or "TITAN" in gpu_name.upper()
            ):  # A10#A100#V100#A40#P40#M40#K80#A4500
                if_gpu_ok = True  # 至少有一张能用的N卡
                gpu_infos.append("%s\t%s" % (i, gpu_name))
                mem.append(
                    int(
                        torch.cuda.get_device_properties(i).total_memory
                        / 1024
                        / 1024
                        / 1024
                        + 0.4
                    )
                )
    if if_gpu_ok is True and len(gpu_infos) > 0:
        gpu_info = "\n".join(gpu_infos)
        default_batch_size = min(mem) // 2
    else:
        gpu_info = i18n("Unfortunately, you don't have a working graphics card to support your training")
        default_batch_size = 1
    gpus = "-".join([i[0] for i in gpu_infos])
    logger.info(gpu_info)
    logger.info("default_batch_size:%s" % default_batch_size)
    logger.info("gpus:%s" % gpus)

    return gpus, default_batch_size

def load_hubert():
    global hubert_model, config
    config = Config()
    logger.info("Loading Hubert model")
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["./app/hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    logger.info("Hubert model loaded")
    hubert_model = hubert_model.to(config.device)
    logger.info("Hubert model loaded to device")
    if config.is_half:
        hubert_model = hubert_model.half()
        logger.info("Hubert model loaded in half precision")
    else:
        hubert_model = hubert_model.float()
        logger.info("Hubert model loaded in single precision")
    hubert_model.eval()

async def vc_single(
    sid,
    input_audio_path,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    crepe_hop_length,
):
    global tgt_sr, net_g, vc, hubert_model, version
    if input_audio_path is None:
        logger.error("You need to upload an audio")
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    logger.info(f"Cloning vocals from {input_audio_path}.  f0_up_key: {f0_up_key}")
    try:
        audio = await load_audio(input_audio_path)
        logger.info(logging.INFO, "audio loaded")
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        times = [0, 0, 0]
        if hubert_model is None:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
        )  # Determine whether there is an N card that can
        # be used to prevent Xiao Bai from writing incorrectly
        # and automatically replace and speed up his reasoning
        # file_big_npy = (
        #     file_big_npy.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        # )
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            input_audio_path,
            times,
            f0_up_key,
            f0_method,
            file_index,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            crepe_hop_length,
            f0_file=f0_file,
        )
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            tgt_sr = resample_sr
        index_info = (
            "Using index:%s." % file_index
            if os.path.exists(file_index)
            else "Index not used."
        )
        return "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (
            index_info,
            times[0],
            times[1],
            times[2],
        ), (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        logger.info(info)
        return info, (None, None)

# 一A tab can have only one tone globally
def get_vc(sid, person):
    global n_spk, tgt_sr, net_g, vc, cpt, version
    config = Config()
    if sid == "" or sid == []:
        global hubert_model
        if hubert_model is not None:  # tabs take into account polling,
            # and you need to add a judgment to see if the SID can
            # only have one tone to switch from model to no model
            logger.info("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr  # ,cpt
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")
            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return {"visible": False, "__type__": "update"}

    weight_root = "./app/weights"
    person = "%s/%s" % (weight_root, sid)
    logger.info("loading %s" % person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    logger.info(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    return {"visible": False, "maximum": n_spk, "__type__": "update"}

def clean():
    return {"value": "", "__type__": "update"}

async def clone_vocals(input_audio: str):

    f0_file = None
    person = "Joel"
    spk_item = get_vc("joel.pth", person)
    vc_transform0 = 0
    file_index1 = './app/logs/joel/added_IVF479_Flat_nprobe_1.index'
    index_rate1 = 0.66
    f0method0 = "rmvpe"
    crepe_hop_length = 120
    filter_radius0 = 3
    resample_sr0 = tgt_sr
    rms_mix_rate0 = 0.21
    protect0 = 0.33
    logging.info(f"Cloning vocals from {input_audio}")
    logger.info(
        f"""Calling vc_single with {spk_item},
        {input_audio}, {vc_transform0}, {f0_file}, {f0method0},
        {file_index1}, {index_rate1}, {filter_radius0}, {resample_sr0},
        {rms_mix_rate0}, {protect0}, {crepe_hop_length}"""
    )
    audio_output, sr = await vc_single(
        spk_item, input_audio,
        vc_transform0,
        f0_file,
        f0method0,
        file_index1,
        index_rate1,
        filter_radius0,
        resample_sr0,
        rms_mix_rate0,
        protect0,
        crepe_hop_length
    )

    return audio_output, sr

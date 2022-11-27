# Run Oneformer Segmentation Inference

import cv2
import torch
import imutils
from tqdm import tqdm
import os
import subprocess
import argparse

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="oneformer")

# Import detectron2 utilities
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from demo.defaults import DefaultPredictor
from demo.visualizer import Visualizer, ColorMode

# import OneFormer Project
from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)

from zmq.constants import DeviceType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")

SWIN_CFG_DICT = {"cityscapes": "configs/cityscapes/oneformer_swin_large_IN21k_384_bs16_90k.yaml",
            "coco": "configs/coco/oneformer_swin_large_IN21k_384_bs16_100ep.yaml",
            "ade20k": "configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml",}

DINAT_CFG_DICT = {"cityscapes": "configs/cityscapes/oneformer_dinat_large_bs16_90k.yaml",
            "coco": "configs/coco/oneformer_dinat_large_bs16_100ep.yaml",
            "ade20k": "configs/ade20k/oneformer_dinat_large_IN21k_384_bs16_160k.yaml",}

def setup_cfg(dataset, model_path, use_swin):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    if use_swin:
      cfg_path = SWIN_CFG_DICT[dataset]
    else:
      cfg_path = DINAT_CFG_DICT[dataset]
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.WEIGHTS = model_path
    cfg.freeze()
    return cfg

def setup_modules(dataset, model_path, use_swin):
    cfg = setup_cfg(dataset, model_path, use_swin)
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
    )
    if 'cityscapes_fine_sem_seg_val' in cfg.DATASETS.TEST_PANOPTIC[0]:
        from cityscapesscripts.helpers.labels import labels
        stuff_colors = [k.color for k in labels if k.trainId != 255]
        metadata = metadata.set(stuff_colors=stuff_colors)
    
    return predictor, metadata

def panoptic_run(img, predictor, metadata, overlay=False):
    predictions = predictor(img, "panoptic")
    panoptic_seg, segments_info = predictions["panoptic_seg"]
    
    visualizer_map = Visualizer(
        img[:, :, ::-1], is_img=False, 
        metadata=metadata, instance_mode=ColorMode.IMAGE
        )
    out_map = visualizer_map.draw_panoptic_seg_predictions(
        panoptic_seg.to(cpu_device), segments_info, alpha=1, is_text=False
    )
    out = None
    if overlay:
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
        out = visualizer.draw_panoptic_seg_predictions(
            panoptic_seg.to(cpu_device), segments_info, alpha=0.5)
    
    return out, out_map
    

def instance_run(img, predictor, metadata, overlay=False):
    predictions = predictor(img, "instance")
    instances = predictions["instances"].to(cpu_device)
    
    visualizer_map = Visualizer(img[:, :, ::-1], is_img=False, metadata=metadata, instance_mode=ColorMode.IMAGE)
    out_map = visualizer_map.draw_instance_predictions(predictions=instances, alpha=1, is_text=False)

    out = None
    if overlay:
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
        out = visualizer.draw_instance_predictions(predictions=instances, alpha=0.5)

    return out, out_map


def semantic_run(img, predictor, metadata, overlay=False):
    predictions = predictor(img, "semantic")
    
    visualizer_map = Visualizer(img[:, :, ::-1], is_img=False, metadata=metadata, instance_mode=ColorMode.IMAGE)
    out_map = visualizer_map.draw_sem_seg(
        predictions["sem_seg"].argmax(dim=0).to(cpu_device), alpha=1, is_text=False)

    out = None
    if overlay:
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
        out = visualizer.draw_sem_seg(
            predictions["sem_seg"].argmax(dim=0).to(cpu_device), alpha=0.5)

    return out, out_map

TASK_INFER = {"panoptic": panoptic_run, 
              "instance": instance_run, 
              "semantic": semantic_run}

def main():
    parser = argparse.ArgumentParser(description="OneFormer Inference")
    parser.add_argument("--use_swin", action="store_false", help="Use Swin-L as backbone")
    parser.add_argument("--task", type=str, default="instance", 
        help="Task to run inference on - pick from panoptic, instance, semantic")
    parser.add_argument("--data_dir", type=str, 
            default="/home/jupyter/839_data/cam17_images", help="Data directory")
    parser.add_argument("--out_dir", type=str, 
            default="/home/jupyter/839_data/", help="Output directory")

    args = parser.parse_args()

    use_swin = args.use_swin
    task = args.task

    # Initialize Model
    if not use_swin:
        # download model checkpoint if it doesn't exist
        if not os.path.exists("150_16_dinat_l_oneformer_coco_100ep.pth"):
            subprocess.run('wget https://shi-labs.com/projects/oneformer/coco/150_16_dinat_l_oneformer_coco_100ep.pth', shell=True)
        predictor, metadata = setup_modules("coco", "150_16_dinat_l_oneformer_coco_100ep.pth", use_swin)
    else:
        if not os.path.exists("150_16_swin_l_oneformer_coco_100ep.pth"):
            subprocess.run('wget https://shi-labs.com/projects/oneformer/coco/150_16_swin_l_oneformer_coco_100ep.pth', shell=True)
        predictor, metadata = setup_modules("coco", "150_16_swin_l_oneformer_coco_100ep.pth", use_swin)

    data_dir = args.data_dir
    out_dir = os.path.join(args.out_dir,task)
    os.makedirs(out_dir, exist_ok=True)
    images = os.listdir(data_dir)

    for img_name in tqdm(images):
        try:
            img_path = os.path.join(data_dir, img_name)
            out_path = os.path.join(out_dir, img_name.replace('jpg','png'))
            if not os.path.isfile(out_path):
                img = cv2.imread(img_path)
                img = imutils.resize(img, width=512)
                _, out_map = TASK_INFER[task](img, predictor, metadata)
                out_map = out_map.get_image()
                cv2.imwrite(out_path, out_map)
        except Exception as e:
            print(e, img_path)
            continue

if __name__ == "__main__":
    main()

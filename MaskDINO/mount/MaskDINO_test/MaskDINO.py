# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py

#fitted for use by Dennis den Hollander (TU/e)

import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.append("/workspace/MaskDINO")
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from maskdino import add_maskdino_config
from predictor import VisualizationDemo


########## added
import json
import torch

from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer, ColorMode
##########


# constants
WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskdino demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=1.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            
            ############ added
            region_data = demo.extract_regions(predictions)

            if region_data is not None:
                num_regions = len(region_data["classes"])
                logger.info(
                    "{}: {} regio's in {:.2f}s".format(
                        path,
                        num_regions,
                        time.time() - start_time,
                    )
                )
            else:
                logger.info(
                    "{}: finished in {:.2f}s (no instances)".format(
                        path, 
                        time.time() - start_time,
                    )
                )
	    ############
	    
            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    base = os.path.splitext(os.path.basename(path))[0]
                    seg_out = os.path.join(args.output, base + "_seg.png")
                    inst_out = os.path.join(args.output, base + "_inst.png")
                    both_out = os.path.join(args.output, base + "_both.png")
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                #visualized_output.save(seg_out)

           ########## added
                inst_vis_img = None
                if region_data is not None:
                    masks = region_data["masks"]      
                    classes = region_data["classes"]  
                    scores = region_data["scores"]   

                    if len(classes) > 0:
                        seg_img = visualized_output.get_image()[:, :, ::-1]  # BGR
                        seg_vis = np.concatenate([img, seg_img], axis=1)
                        cv2.imwrite(seg_out, seg_vis) # save both img
                        
                        h, w = img.shape[:2]
                        instances = Instances((h, w))
                        instances.pred_masks = torch.from_numpy(masks)          
                                   		
                        image_rgb = img[:, :, ::-1] # img is BGR; Visualizer wants RGB
                        inst_visualizer = Visualizer(
                            image_rgb,
                            demo.metadata,
                            instance_mode=ColorMode.IMAGE,
                        )
                        inst_vis = inst_visualizer.draw_instance_predictions(instances.to(demo.cpu_device))
                        inst_img = inst_vis.get_image()[:, :, ::-1]  # back to BGR
                        
                        #instances.pred_classes = torch.from_numpy(classes)      
                        instances.scores = torch.from_numpy(scores)   
                        
                        inst_vis_det = inst_visualizer.draw_instance_predictions(instances.to(demo.cpu_device))
                        inst_img_det = inst_vis_det.get_image()[:, :, ::-1]  # back to BGR

                        inst_img_det_comb = np.concatenate([img, inst_img_det], axis=1)
                        cv2.imwrite(inst_out, inst_img_det_comb) # save detailed pseudo-instance img

                        both = np.concatenate([seg_img, inst_img], axis=1)
                        cv2.imwrite(both_out, both) # save both img

                if region_data is not None:
                    # masks as .npy
                    np.save(
                        os.path.join(args.output, base + "_masks.npy"),
                        region_data["masks"],
                    )

                    # classes + scores as json
                    meta = {
                        "classes": region_data["classes"].tolist(),
                        "scores": region_data["scores"].tolist(),
                    }
                    with open(os.path.join(args.output, base + "_instances.json"), "w") as f:
                        json.dump(meta, f)
            ##########
            
                        
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit


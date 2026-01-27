###########################################################
#
# demo.py
#
# Demo code for running full ROMAN SLAM pipeline including 
# mapping, loop closure, and pose-graph optimization
#
# Authors: Mason Peterson, Yulun Tian, Lucas Jia, Qingyuan Li
#
# Dec. 21, 2024
#
###########################################################

import numpy as np
import matplotlib.pyplot as plt

import argparse
from typing import List
import os
import yaml

from roman.params.submap_align_params import SubmapAlignInputOutput, SubmapAlignParams
from roman.align.submap_align import submap_align
from roman.offline_rpgo.extract_odom_g2o import roman_map_pkl_to_g2o
from roman.offline_rpgo.g2o_file_fusion import create_config, g2o_file_fusion
from roman.offline_rpgo.combine_loop_closures import combine_loop_closures
from roman.offline_rpgo.plot_g2o import plot_g2o, DEFAULT_TRAJECTORY_COLORS, G2OPlotParams
from roman.offline_rpgo.g2o_and_time_to_pose_data import g2o_and_time_to_pose_data
from roman.offline_rpgo.evaluate import evaluate
from roman.offline_rpgo.edit_g2o_edge_information import edit_g2o_edge_information
from roman.params.offline_rpgo_params import OfflineRPGOParams
from roman.params.data_params import DataParams

import mapping_count as mapping

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', default=None, type=str, help='Path to params directory. ' +
                        'Params can include the following files: data.yaml, fastsam.yaml, ' +
                        'mapper.yaml, submap_align.yaml, and offline_rpgo.yaml. Only data.yaml ' +
                        'is required to be provided. Parameter defaults and definitions can be ' +
                        'found in the roman.params module. Additional information can be found ' +
                        'here: https://github.com/mit-acl/ROMAN/blob/main/demo/README.md', required=True)
    parser.add_argument('-o', '--output-dir', type=str, help='Path to output directory', required=True, default=None)
    
    parser.add_argument('-r', '--runs', type=str, nargs='+', required=False, default=None,
                        help='Run names. Overrides runs field in data.yaml')
    parser.add_argument('-m', '--viz-map', action='store_true', help='Visualize map')
    parser.add_argument('-v', '--viz-observations', action='store_true', help='Visualize observations')
    parser.add_argument('-3', '--viz-3d', action='store_true', help='Visualize 3D')
    parser.add_argument('--vid-rate', type=float, help='Video playback rate', default=1.0)
    parser.add_argument('-d', '--save-img-data', action='store_true', help='Save video frames as ImgData class')
    parser.add_argument('-n', '--num-req-assoc', type=int, help='Number of required associations', default=4)
    # parser.add_argument('--set-env-vars', type=str)
    parser.add_argument('--max-time', type=float, default=None, help='If the input data is too large, this allows a maximum time' +
                        'to be set, such that if the mapping will be chunked into max_time increments and fused together')

    parser.add_argument('--skip-map', action='store_true', help='Skip mapping')
    parser.add_argument('--skip-align', action='store_true', help='Skip alignment')
    parser.add_argument('--skip-rpgo', action='store_true', help='Skip robust pose graph optimization')
    parser.add_argument('--skip-indices', type=int, nargs='+', help='Skip specific runs in mapping and alignment')
    parser.add_argument('--skip-self-lc', action='store_true', help='Skip self loop closures in submap_align')
    parser.add_argument('--skip-distance', type=float, help='Skip trying to align submaps that ' +
                        'are farther away than this threshold (meters). By default, all alignment ' +
                        'is attempted for all submaps.', default=np.inf)

    args = parser.parse_args()

    # setup params
    params_dir = args.params
    submap_align_params_path = os.path.join(args.params, f"submap_align.yaml")
    submap_align_params = SubmapAlignParams.from_yaml(submap_align_params_path) \
        if os.path.exists(submap_align_params_path) else SubmapAlignParams()
    offline_rpgo_params_path = os.path.join(args.params, f"offline_rpgo.yaml")
    offline_rpgo_params = OfflineRPGOParams.from_yaml(offline_rpgo_params_path) \
        if os.path.exists(os.path.join(args.params, "offline_rpgo.yaml")) else OfflineRPGOParams()
    data_params = DataParams.from_yaml(os.path.join(args.params, "data.yaml"))
    if args.runs is not None:
        data_params.runs = args.runs
            
    # ground truth pose files
    if os.path.exists(os.path.join(params_dir, "gt_pose.yaml")):
        has_gt = True
        gt_files = [os.path.join(params_dir, "gt_pose.yaml") for _ in range(len(data_params.runs))]
    else:
        has_gt = False
        gt_files = [None for _ in range(len(data_params.runs))]

    # create output directories
    os.makedirs(os.path.join(args.output_dir, "map"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "align"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "offline_rpgo"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "offline_rpgo/sparse"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "offline_rpgo/dense"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "params"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "classes"), exist_ok=True)

    # copy params to output directory
    for param_file in os.listdir(params_dir):
        if param_file.endswith(".yaml"):
            src = os.path.join(params_dir, param_file)
            dst = os.path.join(args.output_dir, "params", param_file)
            os.system(f"cp {src} {dst}")
    
    if not args.skip_map:
        
        for i, run in enumerate(data_params.runs):
            if args.skip_indices and i in args.skip_indices:
                continue
                
            # mkdir $output_dir/map
            args.output = os.path.join(args.output_dir, "map", f"{run}")

            # shell: export RUN=run
            if data_params.run_env is not None:
                os.environ[data_params.run_env] = run
            
            print(f"\n\n----------\nMapping: {run}\n----------\n\n")
            mapping_viz_params = \
                mapping.VisualizationParams(
                    viz_map=args.viz_map,
                    viz_observations=args.viz_observations,
                    viz_3d=args.viz_3d,
                    vid_rate=args.vid_rate,
                    save_img_data=args.save_img_data
                )
            mapping.mapping(
                params_path=args.params,
                output_path=args.output,
                run_name=run,
                max_time=args.max_time,
                viz_params=mapping_viz_params,
                verbose=True
            )
            

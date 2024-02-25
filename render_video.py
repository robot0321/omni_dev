#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import matplotlib
import imageio
import time
# def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
#     render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
#     gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

#     makedirs(render_path, exist_ok=True)
#     makedirs(gts_path, exist_ok=True)

#     for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
#         rendering = render(view, gaussians, pipeline, background)["render"]
#         gt = view.original_image[0:3, :, :]
#         torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
#         torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

cmapper = matplotlib.cm.get_cmap('jet_r')

def depth_colorize_with_mask(depthlist, background=(0,0,0), dmindmax=None):
    """ depth: (H,W) - [0 ~ 1] / mask: (H,W) - [0 or 1]  -> colorized depth (H,W,3) [0 ~ 1] """
    batch, vx, vy = np.where(depthlist!=0)
    if dmindmax is None: # 각 이미지의 min/max로 각자 normalize
        valid_depth = depthlist[batch, vx, vy]
        dmin, dmax = valid_depth.min(), valid_depth.max()
    else:
        dmin, dmax = dmindmax # 전체 scene에서 normalize
    
    norm_dth = np.ones_like(depthlist)*dmax # [B, H, W]
    norm_dth[batch, vx, vy] = (depthlist[batch, vx, vy]-dmin)/(dmax-dmin)
    
    final_depth = np.ones(depthlist.shape + (3,)) * np.array(background).reshape(1,1,1,3) # [B, H, W, 3]
    final_depth[batch, vx, vy] = cmapper(norm_dth)[batch,vx,vy,:3]

    return final_depth


def render_set(model_path, iteration, views, gaussians, pipeline, background, preset):
    cvt = time.time()
    preset = preset+'_skipalpha0.5_divacc'
    render_path = os.path.join(model_path, preset, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, preset, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, preset, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    framelist = []
    depthlist = []
    dmin, dmax = 1e3, 0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        if "depth" in results:
            # depthlist.append((results["depth"]*(results["acc"]>0.9)).detach().cpu().numpy())
            dmask = (results["depth"]>0)
            depth = (results["depth"]*dmask).detach().cpu().numpy()
            depthlist.append(depth)
            if results["depth"][dmask].min().item() < dmin:
                dmin = results["depth"][dmask].min().item()
            if results["depth"][dmask].max().item() > dmax:
                dmax = results["depth"][dmask].max().item()
            
            # import pdb; pdb.set_trace()
            # view.original_depth
            framelist.append(np.round(rendering.permute(1,2,0).detach().cpu().numpy()*255.).astype(np.uint8))
    
    imageio.mimwrite(os.path.join(model_path, preset, "ours_{}".format(iteration), "video60_RGB.mp4"), framelist, fps=60, quality=8)
    pvt = cvt; cvt = time.time(); print(f"Depth colorize time: {cvt-pvt:.4f}")
    # import pdb; pdb.set_trace()
    
    depthlist = np.concatenate(depthlist, axis=0)
    # gtdmindmax = (depthlist.min(), depthlist.max())
    gtdmindmax = (dmin, dmax)
    print("depth range: ", gtdmindmax)
    if "depth" in results:
        if True:
            ### colorize by full scene
            colorized_depth = depth_colorize_with_mask(depthlist, background.detach().cpu().numpy(), dmindmax=gtdmindmax)
            for idx, view in enumerate(tqdm(views, desc="Rendering Depth progress")):
                imageio.imwrite(os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"), np.round(colorized_depth[idx]*255.).astype(np.uint8))
                # np.save(os.path.join(depth_path, view.image_name + ".npy"), depthlist[idx])
                framelist[idx] = np.concatenate((framelist[idx], np.round(colorized_depth[idx]*255.).astype(np.uint8)), axis=1)
        else:
            ### colorize by frame
            for idx, view in enumerate(tqdm(views, desc="Rendering Depth progress")):
                colorized_depth = depth_colorize_with_mask(depthlist[idx,None], background.detach().cpu().numpy())
                imageio.imwrite(os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"), np.round(colorized_depth[0]*255.).astype(np.uint8))
                framelist[idx] = np.concatenate((framelist[idx], np.round(colorized_depth[0]*255.).astype(np.uint8)), axis=1)
            
    imageio.mimwrite(os.path.join(model_path, preset, "ours_{}".format(iteration), "video_RGBD.mp4"), framelist, fps=60, quality=8)
    pvt = cvt; cvt = time.time(); print(f"Depth render time: {cvt-pvt:.4f}")

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, preset : str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, preset=preset)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        render_set(dataset.model_path, scene.loaded_iter, scene.getPresetCameras(), gaussians, pipeline, background, preset)
        

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    # parser.add_argument("--skip_train", action="store_true")
    # parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--preset', default='lookaround', choices=['1440', '360', 'lookaround', 'back', '360_fov1.2', 'llff_d2', 'llff_d4', 'llff_d6', 'llff_d8', 'headbanging_r2', 'headbanging_r3', 'headbanging_circle'], type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.preset)
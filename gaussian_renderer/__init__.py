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

import numpy as np
import matplotlib
import torch
import math
# from depth_diff_gaussian_rasterization_min import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_omni_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

cmapper = matplotlib.cm.get_cmap('jet_r')
def depth_colorize_with_mask(depthlist, background=(0,0,0), dmindmax=None):
    """ depth: (H,W) - [0 ~ 1] / mask: (H,W) - [0 or 1]  -> colorized depth (H,W,3) [0 ~ 1] """
    single_batch = True if len(depthlist.shape)==2 else False
        
    if single_batch:
        depthlist = depthlist[None]
    
    batch, vx, vy = np.where(depthlist!=0)
    if dmindmax is None:
        valid_depth = depthlist[batch, vx, vy]
        dmin, dmax = valid_depth.min(), valid_depth.max()
    else:
        dmin, dmax = dmindmax
    
    norm_dth = np.ones_like(depthlist)*dmax # [B, H, W]
    norm_dth[batch, vx, vy] = (depthlist[batch, vx, vy]-dmin)/(dmax-dmin)
    
    final_depth = np.ones(depthlist.shape + (3,)) * np.array(background).reshape(1,1,1,3) # [B, H, W, 3]
    final_depth[batch, vx, vy] = cmapper(norm_dth)[batch,vx,vy,:3]

    return final_depth[0] if single_batch else final_depth

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    import pdb; pdb.set_trace()
    viewmatrix = torch.eye(4, device=viewpoint_camera.world_view_transform.device) #### For debugging / MUST remove after
    raster_settings = GaussianRasterizationSettings(
        image_height=512,#int(viewpoint_camera.image_height),
        image_width=1024,#int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewmatrix, #viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=torch.zeros_like(viewpoint_camera.camera_center), #viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    opacity[:]=1. #### For debugging / MUST remove after
    scales[:]=10. #### For debugging / MUST remove after
    
    rendered_image, depth, acc, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
        
    from PIL import Image; import numpy as np
    Image.fromarray((rendered_image.permute(1,2,0).detach().cpu().numpy()*255.).astype(np.uint8)).save("hello.png")
    Image.fromarray((depth_colorize_with_mask(depth.detach().cpu().numpy())*255.).astype(np.uint8)).save("hello_depth.png")
    Image.fromarray((acc.detach().cpu().numpy()*255.).astype(np.uint8)).save("hello_acc.png")
    # rendered_image, depth, acc, radii = rasterizer(means3D=means3D,means2D = means2D,shs = shs,colors_precomp = colors_precomp,opacities = opacity,scales = scales,rotations = rotations,cov3D_precomp = cov3D_precomp); Image.fromarray((rendered_image.permute(1,2,0).detach().cpu().numpy()*255.).astype(np.uint8)).save("hello.png"); Image.fromarray((depth.detach().cpu().numpy()/depth.max().item()*255.).astype(np.uint8)).save("hello_depth.png"); Image.fromarray((acc.detach().cpu().numpy()*255.).astype(np.uint8)).save("hello_acc.png")
    hello2=rendered_image.permute(1,2,0).detach().cpu().clone()
    tmp=hello2[:,:512].clone(); hello2[:,:512]=hello2[:,512:]; hello2[:,512:]=tmp
    Image.fromarray((hello2.numpy()*255.).astype(np.uint8)).save("hello2.png")
    import pdb; pdb.set_trace() #### For debugging / MUST remove after


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

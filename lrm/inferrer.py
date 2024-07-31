
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.import torch
import torch
import math
import os
import imageio
import mcubes
import trimesh
import numpy as np
import argparse
from torchvision.utils import save_image
from PIL import Image
import glob
from .models.generator import LRMGenerator
from .cam_utils import build_camera_principle, build_camera_standard, center_looking_at_camera_pose
from functools import partial
from rembg import remove, new_session
from kiui.op import recenter
import kiui

class LRMInferrer:
    def __init__(self, model_name: str, resume:str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _model_kwargs = {'camera_embed_dim': 1024, 'rendering_samples_per_ray': 128, 'transformer_dim': 1024, 'transformer_layers': 16, 'transformer_heads': 16, 'triplane_low_res': 32, 'triplane_high_res': 64, 'triplane_dim': 80, 'encoder_freeze': False}
        
        self.model = self._build_model(_model_kwargs).eval().to(self.device)
        checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        self.model.load_state_dict(state_dict)
        del checkpoint, state_dict
        torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def _build_model(self, model_kwargs):
        model = LRMGenerator(**model_kwargs).to(self.device)
        # model.load_state_dict(model_weights)
        print(f"======== Loaded model from checkpoint ========")
        return model

    @staticmethod
    def get_surrounding_views(M, radius, elevation):
    #   convert spherical coordinates (radius, azimuth, elevation) to Cartesian coordinates (x, y, z).
        camera_positions = []
        rand_theta= np.random.uniform(0, np.pi/180)
        elevation = math.radians(elevation)
        for i in range(M):
            theta = 2 * math.pi * i / M  + rand_theta
            x = radius * math.cos(theta) * math.cos(elevation)
            y = radius * math.sin(theta) * math.cos(elevation)
            z =  radius * math.sin(elevation)
            camera_positions.append([x, y, z])
        camera_positions = torch.tensor(camera_positions, dtype=torch.float32)
        extrinsics = center_looking_at_camera_pose(camera_positions)

        return extrinsics



    @staticmethod
    def _default_intrinsics():
        # return: (3, 2)
        fx = fy = 384
        cx = cy = 256
        w = h = 512
        intrinsics = torch.tensor([
            [fx, fy],
            [cx, cy],
            [w, h],
        ], dtype=torch.float32)
        return intrinsics

    def _default_source_camera(self, batch_size: int = 1):
        # return: (N, D_cam_raw)
        dist_to_center = 1.5
        canonical_camera_extrinsics = torch.tensor([[
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ]], dtype=torch.float32)
        canonical_camera_intrinsics = self._default_intrinsics().unsqueeze(0)

        source_camera = build_camera_principle(canonical_camera_extrinsics, canonical_camera_intrinsics)
        return source_camera.repeat(batch_size, 1)

    def _default_render_cameras(self, batch_size: int = 1):
        render_camera_extrinsics = self.get_surrounding_views(160, 1.5, 0)
        render_camera_intrinsics = self._default_intrinsics().unsqueeze(0).repeat(render_camera_extrinsics.shape[0], 1, 1)
        render_cameras = build_camera_standard(render_camera_extrinsics, render_camera_intrinsics)

        return render_cameras.unsqueeze(0).repeat(batch_size, 1, 1)

    @staticmethod
    def images_to_video(images, output_path, fps, verbose=False):
        # images: (T, C, H, W)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        frames = []
        for i in range(images.shape[0]):
            frame = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            assert frame.shape[0] == images.shape[2] and frame.shape[1] == images.shape[3], \
                f"Frame shape mismatch: {frame.shape} vs {images.shape}"
            assert frame.min() >= 0 and frame.max() <= 255, \
                f"Frame value out of range: {frame.min()} ~ {frame.max()}"
            frames.append(frame)
        imageio.mimwrite(output_path, np.stack(frames), fps=fps)
        if verbose:
            print(f"Saved video to {output_path}")

    def infer_single(self, image: torch.Tensor, render_size: int, mesh_size: int, export_video: bool, export_mesh: bool):
        # image: [1, C_img, H_img, W_img]
        mesh_thres = 1.0
        chunk_size = 1 # You can increase this number if GPU memory allows.
        batch_size = 1

        source_camera = self._default_source_camera(batch_size).to(self.device)
        render_cameras = self._default_render_cameras(batch_size).to(self.device)

        with torch.no_grad():
            planes = self.model.forward(image, source_camera)   
            results = {}

            if export_video:
                frames = []
                for i in range(0, render_cameras.shape[1], chunk_size):
                    frames.append(
                        self.model.synthesizer(
                            planes,
                            render_cameras[:, i:i+chunk_size],
                            render_size,
                            render_size,
                            0,
                            0
                        )
                    )
                frames = {
                    k: torch.cat([r[k] for r in frames], dim=1)
                    for k in frames[0].keys()
                }
                results.update({
                    'frames': frames,
                })
            if export_mesh:

                grid_out = self.model.synthesizer.forward_grid(
                    planes=planes,
                    grid_size=mesh_size,
                )
                vtx, faces = mcubes.marching_cubes(grid_out['sigma'].float().squeeze(0).squeeze(-1).cpu().numpy(), mesh_thres)
                vtx = vtx / (mesh_size - 1) * 2 - 1
                vtx_tensor = torch.tensor(vtx, dtype=torch.float32, device=self.device).unsqueeze(0)
                vtx_colors = self.model.synthesizer.forward_points(planes, vtx_tensor)['rgb'].float().squeeze(0).cpu().numpy()  # (0, 1)
                vtx_colors = (vtx_colors * 255).astype(np.uint8)
                
                mesh = trimesh.Trimesh(vertices=vtx, faces=faces, vertex_colors=vtx_colors)
                results.update({
                    'mesh': mesh,
                })

            return results

    def infer(self, source_image: str, dump_path: str, source_size: int, render_size: int, mesh_size: int, export_video: bool, export_mesh: bool):
        
        session = new_session("isnet-general-use")
        rembg_remove = partial(remove, session=session)
        source_image_size = source_size 
        image_name = os.path.basename(source_image)
        uid = image_name.split('.')[0]

        image = kiui.read_image(source_image, mode='uint8')
        image = rembg_remove(image) 
        mask = rembg_remove(image, only_mask=True) 
        image = recenter(image, mask, border_ratio=0.20)
        os.makedirs(dump_path, exist_ok=True)

        image = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0) / 255.0
        if image.shape[1] == 4:
            image = image[:, :3, ...] * image[:, 3:, ...] + (1 - image[:, 3:, ...])
        image = torch.nn.functional.interpolate(image, size=(source_image_size, source_image_size), mode='bicubic', align_corners=True)
        image = torch.clamp(image, 0, 1)
        save_image(image, os.path.join(dump_path, f'{uid}.png'))
        
        results = self.infer_single(
            image.cuda(),
            render_size=render_size,
            mesh_size=mesh_size,
            export_video=export_video,
            export_mesh=export_mesh,
        )


        # dump video
        if 'frames' in results:
            renderings = results['frames']
            for k, v in renderings.items():
                if k == 'images_rgb':
                    self.images_to_video(
                        v[0],
                        os.path.join(dump_path, f'{uid}.mp4'),
                        fps=40,
                    )
        

        # dump mesh
        if 'mesh' in results:
            mesh = results['mesh']
            mesh.export(os.path.join(dump_path, f'{uid}.obj'), 'obj')


if __name__ == '__main__':

    """
    Example usage:
    python -m lrm.inferrer --export_video --resume ./checkpoints/vfusion3dckpt
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='lrm-base-obj-v1')
    parser.add_argument('--source_path', type=str, default='./assets/40_prompt_images')
    parser.add_argument('--dump_path', type=str, default='./results/40_prompt_images')
    parser.add_argument('--source_size', type=int, default=512)
    parser.add_argument('--render_size', type=int, default=384)
    parser.add_argument('--mesh_size', type=int, default=512)
    parser.add_argument('--export_video', action='store_true')
    parser.add_argument('--export_mesh', action='store_true')
    parser.add_argument('--resume', type=str, help='Path to a checkpoint to resume training from')
    args = parser.parse_args()
    source_path = args.source_path
    images = glob.glob(os.path.join(source_path, "*.jpg"))
    images.extend(glob.glob(os.path.join(source_path, "*.png")))
    images.extend(glob.glob(os.path.join(source_path, "*.jpeg")))

    with LRMInferrer(model_name=args.model_name, resume=args.resume) as inferrer:
        with torch.autocast(device_type="cuda", cache_enabled=False, dtype=torch.float32):
            for img in images:
                inferrer.infer(
                    source_image=img,
                    dump_path=args.dump_path,
                    source_size=args.source_size,
                    render_size=args.render_size,
                    mesh_size=args.mesh_size,
                    export_video=args.export_video,
                    export_mesh=args.export_mesh,
                )

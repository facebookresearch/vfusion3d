# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import gradio as gr
import os
import numpy as np
import trimesh
import mcubes
import imageio
from torchvision.utils import save_image
from PIL import Image
from transformers import AutoModel, AutoConfig
from rembg import remove, new_session
from functools import partial
from kiui.op import recenter
import kiui
from gradio_litmodel3d import LitModel3D

# we load the pre-trained model from HF
class LRMGeneratorWrapper:
    def __init__(self):
        self.config = AutoConfig.from_pretrained("jadechoghari/vfusion3d", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("jadechoghari/vfusion3d", trust_remote_code=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def forward(self, image, camera):
        return self.model(image, camera)

model_wrapper = LRMGeneratorWrapper()

# we preprocess the input image
def preprocess_image(image, source_size):
    session = new_session("isnet-general-use")
    rembg_remove = partial(remove, session=session)
    image = np.array(image)
    image = rembg_remove(image)
    mask = rembg_remove(image, only_mask=True)
    image = recenter(image, mask, border_ratio=0.20)
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0) / 255.0
    if image.shape[1] == 4:
        image = image[:, :3, ...] * image[:, 3:, ...] + (1 - image[:, 3:, ...])
    image = torch.nn.functional.interpolate(image, size=(source_size, source_size), mode='bicubic', align_corners=True)
    image = torch.clamp(image, 0, 1)
    return image

# Copied from https://github.com/facebookresearch/vfusion3d/blob/main/lrm/cam_utils.py and
# https://github.com/facebookresearch/vfusion3d/blob/main/lrm/inferrer.py
def get_normalized_camera_intrinsics(intrinsics: torch.Tensor):
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 0, 1]
    cx, cy = intrinsics[:, 1, 0], intrinsics[:, 1, 1]
    width, height = intrinsics[:, 2, 0], intrinsics[:, 2, 1]
    fx, fy = fx / width, fy / height
    cx, cy = cx / width, cy / height
    return fx, fy, cx, cy

def build_camera_principle(RT: torch.Tensor, intrinsics: torch.Tensor):
    fx, fy, cx, cy = get_normalized_camera_intrinsics(intrinsics)
    return torch.cat([
        RT.reshape(-1, 12),
        fx.unsqueeze(-1), fy.unsqueeze(-1), cx.unsqueeze(-1), cy.unsqueeze(-1),
    ], dim=-1)

def _default_intrinsics():
    fx = fy = 384
    cx = cy = 256
    w = h = 512
    intrinsics = torch.tensor([
        [fx, fy],
        [cx, cy],
        [w, h],
    ], dtype=torch.float32)
    return intrinsics

def _default_source_camera(batch_size: int = 1):
    canonical_camera_extrinsics = torch.tensor([[
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ]], dtype=torch.float32)
    canonical_camera_intrinsics = _default_intrinsics().unsqueeze(0)
    source_camera = build_camera_principle(canonical_camera_extrinsics, canonical_camera_intrinsics)
    return source_camera.repeat(batch_size, 1)

def _center_looking_at_camera_pose(camera_position: torch.Tensor, look_at: torch.Tensor = None, up_world: torch.Tensor = None):
    """
    camera_position: (M, 3)
    look_at: (3)
    up_world: (3)
    return: (M, 3, 4)
    """
    # by default, looking at the origin and world up is pos-z
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32)
    look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
    up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    z_axis = camera_position - look_at
    z_axis = z_axis / z_axis.norm(dim=-1, keepdim=True)
    x_axis = torch.cross(up_world, z_axis)
    x_axis = x_axis / x_axis.norm(dim=-1, keepdim=True)
    y_axis = torch.cross(z_axis, x_axis)
    y_axis = y_axis / y_axis.norm(dim=-1, keepdim=True)
    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    return extrinsics

def compose_extrinsic_RT(RT: torch.Tensor):
    """
    Compose the standard form extrinsic matrix from RT.
    Batched I/O.
    """
    return torch.cat([
        RT,
        torch.tensor([[[0, 0, 0, 1]]], dtype=torch.float32).repeat(RT.shape[0], 1, 1).to(RT.device)
        ], dim=1)

def _build_camera_standard(RT: torch.Tensor, intrinsics: torch.Tensor):
    """
    RT: (N, 3, 4)
    intrinsics: (N, 3, 2), [[fx, fy], [cx, cy], [width, height]]
    """
    E = compose_extrinsic_RT(RT)
    fx, fy, cx, cy = get_normalized_camera_intrinsics(intrinsics)
    I = torch.stack([
        torch.stack([fx, torch.zeros_like(fx), cx], dim=-1),
        torch.stack([torch.zeros_like(fy), fy, cy], dim=-1),
        torch.tensor([[0, 0, 1]], dtype=torch.float32, device=RT.device).repeat(RT.shape[0], 1),
    ], dim=1)
    return torch.cat([
        E.reshape(-1, 16),
        I.reshape(-1, 9),
    ], dim=-1)

def _default_render_cameras(batch_size: int = 1):
    M = 80
    radius = 1.5
    elevation = 0
    camera_positions = []
    rand_theta = np.random.uniform(0, np.pi/180)
    elevation = np.radians(elevation)
    for i in range(M):
        theta = 2 * np.pi * i / M + rand_theta
        x = radius * np.cos(theta) * np.cos(elevation)
        y = radius * np.sin(theta) * np.cos(elevation)
        z = radius * np.sin(elevation)
        camera_positions.append([x, y, z])
    camera_positions = torch.tensor(camera_positions, dtype=torch.float32)
    extrinsics = _center_looking_at_camera_pose(camera_positions)

    render_camera_intrinsics = _default_intrinsics().unsqueeze(0).repeat(extrinsics.shape[0], 1, 1)
    render_cameras = _build_camera_standard(extrinsics, render_camera_intrinsics)
    return render_cameras.unsqueeze(0).repeat(batch_size, 1, 1)

def generate_mesh(image, source_size=512, render_size=384, mesh_size=512, export_mesh=False, export_video=True, fps=30):
    image = preprocess_image(image, source_size).to(model_wrapper.device)
    source_camera = _default_source_camera(batch_size=1).to(model_wrapper.device)

    with torch.no_grad():
        planes = model_wrapper.forward(image, source_camera)

        if export_mesh:
            grid_out = model_wrapper.model.synthesizer.forward_grid(planes=planes, grid_size=mesh_size)
            vtx, faces = mcubes.marching_cubes(grid_out['sigma'].float().squeeze(0).squeeze(-1).cpu().numpy(), 1.0)
            vtx = vtx / (mesh_size - 1) * 2 - 1
            vtx_tensor = torch.tensor(vtx, dtype=torch.float32, device=model_wrapper.device).unsqueeze(0)
            vtx_colors = model_wrapper.model.synthesizer.forward_points(planes, vtx_tensor)['rgb'].float().squeeze(0).cpu().numpy()
            vtx_colors = (vtx_colors * 255).astype(np.uint8)
            mesh = trimesh.Trimesh(vertices=vtx, faces=faces, vertex_colors=vtx_colors)

            mesh_path = "awesome_mesh.obj"
            mesh.export(mesh_path, 'obj')

            return mesh_path, mesh_path

        if export_video:
            render_cameras = _default_render_cameras(batch_size=1).to(model_wrapper.device)
            frames = []
            chunk_size = 1
            for i in range(0, render_cameras.shape[1], chunk_size):
                frame_chunk = model_wrapper.model.synthesizer(
                    planes,
                    render_cameras[:, i:i + chunk_size],
                    render_size,
                    render_size,
                    0,
                    0
                )
                frames.append(frame_chunk['images_rgb'])

            frames = torch.cat(frames, dim=1)
            frames = frames.squeeze(0)
            frames = (frames.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

            video_path = "awesome_video.mp4"
            imageio.mimwrite(video_path, frames, fps=fps)

            return None, video_path

    return None, None

def step_1_generate_obj(image):
    mesh_path, _ = generate_mesh(image, export_mesh=True)
    return mesh_path, mesh_path

def step_2_generate_video(image):
    _, video_path = generate_mesh(image, export_video=True)
    return video_path

def step_3_display_3d_model(mesh_file):
    return mesh_file

# set up the example files from assets folder, we limit to 10
example_folder = "assets"
examples = [os.path.join(example_folder, f) for f in os.listdir(example_folder) if f.endswith(('.png', '.jpg', '.jpeg'))][:10]

with gr.Blocks() as demo:
    with gr.Row():
        
        with gr.Column():
            gr.Markdown("""
            # Welcome to [VFusion3D](https://junlinhan.github.io/projects/vfusion3d.html) Demo

            This demo allows you to upload an image and generate a 3D model or rendered videos from it. 

            ## How to Use:
            1. Click on "Click to Upload" to upload an image, or choose one example image.
            
            2: Choose between "Generate and Download Mesh" or "Generate and Download Video", then click it.
            
            3. Wait for the model to process; meshes should take approximately 10 seconds, and videos will take approximately 30 seconds.
            
            4. Download the generated mesh or video.

            This demo does not aim to provide optimal results but rather to provide a quick look. See our [GitHub](https://github.com/facebookresearch/vfusion3d) for more. 

            """)
            img_input = gr.Image(type="pil", label="Input Image")
            examples_component = gr.Examples(examples=examples, inputs=img_input, outputs=None, examples_per_page=3)
            generate_mesh_button = gr.Button("Generate and Download Mesh")
            generate_video_button = gr.Button("Generate and Download Video")
            obj_file_output = gr.File(label="Download .obj File")
            video_file_output = gr.File(label="Download Video")

        with gr.Column():
            model_output = LitModel3D(
                clear_color=[0.1, 0.1, 0.1, 0],  # can adjust background color for better contrast
                label="3D Model Visualization",
                scale=1.0,
                tonemapping="aces",  # can use aces tonemapping for more realistic lighting
                exposure=1.0,        # can adjust exposure to control brightness
                contrast=1.1,        # can slightly increase contrast for better depth
                camera_position=(0, 0, 2),  # will set initial camera position to center the model
                zoom_speed=0.5,      # will adjust zoom speed for better control
                pan_speed=0.5,       # will adjust pan speed for better control
                interactive=True     # this allow users to interact with the model
            )
            
        
    # clear outputs
    def clear_model_viewer():
        """Reset the Model3D component before loading a new model."""
        return gr.update(value=None)
    
    def generate_and_visualize(image):
        mesh_path = step_1_generate_obj(image)
        return mesh_path, mesh_path

    # first we clear the existing 3D model
    img_input.change(clear_model_viewer, inputs=None, outputs=model_output)

    # then, generate the mesh and video
    generate_mesh_button.click(step_1_generate_obj, inputs=img_input, outputs=[obj_file_output, model_output])
    generate_video_button.click(step_2_generate_video, inputs=img_input, outputs=video_file_output)

demo.launch()

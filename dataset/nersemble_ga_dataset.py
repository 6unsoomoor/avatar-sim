import os
import torch
import json
import math
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from roma import rotvec_to_rotmat
from submodules.flame import FLAME
from concurrent.futures import ThreadPoolExecutor, as_completed
from camera.camera import fov2focal, focal2fov


def proj_mat(fx, fy, cx, cy, w, h, n=0.01, f=100.0):
    P = torch.zeros([4, 4], dtype=torch.float32)
    P[0, 0] = 2 * fx / w
    P[1, 1] = 2 * fy / h

    # consider camera center cx, cy
    P[0, 2] = -1 + 2 * (cx / w)
    P[1, 2] = -1 + 2 * (cy / h)

    # z = zfar, depth = 1.0; z = znear, depth = 0.0
    P[3, 2] = 1.0
    P[2, 2] = f / (f - n)
    P[2, 3] = -(f * n) / (f - n)
    return P


class NeRSembleGADataset(Dataset):
    def __init__(self, 
        flame_model: FLAME,
        json_path: str,
        train: bool = True
    ):
        self.train = train
        self.preload = False

        with open(json_path, 'r') as f:
            json_data = json.load(f)

        self.camera_indices = sorted(list(set([data_item['camera_index'] for data_item in json_data['frames']])))
        self.timestep_indices = sorted(list(set([data_item['timestep_index'] for data_item in json_data['frames']])))

        self.data_dir = os.path.dirname(json_path)

        print(f"Found {len(self.camera_indices)} unique camera indices.")
        print(f'Found {len(self.timestep_indices)} unique frame indices.')

        # camera intrinsics are the same
        self.image_width = json_data['frames'][0]['w']
        self.image_height = json_data['frames'][0]['h']
        self.fov_x = json_data['frames'][0]['camera_angle_x']
        focal = fov2focal(self.fov_x, self.image_width)
        self.fov_y = focal2fov(focal, self.image_height)
        self.camera_intri = np.array([
            [focal, 0.0, self.image_width * 0.5],
            [0.0, focal, self.image_height * 0.5],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        projection_mat = proj_mat(focal, focal, self.image_width * 0.5, self.image_height * 0.5, self.image_width, self.image_height)
        self.image_paths = []
        self.mask_paths = []
        self.cam_view_mats = []
        self.cam_proj_mats = []
        self.cam_positions = []
        self.frame_id = []
        self.camera_names = []

        for data_item in json_data['frames']:
            self.frame_id.append(self.timestep_indices.index(data_item['timestep_index']))
            self.camera_names.append(data_item['camera_id'])
            self.image_paths.append(os.path.join(self.data_dir, data_item['file_path']))
            self.mask_paths.append(os.path.join(self.data_dir, data_item['fg_mask_path']))

            c2w = np.array(data_item['transform_matrix'], dtype=np.float32)
            c2w[:3, 1:3] *= -1
            pos = torch.from_numpy(c2w[:3, 3])
            w2c = torch.from_numpy(np.linalg.inv(c2w))
            full_proj = torch.matmul(projection_mat, w2c)
            self.cam_view_mats.append(w2c)
            self.cam_proj_mats.append(full_proj)
            self.cam_positions.append(pos)

        self.cam_view_mats = torch.stack(self.cam_view_mats, dim=0)
        self.cam_proj_mats = torch.stack(self.cam_proj_mats, dim=0)
        self.cam_positions = torch.stack(self.cam_positions, dim=0)

        translations = []
        rotations = []
        neck_poses = []
        jaw_poses = []
        shapes = []
        exprs = []
        static_offsets = []
        eye_poses = []

        for timestep in tqdm(self.timestep_indices, desc="Loading FLAME parameters"):
            for data_item in json_data['frames']:
                if data_item['timestep_index'] == timestep:
                    flame_data = dict(np.load(os.path.join(self.data_dir, data_item['flame_param_path'])))
                    translations.append(flame_data['translation'])
                    rotations.append(flame_data['rotation'])
                    neck_poses.append(flame_data['neck_pose'])
                    jaw_poses.append(flame_data['jaw_pose'])
                    eye_poses.append(flame_data['eyes_pose'])
                    shapes.append(flame_data['shape'])
                    exprs.append(flame_data['expr'])
                    static_offsets.append(flame_data['static_offset'])
                    break

    
        self.translations = torch.from_numpy(np.concatenate(translations, axis=0)).to(dtype=torch.float32, device='cuda')
        self.rotations = torch.from_numpy(np.concatenate(rotations, axis=0)).to(dtype=torch.float32, device='cuda')
        self.neck_poses = torch.from_numpy(np.concatenate(neck_poses, axis=0)).to(dtype=torch.float32, device='cuda')
        self.jaw_poses = torch.from_numpy(np.concatenate(jaw_poses, axis=0)).to(dtype=torch.float32, device='cuda')
        self.eye_poses = torch.from_numpy(np.concatenate(eye_poses, axis=0)).to(dtype=torch.float32, device='cuda')
        self.shapes = torch.from_numpy(np.stack(shapes, axis=0)).to(dtype=torch.float32, device='cuda')
        self.exprs = torch.from_numpy(np.concatenate(exprs, axis=0)).to(dtype=torch.float32, device='cuda')
        self.static_offsets = torch.from_numpy(np.concatenate(static_offsets, axis=0)).to(dtype=torch.float32, device='cuda')

        self.rotations = rotvec_to_rotmat(self.rotations)
        self.neck_poses = rotvec_to_rotmat(self.neck_poses)
        self.jaw_poses = rotvec_to_rotmat(self.jaw_poses)
        self.eye_poses = rotvec_to_rotmat(self.eye_poses.reshape(-1, 2, 3))

        self.mesh_verts = flame_model(
            shape_params = self.shapes, 
            expression_params = self.exprs, 
            neck_pose_params = self.neck_poses,
            jaw_pose_params = self.jaw_poses, 
            eye_pose_params = self.eye_poses, 
            root_pose_params = self.rotations,
            static_offsets = self.static_offsets
        )
        self.mesh_verts += self.translations.unsqueeze(-2)
        self.mesh_verts = self.mesh_verts.cpu()
        self.blend_weight = self.exprs.clone().cpu()

        if self.train:
            self.image_data = torch.zeros([len(self.image_paths), self.image_height, self.image_width, 3], dtype=torch.uint8)
            self.mask_data = torch.zeros([len(self.image_paths), self.image_height, self.image_width], dtype=torch.uint8)

            def load_single_image(i):
                image = torch.from_numpy(np.array(Image.open(self.image_paths[i]), dtype=np.uint8))
                mask = torch.from_numpy(np.array(Image.open(self.mask_paths[i]), dtype=np.uint8))
                return i, image, mask

            tasks = []
            with ThreadPoolExecutor(max_workers=8) as executor:
                for i in range(len(self.image_paths)):
                    tasks.append(executor.submit(load_single_image, i))

                for future in tqdm(as_completed(tasks), total=len(tasks), desc="Loading training images"):
                    i, image, mask = future.result()
                    self.image_data[i] = image
                    self.mask_data[i] = mask
        else:
            self.image_data = None
            self.mask_data = None



    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        frame_index = self.frame_id[index]
        camera_name = self.camera_names[index]
        mesh_verts = self.mesh_verts[frame_index]
        blend_weights = self.blend_weight[frame_index]
        cam_view_mat = self.cam_view_mats[index]
        cam_proj_mat = self.cam_proj_mats[index]
        cam_position = self.cam_positions[index]

        if self.train:
            # rgb = torch.from_numpy(np.array(Image.open(self.image_paths[index])))
            # mask = torch.from_numpy(np.array(Image.open(self.mask_paths[index]))).unsqueeze(-1)
            rgb = self.image_data[index]
            mask = self.mask_data[index].unsqueeze(-1)
            image = torch.cat([rgb, mask], dim=-1).permute(2, 0, 1)

            return {
                'image': image,
                'mesh': mesh_verts,
                'blend_weight': blend_weights,
                'cam_view_mat': cam_view_mat,
                'cam_proj_mat': cam_proj_mat,
                'cam_position': cam_position
            }
        else:
            return {
                'mesh': mesh_verts,
                'blend_weight': blend_weights,
                'cam_view_mat': cam_view_mat,
                'cam_proj_mat': cam_proj_mat,
                'cam_position': cam_position,
                'frame_index': frame_index,
                'camera_name': camera_name
            }
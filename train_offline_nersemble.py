import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

import sys
import time
import yaml
import torch
import shutil
import random
from tqdm import tqdm
from argparse import ArgumentParser
import nvdiffrast.torch as dr
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Sampler as TorchSampler

from utils import Struct
from model import FLAMEBindingModel, MultiViewReconstruction, BindingModel
from dataset import NeRSembleGADataset
from submodules.flame import FLAME, FlameConfig


class ContinuousRandomSampler(TorchSampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_size = len(dataset)

    def __iter__(self):
        while True:
            yield random.randint(0, self.data_size - 1)


class DataLoader:
    def __init__(self, dataset, batch_size: int = 10):
        self.dataset = dataset
        self.batch_size = batch_size

        sampler = ContinuousRandomSampler(dataset)
        dataloader = TorchDataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=8)
        self.dataloader_iter = iter(dataloader)


    def get_batch(self):
        data = next(self.dataloader_iter)
        for key in data.keys():
            data[key] = data[key].to(device='cuda', non_blocking=True)
        data['image'] = data['image'].to(dtype=torch.float32) / 255.0
        return data


def train_offline(
    dataset: NeRSembleGADataset, gaussian_model: BindingModel, 
    output_path: str, train_config, log: bool = False
):
    batch_size = train_config['batch_size']
    iteration = train_config['iteration']
    tb_writer = SummaryWriter(output_path) if log else None
    dataloader = DataLoader(dataset, batch_size=batch_size)
    recon = MultiViewReconstruction(
        dataset.fov_x, dataset.fov_y,
        dataset.image_width, dataset.image_height,
        gaussian_model, batch_size=batch_size, 
        recon_config=Struct(**train_config['recon']), tb_writer=tb_writer
    )

    print("Iteraion:", iteration)
    print("Dataset size:", len(dataset))
    progress_bar = tqdm(range(iteration), desc="Training")
    for i in range((iteration + batch_size - 1) // batch_size):
        data = dataloader.get_batch()
        recon.step(
            data['image'], data['mesh'], data['blend_weight'], 
            data['cam_view_mat'], data['cam_proj_mat'], data['cam_position']
        )
        progress_bar.update(batch_size)

    training_fps = iteration / progress_bar.format_dict['elapsed']
    if log: tb_writer.close()
    progress_bar.close()
    gaussian_model.prune()
    gaussian_model.save_ply(os.path.join(output_path, "model.ply"))

    result = f"Traning FPS: {training_fps:.2f}"
    print(result)
    with open(os.path.join(output_path, "speed.txt"), "w") as f:
        f.write(result)


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--subject", type=str, default="074")
    parser.add_argument("--work_name", type=str, default=None)
    parser.add_argument("--config", type=str, default="config/nersemble.yaml")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    with open(args.config) as f: config = yaml.load(f, Loader=yaml.FullLoader)

    glctx = dr.RasterizeGLContext()
    if args.work_name is None:
        args.work_name = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(int(time.time())))
    output_path = os.path.join(config['output_dir'], args.subject, args.work_name)
    if os.path.exists(output_path): shutil.rmtree(output_path)
    os.makedirs(output_path)
    shutil.copy(args.config, os.path.join(output_path, "config.yaml"))
    print("Output:", output_path)

    dataset_dir = config['data_dir']
    json_path = f'{dataset_dir}/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION10_{args.subject}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_train.json'

    if config['template_type'] == 'Flame':
        flame_config = FlameConfig()
        flame_config.dtype = torch.float32
        flame_config.flame_geom_path = 'data/FLAME2023/flame2023.pkl'
        flame_model = FLAME(flame_config).cuda()
        
        train_dataset = NeRSembleGADataset(flame_model, json_path)
        gaussian_model = FLAMEBindingModel(Struct(**config['model']), flame_model, glctx)
    else:
        raise NotImplementedError

    gaussian_model.initialize()

    train_offline(train_dataset, gaussian_model, output_path, config['train'], args.log)

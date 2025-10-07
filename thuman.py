import os
import glob
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
import random
random.seed(42)
from pathlib import Path
from tqdm import tqdm
import pickle


from utils.dataset_utils import get_cloth_info
from utils.graphics_utils import focal2fov
from .cameras import Camera


class ThumanDataset(Dataset):
    def __init__(self, cfg, split='train'):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.device = cfg.device

        self.root_dir = cfg.root_dir
        self.item_name = cfg.item_name

        self.camera_views = [f"{i:02d}" for i in range(60)]
        self.select_views = cfg.get('select_views', None)
        self.sample_view_num = cfg.get('sample_view_num', 1)

        self.white_bg = cfg.white_background
        self.H, self.W = 1024, 1024 # hardcoded original size
        self.h, self.w = self.H // cfg.img_res_ratio, self.W // cfg.img_res_ratio
        self.smpl_type = cfg.smpl_type # 'smpl' or 'smplx'

        # gather all take directories of each subject
        all_items = []
        for item_name in sorted(os.listdir(os.path.join(self.root_dir, 'process'))):
            if self.item_name is not None and item_name not in self.item_name:
                continue
            all_items.append(item_name)

        split_idx = int(len(all_items) * 0.9)
        if split == "train":
            subject_ls = all_items[:split_idx]
        else:
            subject_ls = all_items[split_idx:]

        self.data = self.process_data(subject_ls)
        self.preload = cfg.get('preload', False)
        if self.preload:
            self.cameras = [self.getitem(idx) for idx in range(len(self))]
            pass

    def process_data(self, subject_ls, split = 'train'):

        if split in ['train', 'val']:
            all_data = []

            for item_name in tqdm(subject_ls, desc="Loading all subject data in thuman"):

                subject_data = []
                images_path = os.path.join(self.root_dir, 'process', item_name, 'image')
                camera_path = os.path.join(self.root_dir, 'process', item_name, 'camera.pkl')

                camera_info = pickle.load(open(camera_path, 'rb'))
                if self.smpl_type == 'smplx':
                    smpl_param_path = os.path.join(self.root_dir, 'smplx', item_name, 'smplx_param.pkl')
                elif self.smpl_type == 'smpl':
                    smpl_param_path = os.path.join(self.root_dir, 'smpl', item_name, 'smplx_param.pkl')
                smpl_params = pickle.load(open(smpl_param_path, 'rb'))

                # randomly select one view with the first view
                img_files = sorted(glob.glob(os.path.join(images_path, '*.png')))
                remaining_views = [v for v in self.camera_views if v != '00']
                num_random = min(self.sample_view_num, len(remaining_views))
                random_views = random.sample(remaining_views, num_random)
                    
                for random_view in random_views:  # 每个随机view和frontal配对
                    cam_views = ['00', random_view]
                    image_paths = [p for p in img_files if any(v in Path(p).name for v in cam_views)]

                    Ks, Rs, Ts = [], [], []
                    cam_names = []
                    for cam_name in cam_views:
                        cam_names.append(cam_name)
                        
                        R = camera_info[cam_name]['R'].astype(np.float32)
                        T = camera_info[cam_name]['T'].astype(np.float32)
                        K = camera_info[cam_name]['K'].astype(np.float32)

                        Ks.append(K)
                        Rs.append(R)
                        Ts.append(T)

                    Ks = np.stack(Ks, axis=0)
                    Rs = np.stack(Rs, axis=0)
                    Ts = np.stack(Ts, axis=0)
                    cam_params = {'K': Ks, 'R': Rs, 'T': Ts}

                    subject_data.append({
                        'subject_name': item_name,
                        'cam_name': "_".join(cam_names),
                        'image_path': image_paths,
                        'cam_params': cam_params,
                        'smpl_params': smpl_params
                        })
                        
                all_data.append(subject_data)

            all_data = [item for subdata in all_data for item in subdata]

        elif split == 'test':
            all_data = []

            for item_name in tqdm(subject_ls, desc="Loading all subject data in thuman"):

                subject_data = []
                images_path = os.path.join(self.root_dir, 'process', item_name, 'image')
                camera_path = os.path.join(self.root_dir, 'process', item_name, 'camera.pkl')

                camera_info = pickle.load(open(camera_path, 'rb'))
                if self.smpl_type == 'smplx':
                    smpl_param_path = os.path.join(self.root_dir, 'smplx', item_name, 'smplx_param.pkl')
                elif self.smpl_type == 'smpl':
                    smpl_param_path = os.path.join(self.root_dir, 'smpl', item_name, 'smplx_param.pkl')
                smpl_params = pickle.load(open(smpl_param_path, 'rb'))


                image_paths = [p for p in img_files if any(v in Path(p).name for v in self.select_views)]

                for idx, cam_name in enumerate(self.select_views):
                    
                    R = camera_info[cam_name]['R'].astype(np.float32)
                    T = camera_info[cam_name]['T'].astype(np.float32)
                    K = camera_info[cam_name]['K'].astype(np.float32)

                    cam_params = {'K': K[np.newaxis], 'R': R[np.newaxis], 'T': T[np.newaxis]}

                    subject_data.append({
                        'subject_name': item_name,
                        'cam_name': cam_name,
                        'image_path': [image_paths[idx]],
                        'cam_params': cam_params,
                        'smpl_params': smpl_params
                        })
                        
                    all_data.append(subject_data)

            all_data = [item for subdata in all_data for item in subdata]



        return all_data         

    def __len__(self):
        return len(self.data)

    def getitem(self, idx, data_dict=None):
        if data_dict is None:
            data_dict = self.data[idx]

        subject_name = data_dict['subject_name']
        cam_name = data_dict['cam_name']
        img_file = data_dict['image_path']
        
        cam_params = data_dict['cam_params']
        smpl_params = data_dict['smpl_params']

        K = cam_params['K'].copy()
        R = cam_params['R']
        T = cam_params['T'][..., np.newaxis]

        # 对 R 进行转置（每个视角独立转置）
        R = R.transpose(0, 2, 1)  # (n_view, 3, 3)

        # load segmentation label, depth and normal
        images, masks = [], []
    
        for img_path in img_file:

            # 读取图像和掩码
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

            mask_path = img_path.replace('image', 'mask')
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

            image = torch.from_numpy(image).permute(2, 0, 1).float()  # (3, H, W)
            mask = torch.from_numpy(mask).unsqueeze(0).float()       # (1, H, W)

            # append to list
            images.append(image)
            masks.append(mask)

        images = torch.stack(images, dim=0)  # (n_view, C, H, W)
        masks = torch.stack(masks, dim=0)    # (n_view, 1, H, W)


        # update camera parameters
        # 批量缩放 K
        K[:, 0, :] *= self.w / self.W  # 横向缩放
        K[:, 1, :] *= self.h / self.H  # 纵向缩放

        # 提取所有视角的焦距
        focal_length_x = K[:, 0, 0]  # shape: (n_view,)
        focal_length_y = K[:, 1, 1]  # shape: (n_view,)

        FovY = focal2fov(focal_length_y, self.h)
        FovX = focal2fov(focal_length_x, self.w)

        # load smpl data
        # if self.smpl_type == 'smpl':
        #     raise NotImplementedError
        # elif self.smpl_type == 'smplx':
        #     param = np.load(subject_name, allow_pickle=True).item()
        #     smpl_params = param['smpl_params'].reshape(1, -1)
        
        cloth_dir = os.path.join(self.root_dir, 'Meshes_cloth', subject_name)
        cloth_info = get_cloth_info(cloth_dir)


        return Camera(
            subject_name=subject_name.replace("/", "_"),
            cam_id=cam_name,
            K=K, R=R, T=T,
            FoVx=FovX,
            FoVy=FovY,
            image=images,
            mask=masks,
            # semantic_feature=semantic_feature,
            seg_label=None,
            depth=None,
            normal=None,
            # gt_alpha_mask=None,
            data_device=self.cfg.device,
            # human params
            smpl_params=smpl_params,
            cloth_info=cloth_info,
        )

    def __getitem__(self, idx):
        if self.preload:
            return self.cameras[idx]
        else:
            return self.getitem(idx)
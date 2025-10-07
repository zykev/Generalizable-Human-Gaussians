# %%
import smplx
import os
import torch
import numpy as np
import json
import pickle

model_init_params = dict(
        gender='neutral',
        model_type='smplx',
        model_path='.datasets/thuman/models',
        create_global_orient=False,
        create_body_pose=False,
        create_betas=False,
        create_left_hand_pose=False,
        create_right_hand_pose=False,
        create_expression=False,
        create_jaw_pose=False,
        create_leye_pose=False,
        create_reye_pose=False,
        create_transl=False,
        use_pca = False,
        num_pca_comps=12,
        ext='pkl')

smpl_model = smplx.create(**model_init_params)

# read the original THuman smplx parameters
param_root = '.datasets/thuman/smplx'
param_fp = os.path.join(param_root, '0000', 'smplx_param.pkl')

with open(param_fp, 'rb') as f:
    param = pickle.load(f)
# param = np.load(param_fp, allow_pickle=True)
for key in param.keys():
    param[key] = torch.as_tensor(param[key]).to(torch.float32)

model_forward_params = dict(betas=param['betas'],
                            transl=param['transl'],
                            global_orient=param['global_orient'],
                            body_pose=param['body_pose'],
                            left_hand_pose=param['left_hand_pose'],
                            right_hand_pose=param['right_hand_pose'],
                            jaw_pose=param['jaw_pose'],
                            leye_pose=param['leye_pose'],
                            reye_pose=param['reye_pose'],
                            expression=param['expression'],
                            return_verts=True)

smpl_out = smpl_model(**model_forward_params)
smpl_verts = ((smpl_out.vertices[0] * param['scale'])).detach()
# smpl_verts = ((smpl_out.vertices[0] * param['scale'] + param['transl'] * param['scale'])).detach()


# %%
def compute_projections(xyz, K, R, T):
    """
    计算 3D 点在图像平面上的投影，使用 4×4 内外参矩阵合并计算。

    参数：
        xyz: (B, N, 3) - 3D 点，世界坐标系下的点云。
        intrinsics: (B, 3, 3) - 相机内参矩阵。
        train_poses: (B, 4, 4) - 相机外参矩阵 (世界到相机变换 W2C)。
        image_height: int - 图像高度，用于 Y 轴翻转。
        correct_principal: bool - 是否修正主点偏移。

    返回：
        projected_points: (B, N, 2) - 2D 像素坐标。
    """
    B, N, _ = xyz.shape
    xyz = xyz.float()

    intrinsics = K.unsqueeze(0)
    # 构造相机外参矩阵 (4, 4)
    extrinsics = torch.eye(4, dtype=torch.float32).unsqueeze(0)  # (1, 4, 4)
    extrinsics[:, :3, :3] = R.unsqueeze(0) # 旋转部分
    extrinsics[:, :3, 3:] = T.view(1,3,1)  # 平移部分

    # **Step 1: 构造 4×4 内参矩阵**
    K_pad = torch.eye(4, device=intrinsics.device, dtype=intrinsics.dtype).repeat(B, 1, 1)
    K_pad[:, :3, :3] = intrinsics  # 嵌入 3×3 内参


    # **Step 2: 计算投影矩阵 P = K_pad × W2C**
    P = torch.bmm(K_pad, extrinsics)  # (B, 4, 4)

    # **Step 3: 将 3D 点扩展为齐次坐标 (B, N, 4)**
    ones = torch.ones((B, N, 1), device=xyz.device, dtype=xyz.dtype)
    xyz_homogeneous = torch.cat([xyz, ones], dim=2)  # (B, N, 4)

    # **Step 4: 直接应用 P 进行投影变换**
    projected_homogeneous = torch.bmm(xyz_homogeneous, P.transpose(1, 2))  # (B, N, 4)

    # **Step 5: 归一化 (除以 z)**
    eps = 1e-8
    projected_points = projected_homogeneous[:, :, :2] / (projected_homogeneous[:, :, 2:3] + eps)


    return projected_points.squeeze(0)  # (N, 2)

# %%
# read camera parameters
camera_root = '.datasets/thuman/process/0000/camera.json'
with open(camera_root, "r", encoding="utf-8") as f:
    cam = json.load(f)

K = cam["00"]["K"]
R = cam["00"]["R"]
T = cam["00"]["T"]

# %%
points_2d = compute_projections(smpl_verts.unsqueeze(0), 
                                torch.tensor(K).unsqueeze(0), 
                                torch.tensor(R).unsqueeze(0), 
                                torch.tensor(T).unsqueeze(0))
points_2d = points_2d.numpy()

# %%
# import trimesh

# mesh = trimesh.load(".datasets/thuman/smplx/0000/mesh_smplx.obj", process=False)
# verts = mesh.vertices


# %%
import cv2
import matplotlib.pyplot as plt


# 加载原图（BGR → RGB）
ori_image_path = '.datasets/thuman/process/0000/image/00.png'
img = cv2.imread(ori_image_path)[:, :, ::-1]  # 转为 RGB
h, w, _ = img.shape

y = points_2d[:, 1]
x = points_2d[:, 0]

# 可视化叠加
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.scatter(x, y, s=0.5, c='r')
plt.xlim([0, w])
plt.ylim([h, 0])  # 注意 Y 轴反转
plt.axis('off')
plt.tight_layout()
plt.show()


# %%

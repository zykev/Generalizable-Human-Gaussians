import taichi_three as t3
import numpy as np
from taichi_three.transform import *
from tqdm import tqdm
import os
import cv2
import pickle
import pdb

import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def save(pid, data_id, save_path, img, mask):
    img_save_path = os.path.join(save_path, data_id, 'image')
    # depth_save_path = os.path.join(save_path, data_id, 'depth')
    mask_save_path = os.path.join(save_path, data_id, 'mask')
    os.makedirs(img_save_path, exist_ok=True)
    # os.makedirs(depth_save_path, exist_ok=True)
    os.makedirs(mask_save_path, exist_ok=True)

    # depth = depth * 2.0 ** 15
    # cv2.imwrite(os.path.join(depth_save_path, '%02d' % pid + '.png'),
    #             depth.astype(np.uint16))
    img = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)[:, :, ::-1]
    mask = (np.clip(mask, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(os.path.join(img_save_path, '%02d' % pid + '.png'), img)
    cv2.imwrite(os.path.join(mask_save_path, '%02d' % pid + '.png'), mask)


class StaticRenderer:
    def __init__(self):
        ti.init(arch=ti.cuda, device_memory_fraction=0.8)
        self.scene = t3.Scene()
        self.N = 10

    def change_all(self):
        save_obj = []
        save_tex = []
        for model in self.scene.models:
            save_obj.append(model.init_obj)
            save_tex.append(model.init_tex)
        ti.init(arch=ti.cuda, device_memory_fraction=0.8)
        print('init')
        self.scene = t3.Scene()
        for i in range(len(save_obj)):
            model = t3.StaticModel(self.N, obj=save_obj[i], tex=save_tex[i])
            self.scene.add_model(model)

    def check_update(self, obj):
        temp_n = self.N
        self.N = max(obj['vi'].shape[0], self.N)
        self.N = max(obj['f'].shape[0], self.N)
        if not (obj['vt'] is None):
            self.N = max(obj['vt'].shape[0], self.N)

        if self.N > temp_n:
            self.N *= 2
            self.change_all()
            self.camera_light()

    def add_model(self, obj, tex=None):
        self.check_update(obj)
        model = t3.StaticModel(self.N, obj=obj, tex=tex)
        self.scene.add_model(model)

    def modify_model(self, index, obj, tex=None):
        self.check_update(obj)
        self.scene.models[index].init_obj = obj
        self.scene.models[index].init_tex = tex
        self.scene.models[index]._init()

    def camera_light(self):
        camera = t3.Camera(res=(1024, 1024))
        self.scene.add_camera(camera)

        camera_hr = t3.Camera(res=(2048, 2048))
        self.scene.add_camera(camera_hr)

        light_dir = np.array([0, 0, 1])
        light_list = []
        for l in range(6):
            rotate = np.matmul(
                rotationX(math.radians(np.random.uniform(-30, 30))),
                rotationY(math.radians(360 // 6 * l)))
            dir = [*np.matmul(rotate, light_dir)]
            light = t3.Light(dir, color=[1.0, 1.0, 1.0])
            light_list.append(light)
        lights = t3.Lights(light_list)
        self.scene.add_lights(lights)


def render_data(renderer, smplx_path, data_path, data_id, save_path, cam_nums, res):


    obj_path = os.path.join(data_path, data_id, '%s.obj' % data_id)
    texture_path = data_path
    img_path = os.path.join(texture_path, data_id, 'material0.jpeg')
    texture = cv2.imread(img_path)[:, :, ::-1]
    texture = np.ascontiguousarray(texture)
    texture = texture.swapaxes(0, 1)[:, ::-1, :]

    smpl_path = os.path.join(smplx_path, data_id, 'smplx_param.pkl')
    with open(smpl_path, 'rb') as f:
        smpl_para = pickle.load(f)
    
    scale = smpl_para['scale']
    obj = t3.readobj(obj_path, scale=1/scale)

    # height normalization

    verts = obj['vi']  # (N, 3)
    x_min, y_min, z_min = verts.min(0)
    x_max, y_max, z_max = verts.max(0)

    # 水平中心
    x_center = (x_min + x_max) / 2
    z_center = (z_min + z_max) / 2
    y_center = (y_min + y_max) / 2

    look_at_center = np.array([x_center, y_center, z_center])
    base_cam_pitch = -8

    # distance proportional to height
    dis = (y_max - y_min) * 1.05  # 越小越近

    if len(renderer.scene.models) >= 1:
        renderer.modify_model(0, obj, texture)
    else:
        renderer.add_model(obj, texture)


    degree_interval = 360 / cam_nums

    # thuman needs a normalization of orientation
    y_orient = smpl_para['global_orient'][0][1]
    angle_base = (y_orient * 180.0 / np.pi)

    Ks, Rs, Ts, pids = [], [], [], []
    for pid in range(cam_nums):
        pids.append('%02d' % pid)
        angle = angle_base + pid * degree_interval

        def render(dis, angle, look_at_center, p, renderer):
            ori_vec = np.array([0, 0, dis])
            rotate = np.matmul(rotationY(math.radians(angle)),
                               rotationX(math.radians(p)))
            fwd = np.matmul(rotate, ori_vec)
            cam_pos = look_at_center + fwd

            x_min = 0
            y_min = 0 # -25
            cx = res[0] * 0.5
            cy = res[1] * 0.5
            fx = res[0] * 0.8
            fy = res[1] * 0.8
            _cx = cx - x_min
            _cy = cy - y_min
            renderer.scene.cameras[0].set_intrinsic(fx, fy, _cx, _cy)
            renderer.scene.cameras[0].set(pos=cam_pos, target=look_at_center)
            renderer.scene.cameras[0]._init()


            renderer.scene.render()
            camera = renderer.scene.cameras[0]
            extrinsic = camera.export_extrinsic()
            intrinsic = camera.export_intrinsic()
            # depth_map = camera.zbuf.to_numpy().swapaxes(0, 1)
            img = camera.img.to_numpy().swapaxes(0, 1)
            mask = camera.mask.to_numpy().swapaxes(0, 1)
            return extrinsic, intrinsic, img, mask

        extr, intr, img, mask = render(dis, angle, look_at_center,
                                              base_cam_pitch, renderer)
        Ks.append(intr)
        Rs.append(extr[:3, :3])
        Ts.append(extr[:3, 3])

        save(pid, data_id, save_path, img, mask)

    cam_dict = {}
    for cam_id, K, R, T in zip(pids, Ks, Rs, Ts):
        cam_dict[cam_id] = {
            "K": K.tolist(),
            "R": R.tolist(),
            "T": T.tolist()
        }

    camera_json_path = os.path.join(save_path, data_id, 'camera.json')
    with open(camera_json_path, 'w') as f:
        json.dump(cam_dict, f, indent=2)


if __name__ == '__main__':

    np.random.seed(42)

    cam_nums = 60
    res = (1024, 1024)
    smplx_root = '.datasets/THuman/THuman2.1_smplx'
    thuman_root = '.datasets/THuman/THuman2.1_Release'
    save_root = '.datasets/THuman/process'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    renderer = StaticRenderer()

    thuman_list = sorted(os.listdir(thuman_root))
    thuman_list = [f"{i:04d}" for i in range(526)]

    for data_id in tqdm(thuman_list):

        render_data(renderer, smplx_root, thuman_root, data_id, save_root,
                    cam_nums, res)

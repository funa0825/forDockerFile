import os
import sys
import copy
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import utils.pc_util as pc_util
from utils.box_util import (flip_axis_to_camera_np, flip_axis_to_camera_tensor,
                            get_3d_box_batch_np, get_3d_box_batch_tensor)

from utils.pc_util import scale_points, shift_scale_points
from utils.random_cuboid import RandomCuboid
import json
import math
from plyfile import PlyData, PlyElement
import pandas  as pd
import plyfile

import mmcv.ops.furthest_point_sample as furthest_point_sample

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'scannet'))

import scannet_utils


IGNORE_LABEL = -100
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
DATASET_ROOT_DIR = ""  ## Replace with path to dataset
DATASET_METADATA_DIR = "" ## Replace with path to dataset
# boxes_vol = []
# import pickle


# def box3d_vol_batch(corners):
#     """corners: (8,3) no assumption on axis direction"""
#     a = np.sqrt((corners[:, 0, 2] - corners[:, 1, 2]) ** 2)
#     b = np.sqrt((corners[:, 1, 0] - corners[:, 2, 0]) ** 2)
#     c = np.sqrt((corners[:, 0, 1] - corners[:, 4, 1]) ** 2)
#     return a * b * c


class ARKitScenesDatasetConfig(object):
    def __init__(self):
        self.num_semcls = 18
        self.num_angle_bin = 16
        self.max_num_obj = 64

        self.type2class = {
            "cabinet": 0,
            "refrigerator": 1,
            "shelf": 2,
            "stove": 3,
            "bed": 4,
            "sink": 5,
            "washer": 6,
            "toilet": 7,
            "bathtub": 8,
            "oven": 9,
            "dishwasher": 10,
            "fireplace": 11,
            "stool": 12,
            "chair": 13,
            "table": 14,
            "tv_monitor": 15,
            "sofa": 16,
            "build_in_cabinet": 17
        }
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.nyu40ids = np.array(
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
        )
        self.nyu40id2class = {
            nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids))
        }

        self.mean_size_arr = np.array([
            [0.76966726, 0.81160211, 0.92573741],
            [1.876858  , 1.84255952, 1.19315654],
            [0.61327999, 0.61486087, 0.71827014],
            [1.39550063, 1.51215451, 0.83443565],
            [0.97949596, 1.06751485, 0.63296875],
            [0.53166301, 0.59555772, 1.75001483],
            [0.96247056, 0.72462326, 1.14818682],
            [0.83221924, 1.04909355, 1.68756634],
            [0.21132214, 0.4206159 , 0.53728459],
            [1.44400728, 1.89708334, 0.26985747],
            [1.02942616, 1.40407966, 0.87554322],
            [1.37664116, 0.65521793, 1.68131292],
            [0.66508189, 0.71111926, 1.29885307],
            [0.41999174, 0.37906947, 1.75139715],
            [0.59359559, 0.59124924, 0.73919014],
            [0.50867595, 0.50656087, 0.30136236],
            [1.15115265, 1.0546296 , 0.49706794],
            [0.47535286, 0.49249493, 0.58021168],
            [0.47535286, 0.49249493, 0.58021168]
        ])

        self.mean_size_arr_hard_anchor = np.array([
            [1.0, 1.0, 1.0] for i in range(18)
        ]) 
            
        self.type_mean_size = {}
        for i in range(self.num_semcls):
            self.type_mean_size[self.class2type[i]] = self.mean_size_arr[i, :]
        
        # Semantic Segmentation Classes. Not used in V-DETR

    def angle2class(self, angle):
        raise ValueError("ScanNet does not have rotated bounding boxes.")

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        zero_angle = torch.zeros(
            (pred_cls.shape[0], pred_cls.shape[1]),
            dtype=torch.float32,
            device=pred_cls.device,
        )
        return zero_angle

    def class2anglebatch(self, pred_cls, residual, to_label_format=True):
        zero_angle = np.zeros(pred_cls.shape[0], dtype=np.float32)
        return zero_angle

    def param2obb(
        self,
        center,
        heading_class,
        heading_residual,
        size_class,
        size_residual,
        box_size=None,
    ):
        heading_angle = self.class2angle(heading_class, heading_residual)
        if box_size is None:
            box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle * -1
        return obb

    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor(box_size, box_angle, box_center_upright)
        return boxes

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
        boxes = get_3d_box_batch_np(box_size, box_angle, box_center_upright)
        return boxes

    @staticmethod
    def rotate_aligned_boxes(input_boxes, rot_mat):
        centers, lengths = input_boxes[:, 0:3], input_boxes[:, 3:6]
        new_centers = np.dot(centers, np.transpose(rot_mat))

        dx, dy = lengths[:, 0] / 2.0, lengths[:, 1] / 2.0
        new_x = np.zeros((dx.shape[0], 4))
        new_y = np.zeros((dx.shape[0], 4))

        for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
            crnrs = np.zeros((dx.shape[0], 3))
            crnrs[:, 0] = crnr[0] * dx
            crnrs[:, 1] = crnr[1] * dy
            crnrs = np.dot(crnrs, np.transpose(rot_mat))
            new_x[:, i] = crnrs[:, 0]
            new_y[:, i] = crnrs[:, 1]

        new_dx = 2.0 * np.max(new_x, 1)
        new_dy = 2.0 * np.max(new_y, 1)
        new_lengths = np.stack((new_dx, new_dy, lengths[:, 2]), axis=1)

        return np.concatenate([new_centers, new_lengths], axis=1)
    def get_classs(i):
        return self.type2class[i]

class ChromaticAutoContrast(object):
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, data):
        if np.random.random() < self.p:
            lo = np.min(data[:, :3], 0, keepdims=True)
            hi = np.max(data[:, :3], 0, keepdims=True)
            scale = 255 / (hi - lo)
            contrast_feat = (data[:, :3] - lo) * scale
            blend_factor = np.random.random() if self.blend_factor is None else self.blend_factor
            data[:, :3] = (1 - blend_factor) * data[:, :3] + blend_factor * contrast_feat
            """vis
            from openpoints.dataset import vis_points
            vis_points(data['pos'], data['x']/255.)
            """
        return data
    

class ChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, data):
        if np.random.random() < self.p:
            noise = np.random.randn(data.shape[0], 3)
            noise *= self.std * 255
            data[:, :3] = np.clip(noise + data[:, :3], 0, 255)
        return data


class HueSaturationTranslation(object):
    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]

        hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def __init__(self, hue_max=0.5, saturation_max=0.2):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, data):
        # Assume feat[:, :3] is rgb
        hsv = HueSaturationTranslation.rgb_to_hsv(data[:, :3])
        hue_val = (np.random.random() - 0.5) * 2 * self.hue_max
        sat_ratio = 1 + (np.random.random() - 0.5) * 2 * self.saturation_max
        hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
        hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
        data[:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)
        return data
    
    
class ARKitScenesDetectionDataset(Dataset):
    def __init__(
        self,
        dataset_config,
        split_set="train",
        use_height=False,
        augment=False,
        use_random_cuboid=True,
        random_cuboid_min_points=30000,
        args=None,
    ):
        print("Call ARKitScenes Dataloader")
        self.dataset_config = dataset_config
        assert split_set in ["train", "val"]
        root_dir = args.dataset_root_dir
        meta_data_dir = args.meta_data_dir
        if root_dir is None:
            root_dir = DATASET_ROOT_DIR

        if meta_data_dir is None:
            meta_data_dir = DATASET_METADATA_DIR

        self.data_path = root_dir
        self.data_num = args.data_num # 0  mean useallFile
        train_scan_name = list(
            set(
                [
                    os.path.basename(x)
                    for x in os.listdir(self.data_path+"/Training")
                ]
            )
        )
        val_scan_name = list(
            set(
                [
                    os.path.basename(x)
                    for x in os.listdir(self.data_path+"/Validation")
                ]
            )
        )
        all_scan_names = train_scan_name + val_scan_name
        
        # Get Scan Names
        if split_set == "all":
            self.scan_names = all_scan_names
        elif split_set in ["train", "val", "test"]:
            if split_set == "train":
                self.scan_names = train_scan_name
                self.data_path = self.data_path + "/Training"
            if split_set == "val":
                self.scan_names = val_scan_name
                self.data_path = self.data_path + "/Validation"
            if split_set == "test":
                self.scan_names = val_scan_name
                self.data_path = self.data_path + "/Validation"
        else:
            raise ValueError(f"Unknown split name {split_set}")
        if self.data_num > 0:
            self.scan_names = self.scan_names[0:self.data_num]
        #print(self.scan_names)
        self.use_height = use_height
        self.augment = augment
        self.use_random_cuboid = use_random_cuboid
        
        
        self.num_points = args.num_points        
        self.use_color = args.use_color
        self.color_mean = args.color_mean
        self.rot_ratio = args.rot_ratio
        self.scale_ratio = args.scale_ratio
        self.trans_ratio = args.trans_ratio
        self.use_superpoint = args.use_superpoint
        self.args = args
        self.random_cuboid_augmentor = RandomCuboid(min_points=random_cuboid_min_points)
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]
        
        if args.filt_empty:
            # hacking: some problems will happen if there are samples without gt boxes, during training
            print(split_set, len(self.scan_names), "scans before filter")
            empty_box_scannames = self.filter()
            for scan in empty_box_scannames:
                if scan in self.scan_names:
                    #print(scan)
                    self.scan_names.remove(scan)
            print(split_set, len(self.scan_names),"scans after filter")
        
        self.use_normals = args.use_normals
        
    def load_annotation_json(self,json_path):
        with open(json_path, "r") as f:
            json_data = json.load(f)
        bbox_list = []
        angle_list = []
        for label_info in json_data["data"]:
            rotation = np.array(label_info["segments"]["obbAligned"]["normalizedAxes"]).reshape(3, 3)
            transform = np.array(label_info["segments"]["obbAligned"]["centroid"]).reshape(-1, 3)
            scale = np.array(label_info["segments"]["obbAligned"]["axesLengths"]).reshape(-1, 3)
            label = label_info["label"]
            label_int = self.dataset_config.type2class[label]
            cosine_z = math.acos(rotation[0][0])
            angleclass , angleresidual = self.angle2class_residual(cosine_z)
            bbox_list.append([transform[0][0],transform[0][1],transform[0][2],scale[0][0],scale[0][1],scale[0][2],label_int])
            angle_list.append([angleclass,angleresidual,cosine_z])
        return bbox_list,angle_list


    def angle2class_residual(self,angle):
        # -pi< angle < pi -> 0 < angle < 2pi
        angleclass = (angle+math.pi) // (2*math.pi/self.dataset_config.num_angle_bin)
        angleresidual = (angle+math.pi) % (2*math.pi/self.dataset_config.num_angle_bin)
        # -angle_bin/2 < residual < +angle_bin/2
        angleresidual = angleresidual - (2*math.pi/self.dataset_config.num_angle_bin)/2
        return angleclass,angleresidual
    def filter(self):
        empty_box_scannames = []
        for scan_name in self.scan_names:
            json_path = os.path.join(self.data_path, scan_name,scan_name) + "_3dod_annotation.json"
            bboxes,angles = self.load_annotation_json(json_path)
            instance_bboxes = np.array(bboxes)
            #instance_bboxes = np.load(os.path.join(self.data_path, scan_name) + "_bbox.npy")
            if instance_bboxes.shape[0]==0:
                # print("scan_name",scan_name,instance_bboxes.shape[0])
                empty_box_scannames.append(scan_name)
        return empty_box_scannames

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]
        #print(scan_name)
        
        # Get Mesh From Ply
        

        mesh_file = os.path.join(self.data_path, scan_name,scan_name) + "_3dod_mesh.ply"
        #print(mesh_file)
        mesh_vertices = scannet_utils.read_mesh_vertices_rgb(mesh_file)
        #print(mesh_vertices)
        '''
        if self.use_superpoint:
            seg_fileroot = os.path.join((self.data_path + ".." +"/scans"),scan_name) + "/" +scan_name + "_vh_clean_2.0.010000.segs.json"
            with open(seg_fileroot) as f:
                seg_data = json.load(f)
                superpoint_labels = np.array(seg_data['segIndices'])
        if self.use_normals:
            ply_fileroot = os.path.join((self.data_path + ".." +"/raw_data/scans"),scan_name) + "/" +scan_name + "_vh_clean_2.ply"
            # seg_to_verts, num_verts, verts_to_seg = read_segmentation(seg_fileroot)
            vertices, faces = read_plymesh(ply_fileroot)
            def face_normal(vertex, face):
                v01 = vertex[face[:, 1]] - vertex[face[:, 0]]
                v02 = vertex[face[:, 2]] - vertex[face[:, 0]]
                vec = np.cross(v01, v02)
                length = np.sqrt(np.sum(vec ** 2, axis=1, keepdims=True)) + 1.0e-8
                nf = vec / length
                area = length * 0.5
                return nf, area


            def vertex_normal(vertex, face):
                nf, area = face_normal(vertex, face)
                nf = nf * area

                nv = np.zeros_like(vertex)
                for i in range(face.shape[0]):
                    nv[face[i]] += nf[i]

                length = np.sqrt(np.sum(nv ** 2, axis=1, keepdims=True)) + 1.0e-8
                nv = nv / length
                return nv
            coords = vertices[:, :3]
            normals = vertex_normal(coords, faces)
        '''
        # Load BBox & Label 

        #instance_labels = np.load(
        #    os.path.join(self.data_path, scan_name) + "_ins_label.npy"
        #)
        #semantic_labels = np.load(
        #    os.path.join(self.data_path, scan_name) + "_sem_label.npy"
        #)
        #instance_bboxes = np.load(os.path.join(self.data_path, scan_name) + "_bbox.npy")
        json_path = os.path.join(self.data_path, scan_name,scan_name) + "_3dod_annotation.json"
        bboxes,angles = self.load_annotation_json(json_path)
        #print(bboxes)
        instance_bboxes = np.array(bboxes)
        angles_np = np.array(angles)
        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
            pcl_color = mesh_vertices[:, 3:6]
        else:
            point_cloud = mesh_vertices[:, 0:6]
            if self.augment:
                if self.args.color_drop > 0:
                    colors_drop = np.random.random(point_cloud.shape[0]) > self.args.color_drop
                    point_cloud[:, 3:] *= colors_drop[:,None]
                if self.args.color_contrastp > 0:
                    contrast = ChromaticAutoContrast(self.args.color_contrastp)
                    point_cloud[:, 3:] = contrast(point_cloud[:, 3:])
                if self.args.color_jitterp > 0:
                    jitter = ChromaticJitter(self.args.color_jitterp)
                    point_cloud[:, 3:] = jitter(point_cloud[:, 3:])
                if float(self.args.hue_sat.split('_')[-1]) > 0:
                    hue, sat, hue_sat_p = self.args.hue_sat.split('_')
                    hue, sat, hue_sat_p = float(hue), float(sat), float(hue_sat_p)
                    if np.random.random() < hue_sat_p:
                        huesat = HueSaturationTranslation(hue, sat)
                        point_cloud[:, 3:] = huesat(point_cloud[:, 3:])
            
            if self.color_mean < 0:
                point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0
            else:
                point_cloud[:, 3:] = point_cloud[:, 3:] / 255.0 - 0.5
            if self.use_normals:
                point_cloud = np.concatenate((point_cloud,normals),axis=-1)
            pcl_color = point_cloud[:, 3:6]
        #
        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

        # ------------------------------- LABELS ------------------------------
        MAX_NUM_OBJ = self.dataset_config.max_num_obj
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6), dtype=np.float32)
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ), dtype=np.float32)
        angle_classes = np.zeros((MAX_NUM_OBJ,), dtype=np.int64)
        angle_residuals = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)
        size_residuals = np.zeros((MAX_NUM_OBJ, 3), dtype=np.float32)
        raw_sizes = np.zeros((MAX_NUM_OBJ, 3), dtype=np.float32)
        raw_angles = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)

        #sem_seg_labels = np.ones_like(semantic_labels) * IGNORE_LABEL

        #for _c in self.dataset_config.nyu40ids_semseg:
        #    sem_seg_labels[
        #        semantic_labels == _c
        #    ] = self.dataset_config.nyu40id2class_semseg[_c]

        target_bboxes_mask[0 : instance_bboxes.shape[0]] = 1
        target_bboxes[0 : instance_bboxes.shape[0]] = instance_bboxes[:, 0:6]
        angle_classes[0 : instance_bboxes.shape[0]] = angles_np[:, 0]
        angle_residuals[0 : instance_bboxes.shape[0]] = angles_np[:, 1]
        raw_angles[0 : instance_bboxes.shape[0]] = angles_np[:, 2]

        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:

            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:, 1] = -1 * point_cloud[:, 1]
                target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

            # Rotation along up-axis/Z-axis
            rot_angle = ((np.random.random() * np.pi / 18) - np.pi / 36) * self.rot_ratio / 5.0  # -5 ~ +5 degree
            rot_mat = pc_util.rotz(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            target_bboxes = self.dataset_config.rotate_aligned_boxes(
                target_bboxes, rot_mat
            )
            
            if self.trans_ratio > 0.0:
                trans_value = (np.random.random(size=3) - 0.5) * self.trans_ratio / 0.5
                point_cloud[:, 0:3] = point_cloud[:, 0:3] + trans_value
                target_bboxes[:, 0:3] = target_bboxes[:, 0:3] + trans_value
            
            if self.scale_ratio > 0.0:
                scale_value = 1 + (np.random.random() - 0.5) * self.scale_ratio / 0.5
                point_cloud[:, 0:3] = point_cloud[:, 0:3] * scale_value
                target_bboxes[:, :] = target_bboxes[:, :] * scale_value

            if self.args.coloraug_sunrgbd:
                point_cloud[:, 3:6] += 0.5
                point_cloud[:, 3:6] *= (
                    1 + 0.4 * np.random.random(3) - 0.2
                )  # brightness change for each channel
                point_cloud[:, 3:6] += (
                    0.1 * np.random.random(3) - 0.05
                )  # color shift for each channel
                point_cloud[:, 3:6] += np.expand_dims(
                    (0.05 * np.random.random(point_cloud.shape[0]) - 0.025), -1
                )  # jittering on each pixel
                point_cloud[:, 3:6] = np.clip(point_cloud[:, 3:6], 0, 1)
                # randomly drop out 30% of the points' colors
                point_cloud[:, 3:6] *= np.expand_dims(
                    np.random.random(point_cloud.shape[0]) > 0.3, -1
                )
                point_cloud[:, 3:6] -= 0.5
                    
        raw_sizes = target_bboxes[:, 3:6]
        point_cloud_dims_min = point_cloud.min(axis=0)[:3]
        point_cloud_dims_max = point_cloud.max(axis=0)[:3]

        box_centers = target_bboxes.astype(np.float32)[:, 0:3]
        box_centers_normalized = shift_scale_points(
            box_centers[None, ...],
            src_range=[
                point_cloud_dims_min[None, ...],
                point_cloud_dims_max[None, ...],
            ],
            dst_range=self.center_normalizing_range,
        )
        box_centers_normalized = box_centers_normalized.squeeze(0)
        box_centers_normalized = box_centers_normalized * target_bboxes_mask[..., None]
        mult_factor = point_cloud_dims_max - point_cloud_dims_min
        box_sizes_normalized = scale_points(
            raw_sizes.astype(np.float32)[None, ...],
            mult_factor=1.0 / mult_factor[None, ...],
        )
        box_sizes_normalized = box_sizes_normalized.squeeze(0)

        box_corners = self.dataset_config.box_parametrization_to_corners_np(
            box_centers[None, ...],
            raw_sizes.astype(np.float32)[None, ...],
            raw_angles.astype(np.float32)[None, ...],
        )
        box_corners = box_corners.squeeze(0)

        ret_dict = {}
        ret_dict["gt_box_corners"] = box_corners.astype(np.float32)
        
        # global boxes_vol
        # for i in range(instance_bboxes.shape[0]):
        #     boxes_vol.append(box3d_vol_batch(box_corners[i:i+1]))
        #     print(len(boxes_vol))
        # pickle.dump(boxes_vol, open('boxes_vol', 'wb'))
        
        ret_dict["gt_box_centers"] = box_centers.astype(np.float32)
        ret_dict["gt_box_centers_normalized"] = box_centers_normalized.astype(
            np.float32
        )
        ret_dict["gt_angle_class_label"] = angle_classes.astype(np.int64)
        ret_dict["gt_angle_residual_label"] = angle_residuals.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ), dtype=np.int64)
        target_bboxes_semcls[0 : instance_bboxes.shape[0]] = [
            int(x)
            for x in instance_bboxes[:, -1][0 : instance_bboxes.shape[0]]
        ]
        
        size_residuals[0:instance_bboxes.shape[0], :] = \
            raw_sizes[0:instance_bboxes.shape[0], :] - self.dataset_config.mean_size_arr[\
                target_bboxes_semcls[0 : instance_bboxes.shape[0]], :]
        
        ret_dict["gt_box_sem_cls_label"] = target_bboxes_semcls.astype(np.int64)
        ret_dict["gt_box_present"] = target_bboxes_mask.astype(np.float32)
        ret_dict["scan_idx"] = np.array(idx).astype(np.int64)
        # ret_dict["pcl_color"] = pcl_color
        ret_dict["point_clouds"] = torch.from_numpy(point_cloud.astype(np.float32))
        ret_dict["gt_box_sizes"] = raw_sizes.astype(np.float32)
        ret_dict["gt_box_sizes_normalized"] = box_sizes_normalized.astype(np.float32)
        ret_dict["gt_box_sizes_residual_label"] = size_residuals.astype(np.float32)
        ret_dict["gt_box_angles"] = raw_angles.astype(np.float32)
        ret_dict["point_cloud_dims_min"] = point_cloud_dims_min.astype(np.float32)
        ret_dict["point_cloud_dims_max"] = point_cloud_dims_max.astype(np.float32)
        #ret_dict["scan_name"] = scan_name
        if self.use_superpoint:
            ret_dict["superpoint_labels"] = superpoint_labels

        # if self.num_points > 0:
        #     # Introduced from GroupFree3D.
        #     # Note: since there's no map between bbox instance labels and
        #     # pc instance_labels (it had been filtered 
        #     # in the data preparation step) we'll compute the instance bbox
        #     # from the points sharing the same instance label.
        #     point_instance_label = np.zeros(self.num_points) - 1
        #     box_centers_cal_max = copy.deepcopy(box_centers)
        #     box_centers_cal_max[instance_bboxes.shape[0]:, :] += 1000.0  # padding centers with a large number
        #     for i_instance in np.unique(instance_labels):
        #         # find all points belong to that instance
        #         ind = np.where(instance_labels == i_instance)[0]
        #         # find the semantic label            
        #         if semantic_labels[ind[0]] in self.dataset_config.nyu40ids:
        #             x = point_cloud[ind, :3]
        #             center = 0.5 * (x.min(0) + x.max(0))
        #             ilabel = np.argmin(((center - box_centers_cal_max) ** 2).sum(-1))
        #             point_instance_label[ind] = ilabel
        #     ret_dict['point_instance_label'] = point_instance_label.astype(np.int64)   

        return ret_dict

    def collate_fn(self, batch):
        batch_dict = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], np.ndarray):
                batch_dict[key] = torch.stack([torch.from_numpy(sample[key]) for sample in batch],dim=0)
            else:
                batch_dict[key] = [sample[key] for sample in batch]

        return batch_dict

def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        verts_to_seg = data['segIndices']
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts, verts_to_seg


def read_plymesh(filepath):
    """Read ply file and return it as numpy array. Returns None if emtpy."""
    with open(filepath, 'rb') as f:
        plydata = plyfile.PlyData.read(f)
    if plydata.elements:
        vertices = pd.DataFrame(plydata['vertex'].data).values
        faces = np.stack(plydata['face'].data['vertex_indices'], axis=0)
        return vertices, faces


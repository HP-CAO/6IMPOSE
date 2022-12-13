import numpy as np
import os
import pickle

from dataset.dataset_tools.blender_dataset_tools.blender_yolo_preprocessor import validate_bbox
from PIL import Image

import tensorflow as tf
from lib.data.utils import get_crop_index, crop_image, get_bbox_from_mask, pcld_processor_tf, get_offst
from lib.data.augmenter import augment_rgb, augment_depth, rotate_datapoint, add_background_depth


class PvnMixin():
    def __init__(self, use_pvn_kp, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_pvn_kp = use_pvn_kp
        self._centers_list = None
        self._keypoints_list = None

    def get(self, index):
        if self.data_config.use_preprocessed:
            return self.get_data_preprocessed(self.mode, index)
        else:
            data = self.get_dict(index)

            if data is None:
                return None

            return data['rgb'], data['pcld_xyz'], data['pcld_feats'], data['sampled_index'], data['labels'], \
                    data['kpts_targ_offst'], data['ctr_targ_offst'], data['mask_label'], data['crop_factor']

            

    def get_dict(self, index):
        #dpt, rgb_normalized, mask, Rt, xy_ofst, bbox, crop_index, crop_factor, uncropped = self.read_pvn_data(index)
        rgb = self.get_rgb(index)
        mask = self.get_mask(index)
        dpt = self.get_depth(index)

        Rt_list = self.get_RT_list(index)
        Rt = Rt_list[0][0]

        if self.if_augment:
            depth_tensor = tf.expand_dims(dpt, 0)  # add batch
            depth_tensor = tf.expand_dims(depth_tensor, -1)  # channel

            obj_pos = Rt_list[0][0][:3, 3]

            #if self.data_config.background_dir:
            #    bg_dir = self.data_config.background_dir
            #    with Image.open(os.path.join(bg_dir, np.random.choice(os.listdir(bg_dir)))) as random_rgb:
            #        random_rgb = np.array(random_rgb).astype(np.float)/255.
            #else:
            random_rgb = None
            
            depth_tensor = add_background_depth(depth_tensor, obj_pos=obj_pos, 
                            camera_matrix = self.data_config.intrinsic_matrix,rgb2noise=random_rgb)
            depth_tensor = augment_depth(depth_tensor)
            dpt = tf.squeeze(depth_tensor).numpy()
            rgb, mask, dpt, Rt_list = rotate_datapoint(img_likes=[rgb, mask, dpt], Rt=Rt_list)
            Rt = Rt_list[0][0]
        
        # Get bbox from rotated mask!
        bbox = get_bbox_from_mask(mask, gt_mask_value=255)

        if bbox is None:
            return None

        if not validate_bbox(bbox):
            return None

        h, w = self.data_config.rgb_input_shape[:2]

        crop_index, crop_factor = get_crop_index(bbox, rgb_size=rgb.shape, base_crop_resolution=(w, h))

        x1, y1, x2, y2 = crop_index
        xy_ofst =(x1, y1)

        # crop after augmenting depth -> never change the scale
        if self.crop_image:
            rgb = crop_image(rgb, crop_index)
            mask = crop_image(mask, crop_index)
            dpt = crop_image(dpt, crop_index)

        rgb_normalized = rgb / 255.
        if self.if_augment:
            rgb_normalized = augment_rgb(rgb_normalized)
            rgb = (rgb_normalized*255).astype(np.uint8)

        rgb = tf.image.resize(rgb, self.data_config.rgb_input_shape[:2]).numpy() # resize un-normalized rgb to resnet shape

        pcld_xyz, pcld_feats, sampled_index = pcld_processor_tf(dpt.astype(np.float32), rgb_normalized.astype(np.float32), 
            self.data_config.intrinsic_matrix.astype(np.float32), 1, 12288, xy_ofst=xy_ofst) # fixed n_sample points here -> gets sampled online

        if pcld_xyz.shape == (1,):
            return None

        label_list, mask_selected, mask_valid = self.get_label_list(self.cls_type, mask, sampled_index, gt_mask_value=255)

        if not mask_valid:
            return None

        mask_label = (mask_selected == 255).astype('uint8')
        center, kpts = self.get_ct_kpts()
        ctr_targ_offst, kpts_targ_offst = get_offst([Rt], pcld_xyz, mask_selected, self.data_config.n_objects, self.cls_type,
                                                    center, kpts)


        meta_data = {}
        meta_data['rgb'] = rgb.astype(np.uint8)
        meta_data['RT'] = Rt.astype(np.float32)
        meta_data['pcld_xyz'] = pcld_xyz.numpy().astype(np.float32)
        meta_data['pcld_feats'] = pcld_feats.numpy().astype(np.float32)
        meta_data['sampled_index'] = sampled_index.numpy().astype(np.int32)
        meta_data['labels'] = np.array(label_list).astype(np.int32)
        meta_data['mask_label'] = np.array(mask_label).astype(np.int32)
        meta_data['kpts_targ_offst'] = np.array(kpts_targ_offst).astype(np.float32)
        meta_data['ctr_targ_offst'] = np.array(ctr_targ_offst).astype(np.float32)
        meta_data['crop_index'] = np.array(crop_index).astype(np.int32)
        meta_data['crop_factor'] = np.array(crop_factor).astype(np.int32)
        meta_data['data_source'] = np.array('data_blender')
        meta_data['bbox'] = np.array(bbox).astype(int)
        meta_data['image_id'] = np.array(index)
        meta_data['K'] = self.data_config.intrinsic_matrix

        return meta_data

    def get_ct_kpts(self):
        if self._centers_list is None:
            centers_list = []
            keypoints_list = []
            cls_types = [self.cls_type]

            kpts_dir = self.data_config.kps_dir

            for _ in cls_types:
                corners_dir = os.path.join(kpts_dir, self.cls_type, "corners.txt")
                corners = np.loadtxt(corners_dir, dtype=np.float32)
                center = corners.mean(0)
                centers_list.append(center)
                kps_dir = os.path.join(kpts_dir, self.cls_type, "farthest.txt")
                kps = np.loadtxt(kps_dir, dtype=np.float32)
                keypoints_list.append(kps)

            self._centers_list = centers_list
            self._keypoints_list = keypoints_list
            return centers_list, keypoints_list
        else:
            return self._centers_list, self._keypoints_list


    def get_data_preprocessed(self, mode, index):
        get_data = lambda name: np.load(os.path.join(self.data_config.preprocessed_folder, name, f"{index:06}.npy"))

        rgb = get_data("rgb").astype(np.float32)
        crop_down_factor = get_data("crop_factor")
        sampled_index = get_data("sampled_index")
        pcld_xyz = get_data("pcld_xyz")
        pcld_feats = get_data("pcld_feats")

        if self.data_config.online_rgb_aug:
            # TODO augmented twice, might be risky, we intentionally weaken the augmentation effect
            self.p_sat = 0.5 / 2
            self.p_bright = 0.5 / 2
            self.p_noise = 0.1 / 2
            self.p_hue = 0.03 / 2
            self.p_contr = 0.5 / 2
            rgb = self.augment_rgb(rgb)
            pcld_feats = rgb.reshape((-1, 3))[sampled_index]

        if mode == "test":
            if self.use_pvn_kp:
                return rgb, pcld_xyz, pcld_feats, sampled_index, crop_down_factor, [
                    self.obj_cls_id]
            else:
                return rgb, pcld_xyz, pcld_feats, sampled_index, crop_down_factor
        else:


            label_list = get_data("labels")
            mask_label = get_data("mask_label")
            sampled_index = get_data("sampled_index")
            ctr_targ_offst = get_data("ctr_targ_offst")
            kpts_targ_offst = get_data("kpts_targ_offst")

            if self.use_pvn_kp:

                kp_cp_target_path = os.path.join(
                    self.data_config.root, '{:02}/preprocessed/kp_cp_target/{:06}.bin'.format(self.cls_id, index))

                kp_cp_target = pickle.load(open(kp_cp_target_path, 'rb'))

                return rgb, pcld_xyz, pcld_feats, sampled_index, label_list, \
                       kpts_targ_offst, ctr_targ_offst, mask_label, crop_down_factor, [self.obj_cls_id], kp_cp_target
            else:
                return rgb, pcld_xyz, pcld_feats, sampled_index, label_list, \
                       kpts_targ_offst, ctr_targ_offst, mask_label, crop_down_factor


    def get_label_list(self, cls_type, mask, sampled_index, gt_mask_value=255):
        """
        params: index: index for mask.png
                sampled_index: indexes for points selected
        return: a list of point-wise label
        """

        valid = True
        mask_counter = 0

        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        mask_selected = mask.flatten()[sampled_index].astype(np.uint8)

        label_list = []

        if cls_type == 'all':
            raise NotImplementedError("get_label_list for all cls not implemented")
        else:
            # mask_selected = (mask_selected > 0).astype("uint8")
            for i in mask_selected:
                label = np.zeros(2)
                if i == gt_mask_value:
                    label[1] = 1
                    mask_counter += 1
                else:
                    label[0] = 1
                label_list.append(label)

        if mask_counter < 30:
            valid = False

        return label_list, mask_selected, valid
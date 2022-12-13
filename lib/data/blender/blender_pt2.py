import os
import pickle
import numpy as np
from PIL import Image
from lib.utils import CachedProperty
from lib.data.blender.blender import Blender
from lib.data.utils import dpt_2_cld, get_pcld_rgb, get_crop_index, compute_normals
from lib.data.blender.blender_settings import BlenderSettings


class BlenderPt2(Blender):
    def __init__(self, mode, data_name, cls_type, use_preprocessed, size_all, train_size, crop_image, use_pvn_kp=False,
                 shuffle=True, add_noise=True):
        super().__init__(mode, data_name, cls_type, use_preprocessed, size_all, train_size, crop_image, shuffle,
                         add_noise)

        self.use_pvn_kp = use_pvn_kp
        self.crop_down_factor = [1]
        self.data_config = \
            BlenderSettings(data_name, self.cls_type, use_preprocessed=use_preprocessed,
                            crop_image=crop_image, size_all=size_all, train_size=train_size)
        # todo hacking using new directory
        self.data_config.online_rgb_aug = False
        self.data_config.preprocessed_folder = '/home/hongi/dataset/blender/blender/blender_linemod/03/preprocessed'

    def get_item(self):

        """
        return items for train/val
        rgb (add noise or not) [H, W, 3]
        pcld_xyz, [Npts, 3]
        pcld_rgb_nm, [Npts, 9]  where 9 is xyz + rgb + xyz_nrm
        sampled_points_index, [Npts, 1]
        kp_targ_ofst, [Npts, Nkpts, 3]
        ctr_targ_ofst, [Npts, Nctrs, 3]
        """

        index = self.index_lst[self.counter]

        if self.data_config.use_preprocessed:
            data = self.get_data_preprocessed(self.mode, index)
            return data
        else:
            RT_list = self.get_RT_list(index)

            K = self.data_config.intrinsic_matrix

            # get rgb
            rgb = self.get_rgb(index)

            # get depth
            depth = self.get_depth(index)

            # get mask
            mask = self.get_mask(index)

            if self.crop_image:
                bboxes = self.get_gt_bbox(index)
                crop_index, crop_factor = get_crop_index(bboxes[0], rgb_size=self.data_config.rgb_shape[0:2],
                                                         base_crop_resolution=self.data_config.crop_base_shape[0:2])
                x1, y1, x2, y2 = crop_index
                rgb = rgb[y1:y2, x1:x2]
                mask = mask[y1:y2, x1:x2]
                depth = depth[y1:y2, x1:x2]
                pcld_xyz, pcld_index = dpt_2_cld(depth, self.data_config.camera_scale, K, [x1, y1])
            else:
                pcld_xyz, pcld_index = dpt_2_cld(depth, self.data_config.camera_scale, K, [0, 0])

            pcld_rgb = get_pcld_rgb(rgb, pcld_index)
            index_chosen = self.choose_index(pcld_index)
            pcld_xyz_rgb = np.concatenate((pcld_xyz, pcld_rgb), axis=1)[index_chosen, :]

            # pcld_nm = normalize_pcld_xyz(pcld_xyz[index_chosen, :])[:, :3]
            # pcld_nm[np.isnan(pcld_nm)] = 0.0
            pcld_nm = compute_normals(depth.astype(np.float32), K.astype(np.float32)).numpy()
            pcld_nm = pcld_nm[pcld_index[index_chosen]]
            pcld_xyz_rgb_nm = np.concatenate((pcld_xyz_rgb, pcld_nm), axis=1)

            # get sampled_index
            sampled_index = pcld_index[index_chosen]

            if self.mode == "test":
                return rgb, pcld_xyz_rgb_nm[:, :3], pcld_xyz_rgb_nm[:, 3:], sampled_index, self.crop_down_factor

            else:
                # get labels

                label_list, mask_selected = self.get_label_list(mask, sampled_index)

                mask_label = (mask_selected > 0).astype('uint8')

                # get kp_targ_ofst ct_targ_ofst
                ctr_targ_ofst, kp_targ_ofst = self.get_offst(RT_list, pcld_xyz_rgb_nm[:, :3], mask_selected)

                return rgb, pcld_xyz_rgb_nm[:, :3], pcld_xyz_rgb_nm[:, 3:], \
                       sampled_index, label_list, kp_targ_ofst, ctr_targ_ofst, mask_label, self.crop_down_factor

    def get_mask(self, index):
        with Image.open(os.path.join(self.cls_root, "mask/{}.png".format(str(index).rjust(4, "0")))) as mask:
            mask = np.array(mask)
            return mask

    def get_label_list(self, mask, sampled_index):

        """
        params: index: index for mask.png
                sampled_index: indexes for points selected
        return: a list of point-wise label
        """

        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        mask_selected = mask.flatten()[sampled_index].astype("uint8")
        label_list = []

        if self.cls_type == 'all':
            for i in mask_selected:
                label = np.zeros(shape=self.data_config.n_classes)
                cls_id = np.where(self.data_config.mask_value_array == i)[0]
                if len(cls_id) == 0:
                    label[0] = 1
                else:
                    label[cls_id] = 1

                label_list.append(label)
        else:
            mask_selected = (mask_selected > 0).astype("uint8")
            for i in mask_selected:
                label = np.zeros(shape=self.data_config.n_classes)
                cls_id = np.where(self.data_config.mask_binary_array == i)[0]
                label[cls_id] = 1
                label_list.append(label)

        return label_list, mask_selected

    def get_RT_list(self, index):
        """
        return a list of RT matrix [RT_0, RT_1, ..., RT_N]
        """
        meta_list = []
        RT_list = []
        meta = self.meta_lst[index]

        if self.cls_id == 2:
            for i in range(0, len(meta)):
                if meta[i]['obj_id'] == 2:
                    meta_list.append(meta[i])
                    break
        elif self.cls_id == 16:
            for i in range(0, len(meta)):
                meta_list.append(meta[i])
        else:
            meta_list.append(meta[0])

        for mt in meta_list:
            R = np.resize(np.array(mt['cam_R_m2c']), (3, 3))
            T = np.array(mt['cam_t_m2c']) / 1000.0
            RT = np.concatenate((R, T[:, None]), axis=1)
            RT_list.append(RT)

        return RT_list

    def choose_index(self, pcld_index):
        """
        pcld_index: an array: 1 X N
        """
        if len(pcld_index) < 400:
            return None

        pcld_index_id = np.array([i for i in range(len(pcld_index))])

        if len(pcld_index_id) > self.data_config.n_sample_points:
            c_mask = np.zeros(len(pcld_index_id), dtype=int)
            c_mask[:self.data_config.n_sample_points] = 1
            np.random.shuffle(c_mask)
            pcld_index_id = pcld_index_id[c_mask.nonzero()]
        else:
            pcld_index_id = np.pad(pcld_index_id, (0, self.data_config.n_sample_points - len(pcld_index_id)), "wrap")

        return pcld_index_id

    def get_offst(self, RT_list, pcld_xyz, mask_selected):
        # TODO check difference with get_offst in pvn3d.data.utils and remove copied code
        """
            num_obj is == num_classes by default
        """
        RTs = np.zeros((self.data_config.n_objects, 3, 4))
        kp3ds = np.zeros((self.data_config.n_objects, self.data_config.n_key_points, 3))
        ctr3ds = np.zeros((self.data_config.n_objects, self.data_config.n_ctr_points, 3))
        kpts_targ_ofst = np.zeros((self.data_config.n_sample_points, self.data_config.n_key_points, 3))
        ctr_targ_ofst = np.zeros((self.data_config.n_sample_points, self.data_config.n_ctr_points, 3))

        for i in range(len(RT_list)):
            RTs[i] = RT_list[i][0]  # assign RT to each object
            r = RT_list[i][0][:, :3]
            t = RT_list[i][0][:, 3]

            if self.cls_type == 'all':
                cls_index = np.where(mask_selected == self.data_config.mask_value_array[i])[0]
            else:
                cls_index = np.where(mask_selected > 0)[0]

            ctr = self.centers[i][:, None]
            ctr = np.dot(ctr.T, r.T) + t  # ctr [1 , 3]
            ctr3ds[i] = ctr

            ctr_offset_all = []

            for c in ctr:
                ctr_offset_all.append(np.subtract(c, pcld_xyz))

            ctr_offset_all = np.array(ctr_offset_all).transpose((1, 0, 2))
            ctr_targ_ofst[cls_index, :, :] = ctr_offset_all[cls_index, :, :]

            kpts = self.key_points[i]
            kpts = np.dot(kpts, r.T) + t  # [8, 3]
            kp3ds[i] = kpts

            kpts_offset_all = []

            for kp in kpts:
                kpts_offset_all.append(np.subtract(kp, pcld_xyz))

            kpts_offset_all = np.array(kpts_offset_all).transpose((1, 0, 2))  # [kp, np, 3] -> [nsp, kp, 3]
            kpts_targ_ofst[cls_index, :, :] = kpts_offset_all[cls_index, :, :]

        return ctr_targ_ofst, kpts_targ_ofst

    @CachedProperty
    def centers(self):

        centers_list = []

        if self.cls_type == 'all':
            cls_types = [key for key, _ in self.data_config.all_mask_ids.items()]
        else:
            cls_types = [self.cls_type]

        for cls in cls_types:
            corners_dir = os.path.join(self.data_config.kps_dir, cls, 'corners.txt')
            corners = np.loadtxt(corners_dir, dtype=np.float32)
            center = corners.mean(0)
            centers_list.append(center)

        return centers_list

    @CachedProperty
    def key_points(self):

        if self.data_config.n_key_points == 8:
            kp_type = 'farthest'
        else:
            kp_type = 'farthest{}.'.format(self.data_config.n_key_points)

        keypoints_list = []

        if self.cls_type == 'all':
            cls_types = [key for key, _ in self.data_config.all_mask_ids.items()]
        else:
            cls_types = [self.cls_type]

        for cls in cls_types:
            kps_dir = os.path.join(self.data_config.kps_dir, "{}/{}.txt".format(cls, kp_type))
            kps = np.loadtxt(kps_dir, dtype=np.float32)
            keypoints_list.append(kps)

        return keypoints_list

    def get_data_preprocessed(self, mode, index):

        get_data = lambda name: np.load(os.path.join(self.data_config.preprocessed_folder, name, f"{index:06}.npy"))

        rgb = get_data("rgb").astype(np.float32)
        crop_down_factor = get_data("crop_factor")
        pcld_xyz_rgb_nm = get_data("pcld_xyz_rgb_nm")
        sampled_index = get_data("sampled_index")
        crop_down_factor = [1]  # todo fix hard code

        if self.data_config.online_rgb_aug:
            # todo augmented twice, might be risky, we intentionally weaken the augmentation effect

            self.p_sat = 0.5 / 2
            self.p_bright = 0.5 / 2
            self.p_noise = 0.1 / 2
            self.p_hue = 0.03 / 2
            self.p_contr = 0.5 / 2
            rgb = self.augment_rgb(rgb.copy() * (1 / 255.))
            rgb_feats = rgb.reshape((-1, 3))[sampled_index]
            pcld_xyz_rgb_nm[:, 3:6] = rgb_feats

        if mode == "test":
            if self.use_pvn_kp:
                return rgb, pcld_xyz_rgb_nm[:, :3], pcld_xyz_rgb_nm[:, 3:], sampled_index, crop_down_factor, [
                    self.obj_cls_id]
            else:
                return rgb, pcld_xyz_rgb_nm[:, :3], pcld_xyz_rgb_nm[:, 3:], sampled_index, crop_down_factor
        else:

            label_list = get_data("label_list")
            mask_label = get_data("mask_label")
            sampled_index = get_data("sampled_index")
            ctr_targ_offst = get_data("ctr_targ_offst")
            kpts_targ_offst = get_data("kpts_targ_offst")

            if self.use_pvn_kp:

                kp_cp_target_path = os.path.join(
                    self.data_config.root, '{:02}/preprocessed/kp_cp_target/{:06}.bin'.format(self.cls_id, index))

                kp_cp_target = pickle.load(open(kp_cp_target_path, 'rb'))

                return rgb, pcld_xyz_rgb_nm[:, :3], pcld_xyz_rgb_nm[:, 3:], sampled_index, label_list, \
                       kpts_targ_offst, ctr_targ_offst, mask_label, crop_down_factor, [self.obj_cls_id], kp_cp_target
            else:
                return rgb, pcld_xyz_rgb_nm[:, :3], pcld_xyz_rgb_nm[:, 3:], sampled_index, label_list, \
                       kpts_targ_offst, ctr_targ_offst, mask_label, crop_down_factor

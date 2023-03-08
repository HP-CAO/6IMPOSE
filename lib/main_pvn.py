import os
import tqdm
import cv2
import numpy as np
import tensorflow as tf
from typing import List
from lib.data.dataset_params import PvnDatasetParams
from lib.data.dataset_settings import DatasetSettings
from lib.monitor.visualizer import dpt2heat, project2img, vis_accuracy, vis_feature_maps_image, vis_gt_kpts, \
    vis_offset_value, vis_pre_kpts, vis_pts_semantics
from lib.net.utils import match_choose
from lib.trainer.trainer_pvn import TrainerPvnParams, TrainerPvn3d
from lib.net.pvn3d_adp import Pvn3dNet, Pvn3dNetParams
from lib.net.pprocessnet import PProcessParams, InitialPoseModel
from lib.monitor.monitor import MonitorParams
from lib.data.linemod.linemod_settings import LineModSettings
from lib.network import Network
from lib.params import NetworkParams, Networks
from lib.data.utils import expand_dim, get_mesh_diameter, load_mesh, get_diameter_from_mesh
from lib.monitor.evaluator import EvalResult, EvalResultType, cal_accuracy, cal_add_dis, cal_adds_dis, cal_auc
from lib.utils import ensure_fd
from tensorflow.keras.layers import MaxPool2D


class MainPvn3dParams(NetworkParams):
    def __init__(self):
        self.network = Networks.pvn3d
        self.dataset_params = PvnDatasetParams()
        self.trainer_params = TrainerPvnParams()
        self.pvn3d_params = Pvn3dNetParams()
        self.monitor_params = MonitorParams()
        self.pprocess_params = PProcessParams()


class MainPvn3d(Network):
    params: MainPvn3dParams
    trainer: TrainerPvn3d
    model: Pvn3dNet

    def __init__(self, params: MainPvn3dParams):
        super().__init__(params)

        self.cls_type = self.params.dataset_params.cls_type
        self.crop_image = self.params.dataset_params.crop_image
        self.use_preprocessed = self.params.dataset_params.use_preprocessed
        self.augment_per_image = self.params.dataset_params.augment_per_image

        self.is_binary = self.cls_type != 'all'
        self.initial_pose_model = InitialPoseModel(n_point_candidate=self.params.pprocess_params.n_point_candidate)

    def performance_evaluation(self, epoch):

        eval_results = self.performance_eval(self.data_config)
        self.monitor.log_eval_results(epoch, eval_results, name="val")

        if self.params.monitor_params.sim2real_eval:
            sim2real_data_config = LineModSettings(
                'linemod_crop', self.cls_type, self.use_preprocessed, self.crop_image,
                self.params.dataset_params.size_all, self.params.dataset_params.train_size,
                augment_per_image=0)

            test_index = np.loadtxt(sim2real_data_config.test_txt_path).astype(np.int)

            # overwrite with sim2real score
            self.monitor.log_eval_results(epoch, self.performance_eval(sim2real_data_config, test_inds=test_index),
                                          name="sim2real")

    def pre_training(self):
        if self.strategy is not None:
            with self.strategy.scope():
                self.performance_evaluation(-1)
        else:
            self.performance_evaluation(-1)

    def initial_model(self, weights_path=None):
        self.model = Pvn3dNet(self.params.pvn3d_params,
                              rgb_input_shape=self.data_config.rgb_input_shape,
                              num_kpts=self.data_config.n_key_points,
                              num_cls=self.data_config.n_classes,
                              num_cpts=self.data_config.n_ctr_points,
                              dim_xyz=self.data_config.dim_pcld_xyz)

        if weights_path is not None:
            self.model.load_weights(self.params.monitor_params.weights_path)
            print('Pre-trained model loaded successfully')


    def initial_trainer_and_model(self):
        self.trainer = TrainerPvn3d(self.params.trainer_params)

        # self.model = Pvn3dNet(self.params.pvn3d_params,
        #                       rgb_input_shape=self.data_config.rgb_input_shape,
        #                       num_kpts=self.data_config.n_key_points,
        #                       num_cls=self.data_config.n_classes,
        #                       num_cpts=self.data_config.n_ctr_points,
        #                       dim_xyz=self.data_config.dim_pcld_xyz)
        self.initial_model(self.params.monitor_params.weights_path)

        # call model with fake data to build
        n_samples = self.params.pvn3d_params.point_net2_params.n_sample_points
        h, w = self.data_config.rgb_input_shape[:2]
        bs = 5
        pcld_xyz = tf.zeros((bs, n_samples, 3))
        pcld_feats = tf.zeros((bs, n_samples, 6))
        rgb = tf.zeros((bs, h, w, 3))
        sampled_index = tf.zeros((bs, n_samples), tf.int32)
        crop_factor = tf.ones((bs,), tf.int32)
        self.forward_pass((rgb, pcld_xyz, pcld_feats, sampled_index, crop_factor))



    @tf.function
    def train_step(self, inputs):
        rgb, pcld_xyz, pcld_feats, sampled_index, label, kp_targ_ofst, ctr_targ_ofst, mask_label, crop_factor \
            = inputs

        n_samples = self.params.pvn3d_params.point_net2_params.n_sample_points

        sample_inds = tf.random.shuffle(tf.range(12288))[:n_samples]

        pcld_xyz = tf.gather(pcld_xyz, sample_inds, axis=1)
        pcld_feats = tf.gather(pcld_feats, sample_inds, axis=1)
        sampled_index = tf.gather(sampled_index, sample_inds, axis=1)
        label = tf.gather(label, sample_inds, axis=1)
        kp_targ_ofst = tf.gather(kp_targ_ofst, sample_inds, axis=1)
        ctr_targ_ofst = tf.gather(ctr_targ_ofst, sample_inds, axis=1)
        mask_label = tf.gather(mask_label, sample_inds, axis=1)

        input_data = [rgb, pcld_xyz, pcld_feats, sampled_index, crop_factor]

        with tf.GradientTape() as Tape:
            kp_pre_ofst, seg_pre, cp_pre_ofst = self.forward_pass(input_data, training=True)

            if self.params.monitor_params.if_model_summary:
                self.monitor.model_graph_summary(self.model)
                print('model summary saved')
                self.model.summary()
                self.params.monitor_params.if_model_summary = False

            loss, loss_kp, loss_seg, loss_cp = self.trainer.loss_fn_pvn3d(kp_pre_ofst=kp_pre_ofst,
                                                                          kp_targ_ofst=kp_targ_ofst,
                                                                          seg_pre=seg_pre,
                                                                          cp_pre_ofst=cp_pre_ofst,
                                                                          ctr_targ_ofst=ctr_targ_ofst,
                                                                          label=label,
                                                                          mask_label=mask_label,
                                                                          binary_loss=self.is_binary)

        grads = Tape.gradient(loss, self.model.trainable_variables)
        self.trainer.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return {'loss': loss, 'loss_kp': loss_kp, 'loss_seg': loss_seg, 'loss_cp': loss_cp}

    @tf.function
    def val_step(self, inputs):

        rgb, pcld_xyz, pcld_feats, sampled_index, label, kp_targ_ofst, ctr_targ_ofst, mask_label, crop_factor = inputs

        n_samples = self.params.pvn3d_params.point_net2_params.n_sample_points

        sample_inds = tf.random.shuffle(tf.range(12288))[:n_samples]

        pcld_xyz = tf.gather(pcld_xyz, sample_inds, axis=1)
        pcld_feats = tf.gather(pcld_feats, sample_inds, axis=1)
        sampled_index = tf.gather(sampled_index, sample_inds, axis=1)
        label = tf.gather(label, sample_inds, axis=1)
        kp_targ_ofst = tf.gather(kp_targ_ofst, sample_inds, axis=1)
        ctr_targ_ofst = tf.gather(ctr_targ_ofst, sample_inds, axis=1)
        mask_label = tf.gather(mask_label, sample_inds, axis=1)

        input_data = [rgb, pcld_xyz, pcld_feats, sampled_index, crop_factor]

        kp_pre_ofst, seg_pre, cp_pre_ofst = self.forward_pass(input_data, training=False)

        loss, loss_kp, loss_seg, loss_cp = self.trainer.loss_fn_pvn3d(kp_pre_ofst=kp_pre_ofst,
                                                                      kp_targ_ofst=kp_targ_ofst,
                                                                      seg_pre=seg_pre,
                                                                      cp_pre_ofst=cp_pre_ofst,
                                                                      ctr_targ_ofst=ctr_targ_ofst,
                                                                      label=label,
                                                                      mask_label=mask_label,
                                                                      binary_loss=self.is_binary)

        return {'loss': loss, 'loss_kp': loss_kp, 'loss_seg': loss_seg, 'loss_cp': loss_cp}

    @tf.function
    def forward_pass(self, input_data, training=False):
        """
        params:
                rgb [bs, H, W, 3]
                pcld_xyz [bs, npts, 3]
                pcld_feats [bs, npts, 6]
                sampled_index [bs, npts]
        output:
            kpts_pre [bs, npts, n_kpts, 3]
            cpts_pre [bs, npts, n_cpts, 3]
            segs_pre [bs, npts, n_cls]
        """
        rgb, pcld_xyz, pcld_feats, sampled_index, crop_factor = input_data

        pcld = [pcld_xyz, pcld_feats]
        inputs = [pcld, tf.cast(sampled_index, tf.int32), rgb, tf.cast(crop_factor, tf.int32)]

        kp_pre_ofst, seg_pre, cp_pre_ofst = self.model(inputs, training=training)

        return kp_pre_ofst, seg_pre, cp_pre_ofst

    def performance_eval(self, data_config: DatasetSettings, test_inds=None) -> List[EvalResult]:
        dataset_params = self.params.dataset_params
        random_eval_number = 2
        worst_case_number = 2

        mesh_path = data_config.mesh_paths[self.cls_type]
        mesh_points = load_mesh(mesh_path, scale=data_config.mesh_scale, n_points=500)
        obj_id = data_config.obj_dict[data_config.cls_type]

        if dataset_params.data_name == "linemod" or dataset_params.data_name.endswith("linemod"):
            # here we use the computed diameter from https://github.com/j96w/DenseFusion
            mesh_info_path = os.path.join(data_config.mesh_dir, "model_info.yml")
            mesh_diameter = get_mesh_diameter(mesh_info_path, obj_id) * data_config.mesh_scale  # from mm to m
        else:
            mesh_diameter = get_diameter_from_mesh(mesh_points)

        kpts_path = os.path.join(data_config.kps_dir, "{}/farthest.txt".format(data_config.cls_type))
        corner_path = os.path.join(data_config.kps_dir, "{}/corners.txt".format(data_config.cls_type))
        key_points = np.loadtxt(kpts_path)
        center = [np.loadtxt(corner_path).mean(0)]
        mesh_kpts = np.concatenate([key_points, center], axis=0)

        id_list = []
        add_score_list = []
        adds_score_list = []
        gt_pre_list = []
        gt_pre_kpt_list = []
        gt_pre_kpt_offset_list = []
        feature_maps_list = []
        numpy_preprocessed_folder = os.path.join(data_config.preprocessed_folder, 'numpy')

        if test_inds is None:
            n_datapoints = len(os.listdir(os.path.join(numpy_preprocessed_folder, 'rgb')))
            test_inds = np.arange(0, n_datapoints)

        # sample visualization predictions
        random_number = np.random.randint(low=0, high=len(test_inds), size=random_eval_number)
        random_ids = test_inds[random_number]

        for i in tqdm.tqdm(test_inds):
            RT_gt, rgb, dpt, cam_intrinsic, crop_index, pcld_xyz, pcld_feats, sampled_index, crop_factor = \
                self.get_data_preprocessed(numpy_preprocessed_folder, i)
            xy_offset = crop_index[:2]

            n_samples = self.model.num_pts
            sample_inds = tf.random.shuffle(tf.range(12288))[:n_samples]
            pcld_xyz = tf.gather(pcld_xyz, sample_inds)
            pcld_feats = tf.gather(pcld_feats, sample_inds)
            sampled_index = tf.gather(sampled_index, sample_inds)

            input_data = expand_dim(rgb, pcld_xyz, pcld_feats, sampled_index, crop_factor)

            kp_pre_ofst, seg_pre, cp_pre_ofst = self.forward_pass(input_data, training=False)
            R, t, kpts_voted = self.initial_pose_model(
                [input_data[1], kp_pre_ofst, cp_pre_ofst, seg_pre, tf.expand_dims(mesh_kpts, 0)], training=False)
            Rt_pre = tf.concat([R[0], tf.expand_dims(t[0], -1)], axis=-1).numpy()
            kpts_voted = kpts_voted[0].numpy()

            add_score = cal_add_dis(mesh_points, Rt_pre, RT_gt)
            add_score_list.append(add_score)
            adds_score = cal_adds_dis(mesh_points, Rt_pre, RT_gt)
            adds_score_list.append(adds_score)

            h, w = rgb.shape[:2]
            rgb = cv2.resize(rgb, (w * crop_factor, h * crop_factor), interpolation=cv2.INTER_NEAREST)

            rgb = rgb.astype(np.uint8)
            RT_gt = np.array(RT_gt).squeeze()

            if i in random_ids:
                # save prediction

                label_info, kpts_targ_offst, ctr_targ_offst = self.get_data_preprocessed_gt(numpy_preprocessed_folder, i)

                label_list, _ = label_info
                label_list = label_list[sample_inds]
                kpts_targ_offst = kpts_targ_offst[sample_inds]
                ctr_targ_offst = ctr_targ_offst[sample_inds]
                label_segs = np.argmax(label_list, axis=-1).squeeze()

                img_gt_pts_seg = vis_pts_semantics(rgb.copy(), label_segs, sampled_index)
                img_gt_kpts, pts_2d_gt = vis_gt_kpts(rgb.copy(), mesh_kpts, RT_gt, xy_offset, cam_intrinsic)
                img_gt_offset = vis_offset_value(sampled_index, label_segs, [kpts_targ_offst],
                                                 [ctr_targ_offst], pts_2d_gt, img_shape=rgb.shape)

                # log prediction
                img_pre_proj = project2img(mesh_points, Rt_pre, rgb.copy(), cam_intrinsic,
                                           data_config.camera_scale, [0, 0, 255], xy_offset)

                img_pre_proj = cv2.putText(img_pre_proj, "ADD: {:.5f} ADDS: {:.5f}".format(add_score, adds_score),
                                           (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0, 255, 255), 1)

                pre_segs = np.argmax(seg_pre.numpy(), axis=-1).squeeze()

                img_pre_pts_seg = vis_pts_semantics(rgb.copy(), pre_segs, sampled_index)
                img_pre_ktps, pts_2d_pre = vis_pre_kpts(rgb.copy(), kpts_voted, xy_offset, cam_intrinsic)
                img_pre_offset = vis_offset_value(sampled_index, pre_segs, kp_pre_ofst.numpy(),
                                                  cp_pre_ofst.numpy(), pts_2d_gt, pts_2d_pre, rgb.shape)

                try:
                    feature_map_image = self.get_feature_maps_image(input_data)
                except:
                    feature_map_image = np.ones_like([img_pre_pts_seg])

                feature_maps_list.append(feature_map_image)
                gt_pre_list.append([img_gt_pts_seg, img_pre_pts_seg, img_pre_proj])

                if dpt is not None:
                    img_gt_depth = dpt2heat(dpt.copy())
                    gt_pre_list.append(img_gt_depth)

                gt_pre_kpt_offset_list.append([img_gt_offset, img_pre_offset])
                gt_pre_kpt_list.append([img_gt_kpts, img_pre_ktps])

            id_list.append(i)

        add_mean = np.array(add_score_list).mean()
        adds_mean = np.array(adds_score_list).mean()
        add_auc = cal_auc(add_score_list, max_dis=0.1)
        add_accuracy = cal_accuracy(add_score_list, dis_threshold=0.1 * mesh_diameter)
        adds_auc = cal_auc(adds_score_list, max_dis=0.1)
        adds_accuracy = cal_accuracy(adds_score_list, dis_threshold=0.1 * mesh_diameter)

        add_accuracy_image = vis_accuracy(add_score_list, threshold=0.1 * mesh_diameter, name="ADD")
        adds_accuracy_image = vis_accuracy(adds_score_list, threshold=0.1 * mesh_diameter, name="ADDS")
        accuracy_image_list = [add_accuracy_image, adds_accuracy_image]

        # worst_index = np.argpartition(add_score_list, -1 * worst_case_number)[-1 * worst_case_number:]
        # worst_ids = np.array(id_list)[worst_index]

        res = [
            EvalResult('/add_mean', EvalResultType.scalar, add_mean),
            EvalResult('/adds_mean', EvalResultType.scalar, adds_mean),
            EvalResult('/add_auc', EvalResultType.scalar, add_auc),
            EvalResult('/adds_auc', EvalResultType.scalar, adds_auc),
            EvalResult('/add_accuracy', EvalResultType.scalar, add_accuracy),
            EvalResult('/adds_accuracy', EvalResultType.scalar, adds_accuracy)
        ]
        res.extend(
            [EvalResult(f"_{i}_gt_pre/gt_pre", EvalResultType.images, gt_pre) for i, gt_pre in enumerate(gt_pre_list)])
        res.extend([EvalResult(f"_{i}_gt_pre/gt_pre_kpts", EvalResultType.images, gt_pre_kpts) for i, gt_pre_kpts in
                    enumerate(gt_pre_kpt_list)])
        res.extend([EvalResult(f"_{i}_feature_maps/feature_maps", EvalResultType.images, feat_maps) for i, feat_maps in
                    enumerate(feature_maps_list)])

        # for i in range(len(gt_pre_kpt_offset_list)):
        # for j in range(9):
        # gt = gt_pre_kpt_offset_list[i][0][j]
        # pre = gt_pre_kpt_offset_list[i][1][j]
        # res.append(EvalResult(f"_{i}_gt_pre/gt_pre_offset_", EvalResultType.images, np.array([gt, pre]))

        res.extend([EvalResult(f"Accuracy_plot_{i}", EvalResultType.images, acc_img) for i, acc_img in
                    enumerate(accuracy_image_list)])

        return res

    @staticmethod
    def get_data_preprocessed(path, index):
        get_data = lambda name: np.load(os.path.join(path, name, f"{index:06}.npy"))
        rgb = get_data('rgb').astype(np.float32)
        rt = get_data('RT')
        if os.path.isfile(os.path.join(path, 'depth', f"{index:06}.npy")):
            depth = get_data('depth')
        else:
            depth = None
        cam_intrinsic = get_data('K')
        crop_index = get_data('crop_index')
        pcld_xyz = get_data('pcld_xyz')
        pcld_feats = get_data('pcld_feats')
        sampled_index = get_data('sampled_index')
        crop_factor = get_data('crop_factor')

        return rt, rgb, depth, cam_intrinsic, crop_index, pcld_xyz, pcld_feats, sampled_index, crop_factor

    @staticmethod
    def get_data_preprocessed_gt(path, index):
        get_data = lambda name: np.load(os.path.join(path, name, f"{index:06}.npy"))

        label_list = get_data('labels')
        mask_label = get_data('mask_label')
        label_info = [label_list, mask_label]
        kpts_targ_offst = get_data('kpts_targ_offst')
        ctr_targ_offst = get_data('ctr_targ_offst')

        return label_info, kpts_targ_offst, ctr_targ_offst

    def get_feature_maps_image(self, input_data, num_to_vis=16):
        rgb, pcld_xyz, pcld_feats, sampled_index, crop_factor = input_data
        resnet_features = self.model.resnet_model(rgb)
        tf_rgb_features = self.model.psp_model(resnet_features)
        rgb_features = tf_rgb_features.numpy().squeeze()
        tf_pointnet_features = self.model.pointnet2_model([pcld_xyz, pcld_feats])  # bs, n_pts, c

        h, w, c_rgb = rgb_features.shape
        pcld_maps = np.ones(shape=(h * w, c_rgb)) * 0.0
        pointnet_features = tf_pointnet_features.numpy().squeeze()
        pcld_maps[sampled_index] = pointnet_features
        pcld_maps = np.reshape(pcld_maps, newshape=(1, h, w, -1))  # h, w, c
        pcld_maps_ave_pooling = MaxPool2D(pool_size=(2, 2), strides=1, padding='same')(pcld_maps).numpy().squeeze()

        rgb_emb = match_choose(tf_rgb_features, [sampled_index])
        tf_feats_fused = self.model.dense_fusion_model([rgb_emb, tf_pointnet_features], training=False)
        # only visualize first c_rgb maps, since there are too many
        feats_fused = tf_feats_fused.numpy().squeeze()
        fused_maps = np.ones(shape=(h * w, c_rgb)) * 0.0
        fused_maps[sampled_index] = feats_fused[:, :c_rgb]
        fused_maps = np.reshape(fused_maps, newshape=(1, h, w, -1))  # h, w, c

        fused_maps_ave_pooling = MaxPool2D(pool_size=(2, 2), strides=1, padding='same')(fused_maps).numpy().squeeze()

        feats_list = [rgb_features[:, :, :num_to_vis], pcld_maps_ave_pooling[:, :, :num_to_vis],
                      fused_maps_ave_pooling[:, :, :num_to_vis]]

        feats_array = np.concatenate(feats_list, axis=-1)
        image = vis_feature_maps_image(feats_array)
        return image

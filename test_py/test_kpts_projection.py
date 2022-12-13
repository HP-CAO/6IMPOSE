import json
import os
import pickle
import numpy as np
from tqdm import tqdm
from lib.monitor.visualizer import project2img
import cv2
from lib.utils import ensure_fd
from lib.data.utils import get_pointxyz_unity


def check_kpts_on_mesh(kpts, mesh_points):
    """
    Calculate the minimum of the distance between the kpts and mesh_points
    :param kpts:  [8, 3]
    :param mesh_points: [npts, 3]
    """

    for i, kp in enumerate(kpts):
        diff_xyz = np.abs(np.subtract(mesh_points, kp))
        mean_xyz = np.mean(diff_xyz, axis=1)
        points_min = np.min(mean_xyz)
        print("The minimum distance for {} point is {}".format(i, points_min))


def project_kpts_test(file_root, mesh_kp_cp):
    is_flip = False
    path_json_label = os.path.join(file_ori_root, "label.json")
    gt_meta = json.loads(open(path_json_label, "r").read())
    intrinsic_matrix = np.array([[1870.6150197846725, 0, 1920.0],
                                [0, 1870.6148721743875, 1080.0],
                                [0, 0, 1]])

    rgb_folder = os.path.join(file_root, 'preprocessed/rgb')
    crop_info_folder = os.path.join(file_root, "preprocessed/crop_info")

    if is_flip: # flip x-value due to the left-hand system in the unity
        mesh_kp_cp[:, 0] *= -1

    for image_id, label_info in tqdm(gt_meta.items()):

        rt = None

        for i in range(len(label_info)):

            if label_info[i]['obj_id'] == str(cls_id).rjust(2, "0"):
                bbox = np.array(label_info[i]["obj_bb"])
                rt = np.array(label_info[i]['cam_Rt_m2c']).reshape(4, 4)
                break

        crop_info_path = os.path.join(crop_info_folder, '{}.bin'.format(image_id))
        crop_info = pickle.load(open(crop_info_path, "rb"))
        crop_index = crop_info[0:2]
        rt = rt[:3, :]
        bgr = cv2.imread(os.path.join(rgb_folder, "{}.png".format(image_id)))
        bgr = project2img(mesh_kp_cp, rt, bgr, intrinsic_matrix,
                          1, (0, 0, 255), crop_index)

        img_save_path = os.path.join(result_save_folder, "kpts_project_test_{}.png".format(image_id))
        cv2.imwrite(img_save_path, bgr)


if __name__ == '__main__':

    file_root = './dataset/unity/unity/img_crop/06/'
    file_ori_root = './dataset/unity/unity/img_ori/06/'
    result_save_folder = "./kpts_project_test_no_flip"
    ensure_fd(result_save_folder)
    cls_id = 6
    corners_dir = os.path.join(file_root, '../../../unity_obj_kpts/obj_{}/corners.txt'.format(cls_id))
    corners = np.loadtxt(corners_dir, dtype=np.float32)
    kps_dir = os.path.join(file_root, '../../../unity_obj_kpts/obj_{}/farthest.txt'.format(cls_id))
    kpts = np.loadtxt(kps_dir, dtype=np.float32)
    center = corners.mean(0)[np.newaxis, :]
    mesh_kp_cp = np.concatenate([kpts, center], axis=0)
    mesh_points = get_pointxyz_unity(cls_id)
    mesh_points[:, 0] *= -1
    check_kpts_on_mesh(kpts, mesh_points)









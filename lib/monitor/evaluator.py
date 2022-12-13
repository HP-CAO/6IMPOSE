import sys
from collections import Counter
import numpy as np
import tqdm
import cv2
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from lib.geometry.geometry import project_p3d
from lib.monitor.visualizer import draw_p2ds, vis_pts_semantics
from lib.data.utils import expand_dim


class EvalResultType:
    images = 'images'  # list of images
    scalar = 'scalar'  # one scalar


class EvalResult:
    name: str
    type: EvalResultType
    data: None

    def __init__(self, name, type, data):
        self.name = name
        self.type = type
        self.data = data


def cal_distance_pre_gt(rt_pre, rt_gt, mesh_pts):
    """
    calculate the points-wise distance between the transformed 3D points with rt_pre and rt_gt
    :param rt_pre: the predicted rt in batch [bs, 3, 4]
    :param rt_gt: the ground truth rt in batch [bs, 3, 4]
    :param mesh_pts: the points sampled from the mesh_pts [n_pts, 3]
    :return: distance tensor [bs, n_pts, 3]
    """

    rt_pre_perm = np.transpose(rt_pre[:, :, :3], axes=(0, 2, 1))
    rt_gt_perm = np.transpose(rt_gt[:, :, :3], axes=(0, 2, 1))
    pts_with_rt_pre = np.dot(mesh_pts, rt_pre_perm) + rt_pre[:, :, 3]  # [bs, n_pts, 3]
    pts_with_rt_gt = np.dot(mesh_pts, rt_gt_perm) + rt_gt[:, :, 3]  # [bs, n_pts, 3]
    distance = np.linalg.norm(np.subtract(pts_with_rt_gt, pts_with_rt_pre))
    return distance


def cal_adds_dis(cls_ptsxyz, pred_pose, gt_pose):
    pred_pts = np.dot(cls_ptsxyz.copy(), pred_pose[:, :3].T) + pred_pose[:, 3]
    gt_pts = np.dot(cls_ptsxyz.copy(), gt_pose[:, :3].T) + gt_pose[:, 3]
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(gt_pts)
    distances, _ = neigh.kneighbors(pred_pts, return_distance=True)
    return np.mean(distances)


def cal_add_dis(cls_ptsxyz, pred_pose, gt_pose):
    pred_pts = np.dot(cls_ptsxyz.copy(), pred_pose[:, :3].T) + pred_pose[:, 3]
    gt_pts = np.dot(cls_ptsxyz.copy(), gt_pose[:, :3].T) + gt_pose[:, 3]
    mean_dist = np.mean(np.linalg.norm(pred_pts - gt_pts, axis=-1))
    return mean_dist


def single_frame_eval(cls_ptsxyz, pred_pose, gt_pose, rgb, cam_scale, cam_k, xy_ofst=(0, 0)):
    x1, y1 = xy_ofst
    pred_pts = np.dot(cls_ptsxyz.copy(), pred_pose[:, :3].T) + pred_pose[:, 3]
    gt_pts = np.dot(cls_ptsxyz.copy(), gt_pose[:, :3].T) + gt_pose[:, 3]
    add_dis = np.mean(np.linalg.norm(pred_pts - gt_pts, axis=-1))
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(gt_pts)
    distances, _ = neigh.kneighbors(pred_pts, return_distance=True)
    adds_dis = np.mean(distances)

    gt_mesh_projected = project_p3d(gt_pts, cam_scale=cam_scale, K=cam_k)
    pre_mesh_projected = project_p3d(pred_pts, cam_scale=cam_scale, K=cam_k)

    gt_mesh_projected[:, 0] -= x1  # compensate the bounding box offset
    gt_mesh_projected[:, 1] -= y1

    pre_mesh_projected[:, 0] -= x1
    pre_mesh_projected[:, 1] -= y1

    img_gt_project = draw_p2ds(rgb, gt_mesh_projected, r=1, color=[0, 255, 0])
    img_pre_projected = draw_p2ds(rgb, pre_mesh_projected, r=1, color=[255, 0, 0])

    return add_dis, adds_dis, img_gt_project, img_pre_projected

def cal_accuracy(dis, dis_threshold):
    D = np.array(dis)
    D[np.where(D > dis_threshold)] = 0
    non_zeros = np.count_nonzero(D)
    acc = non_zeros / len(D)
    return acc * 100


def batch_add(distance):
    """
    :return: average closest point distance
    """
    average_distance = np.mean(distance)
    return average_distance


def cal_auc(add_dis, max_dis=0.1):
    D = np.array(add_dis)
    D[np.where(D > max_dis)] = np.inf
    D = np.sort(D)
    n = len(add_dis)
    acc = np.cumsum(np.ones((1, n)), dtype=np.float32) / n
    aps = VOCap(D, acc)
    return aps * 100


def VOCap(rec, prec):
    idx = np.where(rec != np.inf)
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx]
    prec = prec[idx]
    mrec = np.array([0.0] + list(rec) + [0.1])
    mpre = np.array([0.0] + list(prec) + [prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i - 1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i]) * 10
    return ap


@tf.function
def pcld_net_forward_pass(input_data, pcld_model, training=False):
    """input_data : pcld_xyz, pcld_feats """
    kp_pre_ofst, seg_pre, cp_pre_ofst = pcld_model(input_data, training=training)
    return kp_pre_ofst, seg_pre, cp_pre_ofst


@tf.function
def pt2_forward_pass(input_data, pt2_model, training=False):
    """
    params:
            input_data = [pcld_xyz, pcld_feats]
    output:
           seg_pre [bs, npts, 2]
    """

    seg_pre = pt2_model(input_data, training=training)
    return seg_pre


def get_coco_metric(gt_bboxes, pred_bboxes):
    return np.mean([get_pascalvoc_metrics(gt_bboxes, pred_bboxes, iou_threshold=thr) for thr in
                    np.arange(0.5, 1.0, 0.05)])  # 10 thresholds


def calculate_ap_every_point(rec, prec):
    mrec = []
    mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)
    mpre = []
    mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]


# from https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/src/evaluators/pascal_voc_evaluator.py
def get_pascalvoc_metrics(gt_boxes, det_boxes, iou_threshold=0.5):
    # coor = np.array(bbox[:4], dtype=np.int32)
    # score = bbox[4]
    # class_ind = int(bbox[5]) if cls_type == 'all' else 0
    # Get classes of all bounding boxes separating them by classes
    gt_classes_only = []
    classes_bbs = v = {'gt': [], 'det': []}
    for bb in gt_boxes:
        gt_classes_only.append(1)
        classes_bbs['gt'].append(bb)
    gt_classes_only = list(set(gt_classes_only))
    for bb in det_boxes:
        classes_bbs['det'].append(bb)

    # Get classes of all bounding boxes separating them by classes
    npos = len(v['gt'])
    # sort detections by decreasing confidence
    dects = [a for a in sorted(v['det'], key=lambda bb: bb['conf'], reverse=True)]
    TP = np.zeros(len(dects))
    FP = np.zeros(len(dects))
    # create dictionary with amount of expected detections for each image
    detected_gt_per_image = Counter([bb['image_index'] for bb in gt_boxes])
    for key, val in detected_gt_per_image.items():
        detected_gt_per_image[key] = np.zeros(val)

    # Loop through detections
    for idx_det, det in enumerate(dects):
        img_det = det['image_index']

        # Find ground truth image
        gt = [gt for gt in classes_bbs['gt'] if gt['image_index'] == img_det]
        # Get the maximum iou among all detectins in the image
        iouMax = sys.float_info.min
        # Given the detection det, find ground-truth with the highest iou
        for j, g in enumerate(gt):
            iou = bboxes_iou(det['coor'][np.newaxis, :], g['coor'][np.newaxis, :])[0]
            if iou > iouMax:
                iouMax = iou
                id_match_gt = j
        # Assign detection as TP or FP
        if iouMax >= iou_threshold:
            # gt was not matched with any detection
            if detected_gt_per_image[img_det][id_match_gt] == 0:
                TP[idx_det] = 1  # detection is set as true positive
                detected_gt_per_image[img_det][
                    id_match_gt] = 1  # set flag to identify gt as already 'matched'
                # print("TP")
            else:
                FP[idx_det] = 1  # detection is set as false positive
                # print("FP")
        # - A detected "cat" is overlaped with a GT "cat" with IOU >= iou_threshold.
        else:
            FP[idx_det] = 1  # detection is set as false positive
            # print("FP")
    # compute precision, recall and average precision
    acc_FP = np.cumsum(FP)
    acc_TP = np.cumsum(TP)
    rec = acc_TP / npos
    prec = np.divide(acc_TP, (acc_FP + acc_TP))
    # Depending on the method, call the right implementation

    [ap, mpre, mrec, ii] = calculate_ap_every_point(rec, prec)

    # add class result in the dictionary to be returned
    ret = {
        'precision': prec,
        'recall': rec,
        'AP': ap,
        'interpolated precision': mpre,
        'interpolated recall': mrec,
        'total positives': npos,
        'total TP': np.sum(TP),
        'total FP': np.sum(FP),
        'iou': iou_threshold,
    }
    return ret['AP'] * 100


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)
    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        # Process 1: Determine whether the number of bounding boxes is greater than 0
        while len(cls_bboxes) > 0:
            # Process 2: Select the bounding box with the highest score according to socre order A
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            # Process 3: Calculate this bounding box A and Remain all iou of the bounding box and remove those
            # bounding boxes whose iou value is higher than the threshold
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def postprocess_boxes(pred_bbox, original_image, input_shape, score_threshold):

    input_h, input_w, _ = input_shape
    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)
    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = original_image.shape[:2]
    resize_ratio = min(input_w / org_w, input_h / org_h)

    dw = (input_w - resize_ratio * org_w) / 2
    dh = (input_h - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # 3. clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)

    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # 4. discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # 5. discard boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)


def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def cal_pts_iou(pre, gt):
    inter = np.logical_and(pre, gt).astype(np.uint8)
    union = np.logical_or(pre, gt).astype(np.uint8)
    iou = np.count_nonzero(inter) / np.count_nonzero(union)

    return iou

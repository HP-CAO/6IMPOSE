import tensorflow as tf
import numpy as np


def match_choose_adp(prediction, choose, crop_down_factor, resnet_input_shape):
    """
    params prediction: feature maps [B, H, W, c] -> [B, H*W, c]
    params crop_down_factor bs,
    params choose: indexes for chosen points [B, n_points]
    return: tensor [B, n_points, c]
    """

    shape = tf.shape(prediction)
    bs = shape[0]
    c = shape[-1]
    prediction = tf.reshape(prediction, shape=(bs, -1, c))
    batch_resnet_shape = tf.repeat(tf.expand_dims(resnet_input_shape, axis=0), repeats=bs, axis=0)  # bs, 2
    crop_down_factor = tf.expand_dims(crop_down_factor, -1)
    image_shape = tf.multiply(batch_resnet_shape, crop_down_factor)  # bs, 2

    feats_inds = map_indices_to_feature_map(choose, resnet_input_shape, image_shape)
    feats_inds = tf.reshape(feats_inds, shape=(bs, -1, 1))
    pre_match = tf.gather_nd(prediction, indices=feats_inds, batch_dims=1)
    return pre_match


def match_choose(prediction, choose):
    """
    params prediction: feature maps [B, H, W, c] -> [B, H*W, c]
    params choose: indexes for chosen points [B, n_points]
    return: tensor [B, n_points, c]
    """
    shape = tf.shape(prediction)
    bs = shape[0]
    c = shape[-1]
    prediction = tf.reshape(prediction, shape=(bs, -1, c))
    choose = tf.reshape(choose, shape=(bs, -1, 1))
    choose = tf.cast(choose, dtype=tf.int32)
    pre_match = tf.gather_nd(prediction, indices=choose, batch_dims=1)
    return pre_match


def bbox_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * tf.math.divide_no_nan(inter_area, union_area)


def bbox_giou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    # Calculate the iou value between the two bounding boxes
    #iou = inter_area / union_area
    iou = tf.math.divide_no_nan(inter_area, union_area)

    # Calculate the coordinates of the upper left corner
    # and the lower right corner of the smallest closed convex surface

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

    # Calculate the area of the smallest closed convex surface C
    enclose_area = enclose[..., 0] * enclose[..., 1]

    # Calculate the GIoU value according to the GioU formula
    #giou = iou - 1.0 * (enclose_area - union_area) / enclose_area
    epsilon = tf.math.divide_no_nan((enclose_area - union_area), enclose_area)
    epsilon = tf.where(tf.math.is_nan(epsilon), 0.0, epsilon)
    giou = iou - 1.0 * epsilon

    return giou


def bbox_ciou(boxes1, boxes2):
    boxes1_coor = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_coor = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left = tf.maximum(boxes1_coor[..., 0], boxes2_coor[..., 0])
    up = tf.maximum(boxes1_coor[..., 1], boxes2_coor[..., 1])
    right = tf.maximum(boxes1_coor[..., 2], boxes2_coor[..., 2])
    down = tf.maximum(boxes1_coor[..., 3], boxes2_coor[..., 3])

    c = (right - left) * (right - left) + (up - down) * (up - down)
    iou = bbox_iou(boxes1, boxes2)

    u = (boxes1[..., 0] - boxes2[..., 0]) * (boxes1[..., 0] - boxes2[..., 0]) + (boxes1[..., 1] - boxes2[..., 1]) * (
            boxes1[..., 1] - boxes2[..., 1])
    d = u / c

    ar_gt = boxes2[..., 2] / boxes2[..., 3]
    ar_pred = boxes1[..., 2] / boxes1[..., 3]

    ar_loss = 4 / (np.pi * np.pi) * (tf.atan(ar_gt) - tf.atan(ar_pred)) * (tf.atan(ar_gt) - tf.atan(ar_pred))
    alpha = ar_loss / (1 - iou + ar_loss + 0.000001)
    ciou_term = d + alpha * ar_loss

    return iou - ciou_term


@tf.function
def map_indices_to_feature_map(indices, resnet_shape, image_shapes):
    """ indices: [b, n_sample_points]
        resnet_shape: (h,w)
        image_shapes: [b, 2]      b x (h,w)
    """
    scales = tf.cast(resnet_shape[0] / image_shapes[:, 0], tf.float32)[..., tf.newaxis]
    rows_inds = tf.cast(tf.floor(tf.cast(indices // image_shapes[:, 1, tf.newaxis], tf.float32) * scales), tf.int32) * \
                resnet_shape[1]
    cols_inds = tf.cast(tf.floor(tf.cast(indices % image_shapes[:, 1, tf.newaxis], tf.float32) * scales), tf.int32)

    return rows_inds + cols_inds

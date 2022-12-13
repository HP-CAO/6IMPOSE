import random
import math
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_addons as tfa
from scipy.spatial.transform import Rotation as R
import tensorflow_addons as tfa

class AugmentSettings:
    """ Set to None to deactivate """
    # --- rgb (Maximum values) ---
    p_sat = 0.15
    p_bright = 0.1
    p_noise = 0.02
    p_hue = 0.02
    p_contr = 0.3
    p_perlin_per_channel = 0.05
    blur_min = 3 # blur kernel size
    blur_max = 7
    sharp_min = 1.0
    sharp_max = 5.0
    
    ops_min = 3  # dont apply all augmentations but between min and max random ops
    ops_max = 5

    random_crop = 0.5

    # --- depth - noise --
    noise_dev_min = 0.0005 #0.00005  # for gaussian noise
    noise_dev_max = 0.0015 #0.0001
    amplitude_min_high = 0.005 # for perlin noise
    amplitude_max_high = 0.008
    amplitude_min_low = 0.001
    amplitude_max_low = 0.005
    perlin_res_low = 0.1  # for perlin frequency
    perlin_res_high = 0.65
    perlin_frequency_std = 0.04
 
    fake_depth_range = 0.2
    plane_elevation = 80./180 * np.pi
    move_background_to_obj = True
    rgb2noise_amplitude = 0 # TODO test this

    edge_range = 2  # shift distance in pixels
    edge_thickness = 4  # pixel width of wobbly edges
    edge_frequency = 0.4  # frequency of wobbly edges

    camera_unit = 0.001  # Kinect V1
    #camera_unit = 0.00025  # Realsense L515

    median_filter = None

    downsample_and_upsample = None # 2 # simulate a lower resolution sensor by down- and upsampling

    cache = {}


def augment_depth(depth):
    """ depth must have batch and channel! [N, H, W, C]
        depth in m!
    """
        # --- per pixel noise ---
    noise_dev_min = AugmentSettings.noise_dev_min
    noise_dev_max = AugmentSettings.noise_dev_max
    if noise_dev_max is not None and noise_dev_min is not None:
        noise_dev = tf.random.uniform((1,), noise_dev_min, noise_dev_max)
        per_pixel_noise = tf.random.normal(depth.shape, mean=0.0, stddev=noise_dev)
        depth = tf.where(depth > 1e-6, depth+per_pixel_noise, depth)

    _, h, w, _ = depth.shape
    res_low, res_high = get_random_perlin_res((h, w))
    depth = add_perlin_noise_to_depth(depth, res_low, res_high)

    # dropout patches
    freq = tf.random.normal((2,), 0.45, 0.08)
    res = get_valid_perlin_res((h,w), freq)
    perlin_noise = perlin_noise_tf((h,w), res,  interpolant = jagged_interpolant)
    perlin_noise = tf.expand_dims(tf.expand_dims(perlin_noise, 0), -1)
    depth = tf.where(perlin_noise<0.6, depth, 0.)

    if AugmentSettings.edge_frequency is not None and AugmentSettings.edge_range is not None:
        depth = warp_edges(depth)

    if AugmentSettings.downsample_and_upsample is not None:
        depth = down_and_upsample(depth)
    
    if AugmentSettings.median_filter is not None:
        depth = tfa.image.median_filter2d(depth, AugmentSettings.median_filter)

    # discretize to camera units
    if AugmentSettings.camera_unit is not None:
        depth = tf.math.floordiv(depth, AugmentSettings.camera_unit) * AugmentSettings.camera_unit

    return depth

def add_background_depth(depth, obj_pos=None, camera_matrix=None, rgb2noise=None):
    """ overlays two background maps to create one noisy fake backgrounds and adds to depth
        optional converts rgb image into depth noise
        rgb2noise: [depth_h, depth_w, 3] == [0...1]
    """

    methods = ['bicubic', 'bilinear', 'gaussian', 'nearest']


    __depth = tf.where(depth < 100.0, depth, 0.)  # set nonexistent depth to 0
    __depth2 = tf.where(__depth > 0.01, depth, 100.)
    plane_max = tf.reduce_max(__depth)
    plane_min = tf.reduce_min(__depth2)

    plane_mid_z = tf.random.uniform((), plane_min, plane_max)

    fake_depth = tf.ones_like(depth) * plane_mid_z

    fake_depth += get_fake_depth(depth, method=None, tilted_plane=True) * plane_mid_z # adds tilted plane

    method = np.random.choice(methods, (2,), replace=False)
    fake_depth += AugmentSettings.fake_depth_range * get_fake_depth(depth, method[0])
    fake_depth += AugmentSettings.fake_depth_range * get_fake_depth(depth, method[1])


    # offset patches
    _, h, w, _ = depth.shape
    freq = tf.random.normal((2,), 0.5, 0.08)
    res = get_valid_perlin_res((h,w), freq)
    perlin_noise = perlin_noise_tf((h,w), res,  interpolant = jagged_interpolant)
    perlin_noise = tf.expand_dims(tf.expand_dims(perlin_noise, 0), -1)
    fake_depth = tf.where(perlin_noise<0.7, fake_depth, fake_depth+tf.random.normal((), 0, .5))

    if AugmentSettings.move_background_to_obj and obj_pos is not None:
        assert len(obj_pos)==3  and camera_matrix is not None, "Specify object position and camera matrix!"
        z = obj_pos[2] # z
        cam_cx, cam_cy = camera_matrix[0][2], camera_matrix[1][2]
        cam_fx, cam_fy = camera_matrix[0][0], camera_matrix[1][1]

        x = int(obj_pos[0] / z * cam_fx + cam_cx) # in pixels
        y = int(obj_pos[1] / z * cam_fy + cam_cy) # in pixels

        fake_depth += z - fake_depth[0, y, x, 0]

    if AugmentSettings.rgb2noise_amplitude > 0:
        rgb2noise = 2*(tf.math.reduce_mean(rgb2noise, -1)-0.5) * AugmentSettings.rgb2noise_amplitude
        fake_depth += tf.expand_dims(tf.expand_dims(rgb2noise, 0), -1)

    fake_depth = tf.abs(fake_depth) # dont allow negative depth

    depth = tf.where(depth > 0.001, depth, fake_depth)
    
    return depth


def augment_rgb(rgb):
    p_sat = AugmentSettings.p_sat
    p_bright = AugmentSettings.p_bright
    p_noise = AugmentSettings.p_noise
    p_hue = AugmentSettings.p_hue
    p_contr = AugmentSettings.p_contr
    ops_min = AugmentSettings.ops_min
    ops_max = AugmentSettings.ops_max
    p_perlin_per_channel = AugmentSettings.p_perlin_per_channel
    

    n_ops_total = 7

    # choose random operations
    ops = tf.range(n_ops_total)
    ops = tf.random.shuffle(ops)
    n_ops_chosen = tf.random.uniform((), ops_min, ops_max + 1, dtype=tf.int32)
    ops = ops[:n_ops_chosen]

    rgb = tf.cast(rgb, tf.float32)

    if p_sat > 0 and 0 in ops:
        rgb = tf.image.random_saturation(rgb, 1. - p_sat, 1. + p_sat)
    if p_hue > 0 and 1 in ops:
        rgb = tf.image.random_hue(rgb, p_hue)
    if p_contr > 0 and 2 in ops:
        rgb = tf.image.random_contrast(rgb, 1. - p_contr, 1. + p_contr)
    if p_bright > 0 and 3 in ops:
        rgb = tf.image.random_brightness(rgb, p_bright)
    if p_noise > 0 and 4 in ops:
        effective_p_noise = tf.random.uniform((), 0, p_noise)
        noise = tf.random.normal(shape=tf.shape(rgb), mean=0.0, stddev=effective_p_noise, dtype=rgb.dtype)
        rgb += noise
    if p_perlin_per_channel > 0 and 5 in ops:
        rgb_size = tf.shape(rgb)[:2].numpy()
        p_amplitude = tf.random.uniform((6,), 0., p_perlin_per_channel)

        r_noise = p_amplitude[0] * perlin_noise_tf(rgb_size, get_random_perlin_res(rgb_size)[0])
        g_noise = p_amplitude[1] * perlin_noise_tf(rgb_size, get_random_perlin_res(rgb_size)[0])
        b_noise = p_amplitude[2] * perlin_noise_tf(rgb_size, get_random_perlin_res(rgb_size)[0])

        perlin_nose = tf.stack([r_noise, g_noise, b_noise], -1)

        r_noise = p_amplitude[3] * perlin_noise_tf(rgb_size, get_random_perlin_res(rgb_size)[1])
        g_noise = p_amplitude[4] * perlin_noise_tf(rgb_size, get_random_perlin_res(rgb_size)[1])
        b_noise = p_amplitude[5] * perlin_noise_tf(rgb_size, get_random_perlin_res(rgb_size)[1])

        perlin_nose += tf.stack([r_noise, g_noise, b_noise], -1)

        rgb += perlin_nose

    if AugmentSettings.blur_max > 0 and 6 in ops:
        ksize = tf.random.uniform((),  AugmentSettings.blur_min,  AugmentSettings.blur_max, dtype=tf.int32)
        rgb = tfa.image.gaussian_filter2d(rgb, (ksize, ksize)).numpy()

    if AugmentSettings.sharp_min > 0 and 7 in ops:
        factor = tf.random.uniform((), AugmentSettings.sharp_min, AugmentSettings.sharp_max)
        rgb = tfa.image.sharpness(rgb, factor).numpy()


    rgb = tf.clip_by_value(rgb, 0., 1.)

    if isinstance(rgb, tf.Tensor):
        rgb = rgb.numpy()

    return rgb

def warp_edges(depth, shift_kernel='RECT'):
    """ applies random pixel shift to gradient in depth image [B,H,W,C] 
        only B == C == 1 !
    """
    range = AugmentSettings.edge_range
    width = AugmentSettings.edge_thickness
    freq = AugmentSettings.edge_frequency
    dx, dy = tf.image.image_gradients(depth)
    depth_edges = tf.squeeze(tf.abs(dx) + tf.abs(dy)).numpy()
    depth_edges = tf.where(depth_edges>0.008, 1.0, 0.).numpy()

    depth_edges = cv2.dilate(depth_edges, np.ones((width, width)).astype(np.float32))

    #cv2.imshow("edges", depth_edges)

    depth_shape = depth.shape[1:-1]

    row_length =depth_shape[1]

    if shift_kernel == 'RECT':
        neighbor_indices = np.array([ -row_length, -row_length+1, 1 ,row_length+1, row_length, row_length-1, -1, -row_length-1]) * range
    elif shift_kernel == 'CROSS':
        neighbor_indices = np.array([- row_length, 1, row_length, -1]) * range

    res = get_valid_perlin_res(depth_shape, (freq, freq))
    dir_map = perlin_noise_tf(depth_shape, res)
    dir_map = np.squeeze(dir_map.numpy())
    dir_map += 1.
    #cv2.imshow("dir_map", dir_map/2.0)
    dir_map *= len(neighbor_indices)/2.
    dir_map = np.floor(dir_map).astype(np.int)

    #pixel_indices = np.random.choice(neighbor_indices, real_depth.shape)  # pick random neighbor
    pixel_indices = neighbor_indices[dir_map]
    pixel_indices += np.arange(np.prod(depth_shape)).reshape(depth_shape)  # offset by own index

    # TODO can pick very wrong indices at the left and right edges

    pixel_indices = np.where(pixel_indices<0, pixel_indices+range*row_length, pixel_indices)  # shift negative indices into img
    pixel_indices = np.where(pixel_indices>=np.prod(depth_shape), pixel_indices-range*row_length - range, pixel_indices)  # shift negative indices into img

    warped = depth.numpy().flatten()[pixel_indices.flatten()].reshape(depth_shape)  # replace pixels with chosen neighbor
        
    modified_depth = tf.where(depth_edges, warped, tf.squeeze(depth)).numpy()

    return tf.expand_dims(tf.expand_dims(modified_depth, 0), -1)

# for linemod noise
def jagged_interpolant(grid):
    return grid * grid*(3-2*grid)

# for L515 noise
def smooth_interpolant(grid):
    return grid * grid * grid * (grid * (grid * 6 - 15) + 10)

def get_perlin_res(size):
    if size in AugmentSettings.cache.keys():
        return AugmentSettings.cache[size]
    else:
        possible_res = list(get_possible_res_generator(size))
        AugmentSettings.cache[size] = possible_res
        return possible_res

def get_possible_res_generator(shape):
    large_divisors = []
    for i in range(1, int(math.sqrt(shape) + 1)):
        if shape % i == 0:
            yield int(i)
            if i * i != shape:
                large_divisors.append(shape / i)
    for divisor in reversed(large_divisors):
        yield int(divisor)

def get_random_perlin_res(shape):
    low_freq = np.random.normal(AugmentSettings.perlin_res_low, scale=AugmentSettings.perlin_frequency_std, size=(2,))
    high_freq = np.random.normal(AugmentSettings.perlin_res_high, scale=AugmentSettings.perlin_frequency_std, size=(2,))
    low_freq = np.clip(low_freq, 0., 1.)
    high_freq = np.clip(high_freq, 0., 1.)

    h, w = shape
    res_low = get_valid_perlin_res((h,w), low_freq)
    res_high = get_valid_perlin_res((h,w), high_freq)

    return res_low, res_high


def get_valid_perlin_res(shape, freqency):
    possible_res_h = get_perlin_res(shape[0])
    possible_res_w = get_perlin_res(shape[1])
    ind_h = tf.cast(tf.round(freqency[0] * (len(possible_res_h)-1)), tf.int32)
    ind_w = tf.cast(tf.round(freqency[1] * (len(possible_res_w)-1)), tf.int32)
    res = (possible_res_h[ind_h], possible_res_w[ind_w])
    return res


def perlin_noise_tf(shape, res, interpolant=smooth_interpolant):
    # ported from https://github.com/pvigier/perlin-numpy
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = tf.meshgrid(tf.range(0, res[1], delta[1]), tf.range(0, res[0], delta[0]))[::-1]
    grid = tf.stack(grid, -1)
    grid = tf.truncatemod(grid, 1.)
    grid = tf.where(tf.abs(grid - 1.0) < 0.001, 0., tf.cast(grid, tf.float32))

    # Gradients
    angles = tf.random.uniform((res[0] + 1, res[1] + 1), 0, 2 * np.pi)

    gradients = tf.stack((tf.math.cos(angles), tf.math.sin(angles)), -1)

    gradients = tf.repeat(tf.repeat(gradients, d[0], 0), d[1], 1)
    g00 = gradients[:-d[0], :-d[1]]
    g10 = gradients[d[0]:, :-d[1]]
    g01 = gradients[:-d[0], d[1]:]
    g11 = gradients[d[0]:, d[1]:]

    # Ramps
    n00 = tf.math.reduce_sum(tf.stack((grid[:, :, 0], grid[:, :, 1]), -1) * g00, 2)
    n10 = tf.math.reduce_sum(tf.stack((grid[:, :, 0] - 1., grid[:, :, 1]), -1) * g10, 2)
    n01 = tf.math.reduce_sum(tf.stack((grid[:, :, 0], grid[:, :, 1] - 1.), -1) * g01, 2)
    n11 = tf.math.reduce_sum(tf.stack((grid[:, :, 0] - 1., grid[:, :, 1] - 1.), -1) * g11, 2)

    # Interpolation
    grid = interpolant(grid)

    n0 = n00 * (1 - grid[:, :, 0]) + grid[:, :, 0] * n10
    n1 = n01 * (1 - grid[:, :, 0]) + grid[:, :, 0] * n11
    return tf.math.sqrt(2.) * ((1 - grid[:, :, 1]) * n0 + grid[:, :, 1] * n1)


def down_and_upsample(depth):
    dpt_shape = tf.shape(depth)[1:3]
    factor = AugmentSettings.downsample_and_upsample
    
    depth = tf.image.resize(depth, (dpt_shape[0]/factor, dpt_shape[1]/factor), 'nearest')
    depth = tf.image.resize(depth, (dpt_shape[0], dpt_shape[1]), 'nearest')
    return depth


def add_perlin_noise_to_depth(depth, res_low, res_high):
    """ depth must have batch and channel! [N, H, W, C]
        depth in m!
    """
    amplitude_min_high = AugmentSettings.amplitude_min_high
    amplitude_max_high = AugmentSettings.amplitude_max_high
    amplitude_min_low = AugmentSettings.amplitude_min_low
    amplitude_max_low = AugmentSettings.amplitude_max_low

    dpt_shape = tf.shape(depth)[1:3]

    amplitude = tf.random.uniform((1,), amplitude_min_low, amplitude_max_low)
    perlin_noise = 0.5 *amplitude * perlin_noise_tf(dpt_shape, res_low)

    amplitude = tf.random.uniform((1,), amplitude_min_high, amplitude_max_high)
    perlin_noise += 0.5 * amplitude * perlin_noise_tf(dpt_shape, res_high)#,  interpolant = jagged_interpolant)

    aug_depth = depth + tf.expand_dims(tf.expand_dims(perlin_noise, 0), -1)
    depth = tf.where(depth > 1e-6, aug_depth, depth)

    return depth


def get_fake_depth(depth, method, tilted_plane=False):
    """ create fake depth centered around 0, in the range [-1.0, 1.0]
        creates sparse [n_nodes,nodes] square, then interpolates it to image size
    """
    n_nodes = tf.random.uniform(shape=(), minval=3, maxval=8, dtype=tf.int32)
    nodes = tf.random.uniform((1, n_nodes, n_nodes, 1), -1.0, 1.0)

    if tilted_plane:
        n_nodes = 2
        azimuth = tf.random.uniform((), 0, np.pi)
        elevation = tf.random.uniform((), -AugmentSettings.plane_elevation, AugmentSettings.plane_elevation)
        top_left = tf.math.sin(elevation) * tf.math.cos(azimuth-np.pi/4)
        top_right = tf.math.sin(elevation) * tf.math.cos(azimuth+np.pi/4)
        bottom_left = tf.math.sin(elevation) * tf.math.cos(azimuth-np.pi*3/4)
        bottom_right = tf.math.sin(elevation) * tf.math.cos(azimuth+np.pi*3/4)

        nodes = tf.expand_dims(tf.expand_dims([[top_left, top_right], [bottom_left, bottom_right]], 0), -1)
        method = 'bilinear'

    img_size = depth.shape[1:3]
    fake_depth = tf.image.resize(nodes, img_size, method=method)
    
    # resizing makes n x n grid in image -> to shift the nodes into the edge of the image, crop and resize again
    # n_nodes = 2 creates a 3x3 grid, where only the middle is "correctly" interpolated
    # therefore (crop n_nodes-1) / (n_nodes+1) portion out of the image
    fake_depth = tf.image.central_crop(fake_depth, (n_nodes-1)/(n_nodes+1))
    fake_depth = tf.image.resize(fake_depth, img_size, 'bilinear')

    return fake_depth

def rotate_datapoint(img_likes, Rt=None):
    """ Rt = list of tuples of (3x4 Transformation matrix and cls_ids)
        bboxes = List[x1, x2, y1, y2, cls]
        img_likes = Tuple[[h, w, c]]
    """
    rotation = np.random.uniform(0, 360)

    (h, w) = img_likes[0].shape[:2]
    center = (h // 2, w // 2)
    M = cv2.getRotationMatrix2D((center[1], center[0]), rotation, 1.0)

    output_data = []
    for img in img_likes:
        if img is not None:
            rot_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST)
            output_data.append(rot_img.copy())
        else:
            output_data.append(None)

    if Rt is not None:
        rot_M = R.from_euler('xyz', [0, 0, rotation], degrees=True).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rot_M
        rotated_rt_list = []
        for (mat, cls_id) in Rt:
            mat = np.append(mat, [[0., 0., 0., 1.]], 0)
            mat = transform.T @ mat
            mat = mat[:3, :]
            rotated_rt_list.append((mat, cls_id))
        output_data.append(rotated_rt_list)

    return tuple(output_data)


def random_horizontal_flip(image, bboxes):
    if random.random() < 0.5:
        _, w, _ = image.shape
        image = image[:, ::-1, :]
        bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

    return image, bboxes


def random_crop(image, bboxes):
    if random.random() < AugmentSettings.random_crop:
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
        crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
        crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
        crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

        image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

    return image, bboxes


def random_translate(image, bboxes):
    if random.random() < 0.5:
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

        M = np.array([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h))

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

    return image, bboxes

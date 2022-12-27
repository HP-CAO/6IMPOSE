import os
from lib.params import Networks

shard_every_n_datapoints = 800  # pvn3d: 200, yolo: 800
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import yaml
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from lib.utils import ensure_fd
from typing import List
import argparse
from utils import read_config, override_params
import math
from enum import Enum


# should be a dataclass but python 3.6.5....
class PreprocessTask:
    indices: List[int]
    params: None
    gpu_inds: None
    num_workers: int
    mode: str
    num_queue: mp.Queue
    lock: mp.Lock

    def __init__(self, indices, params, gpu_inds, num_workers: int, mode: str, queue: mp.Queue, lock: mp.Lock):
        self.indices = indices
        self.params = params
        self.gpu_inds = gpu_inds
        self.num_workers = num_workers
        self.mode = mode
        self.num_queue = queue
        self.lock = lock


class SaveFormat(Enum):
    tfrecord = 1
    numpy = 2
    darknet = 3


def get_num_imgs(params):
    # launch in process to avoid initializing gpus
    def get_num(q: mp.Queue):
        import silence_tensorflow.auto
        from lib.factory import DatasetFactory
        factory = DatasetFactory(params)
        dataset = factory.get_dataset('preprocess')
        if isinstance(dataset.data_config.cls_root, dict):
            num_imgs = math.inf
            for cls_root in dataset.data_config.cls_root.values():
                num_imgs = min(len(os.listdir(os.path.join(cls_root, "rgb"))), num_imgs)
            q.put(num_imgs)

        else:
            q.put(len(os.listdir(os.path.join(dataset.data_config.cls_root, "rgb"))))
        return

    q = mp.Queue()
    worker = mp.Process(target=get_num, args=(q,))
    worker.start()
    worker.join()
    return q.get()


def process_datapoint(task: PreprocessTask):
    import silence_tensorflow.auto
    import tensorflow as tf
    from lib.data.augmenter import AugmentSettings

    try:
        worker_id = mp.current_process()._identity[0] - 1  # substract main_process
    except (KeyError, IndexError):
        worker_id = 0

    num_workers = task.num_workers
    gpu_inds = task.gpu_inds
    gpu = gpu_inds[worker_id % num_workers]  # Tensorflow sees only gpus in the list

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[gpu], 'GPU')
    for dev in physical_devices:
        tf.config.experimental.set_memory_growth(dev, True)

    from lib.factory import DatasetFactory
    factory = DatasetFactory(task.params)

    dataset = factory.get_dataset('preprocess')

    n_aug = max(dataset.data_config.augment_per_image, 1)  # minimum is 1 per image
    preprocessed_save_path = dataset.data_config.preprocessed_folder

    # save augment settings to file
    ensure_fd(preprocessed_save_path)
    augment_dict = {}
    for key, val in AugmentSettings.__dict__.items():
        if not key.startswith('__') and key != 'cache':
            augment_dict[key] = val


    # log only one worker
    if worker_id == 0:
        print("Progress for one worker:")
    iterator = task.indices if worker_id > 0 else tqdm(task.indices)

    processed = 0

    first_run = True

    if used_format == SaveFormat.numpy:
        preprocessed_save_path_numpy = os.path.join(preprocessed_save_path, 'numpy')
        ensure_fd(preprocessed_save_path_numpy)
        with open(os.path.join(preprocessed_save_path_numpy, "augment_settings.yaml"), 'w') as F:
            yaml.dump(augment_dict, F)

        # save n_aug+1 datapoints per image
        for img_id in iterator:
            base_save_id = n_aug * img_id
            for i in range(n_aug):
                # print(f"Preprocessing image {img_id} ({i+1}/{n_aug+1})")
                data = dataset.get_dict(img_id)

                save_id = base_save_id + i
                save_name = "pre_{:06}".format(save_id)

                if data == None:
                    continue

                for key, item in data.items():
                    if first_run:
                        ensure_fd(os.path.join(preprocessed_save_path_numpy, key))

                    np.save(os.path.join(preprocessed_save_path_numpy, key, save_name), item)
                first_run = False
                processed += 1

    elif used_format == SaveFormat.tfrecord:
        # ---- write directly to sharded tfrecord files----
        options = tf.io.TFRecordOptions(compression_type='ZLIB')

        preprocessed_save_path_tfrecord = os.path.join(preprocessed_save_path, 'tfrecord')
        ensure_fd(preprocessed_save_path_tfrecord)
        with open(os.path.join(preprocessed_save_path_tfrecord, "augment_settings.yaml"), 'w') as F:
            yaml.dump(augment_dict, F)

        ensure_fd(preprocessed_save_path_tfrecord)

        writer = tf.io.TFRecordWriter(
            os.path.join(preprocessed_save_path_tfrecord, f"preprocessed_{n_aug * task.indices[0]}.tfrecord"), options=options)

        # save n_aug+1 datapoints per image
        for img_id in iterator:
            base_save_id = n_aug * img_id
            for i in range(n_aug):

                feature_dict = dataset.get_dict(img_id)

                if feature_dict is None:
                    continue

                dtype_dict = {name: val.dtype for name, val in feature_dict.items()}

                serialized_features = {
                    name: tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(val).numpy()]))
                    for name, val in feature_dict.items()}
                example_proto = tf.train.Example(features=tf.train.Features(feature=serialized_features))
                example = example_proto.SerializeToString()

                writer.write(example)
                processed += 1

                if (processed % shard_every_n_datapoints) == 0:
                    writer.close()
                    writer = tf.io.TFRecordWriter(
                        os.path.join(preprocessed_save_path_tfrecord, f"preprocessed_{base_save_id + i}.tfrecord"),
                        options=options)

        writer.close()

        # dump dtypes for parsing binary files
        import pickle
        pickle.dump(dtype_dict, open(os.path.join(preprocessed_save_path_tfrecord, "dtypes_preprocessed.bin"), 'wb'))

    elif used_format == SaveFormat.darknet:
        import cv2
        preprocessed_save_path_darknet = os.path.join(preprocessed_save_path, "darknet")
        ensure_fd(preprocessed_save_path_darknet)
        with open(os.path.join(preprocessed_save_path_darknet, "augment_settings.yaml"), 'w') as F:
            yaml.dump(augment_dict, F)

        img_dir = os.path.join(preprocessed_save_path_darknet, 'obj')

        model_dir = os.path.join(task.params.monitor_params.model_dir, 'darknet_yolo', dataset.data_config.cls_type)

        ensure_fd(img_dir)
        ensure_fd(model_dir)

        if dataset.data_config.cls_type == 'all':
            n_classes = max(dataset.data_config.cls_lst)+1
            with open(os.path.join(preprocessed_save_path_darknet, 'obj.names'), 'w') as F:
                classes = []
                for i in range(n_classes):
                    try:
                        classes.append(dataset.data_config.id2obj_dict[i])
                    except KeyError:
                        classes.append('notdefined')
                F.write('\n'.join(classes))

        else:
            n_classes = 1
            classes = [dataset.data_config.cls_type]
            with open(os.path.join(preprocessed_save_path_darknet, 'obj.names'), 'w') as F:
                F.write('\n'.join(classes))

        data_text = [f"classes = {n_classes}",
                     f"train = {os.path.join(preprocessed_save_path_darknet, 'train.txt')}",
                     f"valid = {os.path.join(preprocessed_save_path_darknet, 'test.txt')}",
                     f"names = {os.path.join(preprocessed_save_path_darknet, 'obj.names')}",
                     f"backup = {model_dir}"
                     ]
        data_text = '\n'.join(data_text)

        with open(os.path.join(preprocessed_save_path_darknet, 'obj.data'), 'w') as F:
            F.write(data_text)

        processed_ids = []
        for img_id in iterator:
            base_save_id = n_aug * img_id
            for i in range(n_aug):

                data = dataset.get_dict(img_id)

                save_id = base_save_id + i
                if data == None:
                    continue

                rgb = data['yolo_rgb_input']  # [h, w, 3] uint8

                h, w = rgb.shape[:2]

                bboxes = data['gt_bboxes']  # [n, 4] uint8
                img_boxes = []
                for box in bboxes:
                    obj_class = int(box[4])
                    x1, y1, x2, y2 = box[:4]
                    cx = (x1 + x2) / 2.0 / w
                    cy = (y1 + y2) / 2.0 / h
                    box_w = (x2 - x1) / w
                    box_h = (y2 - y1) / h
                    img_boxes.append(f"{obj_class} {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}")

                save_name = f"{save_id:08}"

                cv2.imwrite(os.path.join(img_dir, f"{save_name}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                with open(os.path.join(img_dir, f"{save_name}.txt"), 'w') as F:
                    F.write('\n'.join(img_boxes))
                processed_ids.append(os.path.join(img_dir, f"{save_name}.jpg"))
                processed += 1

        # write train/test.txt
        if task.mode == 'train':
            output = 'train.txt'
        elif task.mode == 'val':
            output = 'test.txt'
        elif task.mode == 'full':
            output = 'all.txt'
        else:
            raise AssertionError("Unknown mode for creating train/test.txt: ", task.mode)

        with open(os.path.join(preprocessed_save_path_darknet, output), 'a') as F:
            F.write('\n'.join(processed_ids))
            F.write('\n')

    task.num_queue.put(processed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default_pvn3d.json', help='Path to config file')
    parser.add_argument('--num_workers', default=64, help='Number of workers')
    parser.add_argument('--params', nargs='*', default=None)
    parser.add_argument('--mode', default='train', help='[train|val|full]: generate training or eval set')
    parser.add_argument('--format', default='tfrecord', help='[tfrecord|numpy|darknet]: choose data save format')
    parser.add_argument('--id', default=None, help='If set overrides the logfile name and the save name')
    parser.add_argument('--gpus', default=None, help='Override gpus with 1,2,3...')
    args = parser.parse_args()

    params = read_config(args.config)

    if args.id is not None:
        params.monitor_params.model_name = args.id
        params.monitor_params.log_file_name = args.id

    if args.format == 'tfrecord':
        used_format = SaveFormat.tfrecord
        print("Using tfrecord format")
    elif args.format == 'numpy':
        used_format = SaveFormat.numpy
        print("Using numpy format")
    elif args.format == 'darknet':
        used_format = SaveFormat.darknet
    else:
        raise AssertionError(f"Unknown format: {args.format} [tfrecord|numpy|darknet]")

    if args.gpus is None:
        gpus = params.trainer_params.distribute_train_device
    else:
        gpus = [int(x) for x in args.gpus.split(',')]

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in gpus])

    params.monitor_params.mode = 'preprocess'

    num_workers = int(args.num_workers)  # get distributed over gpus

    max_n_workers_per_gpu = 20 # max can be 25

    n_gpus = len(gpus)
    num_workers = np.clip(num_workers, 0, max_n_workers_per_gpu * n_gpus)  # limit max amount of workers/gpu
    num_workers = np.clip(num_workers, 0, 70)  # dont exceed 64 workers
    gpu_inds = np.floor(np.linspace(0, n_gpus - 0.0001, num_workers)).astype(
        int)  # maps 0 - 4.9999 to 0-4 equally distributed (5 times)

    if args.config is None:
        exit("config file needed")

    if args.params is not None:
        params = override_params(params, args.params)

    if args.mode == 'train':
        imgInds = list(range(0, params.dataset_params.train_size))
    elif args.mode == 'val':
        imgInds = list(range(params.dataset_params.train_size, params.dataset_params.size_all))
    elif args.mode == 'full':
        imgInds = list(range(0, params.dataset_params.size_all))
    else:
        raise AssertionError(f"Unknown mode: {args.mode} [train|val|full]")

    n_datapoints = max(params.dataset_params.augment_per_image, 1) * len(imgInds)

    print(f"Processing {len(imgInds)} images to generate {n_datapoints} datapoints (might be less if data is invalid!)")

    q = mp.Queue()
    l = mp.Lock()

    imgInds = np.array_split(imgInds, num_workers, axis=0)
    tasks = [PreprocessTask(inds, params, gpu_inds, num_workers, args.mode, q, l) for inds in imgInds]
    workers = [mp.Process(target=process_datapoint, args=(task,)) for task in tasks]
    [w.start() for w in workers]
    [w.join() for w in workers]

    num = 0
    while not q.empty():
        num += q.get()

    print("Preprocessing done.")
    print(f"Total: {num} datapoints.")

    from lib.factory import DatasetFactory

    factory = DatasetFactory(params)
    dataset = factory.get_dataset('preprocess')
    preprocessed_save_path = dataset.data_config.preprocessed_folder

    if used_format == SaveFormat.numpy:
        n_aug = max(dataset.data_config.augment_per_image, 1)
        preprocessed_save_path_numpy = os.path.join(preprocessed_save_path, 'numpy')
        print("Renaming to consistent index list: ", preprocessed_save_path_numpy)
        no = 0
        start_ind = imgInds[0][0] * n_aug
        types = [x for x in os.listdir(preprocessed_save_path_numpy) if "." not in x]  # gives folder names
        for ind in tqdm(range(start_ind, start_ind + n_datapoints)):
            is_datapoint = False
            for data_type in types:
                if os.path.isfile(os.path.join(preprocessed_save_path_numpy, data_type, f"pre_{ind:06}.npy")):
                    src_file = os.path.join(preprocessed_save_path_numpy, data_type, f"pre_{ind:06}.npy")
                    tar_file = os.path.join(preprocessed_save_path_numpy, data_type, f"{no:06}.npy")
                    os.rename(src_file, tar_file)
                    is_datapoint = True

            if is_datapoint:
                no += 1  # increment valid file counter

        print(f"Data preprocessing done: {no} datapoints")  # does not correlate with n images anymore

    elif used_format == SaveFormat.tfrecord:
        meta_dict = {'n_datapoints': num}
        import json

        preprocessed_save_path_tfrecord = os.path.join(preprocessed_save_path, 'tfrecord')
        with open(os.path.join(preprocessed_save_path_tfrecord, 'meta.json'), 'w') as F:
            json.dump(meta_dict, F)

from lib.trainer.trainer import TrainerParams
import datetime

""" Param Structs and Constants without importing tf"""

class Datasets:
    blender = 'blender'
    linemod = 'linemod'
    unity_grocieries_real = 'ugreal'

class Networks:
    pvn3d = 'pvn3d'
    yolo = 'yolo'
    darknet = 'darknet'

class DatasetParams:
    dataset: str
    data_name: str
    train_batch_size: int
    val_batch_size: int
    cls_type: str
    train_size: int
    size_all: int
    augment_per_image: int
    use_preprocessed: bool


class MonitorParams:
    def __init__(self):
        self.log_root = './logs/'
        self.model_dir = './models/'
        self.resnet_weights_name = ''
        self.mode = 'test'
        self.force_override = False
        self.model_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.weights_path = None
        self.best_loss = 1e3
        self.if_validation = True
        self.train_epochs = 500
        self.val_frequency = 20
        self.performance_eval_frequency_factor = 2
        self.val_epochs = 1
        self.if_model_summary = True
        self.sim2real_eval = False
        self.model_save_period = 5
        self.write_log = True

class NetworkParams:
    network: str
    dataset_params: DatasetParams
    trainer_params: TrainerParams
    monitor_params: MonitorParams
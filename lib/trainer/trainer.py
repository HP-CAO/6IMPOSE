from abc import abstractmethod
from typing import List, Dict


class TrainerParams:
    distribute_training: bool
    distribute_train_device: List
    learning_rate: float


class Trainer:
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def log(self, losses: Dict[str, float]):
        pass

    @abstractmethod
    def get_overall_loss(self) -> float:
        """ return overall loss """
        pass

    @abstractmethod
    def get(self) -> Dict[str, float]:
        """ get Dict of losses, named for tensorboard """
        pass

    @staticmethod
    def save_weights(model, name):
        model.save_weights('./models/' + name)

    @staticmethod
    def save_model(model, dir_model, name):
        model.save(dir_model + './' + name)
        print("model saved to {}".format(dir_model))

    @staticmethod
    def load_weights(model, path_weights):
        model.load_weights(path_weights)

    @staticmethod
    def load_model(model, path_model):
        model.load_model(path_model)

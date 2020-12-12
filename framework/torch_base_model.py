import torch
import abc
import typing


class BaseModel(torch.nn.Module, abc.ABC):
    def __init__(self, *arg, **kwargs):
        """
        BaseModel is an abstract class of pytorch NN networks which can be used in the experiment framework
         defined in this package. Each network that you want to train with the package training framework should inherit
         BaseModel and implement five functions (forward, prepare_data, loss, score, predict) with the same signatures
          as the abstract functions.
        :param arg:
        :param kwargs:
        """
        super(BaseModel, self).__init__()
        self._device = self.device

    @property
    def init_parameters(self):
        return self._init_parameters

    @property
    def device(self):
        if hasattr(self, '_device'):
            return self._device
        else:
            return 'cuda' if torch.cuda.is_available() else 'cpu'

    def to_device(self, x, device=None):
        device = device if device is not None else self.device
        if device == 'cpu':
            return x.cpu()
        else:
            return x.cuda() if torch.cuda.is_available() else x

    def save_model(self, file_path):
        torch.save(self, file_path)

    @staticmethod
    def load_model(file_path):
        return torch.load(file_path)

    @abc.abstractmethod
    def prepare_data(self, data: typing.Any) -> typing.Tuple[typing.Any, typing.Any, typing.Any]:
        """
        prepare_data should return four variables in the following order:
        ids - which is the ids of the instances.
        x_data - which is the input of the net forward function.
        y_data - which is the ground_truth labels
        loss_data - which is the data used to calculate the loss (using self.loss function)
        :param data: data from pytorch dataloader
        :return: Tuple[ids, x_data, y_data, loss_data]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, x_data):
        raise NotImplementedError

    @abc.abstractmethod
    def loss(self, output, loss_data, criterion: torch.nn.Module):
        """
        calculate the loss
        :param output: output of the self.forward function
        :param loss_data: the data used to calculate the loss
        :param criterion: pytorch loss function, see torch.nn.modules.loss.py
        :return: pytorch loss
        """
        raise NotImplementedError

    @abc.abstractmethod
    def score(self, output):
        """
        calculate the scores, which are the class probabilities
        :param output: output of the self.forward function
        :return: scores
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, output):
        """
        predict the classes
        :param output: output of the self.forward function
        :return: predictions
        """
        raise NotImplementedError


# --------- default functions ---------- #
# if any network didn't implement the four abstract function: prepare_data, loss, score, predict
# then the following default functions will be used in the training procedure

def default_prepare_data(data):
    if isinstance(data, tuple) and len(data) == 3:
        return data[0], data[1], data[2], data[2]
    else:
        return data


def default_loss(output, loss_data, criterion):
    return criterion(output, loss_data)


def default_score(output):
    return output


def default_predict(output):
    return torch.argmax(output, dim=1)

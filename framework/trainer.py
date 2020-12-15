from framework import configs as C, utils, torch_base_model
from framework.torch_layers import init_weights
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler
from tensorboard import program
import numpy as np
import pandas as pd
import datetime
import tqdm
import os
import abc
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
import typing
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class Trainer(abc.ABC):
    def __init__(self, model, configs):
        self._read_configs(configs)
        self._model = model
        self._results = self._init_results_df()
        if self._save_results:
            self._results.to_csv(self.results_file, index=False)
        self._current_epoch = 0
        self._make_last_epoch_results = False
        self._current_datetime = datetime.datetime.now()
        self._basic_results = {'name': self._name, 'model': type(self.model).__name__,
                               'hyperparameters': self._hyperparameters}

    def _read_configs(self, configs):
        self._configs = configs
        self._hyperparameters = configs.get('hyperparameters', {})
        self._folder = configs.get('folder', C.folder)
        self._name = f"{configs.get('name', C.name)}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        self._metrics = configs.get('metrics', C.metrics)
        self._save_results = configs.get('save_results', C.save_results)
        self._save_model = configs.get('save_model', C.save_model)
        self._verbose = configs.get('verbose', C.verbose)

    def _init_results_df(self):
        return pd.DataFrame(columns=['name', 'model', 'hyperparameters', 'datetime', 'epoch', 'fit_time'] +
                                    [f'{m}_train' for m in list(self._metrics)] +
                                    [f'{m}_test' for m in list(self._metrics)])

    @property
    def name(self):
        return self._name

    @property
    def configs(self):
        return self._configs

    @property
    def folder(self):
        if not os.path.exists(self._folder):
            os.makedirs(self._folder)
        return self._folder

    @property
    def results_file(self):
        if self._save_results:
            if not os.path.exists(os.path.join(self.folder, 'results')):
                os.makedirs(os.path.join(self.folder, 'results'))
            return os.path.join(self.folder, 'results', self.name + '_results.csv')
        else:
            return None

    @property
    def model_file(self):
        if self._save_model and not os.path.exists(os.path.join(self.folder, 'models')):
            os.makedirs(os.path.join(self.folder, 'models'))
        return os.path.join(self.folder, 'models', self.name + '_model.pkl')

    @property
    def results(self):
        return self._results

    @property
    def model(self):
        return self._model

    def _append_results(self, a_results):
        self._results = pd.concat([self.results, a_results], axis=0)
        if self._save_results:
            self._results.tail(1).astype(str).to_csv(self.results_file, mode='a', header=False, index=False)

    def save_model(self):
        if self._save_model:
            with open(self.model_file, 'wb') as file:
                pickle.dump(self.model, file)

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def _print(self, io: str, verbose: int = 3):
        if verbose >= 4 and self._verbose >= 4:
            tqdm.tqdm.write(io)
        elif verbose <= self._verbose or (verbose == 4 and self._verbose >= 3):
            print(io)

    def _model_func(self, func_name, *args, **kwargs):
        """
        :param func_name: name of the self.model function that being used. if the self.model doesn't have this function,
        then a default function with the same will be used. see default functions at experiments_framework.base_model.
        :return: return value of self.model.func_name(*args, **kwargs)
        """
        func = getattr(self.model, func_name, getattr(torch_base_model, f'default_{func_name}'))
        return func(*args, **kwargs)

    def calculate_metrics(self, y_scores, y_pred, y_true):
        """
        :param y_scores: class probabilities
        :param y_pred: model's predictions
        :param y_true: true predictions
        :return: a dict where the keys are metric names and the values are metrics' return values.
         see experiments_framework.metrics.py and experiments_framework.configs.py for more details
          on which metrics are being used.
        """
        results = {}
        for metric_name, metric_func in self._metrics.items():
            try:
                results[metric_name] = metric_func(y_scores=y_scores, y_pred=y_pred, y_true=y_true)
            except Exception as e:
                # results[metric_name] = f"{type(e)}: {e}" 0 : causing problems when sorting because types mismatch
                results[metric_name] = None
        return results

    def _prepare_fit(self):
        self._current_epoch = 0
        self._current_datetime = datetime.datetime.now()
        self._make_last_epoch_results = True
        self._print(f"# {'-' * 20} Starting Experiment {self.name} {'-' * 20} #", verbose=1)

    def _make_results(self, train_data, test_data):
        if self._make_last_epoch_results:
            current_datetime = datetime.datetime.now()
            time = (current_datetime - self._current_datetime).total_seconds()
            self._current_datetime = current_datetime
            dt = self._current_datetime.strftime('%Y/%m/%d %H:%M:%S')
            results = self._basic_results.copy()
            results.update({'datetime': dt, 'epoch': self._current_epoch, 'fit_time': time})
            _, train_metrics = self.predict(train_data)
            results.update({f'{k}_train': v for k, v in train_metrics.items()})
            if test_data is not None:
                _, test_metrics = self.predict(test_data)
            else: # same as train
                test_metrics = train_metrics
            results.update({f'{k}_test': v for k, v in test_metrics.items()})
            self._append_results(pd.DataFrame([{k: str(v) if isinstance(v, (dict, list, tuple)) else v
                                                for k, v in results.items()}]))
            return results
        else:
            return self.results.tail(1)

    def _best_results(self):
        return self.results.tail(1)

    def _end_fit(self):
        best_results = self._best_results()
        best_results['results_file_path'] = self.results_file if self._save_results else None
        best_results['model_file_path'] = self.model_file if self._save_model else None
        self.save_model()
        self._print(f"Finished to fit, Experiment {self.name} is over. Best test results are:", verbose=1)
        self._print(str(utils.deep_round(best_results.to_dict('records')[0], 3)), verbose=1)
        return best_results

    @abc.abstractmethod
    def predict(self, data):
        raise NotImplementedError

    @abc.abstractmethod
    def _fit(self, train_data, test_data):
        raise NotImplementedError

    def fit(self, train_data, test_data):
        self._prepare_fit()
        self._fit(train_data, test_data)
        self._make_results(train_data, test_data)
        return self._end_fit()


class SklearnTrainer(Trainer):
    def __init__(self, model, configs):
        """
        Trainer class is an experiment framework to evaluate the fitting procedure of a pytorch nn modelwork.
        Use Train(model, **configs).fit(train_data, test_data) in order to fit the model and get a results file
         containing information on the training procedure, including evaluation metrics scores and running time.
        :param model: pytorch nn modelwork
        :param configs: a dict of configurations. see configs.py file for more information.
         configs should not contain 'model_init' and 'init_parameters'.
        """
        if not isinstance(model, (BaseEstimator, ClassifierMixin, )):
            raise ValueError(f"model should be a sklearn model of type BaseEstimator or ClassifierMixin,"
                             f" got {type(model)}")
        super(SklearnTrainer, self).__init__(model, configs)

    def predict(self, data):
        if len(data) == 3:
            ids, X, y_true = data
        elif len(data) == 2:
            X, y_true = data
            ids = np.array(range(len(y_true)))
        else:
            raise ValueError("data should be a tuple of (X, y_true) or (ids, X, y_true)")
        y_pred = self.model.predict(X)
        if hasattr(self.model, 'predict_proba'):
            y_scores = self.model.predict_proba(X)
        else:
            y_scores = pd.get_dummies(y_pred).astype(float).values
        metrics = self.calculate_metrics(y_scores, y_pred, y_true)
        scores = pd.DataFrame(list(zip(y_true.tolist(), y_pred.tolist(), y_scores.tolist())),
                              columns=['y_true', 'y_pred', 'y_scores'], index=ids.tolist())
        return scores, metrics

    def _fit(self, train_data, test_data):
        X_train, y_train = train_data
        self.model.fit(X_train, y_train)
        self._current_epoch += 1


class TorchTrainer(Trainer):
    def __init__(self, model, configs):
        """
        Trainer class is an experiment framework to evaluate the fitting procedure of a pytorch nn modelwork.
        Use Train(model, **configs).fit(train_data, test_data) in order to fit the model and get a results file
         containing information on the training procedure, including evaluation metrics scores and running time.
        :param model: pytorch nn modelwork
        :param configs: a dict of configurations. see configs.py file for more information.
         configs should not contain 'model_init' and 'init_parameters'.
        """
        if not isinstance(model, torch.nn.Module):
            raise ValueError(f"model should be a pytorch model of type Module,"
                             f" got {type(model)}")
        super(TorchTrainer, self).__init__(model, configs)
        self._read_extra_configs(configs)
        self._device = self.device
        self._tb_writer = None
        self._optimizer = self._optimizer_init(self.model.parameters(), **self._optimizer_kwargs)
        self._criterion = self._criterion_init(**self._criterion_kwargs)
        self._basic_results.update({'optimizer': self._optimizer_init.__name__,
                                    'optimizer_params': self._optimizer_kwargs,
                                    'criterion': self._criterion_init.__name__,
                                    'criterion_params': self._criterion_kwargs})
        self._best_test_metric = float('-inf')
        self._no_improve_iterations = 0

    def _init_results_df(self):
        return pd.DataFrame(columns=['name', 'model', 'hyperparameters', 'optimizer', 'optimizer_params', 'criterion',
                                     'criterion_params', 'datetime', 'epoch', 'fit_time', 'loss', 'grad'] +
                                    [f'{m}_train' for m in ['loss'] + list(self._metrics)] +
                                    [f'{m}_test' for m in ['loss'] + list(self._metrics)])

    def _read_extra_configs(self, configs):
        self._epochs = configs.get('epochs', C.epochs)
        self._convergence_iterations = configs.get('convergence_iterations', C.convergence_iterations)
        self._convergence_metric = configs.get('convergence_metric', C.convergence_metric)
        if self._convergence_metric not in self._metrics:
            self._print(f"The convergence_metric {self._convergence_metric} is not given in configs['metrics']."
                        f"Changing the convergence_metric to loss.", verbose=2)
            self._convergence_metric = 'loss_test'
        else:
            self._convergence_metric += '_test'
        self._loss_every_x_samples = configs.get('loss_every_x_samples', C.loss_every_x_samples)
        self._eval_every_x_epochs = configs.get('test_every_x_epochs', C.eval_every_x_epochs)
        self._optimizer_init = configs.get('optimizer', C.optimizer)
        self._optimizer_kwargs = configs.get('optimizer_kwargs', C.optimizer_kwargs)
        self._criterion_init = configs.get('criterion', C.criterion)
        self._criterion_kwargs = configs.get('criterion_kwargs', C.criterion_kwargs)
        self._tb_port = configs.get('tensorboard_port', C.tensorboard_port)
        if isinstance(self._tb_port, int) and self._tb_port > 1000:
            self._use_tb = True
        else:
            self._use_tb = False

    @property
    def device(self):
        if hasattr(self, '_device'):
            return self._device
        else:
            return 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    def model_file(self):
        if self._save_model and not os.path.exists(os.path.join(self.folder, 'models')):
            os.makedirs(os.path.join(self.folder, 'models'))
        return os.path.join(self.folder, 'models', self.name + '_model.pt')

    def to_device(self, x, device=None):
        device = device if device is not None else self.device
        if device == 'cpu':
            return x.cpu()
        else:
            return x.cuda() if torch.cuda.is_available() else x

    @staticmethod
    def load_model(file_path):
        return torch.load(file_path)

    def calculate_grad(self):
        grad = 0
        for p in self.model.parameters():
            param_norm = p.grad.data.norm(2)
            grad += param_norm.item() ** 2
        grad = grad ** (1. / 2)
        return grad

    def _launch_tensorboard(self):
        if not self._use_tb:
            self._print("tensorboard_port is not an int or is smaller than 1000, TensorBoard will not be launched",
                        verbose=2)
            return
        log_dir = os.path.join(self.folder, 'tb')
        tb = program.TensorBoard()
        tb.configure(logdir=log_dir, port=self._tb_port)
        self._tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, self.name))
        try:
            self._print(f"\t\tTensorBoard is successfully launched and is bind to {tb.launch()}.", verbose=2)
        except Exception as e:
            self._print(f"\t\tCould not launch TensorBoard due to: {e}."
                        f" Using the already running TensorBoard thread at http://localhost:{self._tb_port}/.",
                        verbose=2)
        self._print(f"\t\tIf you are using a virtual machine, "
                    f"run `ssh -N -f -L localhost:{self._tb_port}:localhost:{self._tb_port} <user@remote>`"
                    f" on your local machine and go to http://localhost:{self._tb_port}/", verbose=2)

    def _best_results(self):
        return self.results.sort_values(by=self._convergence_metric, ascending=False).head(1)

    def _check_for_improvement(self, results):
        if not isinstance(results[self._convergence_metric], (int, float)):
            self._print(f"convergence_metric return value is not a number, changing the convergence_metric to loss.",
                        verbose=2)
            self._convergence_metric = 'loss_test'
        current_test_metric = results[self._convergence_metric] * (-1 if 'loss' in self._convergence_metric else 1)
        # if there was an improvement
        if self._best_test_metric <= current_test_metric:
            self._no_improve_iterations = 0
            self._best_test_metric = results[self._convergence_metric]
            self.save_model()
        else:
            self._no_improve_iterations += 1

    def _write_to_tensorboard(self, metric_name, train, metric_score, epoch):
        if not self._use_tb:
            return
        if isinstance(metric_score, (int, float)):
            self._tb_writer.add_scalar(f"{metric_name}/{'Train' if train else 'Test'}",
                                       metric_score, epoch)
        elif isinstance(metric_score, dict):
            for k, v in metric_score.items():
                self._write_to_tensorboard(f"{metric_name}_{k}", train, v, epoch)
        elif isinstance(metric_score, (tuple, list)):
            for i, v in enumerate(metric_score):
                self._write_to_tensorboard(f"{metric_name}_{i}", train, v, epoch)
        else:
            pass

    def _print_results(self, results):
        io = f"{results['datetime']} | [{results['epoch']}/{self._epochs}] |" \
             f" fit-time: {utils.deep_round(results['fit_time'])}sec |" \
             f" loss: {utils.deep_round(results['loss'], 3)} | grad: {utils.deep_round(results['grad'], 3)} |"
        self._write_to_tensorboard(metric_name="Gradient", train=True, metric_score=results['grad'],
                                   epoch=results['epoch'])
        for metric in ['loss'] + sorted(self._metrics.keys()):
            io += f" {metric + '_train'}: {utils.deep_round(results[metric + '_train'], 3)} |"
            io += f" {metric + '_test'}: {utils.deep_round(results[metric + '_test'], 3)} |"
            if metric == 'loss' or self._convergence_metric.startswith(metric):
                self._write_to_tensorboard(metric_name=metric, train=True, metric_score=results[metric + '_train'],
                                           epoch=results['epoch'])
                self._write_to_tensorboard(metric_name=metric, train=False, metric_score=results[metric + '_test'],
                                           epoch=results['epoch'])
        if results[self._convergence_metric] >= self._best_test_metric:
            self._print(io, verbose=4)

    @staticmethod
    def to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        else:
            return np.array(data)

    def predict(self, data):
        """
        Predict the labels of the data samples from the data loader, using the class modelwork. Evaluate the modelwork
        using the metrics defined in the class metrics dict. Returns the predictions and the metrics scores.
        :param data: pytorch data loader
        :return: y_pred, metrics
        """
        losses = []
        self.model.eval()
        ids, y_true, y_pred, y_scores = None, None, None, None
        for batch in data:
            ids_data, x_data, y_data, loss_data = self._model_func('prepare_data', data=batch)
            output = self.model(x_data)
            loss = self._model_func('loss', output=output, loss_data=loss_data, criterion=self._criterion)
            scores = self.to_numpy(self._model_func('score', output=output))
            predictions = self.to_numpy(self._model_func('predict', output=output))
            losses.append(loss.item())
            ids_data = self.to_numpy(ids_data)
            y_data = self.to_numpy(y_data)
            ids = np.concatenate([ids, ids_data], axis=0) if ids is not None else ids_data
            y_true = np.concatenate([y_true, y_data], axis=0) if y_true is not None else y_data
            y_pred = np.concatenate([y_pred, predictions], axis=0) if y_pred is not None else predictions
            y_scores = np.concatenate([y_scores, scores], axis=0) if y_scores is not None else scores
        metrics = self.calculate_metrics(y_scores, y_pred, y_true)
        metrics.update({'loss': np.mean(np.array(losses))})
        scores = pd.DataFrame(list(zip(y_true.tolist(), y_pred.tolist(), y_scores.tolist())),
                              columns=['y_true', 'y_pred', 'y_scores'], index=ids.tolist())
        self.model.train()
        return scores, metrics

    def _fit(self, train_data, test_data):
        """
        Fit and train the model using the data from the train data loader. Evaluating the model with metrics defined
         in the class configs file. Writing the scores and the running time at each step to the class results file.
        :param train_data: pytorch train data_loader of the training dataset
        :param test_data: pytorch test data_loader of the testing dataset
        """
        self._optimizer.zero_grad()
        self._best_test_metric = float('-inf')
        self._no_improve_iterations = 0
        self.model.train()
        self._launch_tensorboard()
        epochs = tqdm.tqdm(range(1, self._epochs + 1), position=0, leave=True) \
            if self._verbose >= 4 else list(range(self._epochs))
        for epoch in epochs:
            losses = []
            self._current_epoch = epoch
            current_samples = 0
            grad = None
            for i, batch in enumerate(train_data):
                ids, x_data, y_data, loss_data = self._model_func('prepare_data', data=batch)
                output = self.model(x_data)
                model_loss = self._model_func('loss', output=output, loss_data=loss_data, criterion=self._criterion)
                losses.append(model_loss.item())
                model_loss.backward()
                current_samples += len(loss_data)
                if current_samples >= self._loss_every_x_samples or i == len(train_data) - 1:
                    model_loss /= current_samples
                    grad = self.calculate_grad()
                    self._optimizer.step()
                    self._optimizer.zero_grad()
                    current_samples = 0
            loss = np.mean(np.array(losses))
            self._basic_results.update({'loss': loss, 'grad': grad})
            if epoch % self._eval_every_x_epochs == 0 or epoch == self._epochs:
                results = self._make_results(train_data, test_data)
                self._check_for_improvement(results)
                self._print_results(results)
                if epoch == self._epochs:
                    self._make_last_epoch_results = False
                    self._save_model = False
            if self._no_improve_iterations >= self._convergence_iterations:
                break


def create_trainer(model, configs: typing.Dict = None):
    if configs is None:
        configs = {}
    if isinstance(model, str):
        if model.endswith('.pt'):
            model = TorchTrainer.load_model(model)
        elif model.endswith('.pkl'):
            model = Trainer.load_model(model)
        else:
            raise ValueError(f"Trainer class only support model file_paths which ends with '.pt' or '.pkl'")
    if isinstance(model, torch.nn.Module):
        return TorchTrainer(model, configs)
    elif isinstance(model, (BaseEstimator, ClassifierMixin)):
        return SklearnTrainer(model, configs)
    else:
        raise ValueError(f"Trainer class doesn't support models of type {type(model)}")


def fit_trainer(configs: dict, parameters: dict, train_data: typing.Union[typing.Tuple, DataLoader],
                test_data: typing.Union[typing.Tuple, DataLoader] = None):
    model_init = configs['model_init']
    model = model_init(**parameters)
    if isinstance(model, torch.nn.Module):
        model.apply(init_weights)
    configs['hyperparameters'] = parameters
    trainer = create_trainer(model, configs)
    return trainer.fit(train_data, test_data)


def fit_trainer_cv(configs: dict, parameters: dict, data: typing.Union[typing.Tuple, DataLoader],
                   cv: int = 5, random_state: int = 42):
    if isinstance(data, typing.Tuple): # sklearn model case
        n = len(data[0])
    elif isinstance(data, DataLoader): # torch dataloader case
        n = len(data.dataset)
    else:
        raise ValueError("data should be tuple of (X, y) or torch dataloader.")
    name = f"{configs.get('name', C.name)}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    verbose = configs.get('verbose', C.verbose)
    if verbose >= 0.5:
        print(f"# {'-' * 20} Starting {cv}-Folds CV Experiment {name} {'-' * 20} #")
    folds_indices = utils.k_folds_indices(n, k=cv, random_state=random_state, shuffle=True)
    results = []
    for i , (train_indices, test_indices) in enumerate(folds_indices):
        if isinstance(data, typing.Tuple):
            X, y = data
            if isinstance(X, (pd.DataFrame, pd.Series)):
                X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            else:
                X_train, X_test = X[train_indices], X[test_indices]
            if isinstance(y, (pd.DataFrame, pd.Series)):
                y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
            else:
                y_train, y_test = y[train_indices], y[test_indices]
            train_data, test_data = (X_train, y_train), (X_test, y_test)
        else: # data is torch.DataLoader
            # some awful coding practise, but torch DataLoader doesn't support creating a sub data loader.
            dl_kwargs = {k: v for k, v in data.__dict__.items()
                         if not k.startswith('_') and k not in ['sampler', 'batch_sampler']}
            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(test_indices)
            train_data = torch.utils.data.DataLoader(sampler=train_sampler, **dl_kwargs)
            test_data = torch.utils.data.DataLoader(sampler=test_sampler, **dl_kwargs)
        # fold_configs = configs.copy()
        # fold_configs['name'] = f"{configs.get('name', C.name)}_fold-{i}"
        results.append(fit_trainer(configs, parameters, train_data, test_data))
    results = pd.concat(results, axis=0)
    agg_results = results.mode().head(1)
    agg_results['name'] = name
    for c in results.columns:
        if pd.api.types.is_numeric_dtype(results[c]):
            agg_results[c] = results[c].mean()
            agg_results[f'{c}_std'] = results[c].std()
        agg_results[f'{c}_values'] = [results[c].tolist()]
    if verbose >= 0.5:
        print(f"{cv}-Folds CV Experiment {name} is over. Results are:")
        print(str(utils.deep_round(agg_results.to_dict('records')[0], 3)))
    return agg_results

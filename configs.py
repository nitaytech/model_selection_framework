"""
configuration variables and descriptions, default values are listed below:

Trainer class configurations:

|---------------------------|-------------------------------------------|-------------|--------------------------------|
| variable                  | description                               | type        | example (default)              |
|---------------------------|-------------------------------------------|-------------|--------------------------------|
| 'name'                    | name of the experiment, will be used in   | str         | 'experiment_104' ('EXP')       |
|                           | the file name                             |             |                                |
|---------------------------|-------------------------------------------|-------------|--------------------------------|
| 'folder'                  | path to the results folder                | path        | 'experiments' ('experiments')  |
|---------------------------|-------------------------------------------|-------------|--------------------------------|
| 'model_init'              | init function of the model, the class     | class       | base_model.BaseModel           |
|---------------------------|-------------------------------------------|-------------|--------------------------------|
| 'init_parameters'         | a dict where the keys are the model init  | dict        | {'input_size': [512, 256, 128],|
|                           | arguments and the values are lists of     | k: str      |'hidden_sizes': [[128, 64], []],|
|                           | candidate values (for the search).        | v: list     |'dropout_p': [0, 0.1, 0.2]}     |
|---------------------------|-------------------------------------------|-------------|--------------------------------|
| 'metrics'                 | a dict where the keys are metric names and| dict        | {'accuracy': metrics.accuracy, |
|                           | the values are the metric functions with  | k: str      | 'recall': metrics.recall}      |
|                           | the following signature:                  | v: function |                                |
|                           | metric_func(y_scores, y_pred, y_true)     |             |                                |
|                           | the metrics are used to evaluate the model|             |                                |
|                           | see metrics.py folder for more examples   |             |                                |
|---------------------------|-------------------------------------------|-------------|--------------------------------|
| 'save_model               | if true, saving the model to a pickle file| bool        | True (False)                   |
|                           | every time there is an improvement        |             |                                |
|---------------------------|-------------------------------------------|-------------|--------------------------------|
| 'save_results'            | if true, saving results dataframe to csv. | bool        | True (True)                    |
|---------------------------|-------------------------------------------|-------------|--------------------------------|
| 'verbose'                 | The verbosity level: if non zero, progress| int         | 3 (4)                          |
|                           | messages are printed. Each level includes |             |                                |
|                           | messages from lower level. 0 - no prints. |             |                                |
|                           | 0.5 - only CV search start and end message|             |                                |
|                           | 1 - only start message and end message.   |             |                                |
|                           | 2 - init and prepare fit. 3 - results each|             |                                |
|                           | improvement. 4 - progressbar.             |             |                                |
|---------------------------|-------------------------------------------|-------------|--------------------------------|

TorchTrainer class extra configurations:

|---------------------------|-------------------------------------------|-------------|--------------------------------|
| variable                  | description                               | type        | example (default)              |
|---------------------------|-------------------------------------------|-------------|--------------------------------|
| 'epochs'                  | number of training epochs                 | int         | 10 (10)                        |
|---------------------------|-------------------------------------------|-------------|--------------------------------|
| 'loss_every_x_samples'    | used as a batch replacement. accumulating | int         | 10 (1)                         |
|                           | the loss and average after x samples and  |             |                                |
|                           | then do the backward step.                |             |                                |
|---------------------------|-------------------------------------------|-------------|--------------------------------|
| 'test_every_x_epochs'     | evaluate the model on the train and test  | int         | 1 (1)                          |
|                           | data every x epochs.                      |             |                                |
|---------------------------|-------------------------------------------|-------------|--------------------------------|
| 'convergence_iterations'  | after this number of evaluation iterations| int         | 10 (10)                        |
|                           | without improvements, the training stops  |             |                                |
|---------------------------|-------------------------------------------|-------------|--------------------------------|
| 'convergence_metric'      | the metric which is used for checking     | str         | 'acc' ('loss')                 |
|                           | convergence (higher is better).           |             |                                |
|---------------------------|-------------------------------------------|-------------|--------------------------------|
| 'optimizer'               | pytorch optimizer used for the training   | torch optim | (torch.optim.Adam)             |
|---------------------------|-------------------------------------------|-------------|--------------------------------|
| 'optimizer_kwargs'        | key words args for initializing optimizer | dict        | {'lr': 0.01} ({})              |
|---------------------------|-------------------------------------------|-------------|--------------------------------|
| 'criterion'               | loss function used for the training       | torch loss  | (torch.nn.NLLLoss)             |
|---------------------------|-------------------------------------------|-------------|--------------------------------|
| 'criterion_kwargs'        | key words args for initializing criterion | dict        | {'weight':torch.tensor([1, 3])}|
|---------------------------|-------------------------------------------|-------------|--------------------------------|
| 'tensorboard_port'        | port to bind tensorboard results.         | int         | 7007 (6006)                    |
|                           | If the value is not int or smaller than   |             |                                |
|                           | 1000, TensorBoard will not be launched    |             |                                |
|---------------------------|-------------------------------------------|-------------|--------------------------------|

# ------------- configurations template  ------------- #


configs = {
'name': 'template',
'folder': 'experiments' ,
'model_init': BaseModel,
'init_parameters': {
'para1' : [val1, val2, val3],
'para2': [val1, val2, val3]
},
'metrics': {'acc': accuracy},
'save_model': True,
'save_results': True,
'verbose': 3,
'epochs': 10,
'loss_every_x_samples': 100,
'test_every_x_epochs': 1,
'convergence_iterations': 10,
'convergence_metric': 'loss',
'optimizer': torch.optim.Adam,
'optimizer_kwargs': {'lr': 0.001},
'criterion': torch.nn.NLLLoss,
'criterion_kwargs': {},
'tensorboard_port': 6006
}

"""

# ------------- configurations defaults  ------------- #


from model_selection_framework.metrics import *
import torch


name = 'EXP'
folder = 'experiments'
epochs = 15
metrics = {'accuracy': accuracy, 'recall': recall, 'precision': precision,
           'pred_lift': pred_lift, 'roc_auc': roc_auc,
           'best_threshold': best_threshold}
save_model = True
save_results = True
verbose = 1
loss_every_x_samples = 1
eval_every_x_epochs = 1
convergence_iterations = 10
convergence_metric = 'roc_auc'
optimizer = torch.optim.Adam
optimizer_kwargs = {'lr': 0.0005}
criterion = torch.nn.NLLLoss
criterion_kwargs = {}
tensorboard_port = 6006

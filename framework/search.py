from framework.trainer import fit_trainer, fit_trainer_cv
from framework import utils as U
from torch.utils.data import DataLoader
import os
from itertools import product
from random import sample
import pandas as pd
import datetime
import typing


def prepare_search(configs: dict):
    if not isinstance(configs, dict):
        raise TypeError(f"configs must be of type dict, got {type(configs)}")
    U.kwargs_must_contain(('model_init', 'init_parameters'), **configs)
    configs = configs.copy()
    if 'folder' in configs:
        if not os.path.exists(configs['folder']):
            os.makedirs(configs['folder'])
        output_path = os.path.join(configs['folder'], configs.get('name', '') +
                                   f"{datetime.datetime.now().strftime('%m%d%H%M')}_search_results.csv")
    else:
        output_path = None
    #     :key output_path: path to the file where the output of each search iteration is saved.
    #     :key header: if True, write header line to output_path file in the first search iteration.
    configs['output_path'] = output_path
    configs['init_parameters'] = {k: U.to_iterable(v) for k, v in configs['init_parameters'].items()}
    return configs


def end_search(search_results: pd.DataFrame, configs: dict, return_best: typing.Union[bool, int] = True,
               metric: str = None):
    output_path = configs.get('output_path', None)
    verbose = configs.get('verbose', 0)
    search_results.to_csv(output_path, index=False)
    if verbose >= 0.5:
        print("Hyperparameters Search has ended",
              f"results are saved to: {output_path}" if output_path is not None else "")
    if isinstance(return_best, bool):
        if return_best:
            return_best = 1
        else:
            return_best = None
    elif isinstance(return_best, int):
        if return_best > search_results.shape[0]:
            return_best = search_results.shape[0]
    else:
        return_best = None
    if return_best is None:
        return search_results
    else:
        if metric is None:
            metric = configs.get('convergence_metric')
        metric = metric + '_test'
        if metric not in search_results:
            raise ValueError(f"metric {metric} is missing from search_results")
        search_results = search_results.sort_values(by=metric, ascending=False)
        best_results = search_results.head(return_best).copy()
        best_results['cv_fold'] = 'mean'
        best_results['rank'] = list(range(best_results.shape[0]))
        # case of cv, keeping the top of each cv
        if metric + '_values' in search_results.columns:
            cvs = len(search_results['name_values'].values[0])
            for i in range(cvs):
                search_results = search_results.sort_values(by=metric + '_values', ascending=False,
                                                            key=lambda values: values.apply(lambda x: x[i]))
                fold_best_results = search_results.head(return_best).copy()
                fold_best_results['cv_fold'] = str(i)
                fold_best_results['rank'] = list(range(fold_best_results.shape[0]))
                best_results = pd.concat([best_results, fold_best_results], axis=0)
        # delete all models and results files not included in top_results
        for i, row in search_results.iterrows():
            if i not in best_results.index:
                model_path = row['model_file_path']
                if os.path.exists(model_path):
                    os.remove(model_path)
                results_path = row['results_file_path']
                if os.path.exists(results_path):
                    os.remove(results_path)
                if 'model_file_path_values' in row:
                    for model_path in row['model_file_path_values']:
                        if os.path.exists(model_path):
                            os.remove(model_path)
                if 'results_file_path_values' in row:
                    for results_path in row['results_file_path_values']:
                        if os.path.exists(results_path):
                            os.remove(results_path)
        return best_results


def base_search(configs: dict, train_data: typing.Union[typing.Tuple, DataLoader],
                test_data: typing.Union[typing.Tuple, DataLoader] = None, search_type: str = 'grid',
                max_iter: int = None, cv: typing.Union[int, typing.List] = 1,
                random_state: int = 42):
    """
    :param configs: experiment configs. Must contain 'model_init' which is the model init function.
     The 'init_parameters' value in configs is the init arguments of the model and should be a Dict[str, List],
     where the key(str) is the parameter name and value(List) is a list of possible values of the parameters.
    :param train_data:
    :param test_data:
    :param search_type: should be 'grid' or 'random'
    :param max_iter: number of search iterations
    :param cv: number of cross-validation folds, or a list of folds indices. default is 1.
     Any non int value is considered as 1.
     When cv > 1, test_data won't be used.
    :param random_state: random seed for cv splits.
    :return: search results pd.DataFrame
    """

    init_parameters = configs.get('init_parameters', {})
    output_path = configs.get('output_path', None)
    header = configs.get('header', True)
    parameters_grid = [dict(zip(init_parameters.keys(), values)) for values in product(*init_parameters.values())]
    n_parameters = len(parameters_grid)
    search_results = []
    # if max_iter is None or it is bigger than the number of parameters combinations, we do grid search
    if max_iter is None or max_iter >= n_parameters:
        parameters = parameters_grid
    else:
        if search_type == 'random':
            parameters = sample(parameters_grid, max_iter)
        elif search_type == 'grid':
            parameters = parameters_grid[:max_iter]
        else:
            raise ValueError(f"search_type should be 'random' or 'grid, got {search_type}")
    for i, parameters in enumerate(parameters):
        if (isinstance(cv, int) and cv > 1) or (isinstance(cv, (list, tuple)) and isinstance(cv[0], (list, tuple)) and
                                                isinstance(cv[0][0], int)):
            results = fit_trainer_cv(configs, parameters, train_data, cv, random_state)
        else:
            results = fit_trainer(configs, parameters, train_data, test_data)
        search_results.append(results)
        if output_path is not None:
            results.astype(str).to_csv(output_path, mode='a', header=(i == 0) and header, index=False)
    return pd.concat(search_results, axis=0)


def grid_search(configs: dict, train_data: typing.Union[typing.Tuple, DataLoader],
                test_data: typing.Union[typing.Tuple, DataLoader] = None, max_iter: int = None,
                cv: typing.Union[int, typing.List] = 1, random_state: int = 42, ):
    """
    A grid hyperparameters search.
    :param configs: experiment configs. Must contain 'model_init' which is the model init function.
     The 'init_parameters' value in configs is the init arguments of the model and should be a Dict[str, List],
     where the key(str) is the parameter name and value(List) is a list of possible values of the parameters.
    :param train_data:
    :param test_data:
    :param max_iter: number of search iterations
    :param cv: number of cross-validation folds, or a list of folds indices. default is 1.
     Any non int value is considered as 1.
     When cv > 1, test_data won't be used.
    :param random_state: random seed for cv splits.
    :return: search results pd.DataFrame
    """
    configs = prepare_search(configs)
    return base_search(configs, train_data, test_data, 'grid', max_iter, cv, random_state)


def random_search(configs: dict, train_data: typing.Union[typing.Tuple, DataLoader],
                  test_data: typing.Union[typing.Tuple, DataLoader] = None, max_iter: int = None,
                  cv: typing.Union[int, typing.List] = 1, random_state: int = 42):
    """
    A random hyperparameters search.
    :param configs: experiment configs. Must contain 'model_init' which is the model init function.
     The 'init_parameters' value in configs is the init arguments of the model and should be a Dict[str, List],
     where the key(str) is the parameter name and value(List) is a list of possible values of the parameters.
    :param train_data:
    :param test_data:
    :param max_iter: number of search iterations
    :param cv: number of cross-validation folds, or a list of folds indices. default is 1.
     Any non int value is considered as 1.
     When cv > 1, test_data won't be used.
    :param random_state: random seed for cv splits.
    :return: search results pd.DataFrame
    """
    configs = prepare_search(configs)
    return base_search(configs, train_data, test_data, 'random', max_iter, cv, random_state)


def search_results_to_parameters_results(search_results: pd.DataFrame, metric: str):
    parameters_results = []
    for h, m in search_results[['hyperparameters', metric]].values:
        if isinstance(h, str):
            h = eval(h)
        h.update({metric: m})
        parameters_results.append(h)
    return pd.DataFrame(parameters_results)


def greedy_search(configs: dict, train_data: typing.Union[typing.Tuple, DataLoader],
                  test_data: typing.Union[typing.Tuple, DataLoader] = None, max_iter: int = None,
                  max_rounds: int = None, random_start: int = None, beam_size: int = 1, metric: str = None,
                  top_repetitions: int = None, repeat_n_tops: int = 0,
                  cv: typing.Union[int, typing.List] = 1, random_state: int = 42):
    """
    A beam-greedy hyperparameters search.
    Each round (round is name of checking all the possible values of all the hyper-parameters),
    we check all the possible values of each single hyper-parameters by using the best `beam_size` (a number)
     combinations of the other hyper-parameters. The best combinations are according to the `metric`,
    therefore higher scores of the metric should indicate a better performances. If it is not the case
    (i.e. lower scores are better), you should create a new metric which returns the -1 * score (see metrics.py file).
    This is why the search is greedy - we always use the best combinations and trying to replace
    a single value at each search iteration.
    If the search is converged - i.e. we checked all the possible combinations or for each hyper-parameter we checked
    all the values all the beam combinations, then the search is stopped.
    We do a random search at first to initialized the combinations selected by the beam.
    :param configs: experiment configs. Must contain 'model_init' which is the model init function.
     The 'init_parameters' value in configs is the init arguments of the model and should be a Dict[str, List],
     where the key(str) is the parameter name and value(List) is a list of possible values of the parameters.
    :param train_data:
    :param test_data:
    :param max_iter: number of search iterations. If None, then there is no limit on the search iterations
    (actually, no more than 9999 iterations).
    :param max_rounds: maximum number of rounds. If None, then there is no limit on the rounds.
    (actually, no more than 999 rounds).
    :param random_start: number of random search iterations before doing the greedy part of the search. Higher number
    can lead to not missing a good combination.
    :param beam_size: an int. The bigger the `beam_size` is, the less greedy the search is.
    :param metric: a string, see the notes in the function documentation.
    :param top_repetitions: after completing all the iterations as defined by max_iter and max_rounds, repeat the top
    best search results. This parameters states the number of repetitions for each top result. default None same as 0.
    :param top_repetitions: an int. the number of top results to repeat.
    :param cv: number of cross-validation folds, or a list of folds indices. default is 1.
     Any non int value is considered as 1.
     When cv > 1, test_data won't be used.
    :param random_state: random seed for cv splits.
    :return: search results pd.DataFrame
    """
    configs = prepare_search(configs)
    if metric is None:
        metric = configs.get('convergence_metric')
    metric = metric + '_test'
    init_parameters = configs.get('init_parameters', {})
    if random_start is None:
        # if random_start is None, then we do one iteration of random, and these parameters will be the base.
        random_start = 1
    if max_iter is None:
        max_iter = 9999
    if max_rounds is None:
        max_rounds = 999
    if random_start > max_iter:
        random_start = max_iter
    current_iter, current_round, parameters = 0, 0, list(init_parameters.keys())
    search_results = []
    parameters_results = pd.DataFrame(columns=parameters + [metric])
    random_results = base_search(configs, train_data, test_data, 'random', random_start, cv, random_state)
    search_results.append(random_results.copy())
    random_results = search_results_to_parameters_results(random_results, metric)
    parameters_results = pd.concat([parameters_results, random_results], axis=0)
    current_iter = random_start
    converged = False
    already_checked_combinations = set([U.dict_to_ordered_str(p)
                                        for p in parameters_results[parameters].to_dict('records')])
    while current_iter < max_iter and current_round < max_rounds and not converged:
        converged = True
        for parameter in parameters:
            # we find the best beam candidates by sorting and then transforming the DF to list of dicts{para: value}
            # the top_beams of a given parameter, are the best other-parameters combinations
            # (and this is why the drop_duplicates)
            parameters_results = parameters_results.reset_index(drop=True)
            top_beams = parameters_results.astype({p: str for p in parameters}).sort_values(by=metric, ascending=False)\
                                          .drop_duplicates([p for p in parameters if p != parameter]).head(beam_size)
            top_beams = parameters_results.loc[top_beams.index, parameters].to_dict('records')
            # we iterate over the top beams combinations, then we will replace the `parameter` value.
            # we use the following code to prepare the possible candidate combinations.
            for beam_parameters in top_beams:
                parameter_candidate_values = []
                for parameter_value in init_parameters[parameter]:
                    # the following lines check if parameter_value + beam_parameters has been checked before
                    current_parameters = beam_parameters.copy()
                    current_parameters[parameter] = parameter_value
                    str_current_parameters = U.dict_to_ordered_str(current_parameters)
                    if str_current_parameters in already_checked_combinations:
                        continue
                    else:
                        already_checked_combinations.add(str_current_parameters)
                        parameter_candidate_values.append(parameter_value)
                if len(parameter_candidate_values) > 0:
                    # preparing the init_parameters value - should be a dict of {parameter_name: [value1, value2, ...]}
                    beam_parameters = {k: [v] for k, v in beam_parameters.items()}
                    beam_parameters[parameter] = parameter_candidate_values
                    beam_configs = configs.copy()
                    beam_configs['header'] = False
                    beam_configs['init_parameters'] = beam_parameters
                    beam_iters = min(max_iter - current_iter, len(parameter_candidate_values))
                    results = base_search(beam_configs, train_data, test_data, 'grid', beam_iters, cv, random_state)
                    search_results.append(results.copy())
                    parameter_results = search_results_to_parameters_results(results, metric)
                    parameters_results = pd.concat([parameters_results, parameter_results], axis=0)
                    converged = False
                    current_iter += beam_iters
                if current_iter >= max_iter:
                    break
            if current_iter >= max_iter:
                break
        current_round += 1
    # repetitions of the top results - only if top_repetitions and repeat_n_tops are used.
    if top_repetitions is not None and top_repetitions > 1 and repeat_n_tops > 0:
        parameters_results = parameters_results.reset_index(drop=True)
        top_beams = parameters_results.astype({p: str for p in parameters})\
                                 .sort_values(by=metric, ascending=False).head(top_repetitions)
        top_beams = parameters_results.loc[top_beams.index, parameters].to_dict('records')
        for i in range(repeat_n_tops):
            for beam_parameters in top_beams:
                beam_parameters = {k: [v] for k, v in beam_parameters.items()}
                beam_configs = configs.copy()
                beam_configs['header'] = False
                beam_configs['init_parameters'] = beam_parameters
                results = base_search(beam_configs, train_data, test_data, 'grid', repeat_n_tops, cv, random_state)
                search_results.append(results.copy())
    search_results = pd.concat(search_results, axis=0)
    return search_results


def search(configs: dict, train_data: typing.Union[typing.Tuple, DataLoader],
           test_data: typing.Union[typing.Tuple, DataLoader] = None, search_type: str = 'random', max_iter: int = None,
           return_best: typing.Union[bool, int] = True, metric: str = None,
           cv: typing.Union[int, typing.List] = 1, random_state: int = 42, **kwargs):
    """
     A hyperparameters search. See random_search(), grid_search() and greedy_search() for more details.
    :param configs: a dict of the experiment configs. Must contain 'model_init' which is the model init function.
     The 'init_parameters' value in configs is the init arguments of the model and should be a Dict[str, List],
     where the key(str) is the parameter name and value(List) is a list of possible values of the parameters.
    :param train_data:
    :param test_data:
    :param search_type: a string. search_type should be one of the following: 'random', 'grid', 'greedy'
    :param max_iter: an int. The number of search iterations
    :param return_best: a bool or an int. If True returns the best result (according to metric),
     If False, returns all results. If an int return all top n results (n is equal to return_best value).
     If cv is used, returns the top n results of each cv, including the top global mean n results.
     Two new columns will be added, rank and cv_fold. Also deletes all the model files and result files of not
     included in the returned results.
    :param metric: A string. The best combination of hyperparameters is selected according to the `metric` score,
    therefore higher scores of the metric should indicate a better performances. If it is not the case
    (i.e. lower scores are better), you should create a new metric which returns the -1 * score (see metrics.py file).
    :param cv: number of cross-validation folds, or a list of folds indices. default is 1.
     Any non int value is considered as 1.
     When cv > 1, test_data won't be used.
    :param random_state: random seed for cv splits.
    :param kwargs: see greedy_search() function parameters.
    :return: a dict of the best hyper-parameter values.
    """
    if search_type == 'random':
        search_results = random_search(configs, train_data, test_data, max_iter, cv, random_state)
    elif search_type == 'grid':
        search_results = grid_search(configs, train_data, test_data, max_iter, cv, random_state)
    elif search_type == 'greedy':
        search_results = greedy_search(configs, train_data, test_data, max_iter, metric=metric, cv=cv,
                                       random_state=random_state, **kwargs)
    else:
        raise ValueError(f"search_type should be 'random', 'grid' or 'greedy', but got {search_type}")
    return end_search(search_results.reset_index(drop=True), configs, return_best, metric)

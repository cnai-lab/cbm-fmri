from typing import NoReturn, DefaultDict
import datetime
import pandas as pd
import numpy as np
from conf_pack.paths import *
from conf_pack.configuration import default_params, c
from collections import defaultdict
from typing import List


def write_time_of_function(func_name: str, old_time: datetime.datetime) -> NoReturn:
    df = pd.DataFrame()
    row = {'function': [func_name], 'time': [datetime.datetime.now() - old_time]}
    row = pd.DataFrame(row)
    name_file = 'function_times.csv'
    if os.path.exists(name_file):
        df = pd.read_csv(name_file)
    df = pd.concat([df, row])
    df.to_csv(name_file, index=False)


def load_graphs_features(filter_type: str, thresh: float):
    return pd.read_pickle(os.path.join('Graphs_pickle', filter_type, f'graph_{thresh}.pkl'))


def write_selected_features(feature_names: List[str], info_gain: List[float]) -> NoReturn:
    df = defaultdict(list)

    full_path = os.path.join(get_results_path(), 'selected_features.csv')
    if os.path.exists(full_path):
        df = pd.read_csv(full_path).to_dict(orient='list')

    for feature, info_feature in zip(feature_names, info_gain):
        if feature in df['feature_names']:
            idx = df['feature_names'].index(feature)
            df['number_of_times_chosen'][idx] = df['number_of_times_chosen'][idx] + 1
            df['summed_inf_gain'][idx] = df['summed_inf_gain'][idx] + info_feature

        else:
            df['feature_names'].append(feature)
            df['number_of_times_chosen'].append(1)
            df['summed_inf_gain'].append(info_feature)

    pd.DataFrame(df).to_csv(full_path, index=False)



def get_y_true() -> np.ndarray:
    df = get_meta_data()
    df.sort_values(by=['Subject'], inplace=True)
    return df['Class'].values


def get_save_path() -> str:
    save_mapping_by_proj = {'stroke': STROKE_SAVE_PATH_PARENT, 'adhd': ADHD_SAVE_PATH_PARENT}
    project_type = default_params.get('project')
    return save_mapping_by_proj[project_type]


def get_data_path() -> str:
    data_path_mapping_by_proj = {'stroke': SCANS_DIR_BEFORE, 'adhd': ADHD_DATA_PATH}
    project_type = default_params.get('project')
    return data_path_mapping_by_proj[project_type]


def get_meta_data() -> pd.DataFrame:
    data_path_mapping_by_proj = {'stroke': STROKE_EXCEL_DATA, 'adhd': ADHD_EXCEL_DATA}
    project_type = default_params.get('project')
    return pd.read_excel(data_path_mapping_by_proj[project_type])


def get_results_path() -> str:
    if default_params.get('result_path') == 'default':
        time = datetime.datetime.now()
        format_time = f'{time.year}_{time.month}_{time.day}_{time.hour}_{time.minute}'
        c.set('Default Params', 'result_path', format_time)
    full_path = os.path.join(get_save_path(), default_params.get('result_path'))
    os.makedirs(full_path, exist_ok=True)
    return full_path


def save_results(perf: DefaultDict) -> NoReturn:
    res = defaultdict(list)
    for (criteria, num_features), acc in perf.items():
        res['criteria'].append(criteria)
        res['number of features'].append(num_features)
        res['Accuracy'].append(acc)
    pd.DataFrame(res).to_csv(os.path.join(get_results_path(), 'results.csv'))


def get_names() -> List[str]:
    path = get_data_path()
    file = open(os.path.join(path, 'names.txt'))
    names = file.read().split('\n')
    return names[:-1]


if __name__ == '__main__':
    print(get_names())
#     d_type = 'correlation'
#     for file in os.listdir(os.path.join(get_data_path(),    )
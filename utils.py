from typing import NoReturn, DefaultDict
import datetime
import pandas as pd
from conf_pack.paths import *
from conf_pack.configuration import *
from collections import defaultdict

def write_time_of_function(func_name: str, old_time: datetime.datetime) -> NoReturn:
    df = pd.DataFrame()
    row = {'function': [func_name], 'time': [datetime.datetime.now() - old_time]}
    row = pd.DataFrame(row)
    name_file = 'function_times.csv'
    if os.path.exists(name_file):
        df = pd.read_csv(name_file)
    df = pd.concat([df, row])
    df.to_csv(name_file, index=False)


def get_y_true() -> np.ndarray:
    df = pd.read_excel(EXCEL_DATA)
    df.sort_values(by=[default_params.get('subject')], inplace=True)
    return df[default_params.get('class_name')].values


def get_results_path() -> str:
    if default_params.get('result_path') == 'default':
        time = datetime.datetime.now()
        format_time = f'{time.year}_{time.month}_{time.day}_{time.hour}_{time.minute}'
        c.set('Default Parameters', 'result_path', format_time)
    full_path = os.path.join(SAVE_PATH_PARENT, default_params.get('result_path'))
    os.makedirs(full_path, exist_ok=True)
    return full_path


def save_results(perf: DefaultDict) -> NoReturn:
    res = defaultdict(list)
    for (criteria, num_features), acc in perf.keys():
        res['criteria'].append(criteria)
        res['number of features'].append(num_features)
        res['Accuracy'].append(acc)
    pd.DataFrame(res).to_csv(os.path.join(get_results_path(), 'results.csv'))
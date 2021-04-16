from configparser import ConfigParser
import numpy as np

c = ConfigParser()
c.read('config.ini')
default_params = c['Default Params']
step = 0.01
tune_parameters = {'threshold':  np.arange(default_params.getfloat('min_thresh'),
                                           default_params.getfloat('max_thresh') + step, step),
                   'density': np.arange(default_params.getfloat('min_dens'),
                                        default_params.getfloat('max_dens') + step, step),
                   }


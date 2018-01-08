import pandas as pd
import pkg_resources
import logging
import numpy as np

INSULIN_PUMP_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/pump_params.csv')
logger = logging.getLogger(__name__)


class InsulinPump(object):
    U2PMOL = 6000

    def __init__(self, params):
        self._params = params

    @classmethod
    def withName(cls, name):
        pump_params = pd.read_csv(INSULIN_PUMP_PARA_FILE)
        params = pump_params.loc[pump_params.Name == name].squeeze()
        return cls(params)

    def bolus(self, amount):
        bol = amount * self.U2PMOL  # convert from U/min to pmol/min
        bol = np.round(bol / self._params['inc_bolus']
                       ) * self._params['inc_bolus']
        bol = bol / self.U2PMOL     # convert from pmol/min to U/min
        bol = min(bol, self._params['max_bolus'])
        bol = max(bol, self._params['min_bolus'])
        return bol

    def basal(self, amount):
        bas = amount * self.U2PMOL  # convert from U/min to pmol/min
        bas = np.round(bas / self._params['inc_basal']
                       ) * self._params['inc_basal']
        bas = bas / self.U2PMOL     # convert from pmol/min to U/min
        bas = min(bas, self._params['max_basal'])
        bas = max(bas, self._params['min_basal'])
        return bas

    def reset(self):
        logger.info('Resetting insulin pump ...')
        pass

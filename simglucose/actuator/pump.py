import pandas as pd
import pkg_resources
import logging
import numpy as np

INSULIN_PUMP_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/pump_params.csv')
logger = logging.getLogger(__name__)


class InsulinPump(object):
    U2PMOL = 6000
    MINUTES_PER_HOUR = 60.0

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
        """Quantize a U/min basal request using the pump's U/hour limits."""
        rate_u_hour = amount * self.MINUTES_PER_HOUR
        rate_u_hour = np.round(
            rate_u_hour / self._params["inc_basal"]
        ) * self._params["inc_basal"]
        rate_u_hour = min(rate_u_hour, self._params["max_basal"])
        rate_u_hour = max(rate_u_hour, self._params["min_basal"])
        return rate_u_hour / self.MINUTES_PER_HOUR

    @property
    def max_basal(self):
        """Maximum basal rate exposed by the controller API, in U/min."""
        return self._params["max_basal"] / self.MINUTES_PER_HOUR

    @property
    def min_basal(self):
        """Minimum basal rate exposed by the controller API, in U/min."""
        return self._params["min_basal"] / self.MINUTES_PER_HOUR

    def reset(self):
        logger.info('Resetting insulin pump ...')
        pass

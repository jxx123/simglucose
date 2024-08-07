import pandas as pd
import pkg_resources
import logging
import numpy as np

INSULIN_PUMP_PARA_FILE = pkg_resources.resource_filename(
    "simglucose", "params/pump_params.csv"
)
logger = logging.getLogger(__name__)


class InsulinPump(object):

    def __init__(self, params):
        self._params = params

    @classmethod
    def withName(cls, name):
        pump_params = pd.read_csv(INSULIN_PUMP_PARA_FILE)
        params = pump_params.loc[pump_params.Name == name].squeeze()
        return cls(params)

    def bolus(self, amount):
        # inc_bolus and max_bolus are in U
        bol = np.round(amount / self._params["inc_bolus"]) * self._params["inc_bolus"]
        bol = min(bol, self._params["max_bolus"])
        bol = max(bol, self._params["min_bolus"])
        return bol

    def basal(self, amount):
        bas = amount * 60.0  # convert from U/min to U/h
        bas = np.round(bas / self._params["inc_basal"]) * self._params["inc_basal"]
        bas = bas / 60.0  # convert from U/h to U/min
        bas = min(bas, self._params["max_basal"])
        bas = max(bas, self._params["min_basal"])
        return bas

    def reset(self):
        logger.info("Resetting insulin pump ...")
        pass

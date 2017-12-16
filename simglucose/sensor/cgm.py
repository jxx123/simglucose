from .noise_gen import CGMNoiseGenerator
import pandas as pd
import logging
import pkg_resources

logger = logging.getLogger(__name__)
SENSOR_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/sensor_params.csv')


class CGMSensor(object):
    def __init__(self, params, seed=None):
        self._params = params
        self.name = params.Name
        self._noise_generator = CGMNoiseGenerator(
            params, seed=seed).gen_noise()
        self.sample_time = params.sample_time
        self.seed = seed
        self._last_CGM = 0

    @classmethod
    def withName(cls, name, **kwargs):
        sensor_params = pd.read_csv(SENSOR_PARA_FILE)
        params = sensor_params.loc[sensor_params.Name == name].squeeze()
        return cls(params, **kwargs)

    def measure(self, patient):
        if patient.t % self.sample_time == 0:
            logger.info(
                't = {} min, CGM is measuring blood glucose'.format(patient.t))
            BG = patient.observation.Gsub
            CGM = BG + next(self._noise_generator)
            CGM = max(CGM, self._params["min"])
            CGM = min(CGM, self._params["max"])
            self._last_CGM = CGM
            return CGM

        # Zero-Order Hold
        return self._last_CGM

    def reset(self):
        logger.info('Resetting CGM sensor ...')
        self._noise_generator = CGMNoiseGenerator(
            self._params, self.seed).gen_noise()


if __name__ == '__main__':
    pass

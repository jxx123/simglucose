import numpy as np
from scipy.interpolate import interp1d
import math
from collections import deque
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def johnson_transform_SU(xi, lam, gamma, delta, x):
    return xi + lam * np.sinh((x - gamma) / delta)


class CGMNoiseGenerator(object):
    PRECOMPUTE = 10  # length of pre-compute noise sequence
    MDL_SAMPLE_TIME = 15

    def __init__(self, params, seed=None):
        self._params = params
        self.seed = seed
        self._noise15_gen = self._noise15_generator()
        self._noise_init = next(self._noise15_gen)

    def _noise15_generator(self, num=np.Inf):
        np.random.seed(self.seed)
        e = np.random.randn()
        count = 0
        while count < num:
            eps = johnson_transform_SU(self._params["xi"],
                                       self._params["lambda"],
                                       self._params["gamma"],
                                       self._params["delta"],
                                       e)
            yield eps
            e = self._params["PACF"] * (e + np.random.randn())
            count += 1

    def _get_noise_seq(self):
        # To make the noise sequence continous, keep the last noise as the
        # beginning of the new sequence
        noise15 = [self._noise_init]
        noise15.extend([next(self._noise15_gen)
                        for _ in range(self.PRECOMPUTE)])
        self._noise_init = noise15[-1]

        noise15 = np.array(noise15)
        t15 = np.array(range(0, len(noise15))) * self.MDL_SAMPLE_TIME

        nsample = int(math.floor(
            self.PRECOMPUTE * self.MDL_SAMPLE_TIME / self._params["sample_time"])) + 1
        t = np.array(range(0, nsample)) * self._params["sample_time"]

        interp_f = interp1d(t15, noise15, kind='cubic')
        noise = interp_f(t)
        noise2return = deque(noise[1:])

        logger.debug('New noise sampled every 15 min:\n{}'.format(noise15))
        logger.debug('New noise sequence:\n{}'.format(noise2return))

        # plt.plot(t15, noise15, 'o')
        # plt.plot(t, noise, '.-')
        # plt.show()

        return noise2return

    def gen_noise(self, num=np.Inf):
        count = 0
        noise = deque()
        while count < num:
            if len(noise) == 0:
                logger.info('Generating a new noise sequence ...')
                noise = self._get_noise_seq()
            yield noise.popleft()
            count += 1


if __name__ == '__main__':
    # logger.setLevel(logging.INFO)
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # ch.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter(
        '%(name)s: %(levelname)s: %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    params = {'PACF': 0.7,
              'gamma': -0.5444,
              'lambda': 15.9574,
              'delta': 1.6898,
              'xi': -5.47,
              'sample_time': 3,
              'min': 39.0,
              'max': 600.0}
    cgm = CGMNoiseGenerator(params, seed=1)
    noise = [n for n in cgm.gen_noise(num=1000)]
    plt.plot(noise)
    plt.show()

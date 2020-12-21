import numpy as np
from scipy.interpolate import interp1d
import math
from collections import deque
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def johnson_transform_SU(xi, lam, gamma, delta, x):
    return xi + lam * np.sinh((x - gamma) / delta)


class CGMNoise(object):
    PRECOMPUTE = 10  # length of pre-compute noise sequence
    MDL_SAMPLE_TIME = 15

    def __init__(self, params, n=np.inf, seed=None):
        self._params = params
        self.seed = seed
        # self._noise15_gen = self._noise15_generator()
        self._noise15_gen = noise15_iter(params, seed=seed)
        self._noise_init = next(self._noise15_gen)

        self.n = n
        self.count = 0
        self.noise = deque()

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

        # logger.debug('New noise sampled every 15 min:\n{}'.format(noise15))
        # logger.debug('New noise sequence:\n{}'.format(noise2return))

        # plt.plot(t15, noise15, 'o')
        # plt.plot(t, noise, '.-')
        # plt.show()

        return noise2return

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < self.n:
            if len(self.noise) == 0:
                logger.debug('Generating a new noise sequence ...')
                self.noise = self._get_noise_seq()
            self.count += 1
            return self.noise.popleft()
        else:
            raise StopIteration()


class noise15_iter:
    def __init__(self, params, seed=None, n=np.inf):
        self.seed = seed
        self.rand_gen = np.random.RandomState(self.seed)
        self._params = params
        self.n = n
        self.e = 0
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.count == 0:
            self.e = self.rand_gen.randn()
        elif self.count < self.n:
            self.e = self._params["PACF"] * (self.e + self.rand_gen.randn())
        else:
            raise StopIteration()
        eps = johnson_transform_SU(self._params["xi"],
                                   self._params["lambda"],
                                   self._params["gamma"],
                                   self._params["delta"],
                                   self.e)
        self.count += 1
        return eps

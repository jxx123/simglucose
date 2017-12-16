import numpy as np
from scipy.stats import truncnorm
from collections import namedtuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
Action = namedtuple('scenario_action', ['meal'])


class ScenarioGenerator(object):
    def __init__(self, seed=None):
        self.seed = seed
        self.scenario = self.create_scenario(self.seed)

    def get_action(self, t):
        # t must be datetime.datetime object
        delta_t = t - datetime.combine(t.date(), datetime.min.time())
        t_sec = delta_t.total_seconds()

        if t_sec < 1:
            logger.info('Creating new one day scenario ...')
            if self.seed:
                # Make scenario repeatable, but varies every day
                self.seed += 1
            self.scenario = self.create_scenario(self.seed)

        t_min = np.floor(t_sec / 60.0)

        if t_min in self.scenario['meal']['time']:
            logger.info('Time for meal!')
            idx = self.scenario['meal']['time'].index(t_min)
            return Action(meal=self.scenario['meal']['amount'][idx])
        else:
            return Action(meal=0)

    @staticmethod
    def create_scenario(seed):
        np.random.seed(seed)
        scenario = {'meal': {'time': [],
                             'amount': []}}

        # Probability of taking each meal
        # [breakfast, snack1, lunch, snack2, dinner, snack3]
        prob = [0.95, 0.3, 0.95, 0.3, 0.95, 0.3]
        time_lb = np.array([5, 9, 10, 14, 16, 20]) * 60
        time_ub = np.array([9, 10, 14, 16, 20, 23]) * 60
        time_mu = np.array([7, 9.5, 12, 15, 18, 21.5]) * 60
        time_sigma = np.array([60, 30, 60, 30, 60, 30])
        amount_mu = [45, 10, 70, 10, 80, 10]
        amount_sigma = [10, 5, 10, 5, 10, 5]

        for p, tlb, tub, tbar, tsd, mbar, msd in zip(
                prob, time_lb, time_ub, time_mu, time_sigma,
                amount_mu, amount_sigma):
            if np.random.rand() < p:
                truncnorm_gen = truncnorm((tlb - tbar) / tsd,
                                          (tub - tbar) / tsd,
                                          loc=tbar, scale=tsd)
                scenario['meal']['time'].append(round(truncnorm_gen.rvs()))
                scenario['meal']['amount'].append(
                    max(round(np.random.normal(mbar, msd)), 0))

        return scenario

    def reset(self):
        self.scenario = self.create_scenario(self.seed)


if __name__ == '__main__':
    from datetime import time
    from datetime import timedelta
    import copy
    now = datetime.now()
    t0 = datetime.combine(now.date(), time(6, 0, 0, 0))
    t = copy.deepcopy(t0)
    sim_time = timedelta(days=2)

    scenario = ScenarioGenerator(seed=1)
    m = []
    T = []
    while t < t0 + sim_time:
        action = scenario.get_action(t)
        m.append(action.meal)
        T.append(t)
        t += timedelta(minutes=1)

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.plot(T, m)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(mdates.AutoDateLocator())
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))
    plt.show()

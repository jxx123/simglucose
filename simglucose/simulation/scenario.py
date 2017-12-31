import logging
from collections import namedtuple
from datetime import datetime
from datetime import timedelta

logger = logging.getLogger(__name__)
Action = namedtuple('scenario_action', ['meal'])


class Scenario(object):
    def __init__(self, start_time=None):
        if start_time is None:
            now = datetime.now()
            start_hour = timedelta(hours=float(
                input('Input simulation start time (hr): ')))
            start_time = datetime.combine(now.date(),
                                          datetime.min.time()) + start_hour
            print('Simulation start time is set to {}.'.format(start_time))
        self.start_time = start_time

    def get_action(self, t):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class CustomScenario(Scenario):
    def __init__(self, start_time=None, scenario=None):
        '''
        scenario - a list of tuples (time, action), where time is a datetime or
                   timedelta or double, action is a namedtuple defined by
                   scenario.Action. When time is a timedelta, it is
                   interpreted as the time of start_time + time. Time in double
                   type is interpreted as time in timedelta with unit of hours
        '''
        Scenario.__init__(self, start_time=start_time)
        if scenario is None:
            scenario = self.input_scenario()
        self.scenario = scenario

    def get_action(self, t):
        times, actions = tuple(zip(*self.scenario))
        times2compare = [parseTime(time, self.start_time) for time in times]
        if t in times2compare:
            idx = times2compare.index(t)
            return Action(meal=actions[idx])
        else:
            return Action(meal=0)

    def reset(self):
        pass

    @staticmethod
    def input_scenario():
        scenario = []

        print('Input a custom scenario ...')
        breakfast_time = float(input('Input breakfast time (hr): '))
        breakfast_size = float(input('Input breakfast size (g): '))
        scenario.append((breakfast_time, breakfast_size))

        lunch_time = float(input('Input lunch time (hr): '))
        lunch_size = float(input('Input lunch size (g): '))
        scenario.append((lunch_time, lunch_size))

        dinner_time = float(input('Input dinner time (hr): '))
        dinner_size = float(input('Input dinner size (g): '))
        scenario.append((dinner_time, dinner_size))

        while True:
            snack_time = float(input('Input snack time (hr): '))
            snack_size = float(input('Input snack size (g): '))
            scenario.append((snack_time, snack_size))

            go_on = input('Continue input snack (y/n)? ')
            if go_on == 'n':
                break
            elif go_on == 'y':
                continue
            else:
                go_on = input('Continue input snack (y/n)? ')
        return scenario


def parseTime(time, start_time):
    if isinstance(time, (int, float)):
        t = start_time + timedelta(minutes=round(time * 60.0))
    elif isinstance(time, timedelta):
        t_sec = time.total_seconds()
        t_min = round(t_sec / 60.0)
        t = start_time + timedelta(minutes=t_min)
    elif isinstance(time, datetime):
        t = time
    else:
        raise ValueError('Expect time to be int, float, timedelta, datetime')
    return t

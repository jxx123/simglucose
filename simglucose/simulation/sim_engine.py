import logging
import time
import os
from collections import Counter
from uuid import uuid4

pathos = True
try:
    from pathos.multiprocessing import ProcessPool as Pool
except ImportError:
    print('You could install pathos to enable parallel simulation.')
    pathos = False

logger = logging.getLogger(__name__)


class SimObj(object):
    def __init__(self,
                 env,
                 controller,
                 sim_time,
                 animate=True,
                 path=None):
        self.env = env
        self.controller = controller
        self.sim_time = sim_time
        self.animate = animate
        self._ctrller_kwargs = None
        self.path = path

    def simulate(self):
        self.controller.reset()
        obs, reward, done, info = self.env.reset()
        tic = time.time()
        while self.env.time < self.env.scenario.start_time + self.sim_time:
            if self.animate:
                self.env.render()
            action = self.controller.policy(obs, reward, done, **info)
            obs, reward, done, info = self.env.step(action)
        toc = time.time()
        logger.info('Simulation took {} seconds.'.format(toc - tic))

    def results(self):
        return self.env.show_history()

    def save_results(self, filename=None):
        df = self.results()
        os.makedirs(self.path, exist_ok=True)
        if filename is None:
            filename = str(self.env.patient.name) + '.csv'
        df.to_csv(os.path.join(self.path, filename))

    def reset(self):
        self.env.reset()
        self.controller.reset()


def sim(sim_object, filename=None):
    """Run and save one simulation, optionally choosing its CSV filename."""
    print("Process ID: {}".format(os.getpid()))
    print('Simulation starts ...')
    sim_object.simulate()
    if filename is None:
        sim_object.save_results()
    else:
        sim_object.save_results(filename=filename)
    print('Simulation Completed!')
    return sim_object.results()


def _batch_filenames(sim_instances):
    """Allocate names in input order, before workers can race to save them."""
    def target(instance, filename):
        # A subclass may implement non-filesystem persistence and keep the
        # constructor's default path=None. Leave that writer in control.
        if instance.path is None:
            return None
        return os.path.normcase(os.path.realpath(
            os.path.join(instance.path, filename)))

    names = [str(s.env.patient.name) + '.csv' for s in sim_instances]
    targets = [target(s, name) for s, name in zip(sim_instances, names)]
    counts = Counter(targets)
    used = set(targets)  # Reserve unique patients' original names, too.
    suffixes = Counter()
    filenames = []
    for instance, name, key in zip(sim_instances, names, targets):
        if key is None or counts[key] == 1:
            filenames.append(None)
            continue
        while True:
            suffixes[key] += 1
            candidate = '{}__{}.csv'.format(name[:-4], suffixes[key])
            destination = target(instance, candidate)
            if destination not in used:
                used.add(destination)
                filenames.append(candidate)
                break
    return filenames


def batch_sim(sim_instances, parallel=False):
    """Run independent experiments and return frames in input order.

    Unique output paths retain ``patient.csv``. Repeated patient/path pairs
    receive deterministic ``patient__N.csv`` names, skipping names reserved
    by other tasks. Repeating a batch updates those same files. Separate
    concurrent batches should use separate output directories.
    """
    tic = time.time()
    sim_instances = list(sim_instances)
    filenames = _batch_filenames(sim_instances)
    if parallel and pathos and sim_instances:
        # pathos caches pools by ID, and its context manager does not clear
        # them. Own a distinct pool and release it even if a task fails.
        with Pool(id=uuid4().hex) as p:
            try:
                results = p.map(sim, sim_instances, filenames)
            finally:
                p.clear()
    else:
        if parallel and not pathos:
            print('Simulation is using single process even though parallel=True.')
        results = [sim(s, filename) for s, filename in
                   zip(sim_instances, filenames)]
    toc = time.time()
    print('Simulation took {} sec.'.format(toc - tic))
    return results

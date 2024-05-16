from .base import Controller
from .base import Action
import cvxpy as cp
import logging
from scipy import sparse
import numpy as np
from fireTS.models import NARX
from sklearn.linear_model import LinearRegression
#  from sklearn.preprocessing import StandardScaler
from simglucose.utils import fetch_patient_params, fetch_patient_quest

logger = logging.getLogger(__name__)


class PatientModel:
    NU = 1

    def __init__(self, patient_name, na, nb, nc, X, y):
        """
        params - parameters of a linear ARX model. params should be an array
        with first na parameters representing the coefficients of the
        auto-regressors, and the rest of the parameters representing the
        coefficients of the exogenous inputs.
        """
        assert na > 0 and nb > 0
        self.patient_name = patient_name
        self._na = na  # auto order
        self._nb = nb  # insulin order
        self._nc = nc  # meal order
        #  self.scaler = StandardScaler()
        self.quests = fetch_patient_quest(patient_name)
        self.patient_params = fetch_patient_params(patient_name)
        self.mdl = NARX(LinearRegression(), auto_order=na, exog_order=[nb, nc])
        self._fit(X, y)
        self._A, self._B = self._construct_state_matrix()

    def _construct_state_matrix(self):
        A = sparse.lil_matrix(np.zeros([self.nx, self.nx]))
        A[0, :self.na] = self.alpha
        A[1:self.na, :(self.na - 1)] = np.identity(self.na - 1)
        if self.nb > 1:
            A[0, self.na:] = self.beta[1:]
            A[(self.na + 1):(self.na + self.nb - 1),
              self.na:(self.na + self.nb - 2)] = np.identity(self.nb - 2)

        # Assume there is only 1 input.
        B = sparse.lil_matrix(np.zeros([self.nx, self.nu]))
        B[0] = self.beta[0]
        B[self.na] = 1
        return A, B

    @property
    def basal(self):
        if self.patient_params:
            # unit: pmol/(L*kg)
            u2ss = self.patient_params['u2ss']
            # unit: kg
            BW = self.patient_params['BW']
        else:
            # NOTE: This need to be revisited. Does it make sense to use
            # average? Use average stats if the params is missing.
            u2ss = 1.43  # unit: pmol/(L*kg)
            BW = 57.0  # unit: kg

        basal = u2ss * BW / 6000  # unit: U/min
        return basal

    @property
    def Gb(self):
        """
        Subcutaneous BG value at the stable state.
        """
        return self.patient_params['Gb']

    def _fit(self, X, y):
        """
        X: [insulin, meal]
        y: CGM readings
        """
        # Centered at the balance points.
        X[:, 0] = X[:, 0] - self.basal
        y = y - self.Gb

        # TODO: verify if standardizing the data helps with the modeling.
        #  X, y = self.scaler.fit_transform(X, y)
        self.mdl.fit(X, y)

    @property
    def params(self):
        # TODO: expose base_estimator coefficients API at the fireTS model level.
        # Going into the internal interface is not a good practice.
        return self.mdl.base_estimator.coef_

    @property
    def alpha(self):
        return self.params[:self.na]

    @property
    def beta(self):
        # NOTE: This assumes the rest of the parameters are the coefficients
        # for insulin. This need to be modified if the model is no longer a
        # simple linear ARX model.
        return self.params[self.na:(self.na + self.nb)]

    @property
    def na(self):
        return self._na

    @property
    def nb(self):
        return self._nb

    @property
    def A(self) -> sparse.lil_matrix:
        return self._A

    @property
    def B(self) -> sparse.lil_matrix:
        return self._B

    @property
    def nx(self):
        return self.na + self.nb - 1

    @property
    def nu(self):
        # Number of inputs is fixed to 1.
        return self.NU


class MPCController(Controller):
    def __init__(self,
                 patient_name,
                 model,
                 N,
                 Q,
                 QN,
                 R,
                 target=140,
                 insulin_max=5):
        self.patient_name = patient_name
        self.N = N
        self.Q = Q
        self.QN = QN
        self.R = R
        self.target = target
        self.quests = fetch_patient_quest(patient_name)
        self.patient_params = fetch_patient_params(patient_name)
        self.u_max = insulin_max  # U/min
        self.model = model
        self.reset()

    @property
    def basal(self):
        if self.patient_params:
            # unit: pmol/(L*kg)
            u2ss = self.patient_params['u2ss']
            # unit: kg
            BW = self.patient_params['BW']
        else:
            # NOTE: This need to be revisited. Does it make sense to use
            # average? Use average stats if the params is missing.
            u2ss = 1.43  # unit: pmol/(L*kg)
            BW = 57.0  # unit: kg

        basal = u2ss * BW / 6000  # unit: U/min
        return basal

    @property
    def Gb(self):
        """
        Subcutaneous BG value at the stable state.
        """
        return self.patient_params['Gb']

    def _define_opt_problem(self):
        obj = 0
        u_min = -self.basal
        yr = self.target - self.Gb
        constraints = [self.x[:, 0] == self.x_init]

        for k in range(self.N):
            #  obj += self.Q * cp.norm(self.x[0, k] - yr, 2) + self.R * cp.norm(
            #      self.u[:, k], 2)
            obj += self.Q * cp.norm(self.x[0, k] - yr, 2)
            if k > 0:
                obj += self.R * cp.norm(self.u[:, k] - self.u[:, k - 1], 2)
            constraints += [
                self.x[:, k + 1] == self.model.A @ self.x[:, k] +
                self.model.B @ self.u[:, k]
            ]
            constraints += [u_min <= self.u[:, k], self.u[:, k] <= self.u_max]
        obj += self.QN * cp.norm(self.x[0, self.N] - yr, 2)
        return cp.Problem(cp.Minimize(obj), constraints)

    def policy(self, observation, reward, done, **kwargs):
        self.state[1:self.model.na] = self.state[:(self.model.na - 1)]
        self.state[0] = observation.CGM - self.Gb
        self.state[(self.model.na + 1):] = self.state[self.model.na:-1]
        self.state[self.model.na] = self.prev_input

        self.x_init.value = self.state
        self.prob.solve(solver=cp.ECOS, warm_start=True, verbose=True)

        opt_input = max(self.u[:, 0].value + self.basal, 0)
        self.prev_input = opt_input
        print(f'opt insulin: {opt_input}')
        return Action(basal=opt_input, bolus=0)

    def reset(self):
        self.state = np.zeros(self.model.nx)
        self.prev_input = 0.0
        self.x_init = cp.Parameter(self.model.nx)
        self.x = cp.Variable((self.model.nx, self.N + 1))
        self.u = cp.Variable((self.model.nu, self.N))
        self.prob = self._define_opt_problem()

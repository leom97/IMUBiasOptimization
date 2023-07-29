import numpy as np
from numba import jit
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import logging
import time
import torch
from lietorch import SO3
from abc import ABC, abstractmethod
import signal
import scipy.sparse.linalg as sparse_linalg
from utils.various import TqdmToLogger

import utils.TLIO_utils.math_utils_TLIO as SO3_TLIO
from utils.pytorch_model import PyTorchNonLinearDbModel, PyTorchLinearDbModel

# logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')


class Optimization(ABC):

    def __init__(self):
        self.pytorch_model = None
        self.it = None
        self.dR_gt = None
        self.I = np.eye(3)
        self.dts_imu = None
        self.dts_gt = None
        self.gt_ts = None
        self.imu_ts = None
        self.rotations = None
        self.raw_gyros = None
        signal.signal(signal.SIGINT, self.catch)
        signal.siginterrupt(signal.SIGINT, False)
        self.graceful_interruption = False  # manual interruption of optimization procedure

    def catch(self, signum, frame):
        self.graceful_interruption = True

    def set_data(self, raw_gyros, rotations, imu_ts, gt_ts):
        self.raw_gyros = raw_gyros
        self.rotations = rotations
        self.imu_ts = imu_ts
        self.gt_ts = gt_ts

        self.dts_imu = np.diff(self.imu_ts)
        self.dts_gt = np.diff(self.gt_ts)

        self.dR_gt = np.einsum("bji,bjk->bik", self.rotations[:-1], self.rotations[1:])  # NB: DR_{ij} = DR_{j->i}!

    @staticmethod
    @jit(nopython=True, parallel=False, cache=True)
    def delta_R_window(w: np.ndarray, dt: np.array, b: np.ndarray) -> np.ndarray:
        """
        Given a sequence of N-1 gyros i->j, and N-1 corresponding dts, compute the quantities:
                            \Delta R_{j->k}(b_i)=
                                                   \Pi_{l=k}^{j-1} Exp((w_l-b_i) \delta t_l)
        for k = i, ..., j
        """

        N = len(dt)
        DR = np.zeros((N + 1, 3, 3))  # this is DR_jj
        DR[0] = np.eye(3)

        for (w_tmp, dt_tmp, l) in zip(w[::-1], dt[::-1], range(1, N + 1)):  # Miracolously correct!
            DR_tmp = SO3_TLIO.mat_exp(dt_tmp * (w_tmp - b.flatten())) @ DR[l - 1]
            DR[l] = DR_tmp

        return DR[::-1]

    def Jr_window(self, w: np.ndarray, b: np.array, eps=1e-3) -> np.ndarray:
        """
        Computes, for k=0, ..., len(w)-1:
                                                  \J_r^k(b) =
                                                  J_r^k(w_k-b)
        """
        phi = w - b  # N x 3

        def hat(v):
            R = np.zeros((len(v), 3, 3))
            R[:, 0, 1] = - v[:, 2]
            R[:, 0, 2] = + v[:, 1]
            R[:, 1, 0] = + v[:, 2]
            R[:, 1, 2] = - v[:, 0]
            R[:, 2, 0] = - v[:, 1]
            R[:, 2, 1] = + v[:, 0]
            return R

        theta = np.linalg.norm(phi, axis=1)[:, None, None]

        Jr = np.zeros((len(w), 3, 3))
        Jr[:, 0, 0] = 1
        Jr[:, 1, 1] = 1
        Jr[:, 2, 2] = 1

        mask = np.squeeze(theta < eps)
        hfm = hat(phi[mask])
        Jr[mask] += - 0.5 * hfm + 1.0 / 6.0 * np.einsum("bij, bjk -> bik", hfm, hfm)

        nmask = np.logical_not(mask)
        hfnm = hat(phi[nmask])

        Jr[nmask] += - (1 - np.cos(theta[nmask])) / np.power(theta[nmask], 2.0) * hfnm + (
                theta[nmask] - np.sin(theta[nmask])) / np.power(theta[nmask], 3.0) * np.einsum("bij, bjk -> bik", hfnm,
                                                                                               hfnm)

        return Jr

    @staticmethod
    @jit(nopython=True, parallel=False, cache=True)
    def Jr_window_slow(w: np.ndarray, b: np.array) -> np.ndarray:
        Jr = np.zeros((len(w), 3, 3))
        for (w_tmp, i) in zip(w, range(len(w))):
            Jr[i] = SO3_TLIO.Jr_exp(w_tmp - b)

        return Jr

    def build_small_linear_system(self, i, b, pre_cached_DRs=None):
        dt_gt = self.dts_gt[i]

        ts = self.gt_ts[i]
        te = self.gt_ts[i + 1]

        ind_imu_s = max(np.searchsorted(self.imu_ts, ts, side="left"),
                        0)  # it is very important for these two to be coherent! If we skip, we lose a lot of precision!
        ind_imu_e = min(np.searchsorted(self.imu_ts, te, side="left"), len(self.imu_ts) - 1)
        w = self.raw_gyros[ind_imu_s:ind_imu_e,
            :3]  # note, we don't do any interpolation, for the moment, and we never need the final IMU measurement
        dts = self.dts_imu[ind_imu_s:ind_imu_e]

        # repeat until convergence
        DR = self.delta_R_window(w, dts, b[i]) if pre_cached_DRs is None else pre_cached_DRs[i]
        Jr = self.Jr_window(w, b[i])

        dR_est = DR[0]
        dR_gt = self.dR_gt[i]

        lhs = SO3_TLIO.mat_log(dR_est.T @ dR_gt)
        mat = np.sum(-np.einsum("bji, bjk -> bik", DR[1:], Jr) * dts[:, None, None], axis=0)

        return dt_gt, mat, lhs, DR, dR_gt

    def get_level_0_quantities(self, b, pre_cached_DRs=None, silent=True):  # in PyTorch! And in lietorch quantities
        # assuming that everything is a NumPy array up to here
        N = len(self.gt_ts)

        mats = []
        lhss = []
        dts_gt = []
        dRs_imu = []
        dRs_gt = []
        pre_cached_DRs_new = []

        if not silent:
            pbar = tqdm(range(N - 1), disable=silent, file=self.tqdm_out)
            pbar.set_description("\t\tComputing rotational increments")
        else:
            # logging.info("\t\tComputing rotational increments")
            pbar = range(N - 1)
        for i in pbar:
            dt_gt, mat, lhs, DR, dR_gt = self.build_small_linear_system(i, b, pre_cached_DRs=pre_cached_DRs)
            mats.append(mat)
            lhss.append(lhs)
            dts_gt.append(dt_gt)
            dRs_imu.append(DR[0])
            dRs_gt.append(dR_gt)
            pre_cached_DRs_new.append(DR)

        # batch
        mats = np.array(mats)
        lhss = np.array(lhss)
        dts_gt = np.array(dts_gt)
        dRs_imu = np.array(dRs_imu)
        dRs_gt = np.array(dRs_gt)

        # and move to PyTorch
        mats = torch.tensor(mats, device="cpu", requires_grad=False, dtype=torch.float64)
        lhss = torch.tensor(lhss, device="cpu", requires_grad=False, dtype=torch.float64)
        dts_gt = torch.tensor(dts_gt, device="cpu", requires_grad=False, dtype=torch.float64)
        dRs_imu = SO3.exp(torch.tensor(Rotation.from_matrix(dRs_imu).as_rotvec(), device="cpu", requires_grad=False,
                                       dtype=torch.float64))
        dRs_gt = SO3.exp(torch.tensor(Rotation.from_matrix(dRs_gt).as_rotvec(), device="cpu", requires_grad=False,
                                      dtype=torch.float64))

        return mats, lhss, dts_gt, dRs_imu, dRs_gt, pre_cached_DRs_new

    @abstractmethod
    def cost_function_at_bias(self, b, ref_step=None):
        pass

    def step_size_search(self, max_step_refinements, b, db, cost_old, accept_always=False):

        terminate = False
        cost = cost_old

        pbar = tqdm(range(max_step_refinements), file=self.tqdm_out)
        for ref_step in pbar:

            if accept_always:
                logging.info("Computing new cost")
            else:
                pbar.set_description(f"\t\tFinding step size")

            b_candidate = b + 2 ** (-ref_step) * db.reshape(-1, 3)

            cost_candidate, pre_cached_level_0_quantities = self.cost_function_at_bias(b_candidate, ref_step)

            if cost_candidate < cost_old or accept_always:  # accept step size
                # if True:  # accept step size
                b = b_candidate
                cost = cost_candidate
                break

            elif ref_step == max_step_refinements - 1:

                terminate = True
                logging.warning("Could not find a suitable step size")

        return b, cost, terminate, pre_cached_level_0_quantities

    def stopping_criterion(self, res_old, res, iterate_tol, maxit, increasing_cost=None):
        terminate = False
        if res_old < res:
            logging.warning("Iterates are diverging")
        if res < iterate_tol:
            logging.info("Convergence reached")
            terminate = True
        if self.it == maxit:
            terminate = True
            logging.warning("Maximum number of iterations reached")
        if self.graceful_interruption:
            terminate = True
            logging.info("Interruption by user")
        if increasing_cost is not None:
            if increasing_cost:
                terminate = True
                logging.warning("Increasing loss, terminating")

        return terminate

    @abstractmethod
    def run_optimization(self, opt_cfg=None):
        pass


class PyTorchOptimization(Optimization):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_db_pytorch(self):
        pass


class LinearMonolevelOptimization(Optimization):

    def __init__(self):
        raise Exception("Go to multilevel optimization, and put 0 levels, this code is not mantained anymore")
        super().__init__()

    def cost_function_at_bias(self, b, ref_step=None):

        raise NotImplementedError("Check this loss again, not sure it is correct")

        pre_cached_DRs = []

        cost = 0

        N = len(self.gt_ts)
        pbar = tqdm(range(N - 1), file=self.tqdm_out)
        for i in pbar:
            if ref_step is not None:
                pbar.set_description(f"Evaluating cost function for step size 1÷{2 ** ref_step}")

            ts = self.gt_ts[i]
            te = self.gt_ts[i + 1]
            ind_imu_s = np.searchsorted(self.imu_ts, ts, side="left") + 1
            ind_imu_e = np.searchsorted(self.imu_ts, te, side="right") - 1
            w = self.raw_gyros[ind_imu_s:ind_imu_e]
            dts = self.dts_imu[ind_imu_s:ind_imu_e]

            DR = self.delta_R_window(w, dts, b[i])
            pre_cached_DRs.append(DR)

            # we only need the big DR
            dR_est = DR[0]
            dR_gt = self.dR_gt[i]

            err = SO3_TLIO.mat_log(dR_est.T @ dR_gt)

            cost += np.linalg.norm(err) / (2 * np.pi) * 360

        if cached_loss is not None:
            cost = cached_loss

        return cost, pre_cached_DRs

    def build_linear_system(self, b, smoothing, pre_cached_DRs=None, pre_cached_mats=None, pre_cached_lhss=None,
                            pre_cached_dts_gt=None):

        N = len(self.gt_ts)
        large_mat = np.eye(3 * (N - 1))
        large_lhs = np.zeros(3 * (N - 1))

        for i in tqdm(range(N - 1), file=self.tqdm_out):

            if pre_cached_mats is not None and pre_cached_lhss is not None and pre_cached_dts_gt is not None:
                dt_gt = pre_cached_dts_gt[i]
                mat = pre_cached_mats[i]
                lhs = pre_cached_lhss[i]
            else:
                dt_gt, mat, lhs, _, _ = self.build_small_linear_system(i, b, pre_cached_DRs=pre_cached_DRs)

            # cost function contribution (multiply by dt for simulating an integral)
            large_mat[3 * i:3 * i + 3, 3 * i:3 * i + 3] = mat.T @ mat * dt_gt
            large_lhs[3 * i:3 * i + 3] = mat.T @ lhs * dt_gt

            # smoothing contribution
            a = smoothing
            if a > 0:
                if i == 0:
                    large_mat[3 * i:3 * i + 3, 3 * i:3 * i + 3] += a / self.gt_ts[i] * self.I
                    large_mat[3 * i:3 * i + 3, 3 * (i + 1):3 * (i + 1) + 3] += -a / self.gt_ts[i] * self.I
                    large_lhs[3 * i:3 * i + 3] += a / self.gt_ts[i] * (b[i + 1] - b[i]).flatten()
                elif i < N - 2:
                    large_mat[3 * i:3 * i + 3, 3 * i:3 * i + 3] += a / self.gt_ts[i] * self.I + a / self.gt_ts[
                        i - 1] * self.I
                    large_mat[3 * i:3 * i + 3, 3 * (i + 1):3 * (i + 1) + 3] += -a / self.gt_ts[i] * self.I
                    large_mat[3 * i:3 * i + 3, 3 * (i - 1):3 * (i - 1) + 3] += -a / self.gt_ts[i - 1] * self.I
                    large_lhs[3 * i:3 * i + 3] += a * (
                            (b[i + 1] - b[i]) / self.gt_ts[i] - (b[i] - b[i - 1]) / self.gt_ts[i]).flatten()
                else:  # end of sequence
                    large_mat[3 * i:3 * i + 3, 3 * i:3 * i + 3] += a / self.gt_ts[i - 1] * self.I
                    large_mat[3 * i:3 * i + 3, 3 * (i - 1):3 * (i - 1) + 3] += -a / self.gt_ts[i - 1] * self.I
                    large_lhs[3 * i:3 * i + 3] += - a / self.gt_ts[i - 1] * (b[i] - b[i - 1]).flatten()

        return large_mat, large_lhs

    def run_optimization(self, opt_cfg=None):

        if opt_cfg is None:
            opt_cfg = {"iterate_tol": 1e-8, "maxit": 20, "smoothing": 0, "max_step_refinements": 10}

        iterate_tol = opt_cfg["iterate_tol"]
        maxit = opt_cfg["maxit"]
        smoothing = opt_cfg["smoothing"]

        # optimization variables
        N = len(self.gt_ts)
        b = np.zeros((N - 1, 3))
        res = np.infty  # initial residual (difference between iterations)
        cost = np.infty

        self.it = 0
        terminate = False
        pre_cached_DRs = None  # this variable might be computed outside of the build_system function: we save it and use it for later, to avoid making the same computation twice
        start_time = -time.time()
        while not terminate:
            logging.info(f"Iteration {self.it}")

            large_mat, large_lhs = self.build_linear_system(b, smoothing, pre_cached_DRs=pre_cached_DRs)
            db = np.linalg.lstsq(large_mat, large_lhs, rcond=None)[0]

            b_old = b
            res_old = res
            cost_old = cost

            b, cost, terminate, pre_cached_DRs = self.step_size_search(opt_cfg["max_step_refinements"], b, db, cost_old)

            res = np.abs(b - b_old).max()
            logging.info(f"Residual in infinity norm: {res}, segment cost in degrees: {cost}\n ")
            terminate = self.stopping_criterion(res_old, res, iterate_tol, maxit)
            self.it += 1

        logging.info(f"Elapsed time: {time.time() + start_time}")

        return b


class NonLinearMultilevelOptimization(PyTorchOptimization):

    def __init__(self):
        super().__init__()

    def run_optimization(self, opt_cfg=None):

        iterate_tol = opt_cfg["iterate_tol"]
        maxit = opt_cfg["maxit"]

        # optimization variables
        N = len(self.gt_ts)
        b = np.zeros((N - 1, 3))
        res = np.infty  # initial residual (difference between iterations)
        cost = np.infty

        # Pytorch interface
        self.pytorch_model = PyTorchLinearDbModel(N - 1, skip_levels_until=opt_cfg["skip_levels_until"])

        self.it = 0
        terminate = False
        pre_cached_DRs = None  # this variable might be computed outside of the build_system function: we save it and use it for later, to avoid making the same computation twice
        start_time = -time.time()
        while not terminate:
            logging.info(f"Iteration {self.it}")

            # get all quantities at level 0
            mats, lhss, dts_gt, dRs_imu, dRs_gt, _ = self.get_level_0_quantities(b, pre_cached_DRs=pre_cached_DRs)

            # solve for db
            repeat = True
            repeat_times = 0
            db = None
            original_lr = opt_cfg["db_opt_cfg"]["lr"]
            while (repeat):
                opt_cfg["db_opt_cfg"]["lr"] *= 2 ** (-repeat_times)
                repeat_times += 1
                db, repeat = self.get_db_pytorch(level_0_quantities=[mats, lhss, dts_gt, dRs_imu, dRs_gt],
                                                 opt_cfg=opt_cfg["db_opt_cfg"])
                if repeat:
                    logging.warning("Bias increment not found, repeating with lower learning rate")

                raise NotImplementedError("Don't repeat during non-linear optimization, else it will be very slow")
                # repeat = False
            raise NotImplementedError("are you sure you want to reset the learning rate every time?")
            opt_cfg["db_opt_cfg"]["lr"] = original_lr
            b_old = b
            res_old = res
            cost_old = cost

            raise NotImplementedError("Add a cost_function_at_bias also here")
            b, cost, terminate, pre_cached_DRs = self.step_size_search(opt_cfg["max_step_refinements"], b, db,
                                                                       cost_old)  # this will not be present

            res = np.abs(b - b_old).max()
            logging.info(f"Residual in infinity norm: {res}, segment cost in degrees: {cost}")
            terminate = self.stopping_criterion(res_old, res, iterate_tol, maxit)
            logging.info('\n')

            self.it += 1

        logging.info(f"Elapsed time: {time.time() + start_time}")

        return b

    def get_db_pytorch(self, level_0_quantities, opt_cfg):
        # see OneNote/Ground truth biases

        self.pytorch_model.set_level_0_quantities(*level_0_quantities)
        self.pytorch_model.compute_tree_quantities(opt_cfg["levels"])
        optimizer = torch.optim.Adam(self.pytorch_model.parameters(), lr=opt_cfg["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               threshold=opt_cfg["threshold"],
                                                               patience=opt_cfg["patience"],
                                                               min_lr=opt_cfg["min_lr"])
        logging.info("Computing bias increment")
        torch.set_printoptions(precision=3)

        repeat_me = False
        loss = None
        old_loss = torch.tensor(torch.inf, device="cpu", dtype=torch.float64, requires_grad=False)
        pbar = tqdm(range(opt_cfg["maxit"]), file=self.tqdm_out)
        for it in pbar:
            loss = self.pytorch_model()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(loss)

            pbar.set_description(f"\t\t\tCurrent sub-problem loss in degrees: {torch.sqrt(loss) / (2 * np.pi) * 360}")

            raise NotImplementedError(
                "Enable the lines below to run non-linear optimization: otherwise it won't work well")
            # if torch.sqrt(loss) / (2 * np.pi) * 360 <= opt_cfg["eps"]:
            #     break
            # else:
            #     old_loss = loss

        if torch.sqrt(loss) / (2 * np.pi) * 360 > opt_cfg["eps"]:
            repeat_me = True

        db = self.pytorch_model.weights_matrix.detach().cpu().numpy()
        return db, repeat_me


class LinearMultilevelOptimization(PyTorchOptimization):

    def __init__(self, tqdm_to_file=None):
        super().__init__()
        self.b = None
        self.opt_cfg = None
        self.db = None
        logger = logging.getLogger()
        self.tqdm_to_file = tqdm_to_file
        self.tqdm_out = TqdmToLogger(logger, level=logging.INFO) if tqdm_to_file else None


    def cost_function_at_bias(self, b, ref_step=None):
        """
        Note, this should also be in charge of computing new DRs, since we're evaluating at a new bias
        """

        loss, penalty, _, _, pre_cached_level_0_quantities = self.compute_loss_pytorch(b, silent=True)

        return np.sqrt(loss) / (2 * np.pi) * 360, pre_cached_level_0_quantities

    def get_db_pytorch(self, b, pre_cached_level_0_quantities):

        db = None
        db_old = np.ones(self.pytorch_model.weights.shape) * np.inf
        res_old = np.inf

        if self.pytorch_model.loss["type"] in ["mse", "gaussian"]:
            logging.info("Solving linear system for bias increment")
            loss, penalty, H, J, _ = self.compute_loss_pytorch(b, pre_cached_level_0_quantities, silent=True)
            db, _ = sparse_linalg.cg(H, -J.T, tol=self.opt_cfg["iterate_tol"] * 1e-1)
        elif self.pytorch_model.loss["type"] == "huber":
            logging.info("Solving for bias increment")

            pbar = tqdm(range(self.pytorch_model.loss["maxit"]), leave=True, file=self.tqdm_out)
            for it in pbar:
                loss, penalty, H, J, pre_cached_level_0_quantities = self.compute_loss_pytorch(b,
                                                                                               pre_cached_level_0_quantities,
                                                                                               silent=True)
                db, _ = sparse_linalg.cg(H, -J.T, tol=self.opt_cfg["iterate_tol"] * 1e-2)
                self.pytorch_model.set_weights(db)

                res = np.abs(db - db_old).max()
                pbar.set_description(f"\t\tIterates residual at step {it} is  {res}")

                if res <= self.opt_cfg["iterate_tol"] * 1e-1 or res_old < res:
                    if res_old < res:
                        logging.warning("Huber iterates are diverging")
                    break

                db_old = db
                res_old = res

        self.pytorch_model.set_weights(0 * db)  # reset model weights to zero
        return db, pre_cached_level_0_quantities

    def compute_loss_pytorch(self, b, pre_cached_level_0_quantities=None, silent=False):
        if pre_cached_level_0_quantities is None:
            mats, lhss, dts_gt, dRs_imu, dRs_gt, pre_cached_DRs = self.get_level_0_quantities(b)
            pre_cached_level_0_quantities = [mats, lhss, dts_gt, dRs_imu, dRs_gt, pre_cached_DRs]
        else:
            mats, lhss, dts_gt, dRs_imu, dRs_gt, pre_cached_DRs = pre_cached_level_0_quantities

        self.pytorch_model.set_level_0_quantities(mats, lhss, dts_gt, dRs_imu, dRs_gt)
        self.pytorch_model.compute_tree_quantities(self.opt_cfg["levels"])
        self.pytorch_model.create_all_dRs()
        loss, penalty, H, J = self.pytorch_model(silent=silent)

        return loss, penalty, H, J, pre_cached_level_0_quantities

    def run_optimization(self, opt_cfg=None):

        if opt_cfg["levels"] > 0:
            raise Exception("The noise model doesn't allow for more than 0 levels")

        iterate_tol = opt_cfg["iterate_tol"]
        maxit = opt_cfg["maxit"]
        self.opt_cfg = opt_cfg

        # optimization variables
        N = len(self.gt_ts)
        b = np.zeros((N - 1, 3))
        res = np.infty  # initial residual (difference between iterations)
        cost = np.infty

        # PyTorch interface
        self.pytorch_model = PyTorchLinearDbModel(N - 1, skip_levels_until=opt_cfg["skip_levels_until"], tqdm_to_file=self.tqdm_to_file)
        self.pytorch_model.loss = opt_cfg["loss"]
        self.pytorch_model.smoothing = opt_cfg["smoothing"]

        self.it = 0
        terminate = False
        pre_cached_level_0_quantities = None  # this variable might be computed outside of the build_system function: we save it and use it for later, to avoid making the same computation twice
        start_time = -time.time()
        while not terminate:
            logging.info(f"Iteration {self.it}")

            # get db
            self.db, pre_cached_level_0_quantities = self.get_db_pytorch(b, pre_cached_level_0_quantities)

            b_old = b
            res_old = res
            cost_old = cost

            b, cost, terminate, pre_cached_level_0_quantities = self.step_size_search(opt_cfg["max_step_refinements"],
                                                                                      b, self.db,
                                                                                      cost_old)
            res = np.abs(b - b_old).max()
            logging.info(f"Residual in infinity norm: {res}, loss: {cost}")
            terminate = self.stopping_criterion(res_old, res, iterate_tol, maxit, cost > cost_old)
            logging.info('End of iteration\n')

            self.it += 1

        logging.info(f"Elapsed time: {time.time() + start_time}")

        self.b = b
        return b

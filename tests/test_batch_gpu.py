"""
Tests for the GPU batch simulation path.

Covers:
- Parity between batch (N=1, CPU) and original single-patient scipy path
- Output shapes for N>1
- Auto-reset on termination
- Patient heterogeneity, meal handling, noise independence
- Seed reproducibility
- Backward compatibility of T1DSimGymnaisumEnv
- Patient pool vs fixed-slot semantics
- Performance benchmark (speedup)
- GPU smoke test (skipped if CUDA unavailable)
"""
import time
import unittest
import numpy as np
import torch

# ---- helpers ----------------------------------------------------------------

def _make_batch_env(n=1, patient_name="adolescent#001", seed=0, device="cpu"):
    from simglucose.envs.batch_env import T1DSimVectorEnv
    return T1DSimVectorEnv(
        n_envs=n,
        patient_name=patient_name,
        seed=seed,
        device=device,
    )


def _make_single_env(patient_name="adolescent#001", seed=0):
    from simglucose.envs.simglucose_gym_env import T1DSimGymnaisumEnv
    return T1DSimGymnaisumEnv(patient_name=patient_name, seed=seed)


def _zero_action_batch(n, env):
    """Return a zero-basal action dict for the batch env."""
    return {
        "basal": np.zeros(n, dtype=np.float32),
        "bolus": np.zeros(n, dtype=np.float32),
    }


def _zero_action_single(env):
    return {"basal": np.float32(0.0), "bolus": np.float32(0.0)}


# =============================================================================
# 1. Parity: RHS matches scalar model
# =============================================================================

class TestRHSParity(unittest.TestCase):

    def test_rhs_matches_scalar(self):
        """t1d_rhs_batch (N=1) must match T1DPatient.model() to < 1e-5."""
        import pandas as pd
        from simglucose.patient.t1dpatient import T1DPatient
        from simglucose.patient.t1dpatient_batch import (
            t1d_rhs_batch, _load_params_for_names,
        )

        name = "adolescent#001"
        p_scalar = T1DPatient.withName(name)
        x_np = p_scalar.init_state.astype(np.float64).copy()
        action_np = type("A", (), {"CHO": 10.0, "insulin": 0.02})()

        last_Qsto = float(x_np[0] + x_np[1])
        last_foodtaken = 0.0

        dxdt_scalar = p_scalar.model(
            0.0, x_np, action_np, p_scalar._params, last_Qsto, last_foodtaken
        )

        # Batch path (N=1)
        params, init = _load_params_for_names([name], "cpu", torch.float64)
        x_t = torch.tensor(x_np, dtype=torch.float64).unsqueeze(0)
        ins_t = torch.tensor([0.02], dtype=torch.float64)
        cho_t = torch.tensor([10.0], dtype=torch.float64)
        lQ_t  = torch.tensor([last_Qsto], dtype=torch.float64)
        lf_t  = torch.tensor([last_foodtaken], dtype=torch.float64)

        dxdt_batch = t1d_rhs_batch(
            0.0, x_t, ins_t, cho_t, params, lQ_t, lf_t
        ).squeeze(0).numpy()

        np.testing.assert_allclose(dxdt_batch, dxdt_scalar, atol=1e-5, rtol=1e-5)


# =============================================================================
# 2. Parity: full trajectory vs scipy path
# =============================================================================

class TestTrajectoryParity(unittest.TestCase):
    """
    Compare batch RK4 integrator against scipy dopri5 on the raw ODE,
    no noise, same initial state, same insulin input.
    """

    def test_batch1_matches_single_env(self):
        """N=1 batch RK4 vs scipy dopri5: BG within 5 mg/dL over 200 steps."""
        from simglucose.patient.t1dpatient import T1DPatient, Action as PatAction
        from simglucose.patient.t1dpatient_batch import T1DPatientBatch

        name = "adolescent#001"

        # --- scipy path (no noise, no random init) ---
        p_scipy = T1DPatient.withName(name, random_init_bg=False, seed=0)
        bg_scipy = []
        basal = float(p_scipy._params.u2ss * p_scipy._params.BW / 6000)
        for _ in range(200 * 3):  # 200 env steps × 3 mini-steps
            act = PatAction(CHO=0, insulin=basal)
            p_scipy.step(act)
            bg_scipy.append(p_scipy.observation.Gsub)
        bg_scipy = np.array(bg_scipy[2::3])  # sample every 3rd (sample_time)

        # --- batch RK4 path ---
        p_batch = T1DPatientBatch(
            [name], device="cpu", random_init_bg=False, seed=0
        )
        bg_batch = []
        ins_t = torch.tensor([basal], dtype=torch.float64)
        cho_t = torch.zeros(1, dtype=torch.float64)
        for _ in range(200 * 3):
            p_batch.step(ins_t, cho_t)
            bg_batch.append(p_batch.observation[0].item())
        bg_batch = np.array(bg_batch[2::3])

        diff = np.abs(bg_scipy[:len(bg_batch)] - bg_batch[:len(bg_scipy)])
        self.assertLess(
            diff.max(), 5.0,
            f"Max BG diff={diff.max():.3f} mg/dL between RK4 and dopri5 over 200 steps"
        )

    def test_trajectory_matches_original_24h(self):
        """288-step (≈24h) trajectory: BG within 5 mg/dL at all points."""
        from simglucose.patient.t1dpatient import T1DPatient, Action as PatAction
        from simglucose.patient.t1dpatient_batch import T1DPatientBatch

        name = "adolescent#001"

        p_scipy = T1DPatient.withName(name, random_init_bg=False, seed=0)
        bg_scipy = []
        basal = float(p_scipy._params.u2ss * p_scipy._params.BW / 6000)
        for _ in range(288 * 3):
            p_scipy.step(PatAction(CHO=0, insulin=basal))
            bg_scipy.append(p_scipy.observation.Gsub)
        bg_scipy = np.array(bg_scipy[2::3])

        p_batch = T1DPatientBatch(
            [name], device="cpu", random_init_bg=False, seed=0
        )
        bg_batch = []
        ins_t = torch.tensor([basal], dtype=torch.float64)
        cho_t = torch.zeros(1, dtype=torch.float64)
        for _ in range(288 * 3):
            p_batch.step(ins_t, cho_t)
            bg_batch.append(p_batch.observation[0].item())
        bg_batch = np.array(bg_batch[2::3])

        diff = np.abs(bg_scipy[:len(bg_batch)] - bg_batch[:len(bg_scipy)])
        self.assertLess(
            diff.max(), 5.0,
            f"Max BG diff={diff.max():.3f} mg/dL over 24h"
        )


# =============================================================================
# 3. Risk batch vs scalar
# =============================================================================

class TestRiskBatch(unittest.TestCase):

    def test_risk_batch_vs_scalar(self):
        """risk_batch must match scalar risk() to < 1e-4 over [20, 600] mg/dL."""
        from simglucose.analysis.risk import risk, risk_batch
        bg_values = np.linspace(21.0, 599.0, 500)
        scalar_ri = np.array([risk(b)[2] for b in bg_values])
        bg_t = torch.tensor(bg_values, dtype=torch.float64)
        batch_ri, _, _ = risk_batch(bg_t)
        np.testing.assert_allclose(
            batch_ri.numpy(), scalar_ri, atol=1e-4, rtol=1e-4
        )


# =============================================================================
# 4. Output shapes
# =============================================================================

class TestOutputShapes(unittest.TestCase):

    def test_step_output_shapes(self):
        N = 4
        env = _make_batch_env(n=N)
        env.reset()
        obs, rew, term, trunc, info = env.step(_zero_action_batch(N, env))
        self.assertEqual(obs["CGM"].shape, (N,))
        self.assertEqual(obs["CHO"].shape, (N,))
        self.assertEqual(rew.shape,  (N,))
        self.assertEqual(term.shape, (N,))
        self.assertEqual(trunc.shape, (N,))

    def test_reset_shapes(self):
        N = 4
        env = _make_batch_env(n=N)
        obs, info = env.reset()
        self.assertEqual(obs["CGM"].shape, (N,))
        self.assertEqual(obs["CHO"].shape, (N,))
        self.assertIsInstance(info, dict)


# =============================================================================
# 5. No Auto-reset on termination
# =============================================================================

class TestTermination(unittest.TestCase):

    def test_termination_signals_with_reset(self):
        """When a patient terminates, term[i] must be True, and auto-reset should occur."""
        env = _make_batch_env(n=2, seed=1)
        env.reset()

        # Force patient 0's BG way out of range (account for Vg scaling)
        env._patient.x[0, 3] = 800.0 * env._patient.params["Vg"][0]
        env._patient.x[0, 12] = 800.0 * env._patient.params["Vg"][0]

        action = _zero_action_batch(2, env)
        obs, rew, term, trunc, info = env.step(action)

        self.assertTrue(term[0], "Patient 0 should have reported termination")
        # After auto-reset, CGM should be a normal initial value (not > 600)
        self.assertLess(obs["CGM"][0], 400.0)
        self.assertIn("final_observation", info)
        self.assertEqual(len(info["final_observation"]), 1)
        self.assertGreaterEqual(info["final_observation"][0]["CGM"], 600.0)



# =============================================================================
# 6. Patient heterogeneity
# =============================================================================

class TestPatientHeterogeneity(unittest.TestCase):

    def test_different_patients_diverge(self):
        """N=3 different patients must have different CGM after 100 steps."""
        from simglucose.envs.batch_env import T1DSimVectorEnv
        env = T1DSimVectorEnv(
            n_envs=3,
            patient_name=["adolescent#001", "adult#001", "child#001"],
            seed=0,
        )
        env.reset()
        for _ in range(100):
            env.step(_zero_action_batch(3, env))
        cgm = env._sensor.last_cgm.cpu().numpy()
        # All three should differ (different physiology)
        self.assertFalse(
            np.allclose(cgm[0], cgm[1]) and np.allclose(cgm[1], cgm[2]),
            "All patients have identical CGM — heterogeneity check failed"
        )


# =============================================================================
# 7. Meal handling
# =============================================================================

class TestMealHandling(unittest.TestCase):

    def test_meal_glucose_rise(self):
        """After a large meal, glucose should rise over 60 steps."""
        env = _make_batch_env(n=2, seed=5)
        env.reset()

        # Run 5 steps to reach steady state
        for _ in range(5):
            env.step(_zero_action_batch(2, env))

        bg_before = env._patient.bg.cpu().numpy().copy()

        # Announce 80g meal to patient 0 by injecting into scenario
        env._scenario.meal_schedule[0, int(env._scenario.start_minutes[0].item())] = 80.0

        for _ in range(60):
            env.step(_zero_action_batch(2, env))

        bg_after = env._patient.bg.cpu().numpy().copy()
        self.assertGreater(
            bg_after[0], bg_before[0],
            "Glucose should rise after a meal"
        )


# =============================================================================
# 8. CGM noise independence
# =============================================================================

class TestCGMNoise(unittest.TestCase):

    def test_cgm_noise_independent(self):
        """Noise residuals between patients must be independent (corr < 0.5)."""
        N = 4
        env = _make_batch_env(n=N, patient_name="adolescent#001", seed=7)
        env.reset()

        # Collect CGM and raw BG to isolate noise residuals
        cgm_records = [[] for _ in range(N)]
        bg_records  = [[] for _ in range(N)]
        for _ in range(200):
            env.step(_zero_action_batch(N, env))
            cgm = env._sensor.last_cgm.cpu().numpy()
            bg  = env._patient.observation.cpu().numpy()
            for i in range(N):
                cgm_records[i].append(cgm[i])
                bg_records[i].append(bg[i])

        for i in range(N):
            for j in range(i + 1, N):
                noise_i = np.array(cgm_records[i]) - np.array(bg_records[i])
                noise_j = np.array(cgm_records[j]) - np.array(bg_records[j])
                corr = float(np.corrcoef(noise_i, noise_j)[0, 1])
                self.assertLess(
                    abs(corr), 0.5,
                    f"Noise between env {i} and {j} too correlated: r={corr:.3f}"
                )


# =============================================================================
# 9. Seed reproducibility
# =============================================================================

class TestSeedReproducibility(unittest.TestCase):

    def test_seed_reproducibility(self):
        """Two envs with same seed must produce identical 100-step trajectories."""
        def run(seed):
            env = _make_batch_env(n=4, seed=seed)
            env.reset(seed=seed)
            cgms = []
            for _ in range(100):
                obs, _, _, _, _ = env.step(_zero_action_batch(4, env))
                cgms.append(obs["CGM"].copy())
            return np.array(cgms)

        a = run(42)
        b = run(42)
        np.testing.assert_array_equal(a, b)


# =============================================================================
# 10. Backward compatibility
# =============================================================================

class TestBackwardCompat(unittest.TestCase):

    def test_original_env_unchanged(self):
        """T1DSimGymnaisumEnv() (no new params) must behave exactly as before."""
        env = _make_single_env(seed=0)
        obs, info = env.reset()
        self.assertIn("CGM", obs)
        self.assertIn("CHO", obs)
        self.assertIsInstance(obs["CGM"], (float, np.floating))
        obs2, rew, done, trunc, info2 = env.step(_zero_action_single(env))
        self.assertIsInstance(rew, (float, np.floating))
        self.assertIn(done, (True, False))

    def test_gymnasium_flag_dispatches_to_batch(self):
        """T1DSimGymnaisumEnv(n_envs=4, device='cpu') must delegate to batch path."""
        from simglucose.envs.simglucose_gym_env import T1DSimGymnaisumEnv
        env = T1DSimGymnaisumEnv(
            patient_name="adolescent#001", n_envs=4, device="cpu"
        )
        self.assertTrue(env._use_batch)
        obs, _ = env.reset()
        self.assertEqual(obs["CGM"].shape, (4,))


# =============================================================================
# 11. Patient pool semantics
# =============================================================================

class TestPatientSemantics(unittest.TestCase):

    def test_pool_semantics_samples_both_patients(self):
        """n_envs=3 with 2-name pool: both patients appear across 20 resets."""
        from simglucose.envs.batch_env import T1DSimVectorEnv
        env = T1DSimVectorEnv(
            n_envs=3,
            patient_name=["adolescent#001", "adolescent#002"],
            seed=0,
        )
        seen = set()
        for _ in range(20):
            env.reset()
            seen.update(env._patient.patient_names)
        self.assertIn("adolescent#001", seen)
        self.assertIn("adolescent#002", seen)

    def test_fixed_slots_keep_assignment(self):
        """n_envs=3 with len=3 list: slots keep their patient across resets."""
        from simglucose.envs.batch_env import T1DSimVectorEnv
        names = ["adolescent#001", "adult#001", "child#001"]
        env = T1DSimVectorEnv(n_envs=3, patient_name=names, seed=0)
        for _ in range(5):
            env.reset()
            self.assertEqual(env._patient.patient_names, names)


# =============================================================================
# 12. GPU smoke test
# =============================================================================

class TestGPU(unittest.TestCase):

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_batch_env_gpu(self):
        """N=8 on CUDA: runs 50 steps without error, tensors on GPU."""
        env = _make_batch_env(n=8, device="cuda")
        env.reset()
        for _ in range(50):
            obs, rew, term, trunc, info = env.step(_zero_action_batch(8, env))
        # Confirm internal tensors live on GPU
        self.assertEqual(env._patient.x.device.type, "cuda")
        self.assertEqual(obs["CGM"].shape, (8,))


# =============================================================================
# 13. Performance benchmark
# =============================================================================

class TestPerformanceSpeedup(unittest.TestCase):

    def _time_env(self, env, n, n_steps=1000):
        env.reset()
        action = _zero_action_batch(n, env)
        start = time.perf_counter()
        for _ in range(n_steps):
            env.step(action)
        return time.perf_counter() - start

    def test_performance_speedup(self):
        """
        N=64 batch (CPU) must be >= 4x faster than 64 sequential single envs.
        Also prints N=1 overhead and, if CUDA available, GPU speedup.
        """
        N = 64
        N_STEPS = 200

        # N=1 batch vs single env
        batch1 = _make_batch_env(n=1, seed=0)
        t_batch1 = self._time_env(batch1, 1, N_STEPS)

        single = _make_single_env(seed=0)
        single.reset()
        act_s = _zero_action_single(single)
        t0 = time.perf_counter()
        for _ in range(N_STEPS):
            single.step(act_s)
        t_single = time.perf_counter() - t0

        print(f"\n[Perf] N=1 batch: {t_batch1:.3f}s  |  single scipy: {t_single:.3f}s  "
              f"|  overhead: {t_batch1/t_single:.2f}x")

        # N=64 batch vs 64 sequential single envs
        batch64 = _make_batch_env(n=N, seed=0)
        t_batch64 = self._time_env(batch64, N, N_STEPS)

        t_seq = 0.0
        for i in range(N):
            e = _make_single_env(seed=i)
            e.reset()
            a = _zero_action_single(e)
            s = time.perf_counter()
            for _ in range(N_STEPS):
                e.step(a)
            t_seq += time.perf_counter() - s

        speedup_cpu = t_seq / t_batch64
        print(f"[Perf] N={N} batch CPU: {t_batch64:.3f}s  |  {N}x sequential: {t_seq:.3f}s  "
              f"|  speedup: {speedup_cpu:.1f}x")

        self.assertGreaterEqual(
            speedup_cpu, 4.0,
            f"Expected >= 4x speedup for N={N}, got {speedup_cpu:.1f}x"
        )

        # GPU speedup (informational only)
        if torch.cuda.is_available():
            batch_gpu = _make_batch_env(n=N, seed=0, device="cuda")
            batch_gpu.reset()
            act_gpu = _zero_action_batch(N, batch_gpu)
            # warm-up
            for _ in range(10):
                batch_gpu.step(act_gpu)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(N_STEPS):
                batch_gpu.step(act_gpu)
            torch.cuda.synchronize()
            t_gpu = time.perf_counter() - t0
            print(f"[Perf] N={N} GPU: {t_gpu:.3f}s  |  GPU vs CPU batch speedup: "
                  f"{t_batch64/t_gpu:.1f}x")


if __name__ == "__main__":
    unittest.main()

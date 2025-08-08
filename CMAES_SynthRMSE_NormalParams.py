import time
import csv
import numpy as np
from scipy.spatial import KDTree
import cma
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

np.set_printoptions(precision=4, suppress=True)

def plot_rmse_vs_param(model_simulator, num_points=15):
    """
    For each parameter, plot RMSE vs that parameter (others fixed at 0.5).
    Plots are arranged horizontally, with no axis ticks, and curves are smoothed.
    """
    dim = model_simulator.model.shape[1]
    fig, axes = plt.subplots(1, dim, figsize=(2.5*dim, 3), sharey=True)
    if dim == 1:
        axes = [axes]
    xvals = np.linspace(0, 1, num_points)
    xvals_smooth = np.linspace(0, 1, 100)
    for i in range(dim):
        rmse_vals = []
        for x in xvals:
            x_query = np.full(dim, 0.5)
            x_query[i] = x
            rmse = model_simulator.evaluate(x_query)
            rmse_vals.append(rmse)
        # Interpolate for smooth curve
        interp_func = interp1d(xvals, rmse_vals, kind='cubic')
        rmse_smooth = interp_func(xvals_smooth)
        axes[i].plot(xvals_smooth, rmse_smooth, color='tab:blue')
        axes[i].set_title(f'Param {i+1}')
        axes[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    axes[0].set_ylabel('RMSE')
    plt.tight_layout()
    plt.show()

class ModelSimulator:
    """
    Simulates an expensive RMSE lookup from a precomputed table of num_models num_optvars-D entries.
    Internally builds a KDTree for nearest-neighbor retrieval
    """
    def __init__(self, num_optvars, num_models, seed):
        t0 = time.time()
        rng = np.random.RandomState(seed)
        # Create random parameter vectors in [0,1]^num_optvars
        self.model= rng.rand(num_models, num_optvars)

        # Generate RMSEs for the model table
        self.rmses = self._gen_model_rmse_map(self.model, num_optvars, num_models, rng)

        # Print top 10 (lowest) RMSEs in table
        top10_idx = np.argsort(self.rmses)[:10]
        print(">> Top 10 RMSEs in table:")
        for rank, idx in enumerate(top10_idx, 1):
            print(f"  {rank}: RMSE={self.rmses[idx]:.4f}")

        # Print worst 2 (highest) RMSEs in table
        worst2_idx = np.argsort(self.rmses)[-2:][::-1]
        print(">> Worst 2 RMSEs in table:")
        for rank, idx in enumerate(worst2_idx, 1):
            print(f"  {rank}: RMSE={self.rmses[idx]:.4f}")

        # Build KDTree for nearest‐neighbor queries
        self.tree = KDTree(self.model)

        print(f"[DEBUG] ModelSimulator initialized in {time.time() - t0:.2f} seconds.")

    def _gen_model_rmse_map(self, model, num_optvars, num_models, rng):
        """
        Generate synthetic RMSE values for the model table.
        """
        # Assign each a random "rmse" value, capped at 0.5 minimum
        raw_rmses = (
            np.sum(np.sin(model * 27.13) + np.cos(model * 19.77), axis=1)
            + np.sum(np.abs(np.sin(model * 50.0)), axis=1)
            + np.sum(model ** 2, axis=1)
        )
        # Add a sharp global minimum at a random location
        center = rng.rand(num_optvars)
        sharpness = 150.0
        raw_rmses += np.exp(-sharpness * np.sum((model - center) ** 2, axis=1))
        # Add plateaus
        raw_rmses = np.round(raw_rmses * 2) / 2.0
        # Add more noise
        noise = rng.normal(0, 0.05, num_models)
        rmses = np.clip(raw_rmses + noise + 5.0, 0.5, None)
        return rmses

    def evaluate(self, x_unit):
        """
        Lookup the nearest neighbor in the table and return its RMSE.
        """
        _, idx = self.tree.query(x_unit)
        return float(self.rmses[idx])

class CMAESCalibration:
    """
    Wraps cma.CMAEvolutionStrategy to calibrate a dim-D vector by minimizing RMSE
    via ModelSimulator.evaluate. Logs each (model, rmse) in evaluation order.
    """
    def __init__(self, simulator, dim, popsize, sigma, seed, param_lower, param_upper, plot_evolution=False):
        self.simulator = simulator
        self.cmaes = cma.CMAEvolutionStrategy([0.5] * dim, sigma,
                                           {'popsize': popsize, 'seed': seed})
        self.log = []  # to record (x_unit, rmse)
        self.param_lower = param_lower
        self.param_upper = param_upper
        self.plot_evolution = plot_evolution

    def run(self, gen_size):
        """
        Run CMA-ES for a given number of generations.
        """
        overall_start = time.time()
        best_rmse = float('inf')
        sigmas = []
        cond_numbers = []
        max_eigenvalues = []
        min_eigenvalues = []

        print("\n    Initial mean: [" + " ".join(f"{v:.4f}" for v in self.cmaes.mean) + "]")
        print(f"    Initial sigma: {self.cmaes.sigma:.4f}\n")

        for gen in range(gen_size):
            gen_start = time.time()
            print(f"> Starting generation {gen+1}/{gen_size}")
            solutions = self.cmaes.ask()
            rmses     = []
            for x_unit in solutions:
                r = self.simulator.evaluate(x_unit)
                self.log.append((x_unit.copy(), r))
                rmses.append(r)
                if r < best_rmse:
                    best_rmse = r
                    print(f">> New best RMSE: {best_rmse:.4f} at generation {gen+1}")
            self.cmaes.tell(solutions, rmses)
            # Print mu, sigma after update
            print(f"[CMAES] Generation {gen+1}:")
            print("  mean : [" + " ".join(f"{v:.4f}" for v in self.cmaes.mean) + "]")
            print(f"  sigma: {self.cmaes.sigma:.4f}")

            # Eigen decomposition and reporting
            eigvals, eigvecs = np.linalg.eigh(self.cmaes.C)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            cond_number = eigvals[0] / eigvals[-1] if eigvals[-1] != 0 else np.inf

            # Track for plotting
            sigmas.append(self.cmaes.sigma)
            cond_numbers.append(cond_number)
            max_eigenvalues.append(eigvals[0])
            min_eigenvalues.append(eigvals[-1])

            print(f"  Condition number of C: {cond_number:.2f}")
            print("  Eigenvalues by principal direction (descending):")
            for i, val in enumerate(eigvals):
                main_param_idx = np.argmax(np.abs(eigvecs[:, i]))
                print(f"    Param {main_param_idx+1}: {val:.2f}")

            print(f">>> End of generation {gen+1} | Best RMSE this gen: {min(rmses):.4f} | Generation time: {time.time() - gen_start:.2f} seconds")

        # Plot condition number, max eigenvalue, min eigenvalue over generations
        if self.plot_evolution:
            generations = np.arange(1, gen_size+1)
            _, ax1 = plt.subplots(figsize=(8,5))
            l1, = ax1.plot(generations, cond_numbers, label='Condition Number')
            l2, = ax1.plot(generations, max_eigenvalues, label='Max Eigenvalue')
            l3, = ax1.plot(generations, min_eigenvalues, label='Min Eigenvalue')
            ax1.set_yscale('log')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Value (log scale)')
            ax1.set_title('Covariance Matrix Evolution')

            # Secondary y-axis for sigma
            ax2 = ax1.twinx()
            l4, = ax2.plot(generations, sigmas, color='tab:red', label='Sigma', linestyle=':', marker='o')
            ax2.set_ylabel('Sigma', color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')

            # Merge legends
            lines = [l1, l2, l3, l4]
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels, loc='upper left')

            plt.tight_layout()
            plt.show()

        print(f"[DEBUG] CMA-ES run complete. Total runtime: {time.time() - overall_start:.2f} seconds.")

        # Print the top 10 RMSEs found by CMA-ES
        log_sorted = sorted(self.log, key=lambda t: t[1])
        print("> Top 10 RMSEs found by CMA-ES:")
        for rank, (_, rmse) in enumerate(log_sorted[:10], 1):
            print(f"  {rank}: RMSE={rmse:.4f}")

    def save_log(self, filename="calibration.csv"):
        """
        Write out the sequence of all evaluated (model → rmse) in order.
        """
        t0 = time.time()
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = [f"param_{i}" for i in range(len(self.log[0][0]))] + ["rmse"]
            writer.writerow(header)
            for x_unit, r in self.log:
                x_scaled = self.param_lower + np.array(x_unit) * (self.param_upper - self.param_lower)
                writer.writerow([f"{v:.4f}" for v in x_scaled] + [f"{r:.4f}"])
        print(f"[DEBUG] Log saved to {filename}. (Elapsed: {time.time() - t0:.2f} seconds)")

if __name__ == "__main__":
# Set random seeds for reproducibility
    seed_model_simulator = 1990
    seed_optimizer = 1993
    seed_numpy_global = 2025

    num_optvars = 12
    param_lower = np.full(num_optvars, 0.0)
    param_upper = np.full(num_optvars, 1.0)
    exhaustive_modelcount = 500_000
    eval_budget = 50 * num_optvars**2

    PLOT_RMSE_LANDSCAPE = False
    PLOT_EVOLUTION = True

    np.random.seed(seed_numpy_global)
    pop_size = 25*num_optvars
    gen_size = int(np.floor(eval_budget/pop_size))
    sigma_init = min(0.3, 1/np.sqrt(num_optvars))

    # 1) Instantiate simulator with 500k dummy entries
    simulator = ModelSimulator(num_models=exhaustive_modelcount, num_optvars=num_optvars, seed=seed_model_simulator)

    # 2) Instantiate and run CMA-ES calibration (in [0,1]^dim)
    calibrator = CMAESCalibration(simulator,
                                  dim=num_optvars,
                                  popsize=pop_size,
                                  sigma=sigma_init,
                                  param_lower=param_lower,
                                  param_upper=param_upper,
                                  plot_evolution=PLOT_EVOLUTION,
                                  seed=seed_optimizer)

    print(f"Starting CMA-ES calibration in {num_optvars}-D: {gen_size} generations x {pop_size} evals/gen")
    calibrator.run(gen_size=gen_size)

    # 3) Save the full evaluation log (scaled back to original space)
    calibrator.save_log("CMAES_Result.csv")

    if PLOT_RMSE_LANDSCAPE:
        plot_rmse_vs_param(simulator, num_points=20)

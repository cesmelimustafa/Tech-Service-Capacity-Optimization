import os
import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any

# Configure robust logging for production-grade execution
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s'
)
logger = logging.getLogger("DES_Input_Analyzer")


class InputModelingPipeline:
    """
    Advanced Statistical Input Modeling pipeline for Discrete Event Simulation (DES).

    This class is responsible for ingesting empirical service data, applying Maximum
    Likelihood Estimation (MLE) to fit theoretical probability distributions, and
    performing Goodness-of-Fit (GoF) tests (e.g., Kolmogorov-Smirnov) to validate
    assumptions for Arena simulation entities.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.dataset = None

        # Set visualization aesthetics
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({'figure.max_open_warning': 0})

    def generate_synthetic_telemetry(self, n_samples: int = 150000) -> pd.DataFrame:
        """
        Generates a synthetic dataset mimicking the Kaggle aftersales repair dataset
        to overcome proprietary data sharing constraints.

        - Interarrival Time: Modeled via Exponential Distribution (Poisson Process).
        - Service Time: Modeled via Lognormal Distribution (Right-skewed, long-tail).
        """
        logger.info(f"Generating synthetic empirical data (N={n_samples}) based on baseline metrics.")
        np.random.seed(self.random_seed)

        # Interarrival times (Target mean ~ 35.0 minutes)
        interarrival = np.random.exponential(scale=35.0, size=n_samples)

        # Service times (Targeting a heavy-tailed distribution observed in hardware repairs)
        service_time = np.random.lognormal(mean=3.5, sigma=0.8, size=n_samples)

        self.dataset = pd.DataFrame({
            'interarrival_min': interarrival,
            'service_time_min': service_time
        })

        # Filter out any non-physical (zero or negative) values
        self.dataset = self.dataset[(self.dataset > 0).all(axis=1)].copy()
        logger.info(f"Data generation complete. Valid records: {len(self.dataset)}")

        return self.dataset

    def _evaluate_goodness_of_fit(self, data: pd.Series, dist_name: str, *args) -> Dict[str, Any]:
        """
        Executes the Kolmogorov-Smirnov test to evaluate the Goodness-of-Fit
        between the empirical data and the estimated theoretical distribution.
        """
        ks_stat, p_value = stats.kstest(data, dist_name, args=args)
        return {
            'KS_Statistic': ks_stat,
            'p_value': p_value
        }

    def fit_distributions(self) -> Dict[str, Dict[str, Any]]:
        """
        Performs Maximum Likelihood Estimation (MLE) for model parameters.
        """
        if self.dataset is None:
            raise ValueError("Dataset is not initialized. Run data generation first.")

        logger.info("Initiating Maximum Likelihood Estimation (MLE) procedures...")
        results = {}

        # 1. Exponential Fit for Interarrival Times
        interarrival_data = self.dataset['interarrival_min']
        loc_exp, scale_exp = stats.expon.fit(interarrival_data)
        gof_exp = self._evaluate_goodness_of_fit(interarrival_data, 'expon', loc_exp, scale_exp)

        results['Interarrival_Time'] = {
            'Distribution': 'Exponential',
            'Parameters': {'Loc': loc_exp, 'Scale (Mean)': scale_exp},
            'GoF_KSTest': gof_exp
        }

        logger.info(f"Interarrival Fit -> Expon(Scale={scale_exp:.4f}) | KS_Stat: {gof_exp['KS_Statistic']:.4f}")

        # 2. Lognormal Fit for Service Times
        service_data = self.dataset['service_time_min']
        shape_ln, loc_ln, scale_ln = stats.lognorm.fit(service_data, floc=0)
        gof_ln = self._evaluate_goodness_of_fit(service_data, 'lognorm', shape_ln, loc_ln, scale_ln)

        results['Service_Time'] = {
            'Distribution': 'Lognormal',
            'Parameters': {'Shape': shape_ln, 'Loc': loc_ln, 'Scale': scale_ln},
            'GoF_KSTest': gof_ln
        }

        logger.info(
            f"Service Time Fit -> Lognorm(Shape={shape_ln:.4f}, Scale={scale_ln:.4f}) | KS_Stat: {gof_ln['KS_Statistic']:.4f}")

        return results

    def generate_diagnostic_plots(self, output_dir: str = "docs"):
        """
        Generates high-resolution diagnostic plots combining Probability Density
        Functions (PDF) with empirical Histograms, and Quantile-Quantile (Q-Q) plots.
        """
        if self.dataset is None:
            raise ValueError("Dataset is not initialized.")

        logger.info("Generating diagnostic visual artifacts (PDF, Q-Q Plots)...")
        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

        # --- Interarrival Time Artifacts ---
        # Histogram & KDE
        sns.histplot(self.dataset['interarrival_min'], bins=60, stat='density',
                     color='steelblue', alpha=0.6, ax=axes[0, 0])
        axes[0, 0].set_title('Empirical Density vs Theoretical PDF (Interarrival)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Interarrival Time (Minutes)')

        # Q-Q Plot
        stats.probplot(self.dataset['interarrival_min'], dist="expon", plot=axes[0, 1])
        axes[0, 1].get_lines()[0].set_markerfacecolor('steelblue')
        axes[0, 1].get_lines()[0].set_markeredgecolor('steelblue')
        axes[0, 1].set_title('Q-Q Plot: Exponential MLE', fontsize=12, fontweight='bold')

        # --- Service Time Artifacts ---
        # Histogram & KDE (Focusing on the long-tail up to 99th percentile for visualization)
        p99 = np.percentile(self.dataset['service_time_min'], 99)
        filtered_service = self.dataset[self.dataset['service_time_min'] < p99]['service_time_min']

        sns.histplot(filtered_service, bins=60, stat='density',
                     color='seagreen', alpha=0.6, ax=axes[1, 0])
        axes[1, 0].set_title('Empirical Density vs Theoretical PDF (Service Time)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Service Time (Minutes) - Trimmed at 99th Percentile')

        # Q-Q Plot
        shape_ln, _, _ = stats.lognorm.fit(self.dataset['service_time_min'], floc=0)
        stats.probplot(self.dataset['service_time_min'], dist=stats.lognorm(shape_ln), plot=axes[1, 1])
        axes[1, 1].get_lines()[0].set_markerfacecolor('seagreen')
        axes[1, 1].get_lines()[0].set_markeredgecolor('seagreen')
        axes[1, 1].set_title('Q-Q Plot: Lognormal MLE', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'statistical_input_diagnostics.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Diagnostic plots successfully exported to: {plot_path}")


if __name__ == "__main__":
    # Execute the Input Modeling Pipeline
    pipeline = InputModelingPipeline(random_seed=1001)

    # Step 1: Synthesize required DES input metrics
    pipeline.generate_synthetic_telemetry(n_samples=169474)  # Mirroring the positive observation count

    # Step 2: Fit Theoretical Distributions & Run GoF tests
    fit_metrics = pipeline.fit_distributions()

    # Step 3: Export Visualization Artifacts
    pipeline.generate_diagnostic_plots(output_dir="docs")
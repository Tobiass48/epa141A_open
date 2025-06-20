#!/usr/bin/env python3
"""
Standalone SOBOL Sensitivity Analysis for IJssel River MORDM
==========================================================

This script runs SOBOL sensitivity analysis independently from the main notebook.
It includes all necessary dependencies and can be run multiple times for testing
different configurations without rerunning the entire MORDM analysis.

Usage:
    python standalone_sobol_analysis.py

Configuration:
    Edit the CONFIGURATION section below to adjust parameters.
"""

import os
import sys
import time
import pickle
import warnings
from datetime import datetime

# Standard scientific libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# EMA Workbench imports
from ema_workbench import (
    ema_logging, 
    SequentialEvaluator, 
    MultiprocessingEvaluator,
    Samplers,
    save_results, 
    load_results
)
from ema_workbench.em_framework import get_SALib_problem

# SALib for sensitivity analysis
from SALib.analyze import sobol

# Custom imports for IJssel River model
from problem_formulation import get_model_for_problem_formulation

# Configure warnings and logging
warnings.filterwarnings('ignore')
ema_logging.log_to_stderr(ema_logging.INFO)

# =============================================================================
# CONFIGURATION
# =============================================================================

class SOBOLConfig:
    """Configuration class for SOBOL analysis parameters"""
    
    # Problem formulation (should match notebook)
    PROBLEM_FORMULATION_ID = 2
    
    # SOBOL sampling parameters
    N_SOBOL_SAMPLES = 16  # Start smaller for testing (was 64 in notebook)
    
    # Execution method
    USE_SEQUENTIAL = True  # True for stability, False for parallel
    N_PROCESSES = 2  # Only used if USE_SEQUENTIAL = False
    
    # Output configuration
    SAVE_RESULTS = True
    SAVE_PLOTS = True
    SHOW_PLOTS = False  # Set to True to display plots interactively
    
    # Fallback configuration
    ENABLE_FALLBACK = True
    FALLBACK_SAMPLES = 8

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_header(message):
    """Print a formatted header message"""
    print("\n" + "="*60)
    print(f" {message}")
    print("="*60)

def print_status(message, status="INFO"):
    """Print a status message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    symbols = {"INFO": "ℹ", "SUCCESS": "✓", "WARNING": "⚠", "ERROR": "✗"}
    symbol = symbols.get(status, "•")
    print(f"[{timestamp}] {symbol} {message}")

def create_output_filename(base_name, n_samples, extension=""):
    """Create a standardized output filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_n{n_samples}_{timestamp}{extension}"

# =============================================================================
# MODEL SETUP AND VALIDATION
# =============================================================================

def setup_dike_model():
    """Set up the IJssel dike model for SOBOL analysis"""
    print_header("MODEL SETUP")
    
    try:
        # Get model from problem formulation
        print_status(f"Loading problem formulation {SOBOLConfig.PROBLEM_FORMULATION_ID}")
        model_result = get_model_for_problem_formulation(SOBOLConfig.PROBLEM_FORMULATION_ID)
        
        # Handle potential tuple return
        if isinstance(model_result, tuple):
            dike_model, planning_steps = model_result
        else:
            dike_model = model_result
            planning_steps = [0, 1, 2]  # Default planning steps
        
        print_status(f"Model '{dike_model.name}' loaded successfully", "SUCCESS")
        print_status(f"Uncertainties: {len(dike_model.uncertainties)}")
        print_status(f"Levers: {len(dike_model.levers)}")
        print_status(f"Outcomes: {len(dike_model.outcomes)}")
        print_status(f"Planning steps: {planning_steps}")
        
        # Display outcomes
        print("\nModel Outcomes:")
        for i, outcome in enumerate(dike_model.outcomes, 1):
            print(f"  {i}. {outcome.name}")
        
        return dike_model, planning_steps
        
    except Exception as e:
        print_status(f"Failed to setup model: {str(e)}", "ERROR")
        raise

def validate_model(dike_model):
    """Validate the model with a small test run"""
    print_header("MODEL VALIDATION")
    
    try:
        print_status("Running small validation test (4 scenarios)")
        
        with SequentialEvaluator(dike_model) as evaluator:
            test_results = evaluator.perform_experiments(scenarios=4)
        
        test_experiments, test_outcomes = test_results
        print_status(f"Validation successful: {len(test_experiments)} experiments", "SUCCESS")
        
        # Check outcome data quality
        for outcome_name in test_outcomes.keys():
            outcome_data = test_outcomes[outcome_name]
            if hasattr(outcome_data, '__len__'):
                print_status(f"  {outcome_name}: {len(outcome_data)} values")
            else:
                print_status(f"  {outcome_name}: {type(outcome_data)}")
        
        return True
        
    except Exception as e:
        print_status(f"Model validation failed: {str(e)}", "ERROR")
        return False

# =============================================================================
# SOBOL ANALYSIS IMPLEMENTATION
# =============================================================================

def run_sobol_experiments(dike_model, n_samples):
    """Run SOBOL experiments using the configured evaluator"""
    print_header("SOBOL EXPERIMENTS")
    
    # Calculate expected experiments
    n_uncertainties = len(dike_model.uncertainties)
    expected_experiments = n_samples * (2 * n_uncertainties + 2)
    
    print_status(f"SOBOL samples: {n_samples}")
    print_status(f"Uncertainties: {n_uncertainties}")
    print_status(f"Expected experiments: {expected_experiments}")
    
    # Choose evaluator based on configuration
    evaluator_class = SequentialEvaluator if SOBOLConfig.USE_SEQUENTIAL else MultiprocessingEvaluator
    evaluator_name = "Sequential" if SOBOLConfig.USE_SEQUENTIAL else f"Multiprocessing ({SOBOLConfig.N_PROCESSES} processes)"
    
    print_status(f"Using {evaluator_name} evaluator")
    
    try:
        start_time = time.time()
        
        # Set up evaluator arguments
        evaluator_kwargs = {}
        if not SOBOLConfig.USE_SEQUENTIAL:
            evaluator_kwargs['n_processes'] = SOBOLConfig.N_PROCESSES
        
        # Run experiments
        with evaluator_class(dike_model, **evaluator_kwargs) as evaluator:
            sobol_results = evaluator.perform_experiments(
                scenarios=n_samples,
                uncertainty_sampling=Samplers.SOBOL
            )
        
        elapsed_time = time.time() - start_time
        sobol_experiments, sobol_outcomes = sobol_results
        
        print_status(f"Experiments completed in {elapsed_time:.1f} seconds", "SUCCESS")
        print_status(f"Total experiments: {len(sobol_experiments)}")
        
        return sobol_results
        
    except Exception as e:
        print_status(f"Experiment execution failed: {str(e)}", "ERROR")
        raise

def analyze_sobol_indices(sobol_results, dike_model):
    """Analyze SOBOL indices for each outcome"""
    print_header("SOBOL INDICES ANALYSIS")
    
    sobol_experiments, sobol_outcomes = sobol_results
    
    # Get SALib problem formulation
    sa_problem = get_SALib_problem(dike_model.uncertainties)
    print_status(f"SALib problem created with {sa_problem['num_vars']} variables")
    
    # Analyze each outcome
    sobol_indices = {}
    outcome_names = [outcome.name for outcome in dike_model.outcomes]
    successful_outcomes = []
    
    print_status(f"Analyzing {len(outcome_names)} outcomes...")
    
    for i, outcome_name in enumerate(outcome_names):
        print_status(f"  {i+1}/{len(outcome_names)}: {outcome_name}")
        
        try:
            # Get model outputs for this outcome
            if isinstance(sobol_outcomes[outcome_name], list):
                Y = np.array(sobol_outcomes[outcome_name])
            else:
                Y = sobol_outcomes[outcome_name].values if hasattr(sobol_outcomes[outcome_name], 'values') else sobol_outcomes[outcome_name]
            
            # Handle different outcome array shapes
            if Y.ndim > 1:
                Y = Y.flatten()
            
            # Remove NaN values for SOBOL analysis
            Y_clean = Y[~np.isnan(Y)]
            
            if len(Y_clean) < len(Y):
                print_status(f"    Removed {len(Y) - len(Y_clean)} NaN values", "WARNING")
            
            if len(Y_clean) < len(Y) * 0.8:  # Less than 80% valid data
                print_status(f"    Only {len(Y_clean)}/{len(Y)} valid samples", "WARNING")
            
            # Compute SOBOL indices
            Si = sobol.analyze(sa_problem, Y_clean, calc_second_order=True, print_to_console=False)
            sobol_indices[outcome_name] = Si
            successful_outcomes.append(outcome_name)
            
            print_status(f"    Analysis complete", "SUCCESS")
            
        except Exception as e:
            print_status(f"    Failed: {str(e)}", "ERROR")
            sobol_indices[outcome_name] = None
    
    print_status(f"Successfully analyzed {len(successful_outcomes)}/{len(outcome_names)} outcomes", "SUCCESS")
    
    return sobol_indices, sa_problem, successful_outcomes

def create_summary_table(sobol_indices, sa_problem, successful_outcomes):
    """Create a summary table of SOBOL indices"""
    print_status("Creating summary table")
    
    summary_data = []
    
    for outcome_name in successful_outcomes:
        Si = sobol_indices[outcome_name]
        uncertainty_names = sa_problem['names']
        
        for i, uncertainty in enumerate(uncertainty_names):
            summary_data.append({
                'Outcome': outcome_name,
                'Uncertainty': uncertainty,
                'S1': Si['S1'][i],
                'S1_conf': Si['S1_conf'][i],
                'ST': Si['ST'][i],
                'ST_conf': Si['ST_conf'][i]
            })
    
    return pd.DataFrame(summary_data)

def visualize_sobol_results(sobol_indices, sa_problem, successful_outcomes, save_plots=True):
    """Create visualization of SOBOL results"""
    print_header("VISUALIZATION")
    
    if len(successful_outcomes) == 0:
        print_status("No successful outcomes to visualize", "WARNING")
        return
    
    print_status(f"Creating plots for {len(successful_outcomes)} outcomes")
    
    # Create subplots
    n_plots = min(6, len(successful_outcomes))  # Max 6 plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, outcome_name in enumerate(successful_outcomes[:n_plots]):
        ax = axes[i]
        
        # Extract indices
        S1 = sobol_indices[outcome_name]['S1']
        ST = sobol_indices[outcome_name]['ST']
        S1_conf = sobol_indices[outcome_name]['S1_conf']
        ST_conf = sobol_indices[outcome_name]['ST_conf']
        
        # Create DataFrame for plotting
        indices_df = pd.DataFrame({
            'S1': S1,
            'ST': ST
        }, index=sa_problem['names'])
        
        err_df = pd.DataFrame({
            'S1_conf': S1_conf,
            'ST_conf': ST_conf
        }, index=sa_problem['names'])
        
        # Create bar plot
        indices_df.plot.bar(yerr=err_df.values.T, ax=ax, capsize=3, alpha=0.7)
        ax.set_title(f'SOBOL Indices: {outcome_name}', fontsize=10)
        ax.set_xlabel('Uncertainties')
        ax.set_ylabel('Sensitivity Index')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend(['First-order (S1)', 'Total-order (ST)'])
    
    # Remove empty subplots
    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    if save_plots:
        plot_filename = create_output_filename("IJssel_SOBOL_plots", SOBOLConfig.N_SOBOL_SAMPLES, ".png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print_status(f"Plots saved to {plot_filename}", "SUCCESS")
    
    if SOBOLConfig.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

def print_top_sensitivities(sobol_indices, sa_problem, successful_outcomes):
    """Print top sensitivities for each outcome"""
    print_header("TOP SENSITIVITIES")
    
    for outcome_name in successful_outcomes:
        ST = sobol_indices[outcome_name]['ST']  # Use total-order indices
        uncertainty_names = sa_problem['names']
        
        # Get top 3 uncertainties
        top_indices = np.argsort(ST)[-3:][::-1]
        
        print(f"\n{outcome_name}:")
        for j, idx in enumerate(top_indices):
            print(f"  {j+1}. {uncertainty_names[idx]}: ST = {ST[idx]:.3f}")

# =============================================================================
# FILE I/O OPERATIONS
# =============================================================================

def save_all_results(sobol_results, sobol_indices, summary_df, n_samples):
    """Save all analysis results to files"""
    print_header("SAVING RESULTS")
    
    if not SOBOLConfig.SAVE_RESULTS:
        print_status("Result saving disabled in configuration")
        return
    
    # Save experimental results
    exp_filename = f'IJssel_SOBOL_results_n{n_samples}.tar.gz'
    save_results(sobol_results, exp_filename)
    print_status(f"Experimental results saved to {exp_filename}", "SUCCESS")
    
    # Save SOBOL indices
    indices_filename = f'IJssel_SOBOL_indices_n{n_samples}.pkl'
    with open(indices_filename, 'wb') as f:
        pickle.dump(sobol_indices, f)
    print_status(f"SOBOL indices saved to {indices_filename}", "SUCCESS")
    
    # Save summary table
    summary_filename = f'IJssel_SOBOL_summary_n{n_samples}.csv'
    summary_df.to_csv(summary_filename, index=False)
    print_status(f"Summary table saved to {summary_filename}", "SUCCESS")
    
    return exp_filename, indices_filename, summary_filename

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def run_fallback_analysis(dike_model):
    """Run fallback analysis with smaller sample size"""
    print_header("FALLBACK ANALYSIS")
    
    print_status(f"Attempting fallback with {SOBOLConfig.FALLBACK_SAMPLES} samples")
    
    try:
        with SequentialEvaluator(dike_model) as evaluator:
            fallback_results = evaluator.perform_experiments(
                scenarios=SOBOLConfig.FALLBACK_SAMPLES,
                uncertainty_sampling=Samplers.SOBOL
            )
        
        print_status("Fallback experiments successful", "SUCCESS")
        print_status(f"Consider using n_samples={SOBOLConfig.FALLBACK_SAMPLES} for full analysis")
        
        return fallback_results
        
    except Exception as e:
        print_status(f"Fallback also failed: {str(e)}", "ERROR")
        return None

def main():
    """Main execution function"""
    print_header("STANDALONE SOBOL SENSITIVITY ANALYSIS")
    print_status("Starting analysis...")
    print_status(f"Configuration: {SOBOLConfig.N_SOBOL_SAMPLES} samples, {'Sequential' if SOBOLConfig.USE_SEQUENTIAL else 'Parallel'} execution")
    
    try:
        # 1. Setup and validate model
        dike_model, planning_steps = setup_dike_model()
        
        if not validate_model(dike_model):
            print_status("Model validation failed. Aborting analysis.", "ERROR")
            return
        
        # 2. Run SOBOL experiments
        try:
            sobol_results = run_sobol_experiments(dike_model, SOBOLConfig.N_SOBOL_SAMPLES)
        except Exception as e:
            if SOBOLConfig.ENABLE_FALLBACK:
                print_status("Main analysis failed, trying fallback", "WARNING")
                sobol_results = run_fallback_analysis(dike_model)
                if sobol_results is None:
                    raise e
            else:
                raise e
        
        # 3. Analyze SOBOL indices
        sobol_indices, sa_problem, successful_outcomes = analyze_sobol_indices(sobol_results, dike_model)
        
        if len(successful_outcomes) == 0:
            print_status("No outcomes were successfully analyzed", "ERROR")
            return
        
        # 4. Create summary and visualizations
        summary_df = create_summary_table(sobol_indices, sa_problem, successful_outcomes)
        visualize_sobol_results(sobol_indices, sa_problem, successful_outcomes, SOBOLConfig.SAVE_PLOTS)
        print_top_sensitivities(sobol_indices, sa_problem, successful_outcomes)
        
        # 5. Save results
        save_all_results(sobol_results, sobol_indices, summary_df, SOBOLConfig.N_SOBOL_SAMPLES)
        
        print_header("ANALYSIS COMPLETE")
        print_status(f"Successfully analyzed {len(successful_outcomes)} outcomes", "SUCCESS")
        print_status("All results saved to files", "SUCCESS")
        
    except Exception as e:
        print_status(f"Analysis failed: {str(e)}", "ERROR")
        print("\nFull error traceback:")
        import traceback
        traceback.print_exc()

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main() 
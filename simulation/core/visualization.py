import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator
import seaborn as sns

# Set up publication-quality plot parameters
def set_publication_style():
    """Set global matplotlib parameters for truly publication-quality figures based on best practices"""
    plt.rcParams.update({
        # Font settings - using professional serif fonts commonly found in scientific publications
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Computer Modern Roman', 'DejaVu Serif'],
        'font.size': 10,  # Optimal size for readability in publications
        'font.weight': 'normal',
        
        # Text settings
        'text.usetex': False,  # True if LaTeX is available
        'text.color': '#333333',  # Dark gray for better contrast
        
        # Axes settings
        'axes.linewidth': 1.0,  # Slightly thicker lines
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'axes.labelweight': 'normal',
        'axes.labelcolor': '#333333',
        'axes.axisbelow': True,  # Grid lines behind data
        'axes.grid': False,  # Grid off by default, added selectively
        'axes.spines.top': False,  # Remove top spine
        'axes.spines.right': False,  # Remove right spine
        'axes.formatter.use_mathtext': True,  # For exponents
        
        # Tick settings
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.visible': False,
        'ytick.minor.visible': False,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # Legend settings
        'legend.frameon': False,  # No frame around legend
        'legend.fontsize': 9,
        'legend.title_fontsize': 10,
        'legend.numpoints': 1,  # Single point in legend
        'legend.markerscale': 0.9,  # Slightly smaller markers in legend
        'legend.handlelength': 1.0,  # Shorter legend lines
        
        # Figure settings
        'figure.figsize': (7, 5),  # Default size for single column in journals
        'figure.dpi': 300,  # Publication quality
        'figure.constrained_layout.use': True,  # Better automatic layout
        'figure.facecolor': 'white',
        
        # Saving settings
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Output settings (for PDF editing)
        'pdf.fonttype': 42,  # Embed fonts (Type 1)
        'ps.fonttype': 42,
    })
    
    # Set a scientific color cycle with high distinguishability
    # Based on ColorBrewer schemes designed for scientific visualization
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf'   # Teal
    ])
    
    # Create output directory if it doesn't exist
    os.makedirs("simulation_graphs", exist_ok=True)

def stylize_axes(ax, with_grid=True):
    """Apply enhanced publication-quality styling to axis"""
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make remaining spines more professional
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    # Set axis below data
    ax.set_axisbelow(True)
    
    # Style ticks
    ax.xaxis.set_tick_params(direction='out', width=1.0, length=4, pad=4, 
                           colors='#333333', which='both')
    ax.yaxis.set_tick_params(direction='out', width=1.0, length=4, pad=4, 
                           colors='#333333', which='both')
    
    # Use scientific notation for large/small numbers, but only if using compatible formatters
    # Check formatter types to avoid errors with categorical axes
    try:
        if isinstance(ax.xaxis.get_major_formatter(), ScalarFormatter):
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.xaxis.get_major_formatter().set_scientific(True)
            ax.xaxis.get_major_formatter().set_powerlimits((-3, 4))
        
        if isinstance(ax.yaxis.get_major_formatter(), ScalarFormatter):
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.yaxis.get_major_formatter().set_scientific(True)
            ax.yaxis.get_major_formatter().set_powerlimits((-3, 4))
    except Exception:
        # Silently continue if there's an issue with formatters
        pass
    
    # Add subtle y-grid when requested
    if with_grid:
        ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.5, color='#cccccc')
    
    # Use clean white background
    ax.set_facecolor('white')
    
    # Ensure tick labels are properly sized and colored
    for label in ax.get_xticklabels():
        label.set_fontsize(9)
        label.set_color('#333333')
    
    for label in ax.get_yticklabels():
        label.set_fontsize(9)
        label.set_color('#333333')
    
    # Ensure axis labels are properly styled
    if ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel(), fontsize=11, color='#333333')
    if ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel(), fontsize=11, color='#333333')
    
    # If there's a title, ensure it's properly styled
    if ax.get_title():
        ax.set_title(ax.get_title(), fontsize=12, fontweight='bold', color='#333333')
    
    return ax

def plot_performance_comparison(results_df, output_file='performance_comparison.pdf'):
    """Plot performance comparison between methods with publication-quality styling"""
    # Select only the most important metrics for clarity
    key_metrics = ['accuracy', 'f1_score', 'far', 'frr']
    
    # Create a subset of the data with only key metrics
    plot_data = results_df[['Method'] + key_metrics].set_index('Method')
    
    # Create a color palette with higher contrast
    colors = sns.color_palette("muted", len(key_metrics))
    
    # Plot
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax = plot_data.plot(kind='bar', color=colors, ax=ax)
    
    # Add minimalist title and labels
    ax.set_title('Performance Comparison', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score', fontsize=10)
    ax.set_xlabel('Method', fontsize=10)
    
    # Rotate x-tick labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    
    # Add subtle value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=7, padding=3)
    
    # Style the axis
    stylize_axes(ax)
    
    # Add subtle y-grid
    ax.grid(axis='y', linestyle=':', alpha=0.3, color='#cccccc')
    
    # Place legend outside the plot for clarity
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"simulation_graphs/{output_file}", dpi=300, bbox_inches='tight')
    plt.savefig(f"simulation_graphs/{output_file.replace('.pdf', '.png')}", dpi=300, bbox_inches='tight')
    plt.close()

def plot_risk_factor_weights(weight_history, factor_names, output_file='factor_weights_evolution.pdf'):
    """Plot risk factor weights over time with publication-quality styling"""
    methods = list(weight_history.keys())
    
    # Create a figure with subplots for each factor
    fig, axes = plt.subplots(len(factor_names), 1, figsize=(8, 2.0*len(factor_names)), sharex=True)
    
    # Ensure axes is always a list for consistent indexing
    if len(factor_names) == 1:
        axes = [axes]
    
    # Create a color palette with better differentiation
    colors = sns.color_palette("deep", len(methods))
    
    # Set a common y-axis range to make factors comparable
    y_min, y_max = float('inf'), float('-inf')
    
    # First pass to determine common y-axis range
    for factor in factor_names:
        for method in methods:
            if factor in weight_history[method] and weight_history[method][factor]:
                weights = [h[1] for h in weight_history[method][factor]]
                if weights:
                    y_min = min(y_min, min(weights))
                    y_max = max(y_max, max(weights))
    
    # Add padding to y-axis
    y_range = y_max - y_min
    y_min = max(0, y_min - y_range * 0.1)  # Don't go below 0 for weights
    y_max = y_max + y_range * 0.1
    
    # Create a common time range
    all_timestamps = []
    for method in methods:
        for factor in factor_names:
            if factor in weight_history[method] and weight_history[method][factor]:
                all_timestamps.extend([h[0] for h in weight_history[method][factor]])
    
    if all_timestamps:
        min_time = min(all_timestamps)
        max_time = max(all_timestamps)
    else:
        # Fallback if no timestamps
        min_time, max_time = 0, 1
    
    # Process the data with significantly reduced smoothing
    for i, factor in enumerate(factor_names):
        # Track which methods are plotted for this factor for legend
        legend_handles = []
        legend_labels = []
        
        for j, method in enumerate(methods):
            if factor in weight_history[method] and weight_history[method][factor]:
                # Get the data for this factor/method
                factor_history = weight_history[method][factor]
                
                # Skip if no data
                if not factor_history:
                    continue
                
                # Perform intelligent downsampling for dense data - reduced to preserve more detail
                if len(factor_history) > 1000:
                    # Less aggressive downsampling to preserve more detail
                    step = len(factor_history) // 500
                    factor_history = factor_history[::step]
                elif len(factor_history) > 200:
                    # Very light downsampling
                    step = len(factor_history) // 150
                    factor_history = factor_history[::step]
                
                # Sort by timestamp to ensure proper plotting
                factor_history.sort(key=lambda x: x[0])
                
                # Extract data
                timestamps = [pd.to_datetime(h[0], unit='s') for h in factor_history]
                weights = [h[1] for h in factor_history]
                
                # Apply much less smoothing to preserve important trends and variations
                if len(weights) > 20:
                    weights_series = pd.Series(weights)
                    # Much smaller window size to preserve more detail
                    window_size = max(2, len(weights) // 100)  # Significantly smaller window
                    smoothed_weights = weights_series.rolling(window=window_size, center=True, min_periods=1).mean()
                    
                    # Plot raw data with higher opacity to show actual variations
                    axes[i].plot(
                        timestamps, weights,
                        color=colors[j],
                        linewidth=0.9,
                        alpha=0.5,  # Increased alpha to show more of the actual data
                        zorder=5-j
                    )
                    
                    # Plot minimally smoothed line on top
                    line, = axes[i].plot(
                        timestamps, smoothed_weights, 
                        label=method, 
                        color=colors[j], 
                        linewidth=1.5,
                        alpha=0.9,
                        marker=None,
                        zorder=10-j
                    )
                else:
                    # For sparse data, show actual points with minimal smoothing
                    line, = axes[i].plot(
                        timestamps, weights, 
                        label=method, 
                        color=colors[j], 
                        linewidth=1.2,
                        alpha=0.9,
                        marker='o' if len(timestamps) < 30 else None,
                        markersize=3 if len(timestamps) < 30 else 0,
                        zorder=10-j
                    )
                
                # Add to legend only if the method has data for this factor
                legend_handles.append(line)
                legend_labels.append(method)
        
        # Set common y-axis limits
        axes[i].set_ylim(y_min, y_max)
        
        # Add labels with improved styling
        axes[i].set_title(f'{factor.capitalize()}', fontsize=11, fontweight='bold')
        axes[i].set_ylabel('Weight', fontsize=10)
        
        # Style the axis
        stylize_axes(axes[i], with_grid=True)
        
        # Add legend if we have plotted data
        if legend_handles:
            leg = axes[i].legend(
                legend_handles, legend_labels,
                loc='upper right', 
                fontsize=8, 
                frameon=True,
                framealpha=0.7,
                facecolor='white',
                edgecolor='lightgray'
            )
            leg.set_zorder(20)  # Ensure legend is on top
    
    # Format the time axis better
    from matplotlib.dates import AutoDateLocator, DateFormatter
    locator = AutoDateLocator(minticks=4, maxticks=8)
    formatter = DateFormatter('%m/%d %H:%M')
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)
    
    # Add common x-label
    axes[-1].set_xlabel('Time', fontsize=11)
    
    # Adjust spacing between subplots
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    # Save figure in multiple formats for different uses
    plt.savefig(f"simulation_graphs/{output_file}", dpi=300, bbox_inches='tight')
    plt.savefig(f"simulation_graphs/{output_file.replace('.pdf', '.png')}", dpi=300, bbox_inches='tight')
    plt.close()

def plot_authentication_methods(auth_decisions, output_file='auth_methods_evolution.pdf'):
    """Plot authentication method usage over time with publication-quality styling"""
    methods = list(auth_decisions.keys())
    
    # Create figure with better dimensions for time series
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    # Define better colors for different authentication levels - improved color scheme
    # Using a more distinct color palette for better differentiation
    auth_colors = ['#2166ac', '#67a9cf', '#d1e5f0']  # Blue sequential palette with higher contrast
    auth_labels = ['Password Only', 'Password + OTP', 'Password + OTP + Face']
    
    # Set line styles for each method to improve differentiation even in grayscale
    line_styles = ['-', '--', '-.', ':']
    
    # For each method, aggregate data by day with improved approach
    all_data = []
    
    # Process each method
    for method_idx, method in enumerate(methods):
        decisions = auth_decisions[method]
        if not decisions:
            continue
            
        # Group by day for clearer visualization
        from collections import defaultdict
        day_groups = defaultdict(list)
        
        for decision in decisions:
            # Round timestamp to nearest day
            timestamp = decision['timestamp']
            day = int(timestamp) // 86400 * 86400  # Seconds in a day
            day_groups[day].append(decision)
        
        # Sort days for consistent time progression
        sorted_days = sorted(day_groups.keys())
        
        # Skip if not enough days
        if len(sorted_days) <= 1:
            continue
            
        # Process each day
        day_data = {'timestamp': [], 'auth_level_counts': [[], [], []]}
        
        for day in sorted_days:
            day_decisions = day_groups[day]
            day_data['timestamp'].append(pd.to_datetime(day, unit='s'))
            
            # Count authentication methods
            level1 = sum(1 for d in day_decisions if len(d['auth_methods']) == 1)
            level2 = sum(1 for d in day_decisions if len(d['auth_methods']) == 2)
            level3 = sum(1 for d in day_decisions if len(d['auth_methods']) == 3)
            
            day_data['auth_level_counts'][0].append(level1)
            day_data['auth_level_counts'][1].append(level2) 
            day_data['auth_level_counts'][2].append(level3)
        
        all_data.append((method, day_data))
    
    # Plot with improved styling and clearer visual hierarchy
    handles, labels = [], []
    
    for method_idx, (method, data) in enumerate(all_data):
        if not data['timestamp']:
            continue
            
        # Use a consistent line style for each method
        line_style = line_styles[method_idx % len(line_styles)]
        
        # For each method, plot percentage of each auth level with better spacing
        for i in range(3):
            # Calculate percentage
            total_counts = [sum(x) for x in zip(*data['auth_level_counts'])]
            percentages = [100 * (count/total if total > 0 else 0) 
                          for count, total in zip(data['auth_level_counts'][i], total_counts)]
            
            # Show raw data with minimal smoothing to preserve variability
            # Only apply minimal smoothing for very noisy data
            raw_percentages = percentages.copy()
            if len(percentages) > 20:  # Only smooth if we have many data points
                percentages_series = pd.Series(percentages)
                # Use minimal smoothing (smaller window)
                percentages = percentages_series.rolling(window=2, center=True, min_periods=1).mean()
            
            # Create label that combines method and auth level
            label = f"{method.title()} - {auth_labels[i]}"
            
            # Plot raw data points with higher transparency
            if len(raw_percentages) > 10:
                ax.scatter(
                    data['timestamp'][::3],  # Plot every third point to reduce clutter
                    [raw_percentages[j] for j in range(0, len(raw_percentages), 3)],
                    color=auth_colors[i],
                    s=15,
                    alpha=0.4,
                    edgecolor='none',
                    zorder=8 + i
                )
            
            # Plot line with a unique style combination
            line = ax.plot(
                data['timestamp'], 
                percentages,
                label=label,
                linestyle=line_style,
                color=auth_colors[i],
                linewidth=1.8,  # Slightly thinner for better visualization
                alpha=0.85,
                marker=None,  # We already show raw data points separately
                zorder=10 + i  # Important: keeps lines from different methods in consistent order
            )
            
            handles.append(line[0])
            labels.append(label)
    
    # Add labels with better positioning and styling
    ax.set_title('Authentication Method Usage Over Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage of Total Authentications (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    
    # Set y-axis to 0-100% range for clarity
    ax.set_ylim(0, 100)
    
    # Format x-axis date labels for better readability
    from matplotlib.dates import AutoDateLocator, DateFormatter
    locator = AutoDateLocator(minticks=5, maxticks=10)
    formatter = DateFormatter('%m/%d')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    
    # Rotate date labels for better fit
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Style the axis with professional appearance
    stylize_axes(ax, with_grid=True)
    
    # Add subtle grid for easier value reading
    ax.grid(axis='y', linestyle=':', alpha=0.3, color='#cccccc')
    
    # Create a cleaner, more organized legend 
    # Group the legend by method for better organization
    if handles:  
        # Customize legend placement and style
        legend = ax.legend(
            handles, labels,
            loc='center left', 
            bbox_to_anchor=(1.02, 0.5), 
            fontsize=10,
            frameon=True,
            framealpha=0.9,
            edgecolor='lightgrey',
            title="Authentication Methods"
        )
        legend.get_title().set_fontsize(11)
        legend.get_title().set_fontweight('bold')
    
    # Add subtle shading for weekend dates
    min_date = min(data['timestamp'][0] for _, data in all_data)
    max_date = max(data['timestamp'][-1] for _, data in all_data)
    for date in pd.date_range(start=min_date, end=max_date, freq='W-SAT'):  # Saturday
        ax.axvspan(date, date + pd.Timedelta(days=2), color='#f5f5f5', alpha=0.5, zorder=0)
    
    # Save figure with ample space for legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.78)  # Make room for legend
    plt.savefig(f"simulation_graphs/{output_file}", dpi=300, bbox_inches='tight')
    plt.savefig(f"simulation_graphs/{output_file.replace('.pdf', '.png')}", dpi=300, bbox_inches='tight')
    plt.close()

def plot_risk_scores_over_time(auth_decisions, output_file='risk_scores_evolution.pdf'):
    """Plot risk scores over time for all methods with publication-quality styling"""
    methods = list(auth_decisions.keys())
    
    # Create figure with dimensions optimized for time series
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Create a professional color palette with high distinguishability
    colors = sns.color_palette("deep", len(methods))
    
    # Line styles to differentiate methods even in grayscale
    line_styles = ['-', '--', '-.', ':']
    
    # Track min and max dates for potential annotations
    min_date, max_date = None, None
    
    # Process and plot each method
    for i, method in enumerate(methods):
        decisions = auth_decisions[method]
        if not decisions:
            continue
            
        # Group by day for clearer visualization
        from collections import defaultdict
        day_groups = defaultdict(list)
        
        for decision in decisions:
            timestamp = decision['timestamp']
            day = int(timestamp) // 86400 * 86400  # Round to day
            day_groups[day].append(decision)
            
        # Sort days for consistent time progression
        sorted_days = sorted(day_groups.keys())
        
        # Skip if not enough days
        if len(sorted_days) <= 1:
            continue
        
        # Convert to datetime for plotting
        days = [pd.to_datetime(day, unit='s') for day in sorted_days]
        
        # Update min/max dates
        if min_date is None or days[0] < min_date:
            min_date = days[0]
        if max_date is None or days[-1] > max_date:
            max_date = days[-1]
            
        # Calculate average risk scores and standard deviations
        avg_risks = []
        std_devs = []
        raw_scores_days = []
        raw_scores_values = []
        
        for day in sorted_days:
            day_decisions = day_groups[day]
            risk_scores = [d['risk_score'] for d in day_decisions]
            avg_risks.append(sum(risk_scores) / len(risk_scores))
            std_devs.append(np.std(risk_scores))
            
            # Store raw scores for scatter plot (sample if too many)
            if len(risk_scores) > 20:
                # Take a sample of raw scores to avoid clutter
                sample_indices = np.linspace(0, len(risk_scores)-1, 15).astype(int)
                for idx in sample_indices:
                    raw_scores_days.append(pd.to_datetime(day, unit='s'))
                    raw_scores_values.append(risk_scores[idx])
            else:
                # For fewer points, show all raw scores
                for score in risk_scores:
                    raw_scores_days.append(pd.to_datetime(day, unit='s'))
                    raw_scores_values.append(score)
        
        # Use much less aggressive smoothing to preserve variability
        raw_avg_risks = avg_risks.copy()
        if len(avg_risks) > 15:
            risk_series = pd.Series(avg_risks)
            # Use smaller window size (was min(5, max(3, len(avg_risks) // 10)))
            window = min(3, max(2, len(avg_risks) // 20))
            avg_risks = risk_series.rolling(window=window, center=True, min_periods=1).mean()
        
        # Plot raw data points with transparency to show distribution
        if raw_scores_days:
            # Add jitter to x-axis to avoid overlap
            jitter_hours = np.random.uniform(-4, 4, len(raw_scores_days))
            jittered_days = [day + pd.Timedelta(hours=j) for day, j in zip(raw_scores_days, jitter_hours)]
            
            # Plot individual data points with small markers
            ax.scatter(
                jittered_days,
                raw_scores_values,
                color=colors[i],
                alpha=0.15,
                s=12,
                edgecolor='none',
                zorder=2
            )
        
        # Plot line with enhanced styling
        line_style = line_styles[i % len(line_styles)]
        
        ax.plot(
            days, 
            avg_risks, 
            label=method.title(), 
            color=colors[i],
            linestyle=line_style,
            linewidth=2.0,  # Slightly thinner for better visibility with raw data
            alpha=0.85,
            marker='o' if len(days) < 15 else None,
            markersize=5 if len(days) < 15 else 0,
            markerfacecolor=colors[i],
            markeredgecolor='white',
            markeredgewidth=0.7,
            zorder=10
        )
        
        # Add extra points to show variability if we have enough data
        if len(sorted_days) < 40 and len(sorted_days) > 3:  # Only for moderate amounts of data
            for j, day in enumerate(sorted_days):
                day_decisions = day_groups[day]
                if len(day_decisions) < 3:  # Skip days with too few decisions
                    continue
                    
                risk_scores = [d['risk_score'] for d in day_decisions]
                
                # Calculate standard error
                std_err = np.std(risk_scores) / np.sqrt(len(risk_scores))
                
                # Plot error bars for key points (avoids clutter)
                if j % max(1, len(sorted_days) // 10) == 0:  # Show ~10 error bars
                    ax.errorbar(
                        days[j],
                        avg_risks[j],
                        yerr=std_err,
                        fmt='none',
                        ecolor=colors[i],
                        elinewidth=1,
                        capsize=3,
                        alpha=0.5,
                        zorder=5
                    )
    
    # Mark environmental changes if we can detect them from the data
    env_changes = set()
    for method in methods:
        for decision in auth_decisions[method]:
            if 'environment' in decision and decision['environment'] != 'normal':
                env_changes.add((decision['timestamp'] // 86400 * 86400, decision['environment']))
    
    # Add subtle annotations for environment changes
    if env_changes and min_date and max_date:
        for timestamp, env in sorted(env_changes):
            date = pd.to_datetime(timestamp, unit='s')
            # Only show if within our plot range
            if min_date <= date <= max_date:
                ax.axvline(date, color='#999999', linestyle=':', linewidth=1, alpha=0.5, zorder=1)
                ax.annotate(
                    f"{env}",
                    xy=(date, 0.02),
                    xycoords=('data', 'axes fraction'),
                    rotation=90,
                    va='bottom',
                    ha='center',
                    fontsize=8,
                    color='#666666',
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="lightgrey", alpha=0.8)
                )
    
    # Add labels with professional styling
    ax.set_title('Daily Average Risk Scores', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Risk Score', fontsize=12)
    
    # Set y-axis to 0-1 range for risk scores
    ax.set_ylim(0, 1)
    
    # Add y-axis grid for easier reading of values
    ax.grid(axis='y', linestyle=':', alpha=0.3, color='#cccccc')
    
    # Style the axis with professional appearance
    stylize_axes(ax, with_grid=True)
    
    # Format x-axis date labels for better readability
    from matplotlib.dates import AutoDateLocator, DateFormatter
    locator = AutoDateLocator(minticks=5, maxticks=10)
    formatter = DateFormatter('%m/%d')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    
    # Rotate date labels for better fit
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add subtle shading for weekend dates
    if min_date and max_date:
        for date in pd.date_range(start=min_date, end=max_date, freq='W-SAT'):  # Saturday
            ax.axvspan(date, date + pd.Timedelta(days=2), color='#f5f5f5', alpha=0.5, zorder=0)
    
    # Add horizontal lines for risk thresholds with labels
    ax.axhline(y=0.3, color='#4daf4a', linestyle='-', linewidth=1, alpha=0.7, zorder=2)
    ax.axhline(y=0.6, color='#e41a1c', linestyle='-', linewidth=1, alpha=0.7, zorder=2)
    
    # Add threshold labels
    ax.text(0.01, 0.3, 'Low Risk Threshold', ha='left', va='bottom', 
            fontsize=8, color='#4daf4a', transform=ax.get_yaxis_transform(),
            bbox=dict(facecolor='white', alpha=0.8, pad=1, edgecolor='none'))
    
    ax.text(0.01, 0.6, 'High Risk Threshold', ha='left', va='bottom', 
            fontsize=8, color='#e41a1c', transform=ax.get_yaxis_transform(),
            bbox=dict(facecolor='white', alpha=0.8, pad=1, edgecolor='none'))
    
    # Add legend with better positioning and styling
    legend = ax.legend(
        loc='upper right',
        fontsize=10,
        frameon=True,
        framealpha=0.9,
        edgecolor='lightgrey',
        title="Authentication Methods"
    )
    legend.get_title().set_fontsize(11)
    legend.get_title().set_fontweight('bold')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"simulation_graphs/{output_file}", dpi=300, bbox_inches='tight')
    plt.savefig(f"simulation_graphs/{output_file.replace('.pdf', '.png')}", dpi=300, bbox_inches='tight')
    plt.close()

def plot_security_metrics(auth_decisions, output_file='security_metrics_evolution.pdf'):
    """Plot security-related metrics over time with publication-quality styling"""
    methods = list(auth_decisions.keys())
    
    # Focus on the most important metrics only
    metrics = ['false_positives', 'false_negatives']  # Security critical metrics
    
    # Create a single figure for both metrics, side by side
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    
    # Create a color palette
    colors = sns.color_palette("deep", len(methods))
    
    for i, metric in enumerate(metrics):
        for j, method in enumerate(methods):
            decisions = auth_decisions[method]
            if not decisions:
                continue
                
            # Group by week for better trend visibility
            from collections import defaultdict
            week_groups = defaultdict(list)
            
            for decision in decisions:
                timestamp = decision['timestamp']
                week = int(timestamp) // (86400 * 7) * (86400 * 7)  # Round to week
                
                # Count this metric
                is_counted = 1 if decision['legitimate'] == (metric == 'false_negatives') and \
                               decision['success'] != (metric == 'false_negatives') else 0
                
                week_groups[week].append(is_counted)
            
            # Process each week
            weeks = []
            rates = []
            
            for week, counts in sorted(week_groups.items()):
                weeks.append(pd.to_datetime(week, unit='s'))
                # Calculate rate
                rate = sum(counts) / len(counts)
                rates.append(rate * 100)  # Convert to percentage
            
            # Plot rate by week
            if weeks:
                axes[i].plot(weeks, rates, label=method, linewidth=1.5, color=colors[j])
        
        # Add labels
        metric_label = 'False Accept Rate' if metric == 'false_positives' else 'False Reject Rate'
        axes[i].set_title(metric_label, fontsize=10, fontweight='bold')
        axes[i].set_ylabel('Rate (%)', fontsize=9)
        
        if i == len(metrics) - 1:
            axes[i].set_xlabel('Date', fontsize=9)
        
        # Style the axis
        stylize_axes(axes[i])
        
        # Format x-axis
        from matplotlib.dates import AutoDateLocator, DateFormatter
        locator = AutoDateLocator(minticks=3, maxticks=5)
        formatter = DateFormatter('%m/%d')
        axes[i].xaxis.set_major_locator(locator)
        axes[i].xaxis.set_major_formatter(formatter)
        
        # Add subtle grid
        axes[i].grid(axis='y', linestyle=':', alpha=0.3, color='#cccccc')
        
        # Add legend to only one subplot
        if i == 0:
            axes[i].legend(loc='best', fontsize=7)
    
    # Add overall title
    fig.suptitle('Security Performance Over Time', fontsize=11, fontweight='bold')
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    plt.savefig(f"simulation_graphs/{output_file}", dpi=300, bbox_inches='tight')
    plt.savefig(f"simulation_graphs/{output_file.replace('.pdf', '.png')}", dpi=300, bbox_inches='tight')
    plt.close()

def plot_user_experience(auth_decisions, output_file='user_experience_metrics.pdf'):
    """Plot user experience metrics over time with publication-quality styling"""
    methods = list(auth_decisions.keys())
    
    # Create a single figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    
    # Create a color palette with better differentiation
    colors = sns.color_palette("deep", len(methods))
    
    # Plot 1: Authentication Success Rate for legitimate users
    for i, method in enumerate(methods):
        legitimate_decisions = [d for d in auth_decisions[method] if d.get('legitimate', True)]
        
        if not legitimate_decisions:
            continue
            
        # Group by week for clearer trends
        from collections import defaultdict
        week_groups = defaultdict(list)
        
        for decision in legitimate_decisions:
            timestamp = decision['timestamp']
            week = int(timestamp) // (86400 * 7) * (86400 * 7)  # Round to week
            week_groups[week].append(decision)
        
        # Process weekly data
        weeks = []
        success_rates = []
        
        for week, week_decisions in sorted(week_groups.items()):
            weeks.append(pd.to_datetime(week, unit='s'))
            success_rate = sum(1 for d in week_decisions if d['success']) / len(week_decisions)
            success_rates.append(success_rate * 100)  # Convert to percentage
        
        # Plot success rate
        if weeks:
            axes[0].plot(weeks, success_rates, label=method, linewidth=1.5, color=colors[i])
    
    # Style first plot
    axes[0].set_title('Authentication Success Rate', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Success Rate (%)', fontsize=9)
    axes[0].set_ylim(0, 100)
    
    # Plot 2: Average Auth Methods for legitimate users (simpler visualization)
    for i, method in enumerate(methods):
        legitimate_decisions = [d for d in auth_decisions[method] if d.get('legitimate', True)]
        
        if not legitimate_decisions:
            continue
            
        # Group by week
        week_groups = defaultdict(list)
        
        for decision in legitimate_decisions:
            timestamp = decision['timestamp']
            week = int(timestamp) // (86400 * 7) * (86400 * 7)
            week_groups[week].append(decision)
        
        # Process weekly data
        weeks = []
        avg_methods = []
        
        for week, week_decisions in sorted(week_groups.items()):
            weeks.append(pd.to_datetime(week, unit='s'))
            avg = sum(len(d['auth_methods']) for d in week_decisions) / len(week_decisions)
            avg_methods.append(avg)
        
        # Plot average methods
        if weeks:
            axes[1].plot(weeks, avg_methods, label=method, linewidth=1.5, color=colors[i])
    
    # Style both plots
    axes[1].set_title('Avg. Authentication Methods', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Number of Methods', fontsize=9)
    axes[1].set_ylim(0.9, 3.1)
    
    for i in range(2):
        # Style the axis
        stylize_axes(axes[i])
        
        # Format x-axis
        from matplotlib.dates import AutoDateLocator, DateFormatter
        locator = AutoDateLocator(minticks=3, maxticks=5)
        formatter = DateFormatter('%m/%d')
        axes[i].xaxis.set_major_locator(locator)
        axes[i].xaxis.set_major_formatter(formatter)
        axes[i].set_xlabel('Date', fontsize=9)
        
        # Add subtle grid
        axes[i].grid(axis='y', linestyle=':', alpha=0.3, color='#cccccc')
    
    # Add legend to the first subplot only
    axes[0].legend(loc='best', fontsize=7)
    
    # Add overall title
    fig.suptitle('User Experience Metrics', fontsize=11, fontweight='bold')
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    plt.savefig(f"simulation_graphs/{output_file}", dpi=300, bbox_inches='tight')
    plt.savefig(f"simulation_graphs/{output_file.replace('.pdf', '.png')}", dpi=300, bbox_inches='tight')
    plt.close()

def plot_environmental_adaptation(metrics, environments, output_file='environmental_adaptation.pdf'):
    """Plot performance in different environments with publication-quality styling"""
    methods = list(metrics.keys())
    
    # Filter environments - if too many, focus on the most distinctive ones
    if len(environments) > 5:
        environments = environments[:5]  # Take first 5
    
    # Prepare data
    env_data = []
    
    for method in methods:
        for env in environments:
            env_key = f'env_{env}'
            if env_key in metrics[method] and metrics[method][env_key]:
                success_rate = sum(metrics[method][env_key]) / len(metrics[method][env_key])
                env_data.append({
                    'Method': method,
                    'Environment': env.replace('_', ' ').title(),
                    'Success Rate': success_rate * 100  # Convert to percentage
                })
    
    if not env_data:
        print("No environment-specific data available for plotting")
        return
        
    df = pd.DataFrame(env_data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Create grouped bar chart with better spacing
    ax = sns.barplot(x='Environment', y='Success Rate', hue='Method', data=df)
    
    # Add labels with more appropriate sizing
    plt.title('Performance Across Environments', fontsize=11, fontweight='bold')
    plt.ylabel('Success Rate (%)', fontsize=10)
    plt.xlabel('Environment', fontsize=10)
    plt.ylim(0, 100)
    
    # Better legend positioning
    plt.legend(title='Method', frameon=False, fontsize=8, loc='best')
    
    # Style the axis
    stylize_axes(ax)
    
    # Handle x-tick rotations for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add data labels on bars for clarity
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=7, padding=3)
    
    # Add subtle grid
    ax.grid(axis='y', linestyle=':', alpha=0.3, color='#cccccc')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"simulation_graphs/{output_file}", dpi=300, bbox_inches='tight')
    plt.savefig(f"simulation_graphs/{output_file.replace('.pdf', '.png')}", dpi=300, bbox_inches='tight')
    plt.close()

def generate_latex_table(results, output_file='simulation_results/results_table.tex'):
    """Generate LaTeX table from simulation results for publication"""
    methods = list(results.keys())
    
    # Define the metrics we want to include in our table
    key_metrics = [
        'accuracy', 'precision', 'recall', 'f1_score', 'far', 'frr', 'eer', 
        'avg_auth_factors', 'avg_risk_score'
    ]
    
    # Create a DataFrame for easier manipulation
    table_data = []
    for method in methods:
        row = {'Method': method.title()}
        for metric in key_metrics:
            if metric in results[method]:
                row[metric] = results[method][metric]
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Format the column names for LaTeX
    column_names = {
        'Method': 'Method',
        'accuracy': 'Accuracy', 
        'precision': 'Precision', 
        'recall': 'Recall',
        'f1_score': 'F1 Score', 
        'far': 'FAR', 
        'frr': 'FRR', 
        'eer': 'EER',
        'avg_auth_factors': 'Avg. Auth Factors',
        'avg_risk_score': 'Avg. Risk Score'
    }
    
    df = df.rename(columns=column_names)
    
    # Generate LaTeX table
    latex_table = df.to_latex(index=False, float_format="%.3f")
    
    # Enhance the LaTeX table with better formatting
    latex_table = latex_table.replace('tabular', 'tabular*{\\textwidth}')
    latex_table = latex_table.replace('\\begin{tabular*}{\\textwidth}', '\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l' + 'c' * (len(df.columns) - 1) + '}')
    
    # Add table caption and label
    latex_header = "\\begin{table}[htbp]\n\\centering\n\\caption{Performance Comparison of Risk Assessment Methods}\n\\label{tab:performance_comparison}\n"
    latex_footer = "\\end{table}"
    
    latex_table = latex_header + latex_table + latex_footer
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to {output_file}")
    return latex_table

def generate_comparison_table(results, baseline_method='fixed', output_file='simulation_results/comparison_table.tex'):
    """Generate LaTeX table comparing methods against a baseline (typically fixed weights)"""
    methods = list(results.keys())
    
    # Define the metrics we want to include in our comparison
    key_metrics = [
        'accuracy', 'precision', 'recall', 'f1_score', 'far', 'frr', 'eer', 
        'avg_auth_factors'
    ]
    
    # Create a DataFrame for comparison
    table_data = []
    
    # Get baseline values
    baseline_values = {}
    if baseline_method in results:
        for metric in key_metrics:
            if metric in results[baseline_method]:
                baseline_values[metric] = results[baseline_method][metric]
    
    # Calculate percentage improvements
    for method in methods:
        if method == baseline_method:
            continue
            
        row = {'Method': method.title()}
        for metric in key_metrics:
            if metric in results[method] and metric in baseline_values and baseline_values[metric] != 0:
                # Calculate percentage improvement
                improvement = (results[method][metric] - baseline_values[metric]) / baseline_values[metric] * 100
                
                # Handle metrics where lower is better (FAR, FRR, EER)
                if metric in ['far', 'frr', 'eer', 'avg_auth_factors']:
                    improvement = -improvement
                    
                row[metric] = improvement
        table_data.append(row)
    
    if not table_data:
        print("No comparison data available")
        return None
    
    df = pd.DataFrame(table_data)
    
    # Format the column names for LaTeX
    column_names = {
        'Method': 'Method',
        'accuracy': 'Accuracy \\%', 
        'precision': 'Precision \\%', 
        'recall': 'Recall \\%',
        'f1_score': 'F1 Score \\%', 
        'far': 'FAR \\%', 
        'frr': 'FRR \\%', 
        'eer': 'EER \\%',
        'avg_auth_factors': 'Auth Factors \\%'
    }
    
    df = df.rename(columns=column_names)
    
    # Generate LaTeX table
    latex_table = df.to_latex(index=False, float_format="%.2f")
    
    # Add highlighting for positive values
    for metric in key_metrics:
        if metric in column_names:
            metric_latex = column_names[metric].replace('\\%', '\\\\%')
            latex_table = latex_table.replace(metric_latex, metric_latex + ' Improvement')
    
    # Enhance the LaTeX table with better formatting
    latex_table = latex_table.replace('tabular', 'tabular*{\\textwidth}')
    latex_table = latex_table.replace('\\begin{tabular*}{\\textwidth}', '\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l' + 'c' * (len(df.columns) - 1) + '}')
    
    # Add table caption and label
    latex_header = f"\\begin{{table}}[htbp]\n\\centering\n\\caption{{Percentage Improvement Compared to {baseline_method.title()} Method}}\n\\label{{tab:improvement_comparison}}\n"
    latex_footer = "\\end{table}"
    
    latex_table = latex_header + latex_table + latex_footer
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX comparison table saved to {output_file}")
    return latex_table

def plot_table(data, output_file='results_table.pdf'):
    """Create a visual table using matplotlib with cleaner presentation"""
    # Convert data dictionary to DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        # Process the data dictionary
        methods = list(data.keys())
        
        # Choose only the most important metrics for clarity
        key_metrics = ['accuracy', 'f1_score', 'far', 'frr', 'eer']
        
        # Create DataFrame
        table_data = []
        for method in methods:
            row = {'Method': method}
            for metric in key_metrics:
                if metric in data[method]:
                    row[metric] = data[method][metric]
                else:
                    row[metric] = None
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
    else:
        df = data.copy()
        # Filter to key metrics if DataFrame has too many columns
        if len(df.columns) > 6:  # Including 'Method'
            key_metrics = ['Method', 'accuracy', 'f1_score', 'far', 'frr', 'eer']
            available_cols = [col for col in key_metrics if col in df.columns]
            df = df[available_cols]
    
    # Rename columns for better display
    column_names = {
        'Method': 'Method',
        'accuracy': 'Accuracy', 
        'precision': 'Precision', 
        'recall': 'Recall',
        'f1_score': 'F1 Score', 
        'far': 'FAR', 
        'frr': 'FRR', 
        'eer': 'EER',
        'avg_auth_factors': 'Auth Factors',
        'avg_risk_score': 'Risk Score'
    }
    
    df = df.rename(columns={k: v for k, v in column_names.items() if k in df.columns})
    
    # Set Method as index if it exists
    if 'Method' in df.columns:
        df = df.set_index('Method')
    
    # Format numeric columns to 3 decimal places
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32]:
            df[col] = df[col].map('{:.3f}'.format)
    
    # Create figure and axis with appropriate size
    fig, ax = plt.subplots(figsize=(len(df.columns) + 1, len(df) * 0.5 + 1))
    
    # Hide axes
    ax.axis('tight')
    ax.axis('off')
    
    # Create table with minimal styling
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc='center',
        loc='center'
    )
    
    # Style the table more appropriately for academic publication
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.3)
    
    # Apply custom styling - simpler, more professional
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor('#e6e6e6')  # Light gray
        elif j == -1:  # Row labels
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f2f2f2')  # Lighter gray
        
        # Add subtle borders
        cell.set_edgecolor('#cccccc')
    
    # Add title
    plt.title('Performance Metrics', fontsize=11, fontweight='bold')
    
    # Save figure - use subplots_adjust instead of tight_layout
    plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.1)
    plt.savefig(f"simulation_graphs/{output_file}", dpi=300, bbox_inches='tight')
    plt.savefig(f"simulation_graphs/{output_file.replace('.pdf', '.png')}", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visual table saved to simulation_graphs/{output_file}")

def plot_heatmap_comparison(data, baseline_method='fixed', output_file='comparison_heatmap.pdf'):
    """Create a heatmap showing the percentage improvement against a baseline method"""
    # Convert data dictionary to DataFrame if needed
    if not isinstance(data, pd.DataFrame):
        methods = list(data.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'far', 'frr', 'eer', 'avg_auth_factors']
        
        # Get baseline values
        baseline = {}
        if baseline_method in data:
            for metric in metrics:
                if metric in data[baseline_method]:
                    baseline[metric] = data[baseline_method][metric]
        
        # Calculate improvements
        improvement_data = {}
        for method in methods:
            if method == baseline_method:
                continue
                
            improvement_data[method] = {}
            for metric in metrics:
                if metric in data[method] and metric in baseline and baseline[metric] != 0:
                    improvement = (data[method][metric] - baseline[metric]) / baseline[metric] * 100
                    
                    # For metrics where lower is better
                    if metric in ['far', 'frr', 'eer', 'avg_auth_factors']:
                        improvement = -improvement
                        
                    improvement_data[method][metric] = improvement
        
        # Convert to DataFrame
        df = pd.DataFrame(improvement_data).T
    else:
        df = data.copy()
    
    # Rename columns for better display
    column_names = {
        'accuracy': 'Accuracy', 
        'precision': 'Precision', 
        'recall': 'Recall',
        'f1_score': 'F1 Score', 
        'far': 'FAR', 
        'frr': 'FRR', 
        'eer': 'EER',
        'avg_auth_factors': 'Auth Factors'
    }
    
    df = df.rename(columns={k: v for k, v in column_names.items() if k in df.columns})
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, len(df) + 1))
    
    # Create a custom colormap - red to white to green
    cmap = sns.diverging_palette(10, 120, as_cmap=True)
    
    # Create heatmap
    ax = sns.heatmap(df, annot=True, cmap=cmap, center=0, fmt='.1f',
                linewidths=.5, cbar_kws={'label': 'Percentage Improvement (%)'},
                vmin=-50, vmax=50)
    
    # Add title and labels
    plt.title(f'Percentage Improvement Compared to {baseline_method.title()}', fontsize=14, fontweight='bold')
    plt.ylabel('Method')
    
    # Style the plot - remove tight_layout() which conflicts with colorbar
    # plt.tight_layout() - This line causes the error with colorbar layout
    
    # Use more compatible approach to adjust layout
    plt.subplots_adjust(bottom=0.15, left=0.15, top=0.9, right=0.95)
    
    # Save figure with bbox_inches='tight' to handle colorbar positioning
    plt.savefig(f"simulation_graphs/{output_file}", dpi=300, bbox_inches='tight')
    plt.savefig(f"simulation_graphs/{output_file.replace('.pdf', '.png')}", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison heatmap saved to simulation_graphs/{output_file}")

def plot_radar_chart(data, output_file='radar_chart.pdf'):
    """Create a radar chart comparing different methods across key metrics"""
    methods = list(data.keys())
    
    # Choose metrics for radar chart
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    available_metrics = [m for m in metrics if all(m in data[method] for method in methods)]
    
    if not available_metrics:
        print("Error: Not enough metrics available for radar chart")
        return
    
    # Extract data
    values = {}
    for method in methods:
        values[method] = [data[method][metric] for metric in available_metrics]
    
    # Create figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of metrics
    N = len(available_metrics)
    
    # Angle of each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create color palette
    colors = sns.color_palette("deep", len(methods))
    
    # Plot each method
    for i, method in enumerate(methods):
        values_method = values[method]
        values_method += values_method[:1]  # Close the loop
        ax.plot(angles, values_method, linewidth=2, linestyle='solid', label=method.title(), color=colors[i])
        ax.fill(angles, values_method, alpha=0.1, color=colors[i])
    
    # Add metrics labels
    nice_names = {
        'accuracy': 'Accuracy', 
        'precision': 'Precision', 
        'recall': 'Recall',
        'f1_score': 'F1 Score'
    }
    
    metric_labels = [nice_names.get(m, m) for m in available_metrics]
    plt.xticks(angles[:-1], metric_labels, fontsize=12, fontweight='bold')
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Add title
    plt.title('Performance Metrics Comparison', size=14, fontweight='bold', y=1.1)
    
    # Style the chart
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    # Use plt.subplots_adjust instead of tight_layout to avoid compatibility issues
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    plt.savefig(f"simulation_graphs/{output_file}", dpi=300, bbox_inches='tight')
    plt.savefig(f"simulation_graphs/{output_file.replace('.pdf', '.png')}", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Radar chart saved to simulation_graphs/{output_file}")

def generate_boxplots(raw_results, baseline_method='fixed', output_prefix='variability_'):
    """Generate publication-quality boxplots to visualize result variability across multiple runs
    
    Args:
        raw_results: Dictionary of raw results from multiple runs
        baseline_method: Method to use as baseline for comparisons
        output_prefix: Prefix for output filenames
    """
    # Set publication style
    set_publication_style()
    
    # Only include methods with data
    methods = [m for m in raw_results.keys() if raw_results[m]]
    
    if not methods:
        print("No data available for boxplots")
        return
    
    # Define important metrics to visualize with proper labels
    metric_labels = {
        'accuracy': 'Accuracy',
        'f1_score': 'F1 Score',
        'far': 'False Acceptance Rate',
        'frr': 'False Rejection Rate',
        'eer': 'Equal Error Rate',
        'avg_auth_factors': 'Average Auth Factors'
    }
    
    # Get available metrics that we have data for
    key_metrics = [m for m in metric_labels.keys() 
                  if all(m in raw_results[method] and raw_results[method][m] for method in methods)]
    
    # Determine number of runs for title
    num_runs = 0
    for method in methods:
        for metric in key_metrics:
            if raw_results[method][metric]:
                num_runs = len(raw_results[method][metric])
                break
        if num_runs > 0:
            break
    
    # Color mapping with professional palette
    method_colors = {}
    palette = sns.color_palette("deep", len(methods))
    for i, method in enumerate(methods):
        method_colors[method] = palette[i]
    
    # Create one boxplot per metric
    for metric in key_metrics:
        # Prepare data for boxplot
        data = []
        labels = []
        
        for method in methods:
            if metric in raw_results[method]:
                values = raw_results[method][metric]
                if values:  # Only include if we have values
                    data.append(values)
                    labels.append(method.title())
        
        if not data:
            continue
            
        # Create figure with dimensions optimized for boxplots
        fig, ax = plt.subplots(figsize=(len(methods)*1.2 + 2, 6))
        
        # Create boxplot with enhanced styling
        boxplot = ax.boxplot(
            data, 
            labels=labels,
            patch_artist=True,
            medianprops={'color': 'black', 'linewidth': 1.5},
            boxprops={'alpha': 0.8, 'linewidth': 1.0},
            whiskerprops={'linewidth': 1.2, 'color': '#333333'},
            capprops={'linewidth': 1.2, 'color': '#333333'},
            flierprops={'marker': 'o', 'markerfacecolor': 'white', 
                      'markeredgecolor': '#666666', 'markersize': 5, 'alpha': 0.7},
            showfliers=True,  # Show outliers
            showmeans=True,  # Show mean as triangle
            meanprops={'marker': '^', 'markerfacecolor': 'white', 
                     'markeredgecolor': 'black', 'markersize': 8}
        )
        
        # Customize boxplot colors
        for patch, method, color in zip(boxplot['boxes'], methods, palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add jittered points to show individual runs with enhanced styling
        for i, (method_data, method) in enumerate(zip(data, methods)):
            # More controlled jitter
            jitter_width = 0.08
            x = np.random.uniform(i+1-jitter_width, i+1+jitter_width, size=len(method_data))
            
            ax.scatter(
                x, method_data, 
                alpha=0.6, 
                s=30, 
                color=method_colors[method],
                edgecolor='white',
                linewidth=0.5,
                zorder=5
            )
        
        # Add title and labels with professional styling
        metric_name = metric_labels.get(metric, metric.replace('_', ' ').title())
        ax.set_title(f'Variability in {metric_name}\nAcross {num_runs} Simulation Runs', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_xlabel('Method', fontsize=12)
        
        # Add reference line for baseline method if available
        if baseline_method in raw_results and metric in raw_results[baseline_method]:
            baseline_values = raw_results[baseline_method][metric]
            if baseline_values:
                baseline_mean = np.mean(baseline_values)
                ax.axhline(
                    y=baseline_mean, 
                    color='#999999', 
                    linestyle='--', 
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=1
                )
                ax.text(
                    len(methods) + 0.3, 
                    baseline_mean,
                    f'{baseline_method.title()} mean',
                    va='center',
                    ha='left',
                    fontsize=9,
                    color='#666666',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1)
                )
        
        # Style the plot
        stylize_axes(ax, with_grid=True)
        
        # Add subtle background coloring to enhance readability
        ax.set_facecolor('#fcfcfc')
        
        # Add subtle grid for easier value reading
        ax.grid(axis='y', linestyle=':', alpha=0.3, color='#cccccc')
        
        # Add statistical annotations if we have enough runs
        if num_runs >= 5:
            # Perform t-tests against baseline
            baseline_idx = None
            for i, method in enumerate(methods):
                if method == baseline_method:
                    baseline_idx = i
                    break
            
            if baseline_idx is not None:
                baseline_data = data[baseline_idx]
                
                for i, (method_data, method) in enumerate(zip(data, methods)):
                    if method == baseline_method:
                        continue
                        
                    # Perform t-test
                    try:
                        from scipy import stats
                        t_stat, p_value = stats.ttest_ind(method_data, baseline_data)
                        
                        # Add significance stars
                        if p_value < 0.001:
                            significance = '***'
                        elif p_value < 0.01:
                            significance = '**'
                        elif p_value < 0.05:
                            significance = '*'
                        else:
                            significance = 'ns'
                            
                        # Add annotation for significant results
                        if p_value < 0.05:
                            y_pos = np.max(method_data) + (np.max(data) - np.min(data)) * 0.05
                            ax.text(
                                i+1, y_pos,
                                significance,
                                ha='center',
                                va='bottom',
                                fontsize=12,
                                color='#333333',
                                fontweight='bold'
                            )
                    except:
                        pass  # Skip if t-test fails
        
        # Add legend for significance if applicable
        if num_runs >= 5:
            ax.text(
                0.98, 0.02,
                "* p<0.05, ** p<0.01, *** p<0.001",
                transform=ax.transAxes,
                ha='right',
                va='bottom',
                fontsize=9,
                color='#666666',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2)
            )
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"simulation_graphs/{output_prefix}{metric}_boxplot.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"simulation_graphs/{output_prefix}{metric}_boxplot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Publication-quality boxplot for {metric} saved to simulation_graphs/{output_prefix}{metric}_boxplot.pdf")
    
    # Create a combined metrics figure for key performance indicators
    create_combined_variability_plot(raw_results, methods, output_prefix, metric_labels)

def create_combined_variability_plot(raw_results, methods, output_prefix, metric_labels=None):
    """Create a professional, publication-ready combined figure showing variability for multiple metrics
    
    Args:
        raw_results: Dictionary of raw results from multiple runs
        methods: List of methods to include
        output_prefix: Prefix for output filenames
        metric_labels: Dictionary mapping metric names to display labels
    """
    # Set default metric labels if not provided
    if metric_labels is None:
        metric_labels = {
            'accuracy': 'Accuracy',
            'f1_score': 'F1 Score',
            'far': 'False Acceptance Rate',
            'frr': 'False Rejection Rate',
            'eer': 'Equal Error Rate'
        }
    
    # Focus on key performance indicators - prefer the most important metrics
    preferred_metrics = ['accuracy', 'f1_score', 'far']
    
    # Find which preferred metrics are available
    available_metrics = []
    for metric in preferred_metrics:
        if all(metric in raw_results[method] and raw_results[method][metric] for method in methods):
            available_metrics.append(metric)
    
    # If we don't have our preferred metrics, use what's available
    if not available_metrics:
        available_metrics = []
        for metric in metric_labels.keys():
            if all(metric in raw_results[method] and raw_results[method][metric] for method in methods):
                available_metrics.append(metric)
                if len(available_metrics) >= 3:
                    break  # Limit to 3 metrics for readability
    
    # Skip if data is not available
    if not available_metrics:
        print("No metrics available for combined variability plot")
        return
    
    # Count number of runs
    num_runs = len(raw_results[methods[0]][available_metrics[0]])
    
    # Create figure with subplots - use golden ratio for aesthetics
    fig_width = 12
    fig_height = fig_width / 1.618 * len(available_metrics) / 3  # Adjust height based on golden ratio
    
    fig, axes = plt.subplots(1, len(available_metrics), figsize=(fig_width, fig_height), sharey=False)
    
    # If single metric, make axes iterable
    if len(available_metrics) == 1:
        axes = [axes]
    
    # Create color palette
    method_colors = {}
    palette = sns.color_palette("deep", len(methods))
    for i, method in enumerate(methods):
        method_colors[method] = palette[i]
    
    # For each metric
    for i, metric in enumerate(available_metrics):
        # Prepare data
        plot_data = []
        plot_labels = []
        
        for method in methods:
            if metric in raw_results[method]:
                values = raw_results[method][metric]
                if values:
                    plot_data.append(values)
                    plot_labels.append(method)
        
        if not plot_data:
            continue
        
        # Get nice label for this metric
        metric_name = metric_labels.get(metric, metric.replace('_', ' ').title())
        
        # Create boxplot with enhanced styling
        boxplot = axes[i].boxplot(
            plot_data, 
            labels=[m.title() for m in plot_labels],
            patch_artist=True,
            medianprops={'color': 'black', 'linewidth': 1.5},
            boxprops={'alpha': 0.8, 'linewidth': 1.0},
            whiskerprops={'linewidth': 1.0, 'color': '#333333'},
            capprops={'linewidth': 1.0, 'color': '#333333'},
            flierprops={'marker': 'o', 'markerfacecolor': 'white', 
                      'markeredgecolor': '#666666', 'markersize': 4, 'alpha': 0.7},
            showfliers=True,  # Show outliers
            showmeans=True,  # Show mean as triangle
            meanprops={'marker': '^', 'markerfacecolor': 'white', 
                     'markeredgecolor': 'black', 'markersize': 7}
        )
        
        # Customize boxplot colors
        for patch, method, color in zip(boxplot['boxes'], plot_labels, palette[:len(plot_labels)]):
            patch.set_facecolor(method_colors[method])
            patch.set_alpha(0.7)
        
        # Add jittered points with better styling
        for j, (method_data, method) in enumerate(zip(plot_data, plot_labels)):
            # More controlled jitter
            jitter_width = 0.08
            x = np.random.uniform(j+1-jitter_width, j+1+jitter_width, size=len(method_data))
            
            axes[i].scatter(
                x, method_data, 
                alpha=0.6, 
                s=25,  # Smaller points for less crowding
                color=method_colors[method],
                edgecolor='white',
                linewidth=0.5,
                zorder=5
            )
        
        # Add title and labels with professional styling
        axes[i].set_title(metric_name, fontsize=13, fontweight='bold')
        
        # Only add y-label for the first subplot
        if i == 0:
            axes[i].set_ylabel('Value', fontsize=12)
        
        # Style the plot
        stylize_axes(axes[i], with_grid=True)
        
        # Add subtle grid for easier value reading
        axes[i].grid(axis='y', linestyle=':', alpha=0.3, color='#cccccc')
        
        # Handle axis limits for better visualization
        if metric in ['accuracy', 'f1_score', 'precision', 'recall']:
            # Metrics that range from 0 to 1
            axes[i].set_ylim(max(0, min(plot_data[0]) - 0.1), min(1, max(plot_data[0]) + 0.1))
        elif metric in ['far', 'frr', 'eer']:
            # Error rates - typically small values
            all_values = [val for sublist in plot_data for val in sublist]
            y_min = max(0, min(all_values) - 0.05)
            y_max = min(1, max(all_values) + 0.05)
            axes[i].set_ylim(y_min, y_max)
        
        # Rotate x labels for better readability
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
        
        # Add statistical annotations if we have enough runs
        if num_runs >= 5 and len(methods) > 1:
            # Find baseline method if it's in our data
            baseline_idx = None
            for j, method in enumerate(plot_labels):
                if method == 'fixed':  # Default baseline
                    baseline_idx = j
                    break
            
            if baseline_idx is not None:
                baseline_data = plot_data[baseline_idx]
                
                for j, (method_data, method) in enumerate(zip(plot_data, plot_labels)):
                    if j == baseline_idx:
                        continue
                        
                    # Perform t-test
                    try:
                        from scipy import stats
                        t_stat, p_value = stats.ttest_ind(method_data, baseline_data)
                        
                        # Add significance stars
                        if p_value < 0.001:
                            significance = '***'
                        elif p_value < 0.01:
                            significance = '**'
                        elif p_value < 0.05:
                            significance = '*'
                        else:
                            continue  # Don't annotate non-significant results
                            
                        # Add annotation
                        y_pos = np.max(method_data) + (axes[i].get_ylim()[1] - axes[i].get_ylim()[0]) * 0.05
                        axes[i].text(
                            j+1, y_pos,
                            significance,
                            ha='center',
                            va='bottom',
                            fontsize=10,
                            color='#333333',
                            fontweight='bold'
                        )
                    except:
                        pass  # Skip if t-test fails
    
    # Add common title
    fig.suptitle(f'Performance Variability Across {num_runs} Simulation Runs', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Add legend for significance
    if num_runs >= 5:
        fig.text(
            0.5, 0.01,
            "Significance vs. fixed: * p<0.05, ** p<0.01, *** p<0.001",
            ha='center',
            va='bottom',
            fontsize=10,
            color='#666666'
        )
    
    # Save figure
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])  # Make room for suptitle and footnote
    plt.savefig(f"simulation_graphs/{output_prefix}combined_metrics.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"simulation_graphs/{output_prefix}combined_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined variability plot saved to simulation_graphs/{output_prefix}combined_metrics.pdf")

def generate_environmental_adaptation_latex(metrics, environments, output_file='simulation_results/environmental_adaptation_table.tex'):
    """Generate a LaTeX table showing performance across different environments
    
    Args:
        metrics: Dictionary containing metrics data for each method and environment
        environments: List of environment names
        output_file: Output file path for the LaTeX table
    """
    methods = list(metrics.keys())
    
    # Filter environments - if too many, focus on the most distinctive ones
    if len(environments) > 8:
        environments = environments[:8]  # Take first 8
    
    # Prepare data
    env_data = []
    
    for method in methods:
        row_data = {'Method': method.title()}
        for env in environments:
            env_key = f'env_{env}'
            if env_key in metrics[method] and metrics[method][env_key]:
                success_rate = sum(metrics[method][env_key]) / len(metrics[method][env_key])
                row_data[env] = success_rate * 100  # Convert to percentage
            else:
                row_data[env] = None
        env_data.append(row_data)
    
    if not env_data:
        print("No environment-specific data available for LaTeX table")
        return None
        
    df = pd.DataFrame(env_data)
    
    # Format environment names for cleaner display
    env_column_names = {}
    for env in environments:
        formatted_env = env.replace('_', ' ').title()
        env_column_names[env] = formatted_env
    
    df = df.rename(columns=env_column_names)
    
    # Generate LaTeX table
    latex_table = df.to_latex(index=False, float_format="%.2f", na_rep="--")
    
    # Enhance the LaTeX table with better formatting
    latex_table = latex_table.replace('tabular', 'tabular*{\\textwidth}')
    latex_table = latex_table.replace('\\begin{tabular*}{\\textwidth}', '\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l' + 'c' * (len(df.columns) - 1) + '}')
    
    # Add table caption and label
    latex_header = "\\begin{table}[htbp]\n\\centering\n\\caption{Performance Across Different Environmental Conditions (\\%)}\n\\label{tab:environmental_adaptation}\n"
    latex_footer = "\\begin{tablenotes}\n\\small\n\\item Note: Values represent correct decision percentage for each authentication method across environments. Higher is better.\n\\end{tablenotes}\n\\end{table}"
    
    latex_table = latex_header + latex_table + latex_footer
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX environmental adaptation table saved to {output_file}")
    return latex_table

def generate_all_visualizations(results, auth_decisions, weight_history, factor_names, metrics, environments, num_runs=1, seed=42):
    """Generate all visualizations with publication-quality styling
    
    Args:
        results: Dictionary of performance metrics for each method
        auth_decisions: Authentication decisions data
        weight_history: Weight history data for each factor
        factor_names: Names of risk factors
        metrics: Additional metrics data
        environments: Environment data for plotting
        num_runs: Number of simulation runs (for plot titles and reproducibility info)
        seed: Random seed used (for reproducibility info)
    """
    # Set publication style
    set_publication_style()
    
    # Convert results to DataFrame for plotting
    methods = list(results.keys())
    metrics_list = [col for col in list(results[methods[0]].keys()) 
                   if not col.endswith('_std') and 
                      not col.endswith('_ci_lower') and
                      not col.endswith('_ci_upper') and
                      not col.endswith('_p_value') and
                      not col.endswith('_significant') and
                      not col.endswith('_effect_size')]
    
    # Create a dataframe for the results
    data = []
    for method in methods:
        row = [method]
        for metric in metrics_list:
            row.append(results[method][metric])
        data.append(row)
        
    columns = ['Method'] + metrics_list
    results_df = pd.DataFrame(data, columns=columns)
    
    # Generate LaTeX tables
    print("Generating LaTeX tables...")
    generate_latex_table(results)
    generate_comparison_table(results)
    
    # Generate environmental adaptation LaTeX table
    print("Generating environmental adaptation LaTeX table...")
    generate_environmental_adaptation_latex(metrics, environments)
    
    # Generate table visualizations
    print("Generating visual result tables...")
    plot_table(results_df)
    plot_heatmap_comparison(results)
    plot_radar_chart(results)
    
    # Generate all plots
    print("Generating performance comparison chart...")
    plot_performance_comparison(results_df)
    
    print("Generating risk factor weights evolution chart...")
    plot_risk_factor_weights(weight_history, factor_names)
    
    print("Generating authentication methods chart...")
    plot_authentication_methods(auth_decisions)
    
    print("Generating risk scores evolution chart...")
    plot_risk_scores_over_time(auth_decisions)
    
    print("Generating security metrics evolution chart...")
    plot_security_metrics(auth_decisions)
    
    print("Generating user experience metrics chart...")
    plot_user_experience(auth_decisions)
    
    print("Generating environmental adaptation chart...")
    plot_environmental_adaptation(metrics, environments)
    
    # Add reproducibility info to the outputs
    if num_runs > 1:
        print(f"All visualizations generated successfully from {num_runs} simulation runs (seed: {seed})")
    else:
        print(f"All visualizations generated successfully (seed: {seed})")
    
    print(f"All visualizations saved in simulation_graphs/")
    print(f"LaTeX tables generated in simulation_results/") 
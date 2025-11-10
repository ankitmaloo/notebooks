#!/usr/bin/env python3
"""
Compare baseline vs post-training results for Password Game evaluation.

Usage:
    python compare_results.py --baseline baseline_export.json --trained trained_export.json

Or import and use programmatically:
    from compare_results import compare_evaluations
    results = compare_evaluations(baseline_path, trained_path)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


def load_evaluation(path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def compute_improvements(baseline: Dict, trained: Dict) -> Dict[str, Any]:
    """Compute improvement metrics between baseline and trained models."""
    b_metrics = baseline['metrics']
    t_metrics = trained['metrics']

    improvements = {
        'success_rate': {
            'baseline': b_metrics['success_rate'],
            'trained': t_metrics['success_rate'],
            'absolute_improvement': t_metrics['success_rate'] - b_metrics['success_rate'],
            'relative_improvement': ((t_metrics['success_rate'] - b_metrics['success_rate']) /
                                    max(b_metrics['success_rate'], 0.001)) * 100
        },
        'avg_rules_satisfied': {
            'baseline': b_metrics['avg_rules_satisfied'],
            'trained': t_metrics['avg_rules_satisfied'],
            'absolute_improvement': t_metrics['avg_rules_satisfied'] - b_metrics['avg_rules_satisfied'],
            'relative_improvement': ((t_metrics['avg_rules_satisfied'] - b_metrics['avg_rules_satisfied']) /
                                    max(b_metrics['avg_rules_satisfied'], 0.001)) * 100
        },
        'avg_final_reward': {
            'baseline': b_metrics['avg_final_reward'],
            'trained': t_metrics['avg_final_reward'],
            'absolute_improvement': t_metrics['avg_final_reward'] - b_metrics['avg_final_reward'],
            'relative_improvement': ((t_metrics['avg_final_reward'] - b_metrics['avg_final_reward']) /
                                    max(abs(b_metrics['avg_final_reward']), 0.001)) * 100
        },
        'avg_password_length': {
            'baseline': b_metrics['avg_password_length'],
            'trained': t_metrics['avg_password_length'],
            'absolute_improvement': t_metrics['avg_password_length'] - b_metrics['avg_password_length'],
            'relative_improvement': ((t_metrics['avg_password_length'] - b_metrics['avg_password_length']) /
                                    max(b_metrics['avg_password_length'], 0.001)) * 100
        },
        'avg_steps_taken': {
            'baseline': b_metrics['avg_steps_taken'],
            'trained': t_metrics['avg_steps_taken'],
            'absolute_improvement': t_metrics['avg_steps_taken'] - b_metrics['avg_steps_taken'],
            'relative_improvement': ((t_metrics['avg_steps_taken'] - b_metrics['avg_steps_taken']) /
                                    max(b_metrics['avg_steps_taken'], 0.001)) * 100
        }
    }

    return improvements


def compare_rule_performance(baseline: Dict, trained: Dict) -> Dict[int, Dict[str, float]]:
    """Compare per-rule performance between baseline and trained models."""
    b_rules = baseline.get('rule_performance', {})
    t_rules = trained.get('rule_performance', {})

    # Convert string keys to int if needed
    b_rules = {int(k): v for k, v in b_rules.items()}
    t_rules = {int(k): v for k, v in t_rules.items()}

    all_rules = set(b_rules.keys()) | set(t_rules.keys())

    rule_comparison = {}
    for rule_idx in sorted(all_rules):
        b_rate = b_rules.get(rule_idx, 0.0)
        t_rate = t_rules.get(rule_idx, 0.0)

        rule_comparison[rule_idx] = {
            'baseline': b_rate,
            'trained': t_rate,
            'improvement': t_rate - b_rate
        }

    return rule_comparison


def print_comparison_report(improvements: Dict, rule_comparison: Dict):
    """Print formatted comparison report."""
    print("\n" + "="*80)
    print("BASELINE vs TRAINED MODEL COMPARISON")
    print("="*80)

    print("\n📊 OVERALL METRICS:\n")

    for metric_name, data in improvements.items():
        print(f"{metric_name.replace('_', ' ').title()}:")
        print(f"  Baseline:  {data['baseline']:.4f}")
        print(f"  Trained:   {data['trained']:.4f}")
        print(f"  Absolute:  {data['absolute_improvement']:+.4f}")
        print(f"  Relative:  {data['relative_improvement']:+.2f}%")
        print()

    print("\n🎯 RULE-LEVEL PERFORMANCE:\n")

    # Find most improved and most declined rules
    improvements_list = [(idx, data['improvement']) for idx, data in rule_comparison.items()]
    improvements_list.sort(key=lambda x: x[1], reverse=True)

    print("Top 5 Most Improved Rules:")
    for idx, improvement in improvements_list[:5]:
        data = rule_comparison[idx]
        print(f"  Rule {idx}: {data['baseline']*100:.1f}% → {data['trained']*100:.1f}% "
              f"({improvement*100:+.1f}%)")

    print("\nTop 5 Most Declined Rules:")
    for idx, improvement in improvements_list[-5:]:
        data = rule_comparison[idx]
        print(f"  Rule {idx}: {data['baseline']*100:.1f}% → {data['trained']*100:.1f}% "
              f"({improvement*100:+.1f}%)")

    print("\n" + "="*80)


def visualize_comparison(improvements: Dict, rule_comparison: Dict, output_dir: str = "."):
    """Create comparison visualizations."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Overall Metrics Comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Baseline vs Trained Model Comparison', fontsize=16, fontweight='bold')

    metrics_to_plot = [
        ('success_rate', 'Success Rate', '%'),
        ('avg_rules_satisfied', 'Avg Rules Satisfied', 'rules'),
        ('avg_final_reward', 'Avg Final Reward', 'reward'),
        ('avg_password_length', 'Avg Password Length', 'chars'),
        ('avg_steps_taken', 'Avg Steps Taken', 'steps')
    ]

    for idx, (metric_key, title, unit) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        data = improvements[metric_key]

        baseline_val = data['baseline']
        trained_val = data['trained']

        # Convert to percentage if needed
        if unit == '%':
            baseline_val *= 100
            trained_val *= 100

        bars = ax.bar(['Baseline', 'Trained'], [baseline_val, trained_val],
                     color=['#3498db', '#2ecc71'], alpha=0.7, edgecolor='black')

        ax.set_ylabel(f'{title} ({unit})')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom')

        # Add improvement text
        improvement = data['absolute_improvement']
        if unit == '%':
            improvement *= 100
        color = 'green' if improvement > 0 else 'red'
        ax.text(0.5, 0.95, f'Δ {improvement:+.2f} {unit}',
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    # Hide last subplot if odd number of metrics
    if len(metrics_to_plot) < 6:
        axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_overall.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/comparison_overall.png")

    # 2. Rule-Level Comparison
    fig, ax = plt.subplots(figsize=(16, 8))

    rule_indices = sorted(rule_comparison.keys())
    baseline_rates = [rule_comparison[i]['baseline'] * 100 for i in rule_indices]
    trained_rates = [rule_comparison[i]['trained'] * 100 for i in rule_indices]

    x = np.arange(len(rule_indices))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_rates, width, label='Baseline',
                   color='#3498db', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, trained_rates, width, label='Trained',
                   color='#2ecc71', alpha=0.7, edgecolor='black')

    ax.set_xlabel('Rule Index', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Rule-Level Performance: Baseline vs Trained', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(rule_indices, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_rules.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/comparison_rules.png")

    # 3. Improvement Heatmap
    fig, ax = plt.subplots(figsize=(16, 6))

    improvements_by_rule = [rule_comparison[i]['improvement'] * 100 for i in rule_indices]
    colors = ['green' if imp > 0 else 'red' for imp in improvements_by_rule]

    bars = ax.bar(rule_indices, improvements_by_rule, color=colors, alpha=0.7, edgecolor='black')

    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('Rule Index', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('Per-Rule Improvement (Trained - Baseline)', fontsize=14, fontweight='bold')
    ax.set_xticks(rule_indices)
    ax.set_xticklabels(rule_indices, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_improvements.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/comparison_improvements.png")


def compare_evaluations(baseline_path: str, trained_path: str,
                       output_dir: str = "./comparison_results") -> Dict[str, Any]:
    """
    Main comparison function.

    Args:
        baseline_path: Path to baseline evaluation JSON
        trained_path: Path to trained model evaluation JSON
        output_dir: Directory to save comparison results

    Returns:
        Dictionary containing all comparison results
    """
    print("Loading evaluations...")
    baseline = load_evaluation(baseline_path)
    trained = load_evaluation(trained_path)

    print("Computing improvements...")
    improvements = compute_improvements(baseline, trained)

    print("Comparing rule performance...")
    rule_comparison = compare_rule_performance(baseline, trained)

    print_comparison_report(improvements, rule_comparison)

    print("\nGenerating visualizations...")
    visualize_comparison(improvements, rule_comparison, output_dir)

    # Save comparison results
    comparison_results = {
        'baseline_file': baseline_path,
        'trained_file': trained_path,
        'improvements': improvements,
        'rule_comparison': rule_comparison
    }

    output_path = f"{output_dir}/comparison_results.json"
    with open(output_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)

    print(f"\n✓ Comparison results saved to: {output_path}")

    return comparison_results


def main():
    parser = argparse.ArgumentParser(
        description='Compare baseline vs trained model evaluations for Password Game'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        required=True,
        help='Path to baseline evaluation JSON file'
    )
    parser.add_argument(
        '--trained',
        type=str,
        required=True,
        help='Path to trained model evaluation JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./comparison_results',
        help='Directory to save comparison results (default: ./comparison_results)'
    )

    args = parser.parse_args()

    compare_evaluations(args.baseline, args.trained, args.output_dir)

    print("\n✅ Comparison complete!")


if __name__ == '__main__':
    main()

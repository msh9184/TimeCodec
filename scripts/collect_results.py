#!/usr/bin/env python3
"""
TimeCodec Baseline Results Collector
=====================================
Collects, parses, and summarizes all baseline experiment results.

Usage:
    python scripts/collect_results.py [--results-dir results/] [--output results/summary/]

Output:
    results/summary/
    ├── baseline_summary.txt       # Human-readable full report
    ├── forecasting_results.csv    # Machine-readable forecasting metrics
    ├── anomaly_results.csv        # Machine-readable anomaly metrics
    ├── imputation_results.csv     # Machine-readable imputation metrics
    └── execution_log.txt          # Merged key log entries
"""

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from datetime import datetime


def parse_forecasting_txt(filepath):
    """Parse forecasting result .txt files.
    Format: | [Test] mse 0.3750 mae 0.3990 corr 0.8500
    """
    results = []
    basename = os.path.basename(filepath)
    # ETTh1_Tin96_Tout96_seed2021.txt
    match = re.match(r'(.+)_Tin(\d+)_Tout(\d+)_seed(\d+)\.txt', basename)
    if not match:
        return results

    dataset, tin, tout, seed = match.group(1), match.group(2), match.group(3), match.group(4)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the LAST [Test] line (best result before early stopping)
    test_lines = [l for l in lines if '[Test]' in l]
    if not test_lines:
        return results

    # Parse the last test line
    line = test_lines[-1]
    mse_m = re.search(r'mse\s+([\d.]+)', line)
    mae_m = re.search(r'mae\s+([\d.]+)', line)
    cor_m = re.search(r'corr\s+([-\d.]+)', line)

    if mse_m and mae_m:
        results.append({
            'dataset': dataset,
            'tin': int(tin),
            'tout': int(tout),
            'seed': int(seed),
            'mse': float(mse_m.group(1)),
            'mae': float(mae_m.group(1)),
            'corr': float(cor_m.group(1)) if cor_m else None,
        })
    return results


def parse_anomaly_log(filepath):
    """Parse anomaly detection log for metrics.
    Format: Accuracy : 0.9500, Precision : 0.8500, Recall : 0.9000, F-score : 0.8750
    """
    with open(filepath, 'r') as f:
        content = f.read()

    match = re.search(
        r'Accuracy\s*:\s*([\d.]+).*?Precision\s*:\s*([\d.]+).*?Recall\s*:\s*([\d.]+).*?F-score\s*:\s*([\d.]+)',
        content
    )
    if match:
        return {
            'accuracy': float(match.group(1)),
            'precision': float(match.group(2)),
            'recall': float(match.group(3)),
            'f_score': float(match.group(4)),
        }
    return None


def parse_imputation_log(filepath):
    """Parse imputation performance log.
    Format: MSE: 0.1234\nMAE: 0.2345
    """
    with open(filepath, 'r') as f:
        content = f.read()

    mse_m = re.search(r'MSE:\s*([\d.]+)', content)
    mae_m = re.search(r'MAE:\s*([\d.]+)', content)

    if mse_m and mae_m:
        return {
            'mse': float(mse_m.group(1)),
            'mae': float(mae_m.group(1)),
        }
    return None


def parse_master_log(filepath):
    """Extract key entries from master.log"""
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if any(kw in line for kw in ['START', 'DONE', 'FAIL', '===', 'Task:', 'ERROR']):
                entries.append(line)
    return entries


def collect_forecasting(results_dir):
    """Collect all forecasting results."""
    all_results = []

    # Search in forecasting/results/ and results/forecasting/
    patterns = [
        os.path.join(results_dir, 'forecasting', '*', '*.txt'),
        os.path.join('forecasting', 'results', '*', '*.txt'),
    ]

    for pattern in patterns:
        for filepath in glob.glob(pattern):
            results = parse_forecasting_txt(filepath)
            all_results.extend(results)

    # Also check pretrained results
    patterns_pre = [
        os.path.join('forecasting', 'results', '*_pretrained', '*.txt'),
    ]
    pretrained_results = []
    for pattern in patterns_pre:
        for filepath in glob.glob(pattern):
            results = parse_forecasting_txt(filepath)
            for r in results:
                r['source'] = 'pretrained'
            pretrained_results.extend(results)

    return all_results, pretrained_results


def collect_anomaly(results_dir):
    """Collect anomaly detection results from step4_detect.log files."""
    all_results = []

    # Search for step4_detect.log in results directories
    patterns = [
        os.path.join(results_dir, 'anomaly_detection', '*', '*', 'logs', 'step4_detect.log'),
    ]

    for pattern in patterns:
        for filepath in glob.glob(pattern):
            parts = filepath.split(os.sep)
            # Find dataset name from path
            idx = parts.index('anomaly_detection') if 'anomaly_detection' in parts else -1
            if idx >= 0 and idx + 1 < len(parts):
                dataset = parts[idx + 1]
            else:
                dataset = 'unknown'

            metrics = parse_anomaly_log(filepath)
            if metrics:
                metrics['dataset'] = dataset
                all_results.append(metrics)

    return all_results


def collect_imputation(results_dir):
    """Collect imputation results from step3_eval_mr*.log files."""
    all_results = []

    patterns = [
        os.path.join(results_dir, 'imputation', '*', '*', 'logs', 'step3_eval_mr*.log'),
    ]

    for pattern in patterns:
        for filepath in glob.glob(pattern):
            parts = filepath.split(os.sep)
            idx = parts.index('imputation') if 'imputation' in parts else -1
            if idx >= 0 and idx + 1 < len(parts):
                dataset = parts[idx + 1]
            else:
                dataset = 'unknown'

            basename = os.path.basename(filepath)
            mr_match = re.search(r'mr([\d.]+)', basename)
            mask_ratio = float(mr_match.group(1)) if mr_match else 0.0

            metrics = parse_imputation_log(filepath)
            if metrics:
                metrics['dataset'] = dataset
                metrics['mask_ratio'] = mask_ratio
                all_results.append(metrics)

    return all_results


def collect_logs(results_dir):
    """Collect and merge all master.log entries."""
    all_entries = []

    for master_log in glob.glob(os.path.join(results_dir, '*', '*', '*', 'logs', 'master.log')):
        parts = master_log.split(os.sep)
        task = parts[-4] if len(parts) > 4 else 'unknown'
        dataset = parts[-3] if len(parts) > 3 else 'unknown'

        all_entries.append(f"\n{'='*60}")
        all_entries.append(f"  {task} / {dataset}")
        all_entries.append(f"{'='*60}")
        all_entries.extend(parse_master_log(master_log))

    return all_entries


def generate_summary(fc_results, fc_pretrained, ad_results, imp_results, output_dir):
    """Generate the comprehensive summary report."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'baseline_summary.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("  TimeCodec Baseline Reproduction Summary\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        # ── FORECASTING ──
        f.write("=" * 80 + "\n")
        f.write("  FORECASTING (Specialist)\n")
        f.write("=" * 80 + "\n\n")

        if fc_results:
            # Group by dataset and tout, average over seeds
            grouped = defaultdict(list)
            for r in fc_results:
                key = (r['dataset'], r['tout'])
                grouped[key].append(r)

            # Header
            f.write(f"{'Dataset':<14} {'Horizon':<8} {'MSE':>8} {'MAE':>8} {'Corr':>8}  (avg over seeds)\n")
            f.write("-" * 60 + "\n")

            datasets_seen = set()
            for (dataset, tout) in sorted(grouped.keys()):
                entries = grouped[(dataset, tout)]
                avg_mse = sum(e['mse'] for e in entries) / len(entries)
                avg_mae = sum(e['mae'] for e in entries) / len(entries)
                corrs = [e['corr'] for e in entries if e['corr'] is not None]
                avg_corr = sum(corrs) / len(corrs) if corrs else float('nan')

                if dataset not in datasets_seen:
                    if datasets_seen:
                        f.write("\n")
                    datasets_seen.add(dataset)

                f.write(f"{dataset:<14} {tout:<8} {avg_mse:>8.4f} {avg_mae:>8.4f} {avg_corr:>8.4f}\n")

            # Per-seed detail
            f.write("\n\n--- Per-Seed Detail ---\n\n")
            f.write(f"{'Dataset':<14} {'Horizon':<8} {'Seed':<6} {'MSE':>8} {'MAE':>8} {'Corr':>8}\n")
            f.write("-" * 65 + "\n")
            for r in sorted(fc_results, key=lambda x: (x['dataset'], x['tout'], x['seed'])):
                corr_str = f"{r['corr']:>8.4f}" if r['corr'] is not None else "     N/A"
                f.write(f"{r['dataset']:<14} {r['tout']:<8} {r['seed']:<6} {r['mse']:>8.4f} {r['mae']:>8.4f} {corr_str}\n")
        else:
            f.write("  No forecasting results found.\n")

        # Pretrained comparison
        if fc_pretrained:
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("  FORECASTING (Pretrained Generalist Comparison)\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"{'Dataset':<14} {'Horizon':<8} {'Seed':<6} {'MSE':>8} {'MAE':>8} {'Corr':>8}\n")
            f.write("-" * 65 + "\n")
            for r in sorted(fc_pretrained, key=lambda x: (x['dataset'], x['tout'], x['seed'])):
                corr_str = f"{r['corr']:>8.4f}" if r['corr'] is not None else "     N/A"
                f.write(f"{r['dataset']:<14} {r['tout']:<8} {r['seed']:<6} {r['mse']:>8.4f} {r['mae']:>8.4f} {corr_str}\n")

        # ── ANOMALY DETECTION ──
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("  ANOMALY DETECTION\n")
        f.write("=" * 80 + "\n\n")

        if ad_results:
            f.write(f"{'Dataset':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F-score':>10}\n")
            f.write("-" * 55 + "\n")
            for r in sorted(ad_results, key=lambda x: x['dataset']):
                f.write(f"{r['dataset']:<10} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
                        f"{r['recall']:>10.4f} {r['f_score']:>10.4f}\n")
        else:
            f.write("  No anomaly detection results found.\n")

        # ── IMPUTATION ──
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("  IMPUTATION\n")
        f.write("=" * 80 + "\n\n")

        if imp_results:
            f.write(f"{'Dataset':<14} {'Mask Ratio':<12} {'MSE':>10} {'MAE':>10}\n")
            f.write("-" * 50 + "\n")

            prev_dataset = None
            for r in sorted(imp_results, key=lambda x: (x['dataset'], x['mask_ratio'])):
                if prev_dataset and r['dataset'] != prev_dataset:
                    f.write("\n")
                prev_dataset = r['dataset']
                f.write(f"{r['dataset']:<14} {r['mask_ratio']:<12.3f} {r['mse']:>10.6f} {r['mae']:>10.6f}\n")
        else:
            f.write("  No imputation results found.\n")

        # ── EXECUTION STATUS ──
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("  EXECUTION STATUS\n")
        f.write("=" * 80 + "\n\n")

        status_counts = {'DONE': 0, 'FAIL': 0}
        for master_log in glob.glob(os.path.join('results', '*', '*', '*', 'logs', 'master.log')):
            parts = master_log.split(os.sep)
            task = parts[1] if len(parts) > 1 else '?'
            dataset = parts[2] if len(parts) > 2 else '?'

            with open(master_log, 'r') as mf:
                content = mf.read()
                fails = content.count('FAIL')
                dones = content.count('DONE')
                status = "PASS" if fails == 0 and dones > 0 else f"FAIL({fails})"
                status_counts['FAIL' if fails > 0 else 'DONE'] += 1

            f.write(f"  {task:<22} {dataset:<14} {status}\n")

        f.write(f"\n  Total: {status_counts['DONE']} passed, {status_counts['FAIL']} failed\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"Summary report saved: {report_path}")
    return report_path


def save_csv(data, filepath, fieldnames):
    """Save results as CSV."""
    if not data:
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"CSV saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Collect TimeCodec baseline results')
    parser.add_argument('--results-dir', default='results', help='Results root directory')
    parser.add_argument('--output', default='results/summary', help='Output directory for summary')
    args = parser.parse_args()

    print("=" * 60)
    print("  TimeCodec Results Collector")
    print("=" * 60)

    # Collect results
    print("\nCollecting forecasting results...")
    fc_results, fc_pretrained = collect_forecasting(args.results_dir)
    print(f"  Found {len(fc_results)} scratch results, {len(fc_pretrained)} pretrained results")

    print("Collecting anomaly detection results...")
    ad_results = collect_anomaly(args.results_dir)
    print(f"  Found {len(ad_results)} results")

    print("Collecting imputation results...")
    imp_results = collect_imputation(args.results_dir)
    print(f"  Found {len(imp_results)} results")

    # Generate summary
    print("\nGenerating summary report...")
    report_path = generate_summary(fc_results, fc_pretrained, ad_results, imp_results, args.output)

    # Save CSVs
    if fc_results:
        save_csv(fc_results,
                 os.path.join(args.output, 'forecasting_results.csv'),
                 ['dataset', 'tin', 'tout', 'seed', 'mse', 'mae', 'corr'])
    if fc_pretrained:
        save_csv(fc_pretrained,
                 os.path.join(args.output, 'forecasting_pretrained.csv'),
                 ['dataset', 'tin', 'tout', 'seed', 'mse', 'mae', 'corr', 'source'])
    if ad_results:
        save_csv(ad_results,
                 os.path.join(args.output, 'anomaly_results.csv'),
                 ['dataset', 'accuracy', 'precision', 'recall', 'f_score'])
    if imp_results:
        save_csv(imp_results,
                 os.path.join(args.output, 'imputation_results.csv'),
                 ['dataset', 'mask_ratio', 'mse', 'mae'])

    # Merge logs
    print("Merging execution logs...")
    log_entries = collect_logs(args.results_dir)
    log_path = os.path.join(args.output, 'execution_log.txt')
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_entries))
    print(f"Merged log saved: {log_path}")

    # Print summary to terminal
    print("\n")
    with open(report_path, 'r') as f:
        print(f.read())


if __name__ == '__main__':
    main()

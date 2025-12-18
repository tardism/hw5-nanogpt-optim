import argparse, os, glob, json
import math
import numpy as np
import matplotlib.pyplot as plt
import csv

def read_metrics_csv(path):
    rows = []
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                'step': int(row['step']),
                'train_loss': float(row['train_loss']),
                'val_loss': float(row['val_loss']),
                'val_ppl': float(row['val_ppl']),
                'lr': float(row['lr']),
            })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs_dir', type=str, default='runs')
    ap.add_argument('--out_dir', type=str, default='report/figures')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    run_dirs = sorted([d for d in glob.glob(os.path.join(args.runs_dir, '*')) if os.path.isdir(d)])

    series = []
    final_rows = []
    for d in run_dirs:
        mpath = os.path.join(d, 'metrics.csv')
        fpath = os.path.join(d, 'final.json')
        cpath = os.path.join(d, 'config.json')
        if not os.path.exists(mpath) or not os.path.exists(fpath) or not os.path.exists(cpath):
            continue
        with open(cpath) as f:
            cfg = json.load(f)
        metrics = read_metrics_csv(mpath)
        with open(fpath) as f:
            fin = json.load(f)
        label = os.path.basename(d)
        series.append((label, metrics))
        final_rows.append({
            'run': label,
            'optimizer': cfg.get('optimizer'),
            'lr': cfg.get('lr'),
            'final_val_loss': fin['val_loss'],
            'final_val_ppl': fin['val_ppl'],
        })

    # Plot loss curves
    plt.figure()
    for label, m in series:
        xs = [r['step'] for r in m]
        ys = [r['val_loss'] for r in m]
        plt.plot(xs, ys, label=label)
    plt.xlabel('step')
    plt.ylabel('validation loss')
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'val_loss_curves.png'), dpi=200)

    plt.figure()
    for label, m in series:
        xs = [r['step'] for r in m]
        ys = [r['train_loss'] for r in m]
        plt.plot(xs, ys, label=label)
    plt.xlabel('step')
    plt.ylabel('training loss')
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'train_loss_curves.png'), dpi=200)

    plt.figure()
    for label, m in series:
        xs = [r['step'] for r in m]
        ys = [r['val_ppl'] for r in m]
        plt.plot(xs, ys, label=label)
    plt.xlabel('step')
    plt.ylabel('validation perplexity (exp(val_loss))')
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'val_ppl_curves.png'), dpi=200)

    # Write final table
    out_csv = os.path.join(args.out_dir, 'final_table.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['run','optimizer','lr','final_val_loss','final_val_ppl'])
        w.writeheader()
        for r in sorted(final_rows, key=lambda x: (str(x['optimizer']), float(x['lr']))):
            w.writerow(r)

    print(f"Wrote figures to {args.out_dir}")
    print(f"Wrote table to {out_csv}")

if __name__ == '__main__':
    main()

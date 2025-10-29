
#!/usr/bin/env python3
"""ml_utils CLI - lightweight ML utilities for datasets and quick benchmarks."""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

def preview(args):
    df = pd.read_csv(args.file)
    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    print("\nSummary:")
    print(df.describe(include='all').to_string())

def split(args):
    df = pd.read_csv(args.file)
    if args.target not in df.columns:
        print(f"Target column '{args.target}' not found in dataset", file=sys.stderr)
        sys.exit(2)
    X = df.drop(columns=[args.target])
    y = df[args.target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y if args.stratify and args._has_classification() else None
    ) if (args.stratify and args._has_classification()) or not args.stratify else train_test_split(X, y, test_size=args.test_size, random_state=args.seed)
    # simpler: do not overcomplicate stratify handling
    # Rebuild train/test DataFrames
    train = X_train.copy()
    train[args.target] = y_train.values
    test = X_test.copy()
    test[args.target] = y_test.values
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_path = outdir / 'train.csv'
    test_path = outdir / 'test.csv'
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"Wrote: {train_path} ({train.shape})\n       {test_path} ({test.shape})")

def benchmark(args):
    df = pd.read_csv(args.file)
    if args.target not in df.columns:
        print(f"Target column '{args.target}' not found in dataset", file=sys.stderr)
        sys.exit(2)
    X = df.drop(columns=[args.target])
    y = df[args.target]
    # simple numeric encoding for non-numeric columns
    for c in X.select_dtypes(include=['object', 'category']).columns:
        X[c] = X[c].astype('category').cat.codes
    # fill na
    X = X.fillna(-999)
    results = []
    models = [
        ('Dummy', DummyClassifier(strategy='most_frequent')),
        ('LogisticRegression', LogisticRegression(max_iter=1000)),
        ('RandomForest', RandomForestClassifier(n_estimators=50))
    ]
    for name, model in models:
        start = time.time()
        try:
            scores = cross_val_score(model, X, y, cv=args.cv, scoring='accuracy')
            duration = time.time() - start
            results.append((name, float(scores.mean()), float(scores.std()), duration))
        except Exception as e:
            results.append((name, None, None, None))
    print(tabulate(results, headers=['model','acc_mean','acc_std','duration_s'], floatfmt='.4f'))

def main():
    parser = argparse.ArgumentParser(prog='ml-utils')
    sub = parser.add_subparsers(dest='cmd', required=True)
    p_preview = sub.add_parser('preview', help='Preview dataset')
    p_preview.add_argument('file', help='CSV file path')
    p_preview.set_defaults(func=preview)

    p_split = sub.add_parser('split', help='Create train/test split')
    p_split.add_argument('file', help='CSV file path')
    p_split.add_argument('--target', required=True, help='Target column name')
    p_split.add_argument('--test-size', type=float, default=0.2)
    p_split.add_argument('--seed', type=int, default=42)
    p_split.add_argument('--out-dir', default='./splits')
    p_split.add_argument('--stratify', action='store_true', help='Stratify split when possible')
    p_split.set_defaults(func=split)

    p_bench = sub.add_parser('benchmark', help='Quick benchmark on dataset')
    p_bench.add_argument('file', help='CSV file path')
    p_bench.add_argument('--target', required=True, help='Target column name')
    p_bench.add_argument('--cv', type=int, default=3)
    p_bench.set_defaults(func=benchmark)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()

# ML Utilities

Lightweight AI / Machine Learning Utilities CLI & Python package.

**Purpose**: provide small, practical utilities useful for prototyping and demos:
- preview datasets
- create reproducible train/test splits
- run a quick model benchmark (scikit-learn)
- export basic EDA summary

This repo is intended to be published to GitHub Sponsors as a helpful open-source starter utility.

## Features

- `ml-utils preview <file>`: show first rows and basic summary.
- `ml-utils split <file> --test-size 0.2 --out-dir ./splits`: create train.csv and test.csv with a reproducible random seed.
- `ml-utils benchmark <file> --target target_col`: run quick benchmarks with LogisticRegression, RandomForest, and a baseline, reporting accuracy and runtime.

## Quickstart

1. Create a virtualenv and install:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Preview:
```bash
ml-utils preview examples/sample.csv
```

3. Create splits:
```bash
ml-utils split examples/sample.csv --target target --test-size 0.25 --seed 42 --out-dir ./splits
```

4. Benchmark:
```bash
ml-utils benchmark examples/sample.csv --target target --cv 3
```

## License

MIT

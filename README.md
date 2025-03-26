# IRT-on-Bench

A Python package for analyzing language model benchmark data using Item Response Theory (IRT).

## Features

- Load and preprocess language model benchmark results
- Compute score matrices for model evaluation
- Fit classical and Bayesian IRT models (Rasch, 2PL)

## Installation

### From GitHub

```bash
pip install git+https://github.com/lamalab-org/irt-on-bench.git
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/lamalab-org/irt-on-bench.git
cd irt-on-bench

# Install in development mode
pip install -e .
```

## Usage

### Running IRT Analysis

You can run a full IRT analysis on your benchmark data using the command-line interface:

```bash
# Basic usage with classical IRT
irt-analyze --data_path=your_data.pkl --output_path=results/

# Using Bayesian IRT with PyMC
irt-analyze --data_path=your_data.pkl --output_path=results/ --pymc
```

### Python API

```python
from irt_on_bench.data import load_model_scores, create_binary_matrix
from irt_on_bench.analyzer import BenchmarkAnalyzer
from irt_on_bench.metadata import BinaryQuestionMetadata
from irt_on_bench.models import fit_2pl_pymc

# Load model scores
model_scores = load_model_scores("your_data.pkl")

# Create binary matrix
binary_array, binary_df, models = create_binary_matrix(model_scores)

# Using the BenchmarkAnalyzer for classical IRT
analyzer = BenchmarkAnalyzer()

# Create question metadata
for i in range(binary_array.shape[0]):
    analyzer.add_question_metadata(BinaryQuestionMetadata(f"q{i}"))

# Add model results
for i, model in enumerate(models):
    model_df = pd.DataFrame({
        'all_correct_': binary_array[:, i]
    }, index=[f"q{j}" for j in range(binary_array.shape[0])])
    analyzer.add_model_results(model, model_df)

# Fit IRT model
irt_results = analyzer.fit_irt(model='2pl')

# Extract parameters
difficulties = irt_results['difficulties']
discriminations = irt_results['discriminations']
abilities = irt_results['abilities']

# Alternatively, use Bayesian IRT with PyMC
trace = fit_2pl_pymc(binary_array.T)
```

### Analyzing Results

The package provides utilities for analyzing and visualizing IRT results:

```python
from irt_on_bench.analysis import item_fit_statistics, identify_misfitting_items
from irt_on_bench.analyzer import BenchmarkAnalyzer

# Calculate item fit statistics
item_stats = item_fit_statistics(binary_array, abilities, difficulties, discriminations)

# Identify misfitting items
misfitting_items = identify_misfitting_items(item_stats)

# Analyze extreme items
question_ids = [f"q{i}" for i in range(len(difficulties))]
extreme_items = BenchmarkAnalyzer.analyze_extreme_items(
    difficulties, 
    discriminations, 
    question_ids
)
```

## Data Format

The package expects model scores in a specific format:

```
{
  "overall": {
    "model_name_1": {
      "all_correct_": pandas.Series,
      ...
    },
    "model_name_2": {
      "all_correct_": pandas.Series,
      ...
    },
    ...
  },
  ...
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Citation

If you use this software in your research, please cite:

```
@software{irt-on-bench,
  author = {Schilling-Wilhelmi, Mara and Alampara, Nawaf and Jablonka, Kevin M.},
  title = {IRT-on-Bench: Item Response Theory for Language Model Benchmarks},
  year = {2025},
  url = {https://github.com/lamalab-org/irt-on-bench}
}
```

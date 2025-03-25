import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.constants import golden

# Load data
with open('model_score_dict.pkl', 'rb') as f:
    model_score_dicts = pickle.load(f)

# Get first model's data to analyze difficulties
sample_df = next(iter(model_score_dicts['overall'].values()))
has_scores = ~(sample_df['score_precision'].isna() | sample_df['score_recall'].isna())
df_filtered = sample_df[has_scores]

# Count questions per difficulty
difficulties = []
for keywords in df_filtered['keywords_0']:
    diff = [k for k in keywords if 'difficulty' in k]
    difficulties.append(diff[0].replace('-', ' ') if diff else 'intermediate')

difficulty_counts = pd.Series(difficulties).value_counts()
weight_sets = [
    {'difficulty basic': 1, 'intermediate': 5, 'difficulty advanced': 10},
    {cat: len(difficulties) / (3 * count) for cat, count in difficulty_counts.items()}
]

# Rest of analysis remains the same
model_metrics = {}
all_hamming_scores = []
for df in model_score_dicts['overall'].values():
    has_scores = ~(df['score_precision'].isna() | df['score_recall'].isna())
    all_hamming_scores.extend(df.loc[has_scores, 'metrics_hamming'])
min_hamming = min(all_hamming_scores)
max_hamming = max(all_hamming_scores)

for model_name, df in model_score_dicts['overall'].items():
    has_scores = ~(df['score_precision'].isna() | df['score_recall'].isna())
    df_filtered = df[has_scores]

    difficulties = []
    for keywords in df_filtered['keywords_0']:
        diff = [k for k in keywords if 'difficulty' in k]
        if not diff:
            difficulties.append('intermediate')
        else:
            difficulties.append(diff[0].replace('-', ' '))

    df_filtered['difficulty'] = difficulties

    scores = {
        'unweighted_mean': df_filtered['all_correct_'].mean(),
        'weighted_mean_0': np.average(df_filtered['all_correct_'],
                                      weights=[weight_sets[0][d] for d in df_filtered['difficulty']]),
        'weighted_mean_1': np.average(df_filtered['all_correct_'],
                                      weights=[weight_sets[1][d] for d in df_filtered['difficulty']]),
        'hamming_mean': (1 - (df_filtered['metrics_hamming'].mean() - min_hamming) / (max_hamming - min_hamming)),
        'hamming_max': (1 - (df_filtered['metrics_hamming'].max() - min_hamming) / (max_hamming - min_hamming)),
        'precision_mean': df_filtered['score_precision'].mean(),
        'recall_mean': df_filtered['score_recall'].mean()
    }
    model_metrics[model_name] = scores

metrics_df = pd.DataFrame(model_metrics).T

selected_models = [model for model in
                   ['o1', 'Claude-3.5 (Sonnet)', 'Mistral-Large-2', 'GPT-4', 'Claude-2-Zero-T', 'Gemini-Pro',
                    'GPT-3.5 Turbo Zero-T', 'Galatica-120b'] if model in metrics_df.index]

# Constants
ONE_COL_WIDTH_INCH = 3.25
TWO_COL_WIDTH_INCH = 7.2
ONE_COL_GOLDEN_RATIO_HEIGHT_INCH = ONE_COL_WIDTH_INCH / golden
TWO_COL_GOLDEN_RATIO_HEIGHT_INCH = TWO_COL_WIDTH_INCH / golden

# Define color scheme
# Original color scheme
# colors = ['#03071E', '#3A3B73', '#6A040F', '#B20404', '#D63909', '#E27F07', '#F9BB0B', '#FFDA00']
# Updated color scheme
colors = ['#194A81', '#2266AC', '#4093C3', '#92C4DE', '#F4A582', '#D8604E', '#B31C2B', '#661124']

plt.style.use('../../nbs/lamalab.mplstyle')
plt.figure(figsize=(TWO_COL_WIDTH_INCH, TWO_COL_WIDTH_INCH / 2))

# MODIFIED: Changed order of metrics and labels
metrics = ['precision_mean', 'recall_mean', 'hamming_mean', 'hamming_max',
           'unweighted_mean', 'weighted_mean_0', 'weighted_mean_1']
labels = ['Precision', 'Recall', 'Normalized Hamming\n(mean)', 'Normalized Hamming\n(max)',
          'All correct', 'Weighted\nall correct', 'Equal weighted\ndifficulties all correct']

for i, model in enumerate(selected_models):
    if model == 'o1':
        label = 'o1-preview'
    elif model == 'Claude-2-Zero-T':
        label = 'Claude-2'
    elif model == 'GPT-3.5 Turbo Zero-T':
        label = 'GPT-3.5 Turbo'
    else:
        label = model

    # Extract values in the new order for plotting
    values_in_new_order = [metrics_df.loc[model, metric] for metric in metrics]

    plt.plot(range(len(metrics)), values_in_new_order, 'o-',
             label=label,
             alpha=0.7,
             linewidth=1,
             color=colors[i % len(colors)])

plt.xticks(range(len(metrics)), labels, rotation=45, fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("chembench_scoring.pdf", dpi=300, bbox_inches='tight')

# Print the calculated weights
print("\nCalculated weights for equal category representation:")
print(weight_sets[1])
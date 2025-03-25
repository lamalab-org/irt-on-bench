import pymc as pm
import numpy as np
import arviz as az
import pickle
import pandas as pd
import os
from matplotlib import pyplot as plt
from scipy.constants import golden

plt.style.use("lamalab.mplstyle")

ONE_COL_WIDTH_INCH = 3.25
TWO_COL_WIDTH_INCH = 7.2
ONE_COL_GOLDEN_RATIO_HEIGHT_INCH = ONE_COL_WIDTH_INCH / golden
TWO_COL_GOLDEN_RATIO_HEIGHT_INCH = TWO_COL_WIDTH_INCH / golden


def fit_2pl_pymc(response_matrix, n_samples=2000, tune=2000):
    # Original fit_2pl_pymc function remains the same
    n_participants, n_items = response_matrix.shape
    with pm.Model() as irt_2pl:
        abilities = pm.Normal('abilities',
                              mu=0,
                              sigma=0.5,
                              shape=n_participants)

        discriminations = pm.TruncatedNormal('discriminations',
                                             mu=1.0,
                                             sigma=0.3,
                                             lower=0.5,
                                             upper=2.5,
                                             shape=n_items)

        difficulties = pm.TruncatedNormal('difficulties',
                                          mu=0,
                                          sigma=0.5,
                                          lower=-2,
                                          upper=2,
                                          shape=n_items)

        scaled_abilities = abilities[:, None] / 2.0
        scaled_difficulties = difficulties[None, :] / 2.0
        scaled_disc = discriminations[None, :] / 2.0

        logit_p = scaled_disc * (scaled_abilities - scaled_difficulties)

        responses = pm.Bernoulli('responses',
                                 logit_p=logit_p,
                                 observed=response_matrix)

        trace = pm.sample(n_samples,
                          tune=tune,
                          chains=4,
                          target_accept=0.99,
                          init='jitter+adapt_diag',
                          return_inferencedata=True,
                          cores=1)

    return trace


def fit_topic_specific_irt(model_score_dicts, output_dir='../results/topic_irt/'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    models = list(model_score_dicts['overall'].keys())
    sample_model = models[0]
    unique_topics = model_score_dicts['overall'][sample_model]['topic'].unique()

    # Dictionary to store results for each topic
    topic_results = {}

    # For each topic
    for topic in unique_topics:
        print(f"Processing topic: {topic}")

        # Create filename for this topic's results
        # Sanitize topic name for filename (replace both spaces and slashes)
        safe_topic_name = topic.replace(' ', '_').replace('/', '_').lower()
        topic_filename = os.path.join(output_dir, f"{safe_topic_name}_results.pkl")

        # Check if results already exist
        if os.path.exists(topic_filename):
            print(f"Loading existing results for topic: {topic}")
            with open(topic_filename, 'rb') as f:
                topic_results[topic] = pickle.load(f)
            continue

        # Create binary matrix for this topic
        topic_binary_matrix = []

        # For each model
        for model in models:
            topic_data = model_score_dicts['overall'][model]
            topic_mask = topic_data['topic'] == topic
            correct_answers = topic_data['all_correct_'][topic_mask].values
            topic_binary_matrix.append(correct_answers)

        topic_binary_array = np.array(topic_binary_matrix)

        # Fit IRT model for this topic
        trace = fit_2pl_pymc(topic_binary_array)

        # Extract abilities and standard deviations for this topic
        summary = az.summary(trace, var_names=['abilities'])
        topic_abilities_mean = summary['mean'].values
        topic_abilities_std = summary['sd'].values

        print("\nAbilities summary:")
        print(f"Means range: {topic_abilities_mean.min():.3f} to {topic_abilities_mean.max():.3f}")
        print(f"Standard deviations range: {topic_abilities_std.min():.3f} to {topic_abilities_std.max():.3f}")

        # Get all MCMC samples for this topic
        topic_results[topic] = {
            'abilities_mean': topic_abilities_mean,
            'abilities_std': topic_abilities_std,
            'trace': trace,  # Save full trace for potential later analysis
            'models': models  # Save model names for reference
        }

        # Save results for this topic
        with open(topic_filename, 'wb') as f:
            pickle.dump(topic_results[topic], f)

        print(f"Saved results for topic: {topic}")

    return topic_results, models


def plot_model_comparison_by_topic(df, topic_results, topics, models_to_plot=None, colors=None):
    n_topics = len(topics)
    n_cols = 3
    n_rows = (n_topics + 2) // 3

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * ONE_COL_WIDTH_INCH, n_rows * ONE_COL_GOLDEN_RATIO_HEIGHT_INCH))
    axes = axes.flatten()

    if models_to_plot is None:
        models_to_plot = df.index

    if colors is None:
        colors = ['#03071E', '#3A3B73', '#6A040F', '#B20404', '#D63909', '#E27F07', '#F9BB0B', '#FFDA00']

    # Create subplot for each topic
    for topic_idx, topic in enumerate(topics):
        ax = axes[topic_idx]

        # Get topic-specific IRT results
        topic_abilities_mean = topic_results[topic]['abilities_mean']
        topic_abilities_std = topic_results[topic]['abilities_std']

        # Print trace information
        print(f"\nTopic: {topic}")
        if 'trace' in topic_results[topic]:
            trace = topic_results[topic]['trace']
            summary = az.summary(trace, var_names=['abilities'])
            print("Trace summary:")
            print(summary)

        # Extract topic-specific scores
        topic_scores = []
        for model in models_to_plot:
            topic_data = model_score_dicts['overall'][model]
            topic_mask = topic_data['topic'] == topic
            correct_answers = topic_data['all_correct_'][topic_mask]
            topic_score = correct_answers.mean()
            topic_scores.append(topic_score)

        topic_df = pd.DataFrame(topic_scores, index=models_to_plot, columns=['score'])

        # Print raw values for debugging
        print(f"\nTopic: {topic}")
        print(f"\nFor topic {topic}:")
        for model_idx, model in enumerate(models_to_plot):
            print(f"{model}:")
            print(f"  Raw ability: {topic_abilities_mean[model_idx]:.3f} ± {topic_abilities_std[model_idx]:.3f}")
            print(f"  Score: {topic_scores[model_idx]:.3f}")

        # Normalize scores and their standard deviations
        # For scores (current implementation)
        score_range = topic_df.max() - topic_df.min()
        df_norm = (topic_df - topic_df.min()) / score_range

        # For IRT abilities: normalize to [0,1] keeping relative distances
        ability_range = topic_abilities_mean.max() - topic_abilities_mean.min()
        abilities_norm = (topic_abilities_mean - topic_abilities_mean.min()) / ability_range

        # Scale standard deviations proportionally
        abilities_std_norm = topic_abilities_std / ability_range

        print("Normalized abilities range:", abilities_norm.min(), "to", abilities_norm.max())
        print("Normalized std range:", abilities_std_norm.min(), "to", abilities_std_norm.max())

        x = np.array([0.1, 0.9])
        y_values = []

        for i, model in enumerate(models_to_plot):
            idx = df.index.get_loc(model)
            y = np.array([df_norm.loc[model, 'score'], abilities_norm[idx]])
            y_values.extend([y[0], y[1]])

            ax.plot(x, y, '-', label=model, color=colors[i % len(colors)], linewidth=1.0)

            y_err = abilities_std_norm[idx]
            y_values.extend([abilities_norm[idx] - y_err, abilities_norm[idx] + y_err])
            ax.errorbar(0.9, abilities_norm[idx], yerr=y_err,
                        fmt='none', color=colors[i % len(colors)], capsize=0,
                        linewidth=1)

        ax.set_xticks([0.1, 0.9])
        ax.set_xticklabels(['Current\nimplementation', 'IRT'])
        if topic_idx % n_cols == 0:
            ax.set_ylabel('Normalized Score')

        ymin = min(y_values) - 0.05
        ymax = max(y_values) + 0.05
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0, 1)

        ax.set_title(topic, pad=10)
        ax.set_facecolor('white')

        if topic_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Remove empty subplots
    for idx in range(topic_idx + 1, len(axes)):
        fig.delaxes(axes[idx])

    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.savefig("model_comparison_by_topic.pdf", format='pdf', bbox_inches='tight')
    plt.show()


# Main execution
with open('../data/filtered_model_score_dict.pkl', 'rb') as f:
    model_score_dicts = pickle.load(f)

# Fit IRT models for each topic
topic_results, models = fit_topic_specific_irt(model_score_dicts)

# Get unique topics
sample_model = models[0]
unique_topics = model_score_dicts['overall'][sample_model]['topic'].unique()

# Define models to plot
models_to_plot = ['o1', 'Claude-3.5 (Sonnet)', 'Mistral-Large-2', 'GPT-4',
                  'Claude-2-Zero-T', 'Gemini-Pro', 'GPT-3.5 Turbo Zero-T', 'Galatica-120b']

# Create DataFrame for plotting
model_score_df = pd.DataFrame(index=models)

# Plot results
plot_model_comparison_by_topic(model_score_df, topic_results, unique_topics, models_to_plot)
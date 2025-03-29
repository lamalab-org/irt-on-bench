import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.constants import golden

plt.style.use("../data/lamalab.mplstyle")

ONE_COL_WIDTH_INCH = 3.25
TWO_COL_WIDTH_INCH = 7.2
ONE_COL_GOLDEN_RATIO_HEIGHT_INCH = ONE_COL_WIDTH_INCH / golden
TWO_COL_GOLDEN_RATIO_HEIGHT_INCH = TWO_COL_WIDTH_INCH / golden

model_list = [
    "Claude-3.5 (Sonnet)",
    "GPT-4o",
    "Gemini-Flash",
    "Gemini-Pro",
    "LLama-3.2-90B",
    "basline",
]

model_map = {
    "Claude3V": "Claude-3.5 (Sonnet)",
    "GPT4V": "GPT-4o",
    "GeminiFlash": "Gemini-Flash",
    "GeminiPro": "Gemini-Pro",
    "GroqLlama": "LLama-3.2-90B",
    "baseline": "Basline",
}


def range_frame(ax, x, y, pad=0.2):
    y_min, y_max = np.min(y), np.max(y)
    x_min, x_max = np.min(x), np.max(x)

    ax.set_ylim(y_min - pad * (y_max - y_min), y_max + pad * (y_max - y_min))
    ax.set_xlim(x_min * (1 - pad), x_max * (1 + pad))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    ax.spines["left"].set_bounds(y_min, y_max)
    ax.spines["bottom"].set_bounds(x_min, x_max)


def plot_model_comparison(
    df, abilities_mean, abilities_std, models_to_plot=None, colors=None
):
    fig, ax = plt.subplots(
        figsize=(1.5 * ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH)
    )

    abilities_norm = (abilities_mean - abilities_mean.min()) / (
        abilities_mean.max() - abilities_mean.min()
    )

    if models_to_plot is None:
        models_to_plot = df.index

    if colors is None:
        colors = [
            "#03071E",
            "#3A3B73",
            "#6A040F",
            "#B20404",
            "#D63909",
            "#E27F07",
            "#FFBA08",
        ]
        # colors = ['#6A040F', '#03071E', '#D00000', '#DC2F02', '#E85D04', '#FAA307', '#FFBA08']

    x = np.array([0, 1])
    for i, model in enumerate(models_to_plot):
        idx = df.index.get_loc(model)
        y = np.array([df.loc[model, "score"], abilities_norm[idx]])
        plt.plot(
            x,
            y,
            "-",
            label=model_map[model],
            color=colors[i % len(colors)],
            linewidth=1.5,
        )
        plt.errorbar(
            1,
            abilities_norm[idx],
            yerr=abilities_std[idx],
            fmt="none",
            color=colors[i % len(colors)],
            capsize=0,
            linewidth=1,
        )

    plt.xticks([0, 1], ["Average", "IRT"])
    plt.ylabel("Score")
    plt.legend(bbox_to_anchor=(0.9, 0.8), loc="upper left")
    range_frame(
        ax,
        x,
        np.concatenate(
            [
                df.loc[models_to_plot, "score"],
                abilities_norm[[df.index.get_loc(m) for m in models_to_plot]],
            ]
        ),
    )
    plt.tight_layout()
    plt.savefig("model_comparison_macbench.pdf", format="pdf")
    plt.show()
    print(abilities_norm)


def plot_vector_as_square_heatmap(vector, title):
    # Calculate the size of the square matrix
    n = int(np.sqrt(len(vector)))
    # Reshape first n^2 elements into square matrix
    matrix = vector[: n * n].values.reshape(n, n)

    sns.heatmap(matrix, cmap="coolwarm", cbar=True)
    plt.title(title)
    plt.show()

    return n * n  # Return number of elements used


def plot_difficulty_violin(df,name):
   plt.figure(figsize=(ONE_COL_WIDTH_INCH, 1.2 *ONE_COL_GOLDEN_RATIO_HEIGHT_INCH))
   
   # Create violin plot
   ax = sns.violinplot(data=df, x='difficulty_level', y='difficulty', color="#FFBA08",linewidth=0.75)
   
   # Apply range frame
   x = df['difficulty_level'].unique()
   y = df['difficulty'].values
   range_frame(ax, x, y)
   
   # Customize labels
   ax.set_xticklabels(['Easy', 'Intermediate', 'Hard'])

#    plt.title('Difficulty Distribution')
   plt.xlabel('Human Assigned')
   plt.ylabel('Computed Difficulty')
   
   plt.tight_layout()
   plt.savefig(f"{name}_difficulty.pdf", format='pdf')

   plt.show()
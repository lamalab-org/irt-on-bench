import pickle

import pandas as pd
import pymc as pm
from irt import fit_2pl_pymc, get_parameter_estimates

from plots import plot_model_comparison


def create_binary_matrix_from_df(df):
  """
  Creates a binary matrix from a df
  """

  # Pivot the DataFrame to create the matrix
  matrix = df.pivot_table(index='question_name', columns='Model', values='all_correct', fill_value=0)
  # Convert scores to binary (1 for perfect score, 0 otherwise)
  matrix = (matrix == 1).astype(int) 
  return matrix




if __name__ == '__main__':


    macbench_scores = pd.read_csv('../data/macbench_scores_for_irt.csv')
    models_to_keep = ['GroqLlama', 'Claude3V', 'GPT4V', 'GeminiPro']
    # remove all non-multiple choice questions
    macbench_scores = macbench_scores[macbench_scores['preferred_score'] == 'multiple_choice_grade']
    # remove baseline and flash
    filtered_df = macbench_scores[macbench_scores['Model'].isin(models_to_keep)]




    matrix = create_binary_matrix_from_df(filtered_df)
    with pm.Model() as model:
        macbench_trace = fit_2pl_pymc(matrix.T)


    with open('../data/macbench_trace.pkl', 'wb') as f:
        pickle.dump(macbench_trace, f)


    estimates = get_parameter_estimates(macbench_trace)
    abilities_mean = estimates['abilities']['mean']
    abilities_std = estimates['abilities']['std']


    model_score_list = []
    models_in_macbench = list(matrix.columns)
    macbench_binary_array = matrix.to_numpy()
    for i in range(len(models_in_macbench)):
        model_score_list.append(macbench_binary_array[:,i].mean())
    model_score_df = pd.DataFrame(model_score_list, index=models_in_macbench, columns=['score'])


    plot_model_comparison(model_score_df, abilities_mean, abilities_std, models_to_plot=models_to_keep)
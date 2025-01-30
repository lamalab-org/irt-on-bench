import pickle
import numpy as np
import pandas as pd
from irt import fit_2pl_pymc, get_parameter_estimates
import pymc as pm
from plots import plot_difficulty_violin, plot_model_comparison



with open('../data/model_score_dict.pkl', 'rb') as f:
    model_score_dicts = pickle.load(f)

models = list(model_score_dicts['overall'].keys())
binary_matrix = np.zeros((len(model_score_dicts['overall']['Claude-3.5 (Sonnet)']), len(models)))
for i, model in enumerate(models):
   binary_matrix[:,i] = model_score_dicts['overall'][model]['all_correct_'].values

# Convert to pandas DataFrame for better visualization
binary_df = pd.DataFrame(binary_matrix, columns=models)
binary_array = binary_df.to_numpy()



with pm.Model() as model:
    trace_all_correct = fit_2pl_pymc(binary_array.T)
with open('../data/chembench_trace_all_correct.pkl', 'wb') as f:
    pickle.dump(trace_all_correct, f)


### PLOT Difficulty
#get human assigned difficulty
difficulty_dict = {'difficulty-basic': 0.25, 'difficulty-advanced': 1, 'NaN': 0.75}
difficulty_vector = model_score_dicts['overall']['Claude-2-Zero-T']['difficulty'].map(difficulty_dict).fillna(0.75)

# Get the difficulty vector from the IRT model
difficulty_vector_from_irt = trace_all_correct.posterior['difficulties'].mean(dim=['chain', 'draw']).values
difficulty_vector_dataframe = pd.DataFrame(difficulty_vector_from_irt, columns=['difficulty'])

difficulty_categories = {
    'easy': difficulty_vector[difficulty_vector == 0.25].index.tolist(),
    'intermediate': difficulty_vector[difficulty_vector == 0.75].index.tolist(),
    'hard': difficulty_vector[difficulty_vector == 1].index.tolist()
}

level_map = {i: 2 for i in range(len(difficulty_vector))}  # Default to 'hard' (2)
level_map.update({i: 0 for i in difficulty_categories['easy']})
level_map.update({i: 1 for i in difficulty_categories['intermediate']})

difficulty_vector_dataframe['difficulty_level'] = difficulty_vector_dataframe.index.map(level_map)


plot_difficulty_violin(difficulty_vector_dataframe,'chembench')




### PLOT Model Comparison
estimates = get_parameter_estimates(trace_all_correct)
abilities_mean = estimates['abilities']['mean']
abilities_std = estimates['abilities']['std']


model_list = list(model_score_dicts['overall'].keys())

model_score_list = []
for model in model_list:
    model_score_list.append(sum(model_score_dicts['overall'][model]['all_correct_'])/len(model_score_dicts['overall'][model]['all_correct_']))
model_score_df = pd.DataFrame(model_score_list, index=model_list, columns=['score'])

models_to_plot = ['o1', 'Claude-3.5 (Sonnet)', 'GPT-4o', 'GPT-4','Llama-2-70B Chat', 'Mistral-Large-2', 'Galatica-120b']

plot_model_comparison(model_score_df, abilities_mean, abilities_std, models_to_plot=models_to_plot)


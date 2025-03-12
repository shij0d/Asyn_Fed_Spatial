#%%
import os
import pickle
import torch
import matplotlib.pyplot as plt
from typing import Dict,Tuple
import sys
from matplotlib.ticker import MaxNLocator

#%%
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define the path to the results directory and the .pkl files
seed_value = 1  # Replace <seed_value> with the actual seed value
result_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'scenario_1', f'seed_{seed_value}','test')  # Replace <seed_value> with the actual seed value
params_dis_file = os.path.join(result_dir, 'params_dis.pkl')
params_non_dis_file = os.path.join(result_dir, 'params_non_dis.pkl')

# Load the results from the .pkl files
with open(params_dis_file, 'rb') as f:
    results_dis: Dict = pickle.load(f)

with open(params_non_dis_file, 'rb') as f:
    non_dis_param = pickle.load(f)

# Calculate distances
distances_dict = {}
times_accumulated_dict = {}
for (concurrency, step_size), (params_GDT, times_elapsed) in results_dis.items():
    distances = []
    for param in params_GDT:
        distance = torch.sqrt(torch.norm(param.gamma - non_dis_param.gamma)**2 + \
                              torch.norm(param.delta - non_dis_param.delta)**2 + \
                              torch.norm(param.theta - non_dis_param.theta)**2)
        distances.append(distance.item())
    distances = torch.tensor(distances)
    times_accumulated = torch.cumsum(torch.tensor(times_elapsed), dim=0)
    distances_dict[(concurrency, step_size)] = distances
    times_accumulated_dict[(concurrency, step_size)] = times_accumulated



    

# Plot the results using subplots: each subplot corresponds to a different step size, and each line corresponds to a different concurrency
step_sizes=sorted(set(key[1] for key in results_dis.keys()))
num_step_sizes = len(step_sizes)
fig, axes = plt.subplots(1,num_step_sizes, figsize=(6*num_step_sizes, 5), sharex=True)

step_size_to_index = {step_size: i for i, step_size in enumerate(step_sizes)}

for (concurrency, step_size), distances in distances_dict.items():
    distances = distances_dict[(concurrency, step_size)]
    iterations=range(len(distances))
    ax = axes[step_size_to_index[step_size]]
    ax.plot(iterations[1:], torch.log10(distances[1:]), label=f'concurrency={concurrency}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Distance(log10)')
    ax.set_title(f'Step size:{step_size}')
    ax.set_ylim([-8.5, 1])
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'distance_plots_fixed_step_size.pdf'))
plt.show()

#enlarge at 0:20
fig, axes = plt.subplots(1,num_step_sizes, figsize=(6*num_step_sizes, 5), sharex=True)

step_size_to_index = {step_size: i for i, step_size in enumerate(step_sizes)}

for (concurrency, step_size), distances in distances_dict.items():
    distances = distances_dict[(concurrency, step_size)]
    iterations=range(len(distances))
    ax = axes[step_size_to_index[step_size]]
    ax.plot(iterations[1:21], torch.log10(distances[1:21]), label=f'concurrency={concurrency}')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Distance(log10)')
    ax.set_title(f'Step size:{step_size}')
    ax.set_ylim([-2, 0])
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'distance_plots_fixed_step_size_enlarged.pdf'))
plt.show()




# Plot the results using subplots: each subplot corresponds to a different concurrency, and each line corresponds to a different step_size


concurrencies=sorted(set(key[0] for key in results_dis.keys()))
num_concurrencies = len(concurrencies)
fig, axes = plt.subplots(1,num_concurrencies, figsize=(6*num_concurrencies, 5), sharex=True)

concurrency_to_index = {concurrency: i for i, concurrency in enumerate(concurrencies)}

for (concurrency, step_size), distances in distances_dict.items():
    distances = distances_dict[(concurrency, step_size)]
    iterations=range(len(distances))
    ax = axes[concurrency_to_index[concurrency]]
    ax.plot(iterations[0:], torch.log10(distances[0:]), label=f'step size={step_size}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Distance(log10)')
    ax.set_title(f'Concurrency:{concurrency}')
    ax.set_ylim([-8.5, 1])
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'distance_plots_fixed_concurrency.pdf'))
plt.show()



#time for each iteration

step_sizes=sorted(set(key[1] for key in results_dis.keys()))
num_step_sizes = len(step_sizes)
fig, axes = plt.subplots(1,num_step_sizes, figsize=(6*num_step_sizes, 5), sharex=True)

step_size_to_index = {step_size: i for i, step_size in enumerate(step_sizes)}

for (concurrency, step_size), times_accumulated in times_accumulated_dict.items():
    iterations=range(len(times_accumulated))
    ax = axes[step_size_to_index[step_size]]
    ax.plot(iterations[0:], times_accumulated[0:], label=f'concurrency={concurrency}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Time')
    ax.set_title(f'Step size:{step_size}')
    #ax.set_ylim([-8, 1])
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'times_accumulated_plots_fixed_step_size.pdf'))
plt.show()
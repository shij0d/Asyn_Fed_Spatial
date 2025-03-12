#%%
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List

from src.data_generation import GPPSampleGenerator
from src.estimation import Server,Worker,LocalComputation,GlobalComputation,StepType,Parameter,NonDisEstimation
from src.utils import set_times
import math
import torch
from sklearn.gaussian_process.kernels import Matern
from src.kernel import exponential_kernel,onedif_kernel
import yaml
import pickle
import logging

#initialization
with open(os.path.join(os.path.dirname(__file__), '..', 'config', 'scenario_1.yaml')) as f:
    config = yaml.safe_load(f)
r=config["seed"]
J=config["num_workers"]
alpha=config["alpha"]
length_scale=config["length_scale"]
nu=config["nu"]
n=config["num_samples"]
N=n*J
mis_dis=config["min_dis"]
l=math.sqrt(2*N)*mis_dis
extent=-l/2,l/2,-l/2,l/2,
coefficients=config["coefficients"]
noise_level=config["noise_level"]
m=config["num_knots"]
kernel=alpha*Matern(length_scale=length_scale,nu=nu)
if nu==0.5:
        kernel_est =exponential_kernel
elif nu==1.5:
        kernel_est = onedif_kernel
else:
    raise("incompleted")   
time_mean_dict={
    StepType.MU_SIGMA:3,
    StepType.GAMMA:2,
    StepType.DELTA:1,
    StepType.THETA:5
}
time_dicts=set_times(time_mean_dict,J)

gamma=torch.tensor(coefficients,dtype=torch.float64)
delta=torch.tensor(1/noise_level**2,dtype=torch.float64)
theta=torch.tensor([alpha,length_scale],dtype=torch.float64)
param0=Parameter(None,None,gamma,delta,theta) # the initial value for gamma, theta, and delta

concurrencies:List[int]=config["concurrencies"]
step_sizes:List[float]=config["step_sizes"] # the step sizes for update theta



########## 1. Generate data ##########
sampler=GPPSampleGenerator(num=N,min_dis=mis_dis,extent=extent,kernel=kernel,coefficients=coefficients,noise=noise_level,seed=r)
data,knots=sampler.generate_obs_gpp(m=m,method="random")
dis_data=sampler.data_split(data,J)
data = torch.tensor(data, dtype=torch.float64)
knots = torch.tensor(knots, dtype=torch.float64)
dis_data = [torch.tensor(d, dtype=torch.float64) for d in dis_data]
result_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'scenario_1')
os.makedirs(result_dir, exist_ok=True)

########## 2. Define local workers and their local computation ##########

workers=[]
for j in range(J):
    local_data=dis_data[j]
    local_computation=LocalComputation(local_data,knots,kernel_est)
    worker=Worker(j,local_computation,time_dicts[j])
    workers.append(worker)

#############3. define the server and then run the algorithm############

logging.basicConfig(filename=os.path.join(result_dir, 'intermediate_results.log'), level=logging.INFO, format='%(message)s')
local_params_path=os.path.join(result_dir,'local_params')
results_dis={}     
for step_size in step_sizes:
    global_computation=GlobalComputation(knots,kernel_est,step_size)
    for concurrency in concurrencies:
        logging.info(f"#############step_size:{step_size},concurrency:{concurrency}######################")
        server=Server(workers,concurrency,global_computation)
        server.run(param0,T=100,S=10,local_params_path=local_params_path)
        results_dis[(concurrency,step_size)]=(server.params_GDT,server.times_elapsed)

##########4. save the results################
filename=os.path.join(result_dir,f'params_dis.pkl')
with open(filename, 'wb') as f:
    pickle.dump(results_dis, f)
###########5. Non-distributed estimation################                 
non_dis_estimation=NonDisEstimation(data,knots,kernel_est)
non_dis_param=non_dis_estimation.get_minimizer(param0,tol=1e-8)
filename=os.path.join(result_dir,f'params_non_dis.pkl')
with open(filename, 'wb') as f:
    pickle.dump(non_dis_param, f)


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_generation import GPPSampleGenerator
from src.estimation import Server,Worker,LocalComputation,GlobalComputation,StepType,Parameter
from src.utils import set_times
import math
import torch
from sklearn.gaussian_process.kernels import Matern
from src.kernel import exponential_kernel,onedif_kernel
import yaml

########## 1. Generate data ##########
with open(os.path.join(os.path.dirname(__file__), '..', 'config', 'scenarios_1.yaml')) as f:
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
sampler=GPPSampleGenerator(num=N,min_dis=mis_dis,extent=extent,kernel=kernel,coefficients=coefficients,noise=noise_level,seed=r)
data,knots=sampler.generate_obs_gpp(m=m,method="random")
dis_data=sampler.data_split(data,J)
data = torch.tensor(data, dtype=torch.float64)
knots = torch.tensor(knots, dtype=torch.float64)
dis_data = [torch.tensor(d, dtype=torch.float64) for d in dis_data]

########## 2. Define the server and local workers ##########
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
workers=[]
for j in range(J):
    local_data=dis_data[j]
    local_computation=LocalComputation(local_data,knots,kernel_est)
    worker=Worker(j,local_computation,time_dicts[j])
    workers.append(worker)

global_computation=GlobalComputation(knots,kernel_est)
concurrency=config["concurrency"]
server=Server(workers,concurrency,global_computation)



######## 3. Run the algorithm ##########
gamma=torch.tensor(coefficients,dtype=torch.float64)
delta=torch.tensor(1/noise_level**2,dtype=torch.float64)
theta=torch.tensor([alpha,length_scale],dtype=torch.float64)
param0=Parameter(None,None,gamma,delta,theta) # the initial value for gamma, theta, and delta
server.run(param0,T=100,S=10)
server.save_params_GDT(os.path.join(os.path.dirname(__file__), '..', 'results', 'scenario_1','params.pkl'))
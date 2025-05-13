#%%
project_path='/scratch/shij0d/projects/Asyn_Fed_Spatial'
import sys
import os
import argparse
sys.path.append(project_path)
from typing import List

import yaml
from src.data_generation import GPPSampleGeneratorUnitSquare

from src.estimation import Parameter
from src.utils import set_times
import torch
from sklearn.gaussian_process.kernels import Matern
from src.kernel import exponential_kernel,onedif_kernel
import pickle
import time
from functools import partial
#initialization
#load the config

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run MPI-based federated spatial regression')
    parser.add_argument('--num_workers', type=int, 
                        help='the number of workers')
    parser.add_argument('--num_samples', type=int, 
                        help='the number of samples')
    parser.add_argument('--alpha', type=float, default=1, 
                        help='the variance of matern kernel')
    parser.add_argument('--length_scale', type=float, default=0.1, 
                        help='the length scale of the matern kernel')
    parser.add_argument('--nu', type=float, default=0.5, 
                        help='the smoothness of the matern kernel')
    parser.add_argument('--coefficients', type=str, default='[-1, 2, 3, -2, 1]', 
                        help='the coefficients of the covariates')
    parser.add_argument('--noise_level', type=float, default=2, 
                        help='the noise level')
    parser.add_argument('--m', type=int, default=100, 
                        help='the number of knots')
    parser.add_argument('--data_dir', type=str, 
                        help='the directory of the generated data to be saved')
    parser.add_argument('--seed', type=int, default=123,
                        help='the seed')
    return parser.parse_args()

args=parse_arguments()
r=args.seed
J=args.num_workers
alpha=args.alpha
length_scale=args.length_scale
nu=args.nu
n=args.num_samples
N=n*J
coefficients=eval(args.coefficients)
noise_level=args.noise_level
m=args.m
kernel=alpha*Matern(length_scale=length_scale,nu=nu)
theta=torch.tensor([alpha,length_scale],dtype=torch.float32)
kernel=partial(exponential_kernel,theta=theta)
if nu==0.5:
        kernel_est = exponential_kernel
elif nu==1.5:
        kernel_est = onedif_kernel
else:
    raise ValueError("incompleted")   

gamma=torch.tensor(coefficients,dtype=torch.float64)
delta=torch.tensor(1/noise_level**2,dtype=torch.float64)
theta=torch.tensor([alpha,length_scale],dtype=torch.float64)
param0=Parameter(None,None,gamma,delta,theta) # the initial value for gamma, theta, and delta


sampler=GPPSampleGeneratorUnitSquare(num=N,kernel=kernel,coefficients=coefficients,noise_level=noise_level,seed=r)
start_time = time.time()
res=sampler.generate_obs_gp(m=m,method="random")
end_time = time.time()
print("Time taken to generate data: ", end_time - start_time)
data,knots=res[0]
dis_data=sampler.data_split(data,J)
data_dir = args.data_dir
os.makedirs(data_dir, exist_ok=True)
#save distributed data into different pickle files: server: knots, worker 1: data1, worker 2: data2, ...
with open(os.path.join(data_dir, 'data_server.pkl'), 'wb') as f:
    data_dict={'knots':knots,'kernel_est':kernel_est,'param0':param0}
    pickle.dump(data_dict, f)
for i in range(J):
    with open(os.path.join(data_dir, 'data_worker_{}.pkl'.format(i)), 'wb') as f:
        data_dict={'local_data':dis_data[i],'knots':knots,'kernel_est':kernel_est}
        pickle.dump(data_dict, f)

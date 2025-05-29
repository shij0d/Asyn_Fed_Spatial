import sys
import os

import numpy as np
#print(f"NumPy is using {np.__config__.show()} for multithreading")
from collections import deque,OrderedDict
from typing import Deque,Dict,List,Tuple,Callable
from enum import Enum
import torch
#torch.set_num_threads(192)
#print(f"Using {torch.get_num_threads()} threads in PyTorch")
from scipy.optimize import minimize
from src.utils import softplus, inv_softplus
from warnings import warn
import math
import random
import pickle
import psutil
import logging
from mpi4py import MPI
from queue import Queue

import threading
import time 
from torch.autograd.functional import hessian
import traceback
import numexpr

process = psutil.Process()

class StepType(Enum):
    MU_SIGMA = 1
    GAMMA = 2
    DELTA = 3
    THETA = 4 # The step for updating theta
    DELTA_THETA = 5 # The step for updating delta and theta
    #CHECK_THETA = 5 # The step for checking theta


class Parameter():
    def __init__(self, mu:torch.Tensor=None,sigma:torch.Tensor=None,gamma:torch.Tensor=None,delta:torch.Tensor=None,theta:torch.Tensor=None,index:Tuple[int,StepType,int]=None):
        self.mu=mu
        self.sigma=sigma
        self.gamma=gamma
        self.theta=theta
        self.delta=delta
        self.index=index # The index of the parameter, which is a tuple (t,i,s) with t being the round, i being the step and s being the iteration number of Newton's method for optimizing theta

def compute_average_param(params: List[Parameter]) -> Parameter:
        # Compute the average of the parameters
        valid_params = [param for param in params if param is not None]
        mu_avg = torch.mean(torch.stack([param.mu for param in valid_params]), dim=0)
        sigma_avg = torch.mean(torch.stack([param.sigma for param in valid_params]), dim=0)
        gamma_avg = torch.mean(torch.stack([param.gamma for param in valid_params]), dim=0)
        delta_avg = torch.mean(torch.stack([param.delta for param in valid_params]), dim=0)
        theta_avg = torch.mean(torch.stack([param.theta for param in valid_params]), dim=0)
        index = valid_params[0].index
        return Parameter(mu_avg, sigma_avg, gamma_avg, delta_avg, theta_avg, index)
    
class LocalQuantity():
    # The local quantity that is computed by the worker and depends on the parameter
    def __init__(self, worker_id:int,value:Dict[str,torch.Tensor],index:Tuple[int,StepType,int]):
        self.worker_id=worker_id
        self.value=value
        self.index=index # The index of the parameter, which is a tuple (t,i,s) with t being the round, i being the step and s being the iteration number of Newton's method for optimizing theta 

class LocalQuantities():
    #collection of local quantities, that keep only the newest local quantity for each worker and step
    def __init__(self, local_quantities_list:List[LocalQuantity]=None):
        self.local_quantities_dict: Dict[int, Dict[StepType, LocalQuantity]] = {}
        if local_quantities_list is not None:
            for local_quantity in local_quantities_list:
                self.merge_new(local_quantity)
            
            
    def merge_new(self,local_quantity:LocalQuantity):
        worker_id = local_quantity.worker_id
        step = local_quantity.index[1]
        if worker_id not in self.local_quantities_dict:
            self.local_quantities_dict[worker_id] = {}
        if step not in self.local_quantities_dict[worker_id]:
            self.local_quantities_dict[worker_id][step] = local_quantity
        else:
            existing_quantity = self.local_quantities_dict[worker_id][step]
            if (local_quantity.index[0] > existing_quantity.index[0]) or \
               (local_quantity.index[0] == existing_quantity.index[0] and local_quantity.index[2] > existing_quantity.index[2]):
                self.local_quantities_dict[worker_id][step] = local_quantity
    
    def merge_new_dif(self,local_quantity: LocalQuantity):
        worker_id = local_quantity.worker_id
        step = local_quantity.index[1]
        if worker_id not in self.local_quantities_dict:
            self.local_quantities_dict[worker_id] = {}
        if step not in self.local_quantities_dict[worker_id]:
            self.local_quantities_dict[worker_id][step] = local_quantity
        else:
            existing_quantity = self.local_quantities_dict[worker_id][step]
            if (local_quantity.index[0] > existing_quantity.index[0]) or \
               (local_quantity.index[0] == existing_quantity.index[0] and local_quantity.index[2] > existing_quantity.index[2]):
                for key in local_quantity.value.keys():
                    existing_quantity.value[key] += local_quantity.value[key]
                existing_quantity.index = local_quantity.index
                
    
    def get_count_by_worker(self,worker_id:int)->int:
        return len(self.local_quantities_dict.get(worker_id,{}))
    
    
    
    def get_count_by_step(self,step:StepType)->int:
        count = 0
        for worker_quantities_dict in self.local_quantities_dict.values():
            if step in worker_quantities_dict:
                count += 1
        return count
    
    
    def get_by_step(self,step:StepType)->Dict[int,LocalQuantity]:
        #return {key:worker_id, value:LocalQuantity} for given step
        step_quantities_dict = {}
        for worker_id, worker_quantities in self.local_quantities_dict.items():
            if step in worker_quantities:
                step_quantities_dict[worker_id]=worker_quantities[step]
        return step_quantities_dict
    
    def remove_by_step(self,step:StepType):
        for worker_id in self.local_quantities_dict.keys():
            if step in self.local_quantities_dict[worker_id]:
                del self.local_quantities_dict[worker_id][step]
                
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_gb = memory_info.rss / 1024 ** 3  # Convert bytes to GB
    return memory_usage_gb
          
class LocalComputation():
    # The class that implements the computation in local workers
    def __init__(self,local_data:torch.Tensor, knots:torch.Tensor,\
        kernel:Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],type_LR='F'):
        self.local_data=local_data
        self.knots=knots
        self.kernel=kernel
        self.type_LR=type_LR # there are three types for low-rank structure: 'F', 'P', 'H', For fully low-rank, P for partially low-rank where the dependence inside workers is kept, H for hybrid low-rank where dependence for points are kept
        self.n=self.local_n()
        self.IM_quantities:OrderedDict[int,Dict[str,torch.Tensor]]=OrderedDict() # store the intermediate quantity that can be reused, it is dictionary in the format of {t,{name,value}}
        self.IM_quantities_count=10 # the number of intermediate quantities that are stored, it is used to control the memory usage
    def K_f(self,theta: torch.Tensor)->torch.Tensor:
        K=self.kernel(self.knots, self.knots, theta)
        return K
    
    def local_B(self,theta: torch.Tensor)->torch.Tensor:
        K=self.K_f(theta)
        invK:torch.Tensor = torch.linalg.inv(K)
        local_locs = self.local_data[:, :2]
        local_B = self.kernel(local_locs, self.knots, theta) @ invK
        return local_B,K
    def local_B_with_IM_quantity(self,theta: torch.Tensor,t:int)->torch.Tensor:
        IM_quantity = self.IM_quantities.get(t, {})

        # Retrieve or compute 'local_B'
        if 'local_B' in IM_quantity and 'K' in IM_quantity:
            local_B = IM_quantity['local_B']
            K = IM_quantity['K']
        else:
            local_B, K = self.local_B(theta)
            IM_quantity['local_B'] = local_B
            IM_quantity['K'] = K
            self.IM_quantities[t] = IM_quantity  # Update the cache
        # Check if the cache size exceeds the limit
        while len(self.IM_quantities) > self.IM_quantities_count:
            # Remove the oldest item from the cache
            self.IM_quantities.popitem(last=False)
            
        return local_B,K
    
    def local_errorv(self,gamma)->torch.Tensor:
        local_X = self.local_data[:, 3:]
        local_z = self.local_data[:, 2].unsqueeze(1)
        local_errorv = local_X@gamma-local_z        
        return local_errorv
    
    def local_XX(self)->torch.Tensor:
        local_X = self.local_data[:, 3:]
        local_XX = local_X.T@local_X
        return local_XX
    def local_Xz(self)->torch.Tensor:
        local_X=self.local_data[:, 3:]
        local_z=self.local_data[:, 2].unsqueeze(1)
        local_Xz=local_X.T@local_z
        return local_Xz
    def local_n(self)->torch.Tensor:
        return torch.tensor(self.local_data.shape[0])
    def local_inv(self,theta: torch.Tensor,delta: torch.Tensor):
        # (delta*(C-BKB^T)*I)^{-1}
        local_S=self.local_data[:, :2]
        local_cov=self.kernel(local_S,local_S,theta)
        local_B,K=self.local_B(theta)
        local_BKBT=local_B@K@local_B.T
        local_M=delta*(local_cov-local_BKBT)
        local_M.diagonal().add_(1) #in place add to the diagonal
        local_inv=torch.linalg.inv(local_M)
        return local_inv
    def local_inv_with_IM_quantity(self,theta: torch.Tensor,delta:torch.Tensor,t:int):
        # (delta*(C-BKB^T)*I)^{-1}
        IM_quantity = self.IM_quantities.get(t, {})
        if 'local_inv' in IM_quantity:
            local_inv = IM_quantity['local_inv']
        else:
            local_inv=self.local_inv(theta,delta)
            IM_quantity['local_inv'] = local_inv
            self.IM_quantities[t] = IM_quantity
        # Check if the cache size exceeds the limit
        while len(self.IM_quantities) > self.IM_quantities_count:
            # Remove the oldest item from the cache
            self.IM_quantities.popitem(last=False)
        return local_inv
        
    def local_cov(self,theta: torch.Tensor):
        # C
        local_S=self.local_data[:, :2]
        local_cov=self.kernel(local_S,local_S,theta)
        return local_cov
    
    def local_inv_diag_v(self,theta: torch.Tensor,delta: torch.Tensor):
        # (delta*diag(C-BKB^T)+I))^{-1}
        #diagonal of self.kernel(local_S,local_S,theta)
        alpha=theta[0]
        local_B,K= self.local_B(theta)
        local_BK = local_B @ K    
        local_BKBT_diag_V=torch.sum(local_B * local_BK, dim=1).unsqueeze(1)
        local_inv_diag_V=1/(delta*(alpha-local_BKBT_diag_V)+1)
        return local_inv_diag_V
    
    def local_inv_diag_v_with_IM_quantity(self,theta: torch.Tensor,delta: torch.Tensor,t:int):
        # (delta*diag(C-BKB^T)+I))^{-1}
        #diagonal of self.kernel(local_S,local_S,theta)
        IM_quantity = self.IM_quantities.get(t, {})
        if 'local_inv_diag_v' in IM_quantity:
            local_inv_diag_v = IM_quantity['local_inv_diag_v']
        else:
            local_inv_diag_v=self.local_inv_diag_v(theta,delta)
            IM_quantity['local_inv_diag_v'] = local_inv_diag_v
            self.IM_quantities[t] = IM_quantity
        # Check if the cache size exceeds the limit
        while len(self.IM_quantities) > self.IM_quantities_count:
            # Remove the oldest item from the cache
            self.IM_quantities.popitem(last=False)
        return local_inv_diag_v
    
    
    def compute_local_for_mu_sigma(self,para:Parameter)->Dict[str,torch.Tensor]:
        
        theta=para.theta
        gamma=para.gamma 
        # if up_im_local_quantity is not None and up_im_local_quantity.index[0]==para.index[0]:
        #     local_B=up_im_local_quantity.value.get('local_B',self.local_B(theta))
        # else:
        #     local_B = self.local_B(theta)
        # up_im_local_quantity['local_B']=local_B
        # up_im_local_quantity.index=para.index
        
        t=para.index[0]
        local_B,_=self.local_B_with_IM_quantity(theta,t)
        
        if self.type_LR=='F':
            local_for_sigma=local_B.T@local_B
            local_errorv = self.local_errorv(gamma)
            local_for_mu=local_B.T@local_errorv
        elif self.type_LR=='P':
            delta=para.delta
            local_inv=self.local_inv_with_IM_quantity(theta,delta,t)
            #local_inv=self.local_inv(theta,delta)
            local_for_sigma=local_B.T@local_inv@local_B
            local_errorv = self.local_errorv(gamma)
            local_for_mu=local_B.T@local_inv@local_errorv
        elif self.type_LR=='H':
            delta=para.delta
            local_inv_diag_V=self.local_inv_diag_v_with_IM_quantity(theta,delta,t)
            #local_inv_diag_V=self.local_inv_diag_v(theta,delta)
            local_for_sigma=local_B.T@(local_inv_diag_V*local_B)
            local_errorv = self.local_errorv(gamma)
            local_for_mu=local_B.T@(local_inv_diag_V*local_errorv)        
        else:
            raise ValueError("Invalid type_LR. Choose from 'F', 'P', or 'H'.")
        
        value={}
        value['mu']=local_for_mu
        value['sigma']=local_for_sigma
        
        return value
    
    def compute_local_for_gamma(self,para:Parameter)->Dict[str,torch.Tensor]:
        theta=para.theta
        mu=para.mu
        local_X =self.local_data[:, 3:]
        
        
        # if up_im_local_quantity is not None and up_im_local_quantity.index[0]==para.index[0]:
        #     local_B=up_im_local_quantity.value.get('local_B',self.local_B(theta))
        # else:
        #     local_B = self.local_B(theta) 
        # up_im_local_quantity['local_B']=local_B
        # up_im_local_quantity.index=para.index
        
        value={}
        t=para.index[0]
        local_B,_=self.local_B_with_IM_quantity(theta,t)
        if self.type_LR=='F':
            local_for_gamma=local_X.T @ local_B @ mu 
            value['gamma']=local_for_gamma
        elif self.type_LR=='P':
            delta=para.delta
            local_z = self.local_data[:, 2].unsqueeze(1)
            local_inv=self.local_inv_with_IM_quantity(theta,delta,t)
            #local_inv=self.local_inv(theta,delta)
            local_for_gamma=local_X.T@local_inv@(local_z-local_B@mu)
            local_for_XX=local_X.T@local_inv@local_X
            value['gamma']=local_for_gamma
            value['XX']=local_for_XX
        elif self.type_LR=='H':
            delta=para.delta
            local_z = self.local_data[:, 2].unsqueeze(1)
            local_inv_diag_V=self.local_inv_diag_v_with_IM_quantity(theta,delta,t)
            #local_inv_diag_V=self.local_inv_diag_v(theta,delta)
            local_for_gamma=local_X.T@(local_inv_diag_V*(local_z-local_B@mu))
            local_for_XX=local_X.T@(local_inv_diag_V*local_X)
            value['gamma']=local_for_gamma
            value['XX']=local_for_XX
        else:
            raise ValueError("Invalid type_LR. Choose from 'F', 'P', or 'H'.")
        
        return value
    
    def compute_local_for_delta(self,para:Parameter)->Dict[str,torch.Tensor]:
        theta=para.theta
        gamma=para.gamma   
        mu=para.mu
        sigma=para.sigma
        
        value={}
        
        # if up_im_local_quantity is not None and up_im_local_quantity.index[0]==para.index[0]:
        #     local_B=up_im_local_quantity.value.get('local_B',self.local_B(theta))
        # else:
        #     local_B = self.local_B(theta) 
        # up_im_local_quantity['local_B']=local_B
        # up_im_local_quantity.index=para.index
        #local_B,_ = self.local_B(theta)
        t=para.index[0]
        local_B,_=self.local_B_with_IM_quantity(theta,t)
        if self.type_LR=='F':
            local_errorv =self.local_errorv(gamma) 
            reminder_v=local_errorv+local_B@mu
            term1=sum(reminder_v*reminder_v)
            local_BSigma = local_B @ sigma
            term2=torch.sum(local_B * local_BSigma) # this method (complexity is n*m^2+2n*m) is faster than torch.trace(local_B.T @ local_B @sigma) (complexity is 2*n*m^2)
            local_for_delta=term1+term2 
            value['delta']=local_for_delta
            # term1 = torch.trace(local_B.T @ local_B @
            #                             (sigma + mu @ mu.T))
            # term2 = 2 * local_errorv.T @ local_B @ mu
            # term3 = local_errorv.T @ local_errorv
            # local_for_delta=term1+term2+term3
        elif self.type_LR=='P':
            delta=para.delta
            local_inv=self.local_inv_with_IM_quantity(theta,delta,t)
            #local_inv=self.local_inv(theta,delta)
            local_inv_square=local_inv@local_inv
            local_errorv =self.local_errorv(gamma)
            reminder_v=local_errorv+local_B@mu
            term1=reminder_v.T@local_inv_square@reminder_v
            local_BsigmaB=local_B@sigma@local_B.T
            term2=torch.trace(local_inv_square@local_BsigmaB)
            local_for_delta=term1+term2
            value['delta']=local_for_delta
            local_tr_inv=torch.trace(local_inv)
            value['tr_inv']=local_tr_inv
        
        elif self.type_LR=='H':
            delta=para.delta
            local_errorv =self.local_errorv(gamma)
            reminder_v=(local_errorv+local_B@mu)
            reminder_v_square=reminder_v*reminder_v
            
            local_inv_diag_v=self.local_inv_diag_v_with_IM_quantity(theta,delta,t)
            #local_inv_diag_v=self.local_inv_diag_v(theta,delta)
            local_inv_diag_v_square=local_inv_diag_v*local_inv_diag_v
            
            term1=sum(local_inv_diag_v_square*reminder_v_square)
            
            local_BSigmaB_diag_V=torch.sum(local_B * (local_B @ sigma), dim=1).unsqueeze(1)
            term2=sum(local_inv_diag_v_square*local_BSigmaB_diag_V)
            local_for_delta=term1+term2
            value['delta']=local_for_delta
            
            local_tr_inv=sum(local_inv_diag_v)
            value['tr_inv']=local_tr_inv
        else:
            raise ValueError("Invalid type_LR. Choose from 'F', 'P', or 'H'.")
            

        return value
    
    def compute_local_for_theta(self,para:Parameter)-> Dict[str,torch.Tensor]:
        #gradient and Hessian
        mu=para.mu
        sigma=para.sigma
        gamma=para.gamma 
        delta=para.delta
        theta=para.theta
        
        n = self.local_data.shape[0]
        
        theta = theta.clone()
        if theta.grad is not None:
            theta.grad.data.zero_()
        theta.requires_grad_(True)

        local_B,_ =self.local_B(theta) 
        local_errorv=self.local_errorv(gamma)
        reminder_v=local_errorv+local_B@mu
        
        
        if self.type_LR=='F':
            f_value:torch.Tensor = -n * torch.log(delta) + delta * (
                torch.sum(local_B * (local_B@sigma))
                + reminder_v.T @ reminder_v
            )
        elif self.type_LR=='P':            
            local_inv=self.local_inv(theta,delta)
            local_BsigmaB=local_B@sigma@local_B.T
            f_value:torch.Tensor=-n * torch.log(delta)-torch.logdet(local_inv)+delta *(reminder_v.T@local_inv@reminder_v+torch.trace(local_inv@local_BsigmaB))
            
        elif self.type_LR=='H':
            local_inv_diag_v=self.local_inv_diag_v(theta,delta)
            local_BsigmaB_diag_V=torch.sum(local_B * (local_B@sigma), dim=1).unsqueeze(1)
            f_value:torch.Tensor = -n * torch.log(delta)-torch.sum(torch.log(local_inv_diag_v))+delta*(sum(reminder_v*local_inv_diag_v*reminder_v)+sum(local_BsigmaB_diag_V*local_inv_diag_v))
        else:
            raise ValueError("Invalid type_LR. Choose from 'F', 'P', or 'H'.")
        # First backward pass: compute first-order gradient
        grad = torch.autograd.grad(f_value, theta, create_graph=True)[0]  # Retain graph for Hessian
        
        # Initialize Hessian
        hessian = torch.zeros((theta.numel(), theta.numel()), dtype=torch.float64)

        # Compute Hessian row-by-row
        for i in range(theta.numel()):
            # Compute gradient of grad[i] w.r.t theta
            g2 = torch.autograd.grad(
                grad[i], 
                theta, 
                retain_graph=True,  # Retain graph for subsequent iterations
                allow_unused=False  # Ensure all elements are used
            )[0]
            hessian[i] = g2.view(-1)  # Flatten to match Hessian shape
        
        # Detach theta to prevent further graph retention
        theta = theta.detach()
        
        value={}
        value['grad']=grad.detach()
        value['hessian']=hessian
        
        return value # Return both gradient and Hessian  
   
   

    

    def compute_local_for_theta_delta(self, para: Parameter) -> Dict[str, torch.Tensor]:
        # Unpack parameters
        mu, sigma, gamma = para.mu, para.sigma, para.gamma
        theta = para.theta.clone().detach()
        delta = para.delta.clone().detach()
        n = self.local_data.shape[0]
        
        # Combine delta and theta into a single vector of size (d+1)
        var = torch.cat([delta.reshape(1), theta.reshape(-1)])
        var = var.detach().clone().requires_grad_(True)

        def f_scalar(v):
            # v[0] = delta, v[1:] = theta
            dl=v[0]
            th=v[1:]
            local_B, _ = self.local_B(th)
            err = self.local_errorv(gamma)
            r = err + local_B @ mu

            if self.type_LR == 'F':
                Bsig = local_B @ sigma
                return (-n * torch.log(dl)
                        + dl * (torch.sum(local_B * Bsig)
                                + r.T @ r)).squeeze()
            elif self.type_LR == 'P':
                inv = self.local_inv(th, dl)
                BsigB = local_B @ sigma @ local_B.T
                return (-n * torch.log(dl)
                        - torch.logdet(inv)
                        + dl * (r.T @ inv @ r
                                + torch.trace(inv @ BsigB))).squeeze()
            elif self.type_LR == 'H':
                inv_diag = self.local_inv_diag_v(th, dl)
                BsigB_diag = torch.sum(local_B * (local_B @ sigma),
                                    dim=1, keepdim=True)
                return (-n * torch.log(dl)
                        - torch.sum(torch.log(inv_diag))
                        + dl * (torch.sum(r**2 * inv_diag)
                                + torch.sum(BsigB_diag * inv_diag))).squeeze()
            else:
                raise ValueError("Invalid type_LR")

        # 1) gradient w.r.t. var
        f0 = f_scalar(var)
        grad = torch.autograd.grad(f0,var)[0]

        # 2) Hessian w.r.t. var
        H:torch.Tensor = hessian(f_scalar, var)

        return {'grad': grad.detach(), 'hessian': H.detach()}

    def __FreeVector2Parameter_GDT(self, free_vector_GDT: torch.Tensor) -> Parameter:
        p = self.local_data.shape[1] - 3
        start = 0
        gamma = free_vector_GDT[start:start + p].unsqueeze(1)
        start += p
        
        # Extracting and transforming delta from compact_param
        delta_ol = free_vector_GDT[start]
        delta = softplus(delta_ol)
        start += 1
        
        # Extracting and transforming theta values from compact_param
        theta_ol = free_vector_GDT[start:]
        theta = softplus(theta_ol).unsqueeze(1)
        param=Parameter(None,None,gamma,delta,theta)
        return param
    def __Parameter2FreeVector_GDT(self, param: Parameter) -> torch.Tensor:
        p=param.gamma.shape[0]
        free_vector_GDT = torch.empty(p+1+param.theta.shape[0], dtype=torch.float64)

        start = 0
        if p > 0:
            free_vector_GDT[start:(start+p)] = param.gamma.squeeze()
            start += p

        free_vector_GDT[start] = inv_softplus(param.delta)
        start += 1

        free_vector_GDT[start:] = inv_softplus(param.theta).squeeze()
        
        return free_vector_GDT
    
    
    def local_neg_log_lik(self,free_vector_GDT:torch.Tensor,require_grad:bool)->torch.Tensor:
        #compact_para is a postive tensor that only corresponds to gamma, delta, and theta
        free_vector_GDT.requires_grad_(require_grad)
        if free_vector_GDT.grad is not None:
            free_vector_GDT.grad.data.zero_()
        
        param = self.__FreeVector2Parameter_GDT(free_vector_GDT)
        gamma=param.gamma
        delta=param.delta
        theta=param.theta
                
        n=self.local_data.shape[0]
        local_errorv=self.local_errorv(gamma)
        
        if self.type_LR=='F':
            
            local_B,K=self.local_B(theta)
            invK=torch.linalg.inv(K)
            tempM = invK+delta*local_B.T@local_B
            f_value:torch.Tensor = torch.logdet(tempM)+torch.logdet(K)-n*torch.log(delta)+delta*local_errorv.T@local_errorv-(delta**2)*(local_errorv.T@local_B)@torch.linalg.inv(tempM)@(local_B.T@local_errorv)
            f_value = f_value/n
            
        elif self.type_LR=='P': 
            local_cov_nugget=self.local_cov(theta)
            local_cov_nugget.diagonal().add_(1/delta) #in place add to the diagonal
            f_value:torch.Tensor = torch.logdet(local_cov_nugget)+local_errorv.T@torch.linalg.inv(local_cov_nugget)@local_errorv
            f_value = f_value/n
            
        elif self.type_LR=='H':
            
            local_inv_diag_v=self.local_inv_diag_v(theta,delta)
            local_B,K=self.local_B(theta)
            invK=torch.linalg.inv(K)
            tempM = invK+delta*local_B.T@(local_inv_diag_v*local_B)
            local_errorv_scaled=local_errorv*local_inv_diag_v
            # det(D+BKB^T)=det(K^{-1}+B^TD^{-1}B)*det(K)*det(D)
            # (D+BKB^T)^{-1}=D^{-1}-D^{-1}B(K^{-1}+B^TD^{-1}B)^{-1}B^TD^{-1}
            f_value:torch.Tensor = torch.logdet(tempM)+torch.logdet(K)-n*torch.log(delta)-sum(torch.log(local_inv_diag_v))+delta*(local_errorv.T)@(local_errorv_scaled)-(delta**2)*(local_errorv_scaled.T@local_B)@torch.linalg.inv(tempM)@(local_B.T@local_errorv_scaled)
            f_value = f_value/n
        else:
            raise ValueError("Invalid type_LR. Choose from 'F', 'P', or 'H'.")
        
        if require_grad:
            # Compute the gradients
            f_value.backward()
            grad = free_vector_GDT.grad
            return f_value, grad
        else:
            return f_value
    def local_neg_log_lik_wrapper(self, requires_grad=True):
        """
        the negative local log likelihood function for the low rank model
        """
        if requires_grad:
            def fun(free_vector_GDT: np.ndarray):
                free_vector_GDT = torch.tensor(free_vector_GDT, dtype=torch.float64)
                value = self.local_neg_log_lik(free_vector_GDT, False)
                return value.numpy().flatten()

            def gradf(free_vector_GDT: np.ndarray):
                free_vector_GDT = torch.tensor(free_vector_GDT, dtype=torch.float64)
                _, grad = self.local_neg_log_lik(free_vector_GDT, True)
                return grad.numpy()

            return fun, gradf
        
        else:
            def fun(free_vector_GDT: np.ndarray):
                free_vector_GDT = torch.tensor(free_vector_GDT, dtype=torch.float64)
                value = self.local_neg_log_lik(free_vector_GDT, False)
                return value.numpy().flatten()
            return fun
        
    def get_local_minimizer(self,param0:Parameter,tol=None)->Parameter:
        #param0 is the initial parameter, and mu and Sigma can be ignored
        free_vector_GDT0=self.__Parameter2FreeVector_GDT(param0).numpy() # transfer to the free vector, should be in numpy form
        nllikf, nllikgf = self.local_neg_log_lik_wrapper(requires_grad=True)
        if tol:
            result = minimize(fun=nllikf,
                            x0=free_vector_GDT0,
                            method="BFGS",
                            tol=tol,
                            jac=nllikgf)
        else: # use the default tolerance 
            result = minimize(fun=nllikf,
                            x0=free_vector_GDT0,
                            method="BFGS",
                            jac=nllikgf)
        minimizer_lik = torch.tensor(result.x, dtype=torch.float64)
        param= self.__FreeVector2Parameter_GDT(minimizer_lik)
        param.index=(-1,StepType.MU_SIGMA,0) # the index is set to -1,0,0, which means that it is not used in the computation
        local_for_mu_sigma=self.compute_local_for_mu_sigma(param)
        
        K=self.K_f(param.theta)
        invK=torch.linalg.inv(K)
        tempM = invK+param.delta*local_for_mu_sigma['sigma']
        sigma = torch.linalg.inv(tempM)
        mu = -sigma@(param.delta*local_for_mu_sigma['mu'])
        param.mu=mu
        param.sigma=sigma
        return param
       
class GlobalComputation():
    # The class that implements the computation in the server
    def __init__(self,knots:torch.Tensor,\
        kernel:Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],step_size_inner:float,type_LR='F'):
        self.knots=knots
        self.kernel=kernel
        self.step_size_inner=step_size_inner
        self.global_para_ind_quantities:Dict[str,torch.Tensor]={}
        self.type_LR=type_LR #same as the type_LR in LocalComputation
    def reset(self):
        self.global_para_ind_quantities:Dict[str,torch.Tensor]={}
    def set_global_para_ind_quantities(self,global_para_ind_quantities:Dict[str,torch.Tensor]):
        self.global_para_ind_quantities=global_para_ind_quantities
        
    def K_f(self,theta: torch.Tensor)->torch.Tensor:
        K=self.kernel(self.knots, self.knots, theta)
        return K
    
    def Grad_Hessian_f(self, param: Parameter)->Dict[str,torch.Tensor]:
        theta=param.theta
        mu=param.mu
        sigma=param.sigma
        
        theta=theta.clone()
        if theta.grad is not None:
            theta.grad.data.zero_()
        theta.requires_grad_(True)
        K=self.K_f(theta)
        invK = torch.inverse(K)
        # f_value = mu.T @ invK @ mu + torch.trace(invK @ Sigma) - torch.log(torch.det(Sigma) / torch.det(K))
        f_value = mu.T @ invK @ mu + \
            torch.trace(invK @ sigma) + torch.logdet(K)
        f_value.backward(create_graph=True)
        grad = theta.grad.clone() if theta.grad is not None else None
        hessian = torch.zeros((theta.numel(), theta.numel()), dtype=torch.float64)
        for i in range(theta.numel()):
            g2 = torch.autograd.grad(
                grad[i] if grad is not None else None, 
                theta, 
                retain_graph=True, 
                allow_unused=False
            )[0]
            hessian[i] = g2.view(-1)
        theta = theta.detach()
        
        value={}
        value['grad']=grad.detach()
        value['hessian']=hessian
        
        return value    
        
    def update_mu_Sigma(self,param:Parameter,global_quantity_mu_sigma:Dict[str,torch.Tensor]):
        delta=param.delta
        theta=param.theta
        
        global_for_mu,global_for_sigma=global_quantity_mu_sigma['mu'],global_quantity_mu_sigma['sigma']
         
        J=self.global_para_ind_quantities['J']
        
        K=self.K_f(theta)
        invK=torch.linalg.inv(K)
        tempM = invK+J*delta*global_for_sigma
        sigma = torch.linalg.inv(tempM)
        mu = -sigma@(J*delta*global_for_mu)
        param=Parameter(mu,sigma,param.gamma,param.delta,param.theta)
        return param
    def update_gamma(self,param: Parameter,global_quantity_gamma:Dict[str,torch.Tensor]):
        global_for_gamma=global_quantity_gamma['gamma']
        if self.type_LR=='F':
            global_XX=self.global_para_ind_quantities['XX']
            global_Xz=self.global_para_ind_quantities['Xz']
            gamma=torch.linalg.inv(global_XX)@(global_Xz-global_for_gamma)
        elif self.type_LR=='P' or self.type_LR=='H':
            global_XX=global_quantity_gamma['XX']
            gamma=torch.linalg.inv(global_XX)@(global_for_gamma)
        param=Parameter(param.mu,param.sigma,gamma,param.delta,param.theta)
        return param
    
    def update_delta(self,param: Parameter,global_quantity_delta:Dict[str,torch.Tensor]):
        global_for_delta=global_quantity_delta['delta']
        if self.type_LR=='F':
            n=self.global_para_ind_quantities['n']
            delta=n/global_for_delta
        elif self.type_LR=='P' or self.type_LR=='H':
            tr_inv=global_quantity_delta['tr_inv']
            delta=tr_inv/global_for_delta
        else:
            raise ValueError("Invalid type_LR. Choose from 'F', 'P', or 'H'.")
        param=Parameter(param.mu,param.sigma,param.gamma,delta,param.theta)
        return param
    
    def update_theta(self,param:Parameter,global_quantity_theta:Dict[str,torch.Tensor]):        
        global_for_theta_gradient=global_quantity_theta['grad']
        global_for_theta_hessian=global_quantity_theta['hessian']
        J=self.global_para_ind_quantities['J']
        com_gradient_hessian=self.Grad_Hessian_f(param)
        com_gradient=com_gradient_hessian['grad']
        com_hessian=com_gradient_hessian['hessian']
        
        gradient=global_for_theta_gradient*J+com_gradient
        hessian=global_for_theta_hessian*J+com_hessian
        eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
        
        abs_eigenvalues = eigenvalues.abs()
        
        threshold = 0.01  # Define a positive threshold
        
        # Replace elements smaller than the threshold with the threshold value
        modified_eigenvalues = torch.where(
                    abs_eigenvalues < threshold, torch.tensor(threshold), abs_eigenvalues)
        
        # Construct the diagonal matrix of modified eigenvalues
        modified_eigenvalue_matrix = torch.diag(modified_eigenvalues)
        modified_hess = eigenvectors@modified_eigenvalue_matrix@eigenvectors.T
        # if self.step_size_inner<0.1:
        #     self.step_size_inner=self.step_size_inner*2        
        step_size=self.step_size_inner
        
        
        invh_m_grad = torch.linalg.inv(modified_hess)@gradient
        theta = param.theta-step_size*invh_m_grad
        while theta[0]<=0 or theta[1]<=0:
            step_size=step_size/2
            self.step_size_inner=step_size
            logging.info(f"step_size:{step_size}")
            theta = param.theta-step_size*invh_m_grad
            
        param=Parameter(param.mu,param.sigma,param.gamma,param.delta,theta)
        return param
    def update_delta_theta(self,param:Parameter,global_quantity_theta:Dict[str,torch.Tensor]):        
        global_for_gradient=global_quantity_theta['grad'].unsqueeze(1)
        global_for_hessian=global_quantity_theta['hessian']
        J=self.global_para_ind_quantities['J']
        com_gradient_hessian=self.Grad_Hessian_f(param)
        com_gradient=com_gradient_hessian['grad']
        com_hessian=com_gradient_hessian['hessian']
        
        d = com_gradient.numel()

        # 1) Add 0 in front of gradient
        zero_grad = torch.zeros((1,1), dtype=com_gradient.dtype, device=com_gradient.device)
        com_gradient = torch.cat([zero_grad, com_gradient], dim=0)  # shape: (d+1,)

        # 2) Add a zero row at the top of the Hessian
        zero_row = torch.zeros((1, d), dtype=com_hessian.dtype, device=com_hessian.device)
        com_hessian = torch.cat([zero_row, com_hessian], dim=0)     # shape: (d+1, d)

        # 3) Add a zero column at the left of the Hessian
        zero_col = torch.zeros((d + 1, 1), dtype=com_hessian.dtype, device=com_hessian.device)
        com_hessian = torch.cat([zero_col, com_hessian], dim=1)     # final shape: (d+1, d+1)
        
        
        gradient=global_for_gradient*J+com_gradient
        hessian=global_for_hessian*J+com_hessian
        eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
        
        abs_eigenvalues = eigenvalues.abs()
        
        threshold = 0.01  # Define a positive threshold
        
        # Replace elements smaller than the threshold with the threshold value
        modified_eigenvalues = torch.where(
                    abs_eigenvalues < threshold, torch.tensor(threshold), abs_eigenvalues)
        
        # Construct the diagonal matrix of modified eigenvalues
        modified_eigenvalue_matrix = torch.diag(modified_eigenvalues)
        modified_hess = eigenvectors@modified_eigenvalue_matrix@eigenvectors.T
        # if self.step_size_inner<0.1:
        #     self.step_size_inner=self.step_size_inner*2        
        step_size=self.step_size_inner
        
        
        invh_m_grad = torch.linalg.inv(modified_hess)@gradient
        new_theta:torch.Tensor = param.theta - step_size * invh_m_grad[1:]
        new_delta:torch.Tensor = param.delta - step_size * invh_m_grad[0]
        
        while (new_delta <= 0) or (new_theta <= 0).any():
            step_size=step_size/2
            self.step_size_inner=step_size
            logging.info(f"step_size:{step_size}")
            new_theta = param.theta - step_size * invh_m_grad[1:]
            new_delta = param.delta - step_size * invh_m_grad[0]
            
        param=Parameter(param.mu,param.sigma,param.gamma,new_delta, new_theta)
        return param


class InitializationWorker():
    def __init__(self,comm: MPI.Intracomm, server_rank:int, local_computation: LocalComputation):
        self.comm=comm
        self.rank=comm.Get_rank()
        self.server_rank=server_rank
        self.size=comm.Get_size()
        self.worker_id=self.rank
        self.update_param:Parameter=Parameter() #store the newest parameter
        self.local_params: Deque[Parameter]= deque()  # is a queue of local parameters, in the principal of "first in and first out"
        self.pre_local_quantity_dict:Dict[StepType,LocalQuantity]={}
        self.local_computation=local_computation
    def set_pre_local_quantity(self,local_quantity:LocalQuantity):
        step=local_quantity.index[1]
        self.pre_local_quantity_dict[step]=local_quantity
    
    def get_initial_param(self,param0: Parameter)->Parameter:
        
        try:
            param = self.local_computation.get_local_minimizer(param0)
            return param
        except Exception as e:
            logging.error(f"Error in get_initial_param: {e}")
            return None
    def set_local_params(self, param:Parameter):
        #when a new parameter is received, add it to the local parameters
        for name in ['mu', 'sigma', 'gamma', 'delta', 'theta']:
            value = getattr(param, name)
            if value is not None:
                setattr(self.update_param, name, value)
            else:
                setattr(param, name, getattr(self.update_param, name))
        id=None
        if self.local_params is not None and len(self.local_params)>0:
            for i, local_param in enumerate(self.local_params):
                if local_param.index[1]==param.index[1]:
                    id=i
        if id is None:
            self.local_params.append(param)
        else:
            self.local_params[id]=param
      
    def compute_local_quantity(self, param: Parameter,step: StepType=None)->LocalQuantity:
        if step is None:
            step=param.index[1]
        elif param.index[1] is None:
            param.index[1]=step
        elif step!=param.index[1]:
            warn("the step is not consistent with the step in parameter", category=None, stacklevel=1)
        if step==StepType.MU_SIGMA:
            value=self.local_computation.compute_local_for_mu_sigma(param)
        elif step==StepType.GAMMA:
            # compute local quantity for gamma
            value=self.local_computation.compute_local_for_gamma(param)
        elif step==StepType.DELTA:
            # compute local quantity for delta
            value=self.local_computation.compute_local_for_delta(param)
        elif step==StepType.THETA:
            # compute local quantity for theta
            value=self.local_computation.compute_local_for_theta(param)
        elif step==StepType.DELTA_THETA:
            # compute local quantity for theta and delta
            value=self.local_computation.compute_local_for_theta_delta(param)    
        return LocalQuantity(self.worker_id, value, param.index)
    def compute_para_ind_local_quantity(self)->Dict[str,torch.Tensor]: 
        #compute local quantity that is independent of the parameter
        para_ind_local_quantity={}
        para_ind_local_quantity['XX']=self.local_computation.local_XX()
        para_ind_local_quantity['Xz']=self.local_computation.local_Xz()
        para_ind_local_quantity['n']=self.local_computation.local_n()
        return para_ind_local_quantity
    def initialization(self):
        while True:
            information=self.comm.bcast(None, root=self.server_rank)
            if information['message']=='loc_opt':
                param0=information['param0']
                initial_param=self.get_initial_param(param0=param0)
                self.comm.gather(initial_param,root=self.server_rank)
            elif information['message']=='compute_para_ind_quantity':
                para_ind_local_quantity=self.compute_para_ind_local_quantity()
                self.comm.gather(para_ind_local_quantity,root=self.server_rank)
            elif information['message']=='compute_local_quantity':
                param=information['param']
                step=information['step']
                local_quantity=self.compute_local_quantity(param,step)
                self.set_pre_local_quantity(local_quantity)
                self.comm.gather(local_quantity,root=self.server_rank)
            elif information['message']=='set_type_LR':
                type_LR=information['type_LR']
                self.local_computation.type_LR=type_LR
                
            elif information['message']=='stop':
                param=information['param']
                self.set_local_params(param=param)
                break
    def save(self,filename:str):
        #save self.local_params, self.pre_local_quantity_dict
        with open(filename, 'wb') as f:
            pickle.dump({'local_params': self.local_params, 'pre_local_quantity_dict': self.pre_local_quantity_dict, 'update_param': self.update_param}, f)
            
class Worker():
    def __init__(self,comm: MPI.Intracomm, server_rank:int, local_computation: LocalComputation,logger:logging.Logger,iflog=True):
        self.comm=comm
        self.rank=comm.Get_rank()
        self.server_rank=server_rank
        self.size=comm.Get_size()
        self.worker_id=self.rank
        self.cur_param=None # used for computing local quantity
        self.update_param:Parameter=Parameter() #store the newest parameter
        self.local_params: Deque[Parameter]= deque()  # is a queue of local parameters, in the principal of "first in and first out"
        self.coming_time=None
        self.pre_local_quantity_dict:Dict[StepType,LocalQuantity]={}
        self.local_computation=local_computation
        self.local_quantities:Deque[LocalQuantity]=deque()
        if iflog:
            self.logger=logger
        else:
            self.logger=None
        self.condition_send=threading.Condition()
        self.condition_receive=threading.Condition()
        self.stop_flag = threading.Event()
        
    def reset(self):
        self.cur_param=None
        self.local_params: Deque[Parameter]= deque()  # is a queue of local parameters, in the principal of "first in and first out"
        self.coming_time=None
        self.pre_local_quantity_dict:Dict[StepType,LocalQuantity]={}
        
    def set_pre_local_quantity(self,local_quantity:LocalQuantity):
        step=local_quantity.index[1]
        self.pre_local_quantity_dict[step]=local_quantity
    
    def get_initial_param(self,param0: Parameter)->Parameter:
        
        try:
            param = self.local_computation.get_local_minimizer(param0)
            
            return param
        except Exception as e:
            logging.error(f"Error in get_initial_param: {e}")
            return None
        
    
    def set_local_params(self, param:Parameter):
        #when a new parameter is received, add it to the local parameters
        for name in ['mu', 'sigma', 'gamma', 'delta', 'theta']:
            value = getattr(param, name)
            if value is not None:
                setattr(self.update_param, name, value)
            else:
                setattr(param, name, getattr(self.update_param, name))
        id=None
        if self.local_params is not None and len(self.local_params)>0:
            for i, local_param in enumerate(self.local_params):
                if local_param.index[1]==param.index[1]:
                    id=i
        if id is None:
            self.local_params.append(param)
        else:
            self.local_params[id]=param
                
            
    def compute_local_quantity(self, param: Parameter,step: StepType=None)->LocalQuantity:
        if step is None:
            step=param.index[1]
        elif param.index[1] is None:
            param.index[1]=step
        elif step!=param.index[1]:
            warn("the step is not consistent with the step in parameter", category=None, stacklevel=1)
        if step==StepType.MU_SIGMA:
            value=self.local_computation.compute_local_for_mu_sigma(param)
        elif step==StepType.GAMMA:
            # compute local quantity for gamma
            value=self.local_computation.compute_local_for_gamma(param)
        elif step==StepType.DELTA:
            # compute local quantity for delta
            value=self.local_computation.compute_local_for_delta(param)
        elif step==StepType.THETA:
            # compute local quantity for theta
            value=self.local_computation.compute_local_for_theta(param)
        elif step==StepType.DELTA_THETA:
            # compute local quantity for theta and delta
            value=self.local_computation.compute_local_for_theta_delta(param)    
        return LocalQuantity(self.worker_id, value, param.index)
    
    def compute_new_local_quantity(self)->LocalQuantity:
        if self.cur_param:
            local_quantity=self.compute_local_quantity(self.cur_param)
            return local_quantity
    def compute_dif_and_update_pre(self)->LocalQuantity:
        local_quantity=self.compute_new_local_quantity()
        step=local_quantity.index[1]
        dif_value={}
        for key in local_quantity.value.keys():
            dif_value[key]=local_quantity.value[key]-self.pre_local_quantity_dict[step].value[key]
        dif_quantity=LocalQuantity(local_quantity.worker_id,dif_value,local_quantity.index)
        self.set_pre_local_quantity(local_quantity)
        return dif_quantity         
    
    def initialization(self,initial_worker_path:str):
        with open(initial_worker_path, 'rb') as f:
            initial_worker_dict:Dict= pickle.load(f)
        self.pre_local_quantity_dict=initial_worker_dict['pre_local_quantity_dict']
        self.local_params=initial_worker_dict['local_params']
        self.update_param=initial_worker_dict['update_param']

    
    def send_local_quantity(self):
        #the thread sending local quantity to the server
        while not self.stop_flag.is_set():
            with self.condition_send:
                while len(self.local_quantities)==0:
                    wait_result=self.condition_send.wait(timeout=600)
                    if not wait_result:
                        if self.logger is not None:
                            self.logger.info(f"Timeout waiting for send local quantity to server")
                    if self.stop_flag.is_set():
                        information={'message':'stop'}
                        self.comm.send(information,dest=self.server_rank)
                        return
                local_quantity=self.local_quantities.popleft()
            start_time_send = time.time()
            local_quantity.value['start_time_send']=start_time_send
            information={'message':'local_quantity','param':local_quantity}
            self.comm.send(information,dest=self.server_rank)
            
                
                
    def compute_local_quantity_thread(self):
        #the thread computing local quantity
        while not self.stop_flag.is_set():
            with self.condition_receive:
                while len(self.local_params)==0:
                    if self.logger is not None:
                        self.logger.info(f"worker {self.worker_id} is waiting for receiving param from server")
                    wait_result=self.condition_receive.wait(timeout=600)
                    if not wait_result:
                        if self.logger is not None:
                            self.logger.info(f"Timeout of worker {self.worker_id} waiting for receiving param from server")
                    
                    if self.stop_flag.is_set():  # Check again after waking up
                        with self.condition_send:
                            self.condition_send.notify()
                            return 
                self.cur_param=self.local_params.popleft() 
                if self.logger is not None:
                    self.logger.info(f"worker {self.worker_id} uses the received param from server or use local params to compute the local quantity with index {self.cur_param.index}")
            
            start_time_compuation = time.time()  
            local_quantity=self.compute_dif_and_update_pre()
            end_time_compuation = time.time()
            #if self.logger is not None:
            #    self.logger.info(f" worker {self.worker_id} computes the local quantity {local_quantity.index} end with time {end_time_compuation}")
            local_quantity.value['start_time_compuation']=start_time_compuation
            local_quantity.value['end_time_compuation']=end_time_compuation
            if self.logger is not None:
                self.logger.info(f" worker {self.worker_id} computes the local quantity {local_quantity.index} end with time {end_time_compuation-start_time_compuation} and the torch threads: {torch.get_num_threads()}; total number of threads: {psutil.Process().num_threads()}, torch interop threads: {torch.get_num_interop_threads()}, python thread: {len(threading.enumerate())}")
                #for pool in threadpool_info():
                #    self.logger.info(f"pool: {pool}")
            #sleep(1)
            with self.condition_send:
                self.local_quantities.append(local_quantity)
                self.condition_send.notify()
                
    def receive_param(self):
        #the thread receiving parameter from server
        while not self.stop_flag.is_set():
            information=self.comm.recv(source=self.server_rank)
            if information['message']=='compute_local_quantity':
                param:Parameter=information['param']
                with self.condition_receive:
                    self.set_local_params(param)
                    
                    self.condition_receive.notify()
            elif information['message']=='stop':
                self.stop_flag.set()
                with self.condition_receive:
                    self.condition_receive.notify()
                break
    
    
    def start(self,initialization_worker_path:str):
        #start the worker
        if self.logger is not None:
            self.logger.info(f"worker {self.worker_id} starts initialization")
        self.initialization(initialization_worker_path)
        
        if self.logger is not None:
            self.logger.info(f"worker {self.worker_id} finishes initialization")
        
        send_thread=threading.Thread(target=self.send_local_quantity)
        receive_thread=threading.Thread(target=self.receive_param)
        compute_thread=threading.Thread(target=self.compute_local_quantity_thread)
        send_thread.start()
        receive_thread.start()
        compute_thread.start()
        send_thread.join()
        receive_thread.join()
        compute_thread.join()

class InitializationServer():
    def __init__(self,comm: MPI.Intracomm,global_computation:GlobalComputation,type_LR='F',dl_th_together=False):
        self.param:Parameter = None # The global parameter for the model
        self.params:List[Parameter]=[]
        self.comm=comm
        self.rank=comm.Get_rank()
        self.size=comm.Get_size() # The number of workers+1 (server)
        self.J=self.size-1 # The number of workers
        #get the rank of local workers: from 0 to self.size, except self rank
        self.worker_ranks = [rank for rank in range(self.size) if rank != self.rank]        
        self.local_quantities:LocalQuantities=LocalQuantities() # The local quantities (in difference form) received from the workers and group by the 
        self.global_quantities:Dict[StepType,Dict[str,torch.Tensor]]={} # The global quantities that are computed from the local quantities
        global_computation.reset()
        self.global_computation=global_computation
        
        if type_LR in ['F', 'P', 'H']:
            self.type_LR=type_LR
            self.global_computation.type_LR=self.type_LR
        else:
            raise(ValueError("Invalid type_LR. Choose from 'F', 'P', or 'H'."))
        
        self.dl_th_together=dl_th_together # if True, the delta and theta are updated together, otherwise, they are updated separately
        self.choice_dict={}
    def initial_step(self,step: StepType=None):
        if step is None:
            step=self.param.index[1]
        elif self.param.index[1] is None:
            self.param.index[1]=step
        elif step!=self.param.index[1]:
            warn("the step is not consistent with the step in parameter", category=None, stacklevel=1)
        local_quantities_list = []
        
        
        self.comm.bcast({"message":"compute_local_quantity","param":self.param,"step":step}, root=self.rank)
        local_quantity=None
        gathered_local_quantities=self.comm.gather(local_quantity, root=self.rank)
        local_quantities_list=[local_quantity for local_quantity in gathered_local_quantities if local_quantity is not None]
        

        local_quantities = LocalQuantities(local_quantities_list)
        local_quantities_step = local_quantities.get_by_step(step)
        step_keys_dict={StepType.MU_SIGMA:['mu','sigma'],StepType.GAMMA:['gamma'],StepType.DELTA:['delta'],StepType.THETA:['grad','hessian'],StepType.DELTA_THETA:['grad','hessian']} 
        if self.type_LR=='P' or self.type_LR=='H':
            step_keys_dict[StepType.GAMMA].append('XX')
            if not self.dl_th_together:
                step_keys_dict[StepType.DELTA].append('tr_inv')
                
        sum_dict = {key: sum(lq.value[key] for lq in local_quantities_step.values()) 
                    for key in step_keys_dict[step]}
        n = len(local_quantities_step)
        global_quantity_step = {key: value / n for key, value in sum_dict.items()}
        self.global_quantities[step]=global_quantity_step
        # Update parameters
        step_func_dict={StepType.MU_SIGMA:self.global_computation.update_mu_Sigma,StepType.GAMMA:self.global_computation.update_gamma,StepType.DELTA:self.global_computation.update_delta,StepType.THETA:self.global_computation.update_theta,StepType.DELTA_THETA:self.global_computation.update_delta_theta}
        self.param=step_func_dict[step](self.param,global_quantity_step)
        
    def initialization(self,param0: Parameter,params_path:str=None):
        #initialize the global parameter and the local quantities
        information={'type_LR':self.type_LR,'message':'set_type_LR'}
        self.comm.bcast(information, root=self.rank)
        
        if params_path and os.path.exists(params_path):
            with open(params_path, 'rb') as f:
                params = pickle.load(f)
        else:       
            #send param0
            information={'param0':param0,'message':'loc_opt'}
            self.comm.bcast(information, root=self.rank)
            initial_param=None
            gathered_initial_params=self.comm.gather(initial_param, root=self.rank)
            params=[initial_param for initial_param in gathered_initial_params if initial_param is not None]
            #save the resutls
            if params_path:
                with open(params_path, 'wb') as f:
                    pickle.dump(params, f)
            
        self.param =compute_average_param(params)
        self.param.index=(0,StepType.MU_SIGMA,0)
        #initialize the global quantities that are independent of the parameter
        if self.type_LR=='F':
            keys={'XX','Xz','n'} #n is the average local sample size
        elif self.type_LR=='P' or self.type_LR=='H':
            keys={'n'} #n is the average local sample size
        para_ind_quantity=None
        information={'message':'compute_para_ind_quantity'}
        self.comm.bcast(information, root=self.rank)
        gathered_para_ind_quantities=self.comm.gather(para_ind_quantity, root=self.rank)
        gathered_para_ind_quantities=[para_ind_quantity for para_ind_quantity in gathered_para_ind_quantities if para_ind_quantity is not None]
        sum_dict={key:sum(para_ind_quantity[key] for para_ind_quantity in gathered_para_ind_quantities) for key in keys}
        global_para_ind_quantities={key: value / self.J for key, value in sum_dict.items()}
        global_para_ind_quantities['J']=self.J
        self.global_computation.set_global_para_ind_quantities(global_para_ind_quantities)
        
        #update mu and sigma
        self.initial_step(StepType.MU_SIGMA)
        self.param.index=(0,StepType.GAMMA,0) 
        #update gamma
        self.initial_step(StepType.GAMMA)
        if self.dl_th_together:
            self.param.index=(0,StepType.DELTA_THETA,0)
            S=5
            for s in range(S):
                self.initial_step(StepType.DELTA_THETA)
                self.param.index=(0,StepType.DELTA_THETA,s+1)
        else:
            self.param.index=(0,StepType.DELTA,0)
            #update delta
            self.initial_step(StepType.DELTA)
            self.param.index=(0,StepType.THETA,0)
            #update theta
            S=5
            for s in range(S):
                self.initial_step(StepType.THETA)
                self.param.index=(0,StepType.THETA,s+1)
                
        self.param.index=(1,StepType.MU_SIGMA,0)
        self.comm.bcast({"message":"stop","param":self.param}, root=self.rank)
    def save(self,filename:str):
        #save self.param, self.global_quantities, self.global_computation.global_para_ind_quantities
        with open(filename, 'wb') as f:
            pickle.dump({'param': self.param, 'global_quantities': self.global_quantities, 'global_para_ind_quantities': self.global_computation.global_para_ind_quantities}, f)
     
class  Server():
    def __init__(self,comm: MPI.Intracomm,concurrency:int,global_computation:GlobalComputation,logger: logging.Logger,theta_logger: logging.Logger,type_LR='F',dl_th_together=False,iflog=True):
        self.param:Parameter = None # The global parameter for the model
        self.params:List[Parameter]=[]
        self.params_GDT:List[Parameter]=[] # The global parameters for the model
        self.comm=comm
        self.rank=comm.Get_rank()
        self.size=comm.Get_size() # The number of workers+1 (server)
        self.J=self.size-1 # The number of workers
        #get the rank of local workers: from 0 to self.size, except self rank
        self.worker_ranks = [rank for rank in range(self.size) if rank != self.rank]        
        self.local_quantities:LocalQuantities=LocalQuantities() # The local quantities (in difference form) received from the workers and group by the 
        self.global_quantities:Dict[StepType,Dict[str,torch.Tensor]]={} # The global quantities that are computed from the local quantities
        self.concurrency=concurrency # The number of workers that are needed to compute the global quantities
        global_computation.reset()
        self.global_computation=global_computation
        
        
        self.condition_receive=threading.Condition()
        self.condition_send=threading.Condition()
        self.params_index:Dict[int,int]={worker_rank:0 for worker_rank in self.worker_ranks}
        self.stop_flag = threading.Event()
        if iflog:
            self.logger=logger
        else:
            self.logger=None
        self.theta_logger=theta_logger
        if type_LR in ['F', 'P', 'H']:
            self.type_LR=type_LR
            self.global_computation.type_LR=self.type_LR
        else:
            raise(ValueError("Invalid type_LR. Choose from 'F', 'P', or 'H'."))
        
        self.dl_th_together=dl_th_together # if True, the delta and theta are updated together, otherwise, they are updated separately
        self.choice_dict={}
        
    def initialization(self,initialization_server_path:str):
        with open(initialization_server_path, 'rb') as f:
            initialization_server_dict:Dict= pickle.load(f)
        self.param=initialization_server_dict['param']
            
        self.global_quantities=initialization_server_dict['global_quantities']
        self.global_computation.set_global_para_ind_quantities(initialization_server_dict['global_para_ind_quantities'])
        
    def update_parameter_by_step(self,step:StepType):
        with self.condition_receive:
            time_start=time.time()
            while self.local_quantities.get_count_by_step(step)<self.concurrency:
                wait_result = self.condition_receive.wait(timeout=600)  # 5 minutes timeout
                if not wait_result:
                    raise(f"Timeout waiting for waiting local quantities for step {step}")
            local_quantities_by_step=self.local_quantities.get_by_step(step)
            self.local_quantities.remove_by_step(step)  
            time_end=time.time()
            if self.logger is not None:
                self.logger.info(f"server spends {time_end-time_start} seconds for waiting {self.concurrency} local quantities for step {step}")
        
        #take a summation of local_quantities_by_step, then get the global quantity
        step_keys_dict = {
            StepType.MU_SIGMA: ['mu', 'sigma'],
            StepType.GAMMA: ['gamma'],
            StepType.DELTA: ['delta'],
            StepType.THETA: ['grad', 'hessian'],
            StepType.DELTA_THETA: ['grad', 'hessian']
        }
        
        if self.type_LR=='P' or self.type_LR=='H':
            step_keys_dict[StepType.GAMMA].append('XX')
            if not self.dl_th_together:
                step_keys_dict[StepType.DELTA].append('tr_inv')
            
        # Initialize sum dictionary
        sum_dict = {key: 0 for key in step_keys_dict[step]}

        # Sum up all values for each key
        for work_id in local_quantities_by_step.keys():
            local_quantity = local_quantities_by_step[work_id]
            for key in step_keys_dict[step]:
                sum_dict[key] += local_quantity.value[key]
        #     end_time_compuations.append(local_quantity.value['end_time_compuation'])
        # max_time_compuation=max(end_time_compuations)
        # if max_time_compuation>start_time:
        #     time_for_computation=max_time_compuation-start_time
        # else:
        #     time_for_computation=0
        # time_for_commmunication=end_time-max_time_compuation
        # self.logger.info(f"server spends {time_for_computation} (end_time_compuations:{end_time_compuations}) seconds for waiting local computation finished and {time_for_commmunication} seconds for remaining communication")
        
        for key in step_keys_dict[step]:
            self.global_quantities[step][key]=self.global_quantities[step][key]+sum_dict[key]/self.J 
        time_start=time.time()
        step_func_dict={StepType.MU_SIGMA:self.global_computation.update_mu_Sigma,StepType.GAMMA:self.global_computation.update_gamma,StepType.DELTA:self.global_computation.update_delta,StepType.THETA:self.global_computation.update_theta,StepType.DELTA_THETA:self.global_computation.update_delta_theta}
        self.param=step_func_dict[step](self.param,self.global_quantities[step])
        time_end=time.time()
        if self.logger is not None:
            self.logger.info(f"server spends {time_end-time_start} seconds for updating the parameter after receiving {self.concurrency} local quantities")
        
   
    # def receive_local_quantity(self):
    #     while not self.stop_flag.is_set():
    #         information=self.comm.recv(source=MPI.ANY_SOURCE)
    #         if information['message']=='local_quantity':
    #             local_quantity:LocalQuantity=information['param']
    #         else: #other message from the worker
    #             return
    #         self.logger.info(f"server receives the local quantity {local_quantity.index}")
    #         with self.condition_receive:
    #             self.local_quantities.merge_new_dif(local_quantity)
    #             self.condition_receive.notify()
                
    def receive_local_quantity_from_worker(self,worker_rank):
        while not self.stop_flag.is_set():
            information=self.comm.recv(source=worker_rank)
            if information['message']=='local_quantity':
                local_quantity:LocalQuantity=information['param']
                recieve_time=time.time()
                local_quantity.value['recieve_time']=recieve_time
            else: #other message from the worker
                return
            #self.logger.info(f"server receives the local quantity {local_quantity.index}")
            with self.condition_receive:
                self.local_quantities.merge_new_dif(local_quantity)
                self.condition_receive.notify_all()
    
    def send_param_to_worker(self,worker_rank):
            while not self.stop_flag.is_set():
                with self.condition_send:
                    index=self.params_index[worker_rank]
                    while index>=len(self.params):
                        # wait for new param to be added, then check if the number is sufficient
                        wait_result=self.condition_send.wait(timeout=600)  # 5 minutes timeout
                        if not wait_result:
                            if self.logger is not None:
                                self.logger.info(f"Timeout waiting for send param to worker {worker_rank}")
                        if self.stop_flag.is_set():  # Check again after waking up
                            information={'message':'stop'}
                            self.comm.send(information,dest=worker_rank)
                            return 
                    param=self.params[index]
                    information={'message':'compute_local_quantity','param':param}
                    # this should be put inside self.condition_send since other othread can modify it
                    self.params_index[worker_rank]=index+1
                t=param.index[0]
                choice=self.choice_dict[t]
                if worker_rank in choice:
                    self.comm.send(information,dest=worker_rank) 
                    #self.logger.info(f"server sends the parameter {param.index} to worker {worker_rank}")  
            information={'message':'stop'}
            self.comm.send(information,dest=worker_rank)
    def update_parameter(self,T,S,ratio=1,inner_precision=1e-5):
        times_elapsed = [0] # start from 0
        params_GDT=[]
        params_GDT.append(Parameter(None,None,self.param.gamma,self.param.delta,self.param.theta)) # add the initial parameter
        start_time_g = time.time()
        for t in range(T):
            choice=random.sample(self.worker_ranks, int(ratio*len(self.worker_ranks)))
            #sort choices
            choice.sort()
            self.choice_dict[t+1]=choice
            if self.logger is not None:
                self.logger.info(f"iteration:{t}")
            self.theta_logger.info(f"iteration:{t}")
            # step MU_SIGMA
            #time_start=time.time()
            self.update_parameter_by_step(StepType.MU_SIGMA) 
            #time_end=time.time()
            #self.theta_logger.info(f"time for mu_sigma:{time_end-time_start}")
            self.param.index=(t+1,StepType.GAMMA,0) 
            param=Parameter(mu=self.param.mu,sigma=self.param.sigma,index=self.param.index) 
            
            with self.condition_send:
                self.params.append(param)
                self.condition_send.notify_all()
            # step GAMMA
            #time_start=time.time()
            self.update_parameter_by_step(StepType.GAMMA)
            #time_end=time.time()
            #self.theta_logger.info(f"time for gamma:{time_end-time_start}")
            if self.dl_th_together:
                self.param.index=(t+1,StepType.DELTA_THETA,0)
                param=Parameter(gamma=self.param.gamma,index=self.param.index)
                with self.condition_send:
                    self.params.append(param)
                    self.condition_send.notify_all()
                # step DELTA_THETA
                for s in range(S):
                    old_theta = self.param.theta.clone()
                    old_delta = self.param.delta.clone()
                    
                    self.update_parameter_by_step(StepType.DELTA_THETA)
                    self.theta_logger.info(f"theta:{self.param.theta.tolist()}")
                    theta_change = (self.param.theta - old_theta).norm()
                    delta_change = (self.param.delta - old_delta).norm()
                    if theta_change < inner_precision and delta_change < inner_precision:
                        if self.logger is not None:
                            self.logger.info(f"Early stopping at step {s} due to small changes")
                        self.param.index = (t+2, StepType.MU_SIGMA, 0)
                        param=Parameter(delta=self.param.delta,theta=self.param.theta,index=self.param.index)
                        with self.condition_send:
                            self.params.append(param)
                            self.condition_send.notify_all()
                        break
                    else:
                        if s<S-1:
                            self.param.index=(t+1,StepType.DELTA_THETA,s+1)  
                        else: 
                            self.param.index=(t+2,StepType.MU_SIGMA,0)
                        param=Parameter(delta=self.param.delta,theta=self.param.theta,index=self.param.index)
                        with self.condition_send:
                            self.params.append(param)
                            self.condition_send.notify_all()
            else:
                self.param.index=(t+1,StepType.DELTA,0)
                param=Parameter(gamma=self.param.gamma,index=self.param.index)
                with self.condition_send:
                    self.params.append(param)
                    self.condition_send.notify_all()
                # step DELTA
                self.update_parameter_by_step(StepType.DELTA)
                self.param.index=(t+1,StepType.THETA,0)
                param=Parameter(delta=self.param.delta,theta=self.param.theta,index=self.param.index)  # Add theta
                with self.condition_send:
                    self.params.append(param)  # Use param instead of self.param
                    self.condition_send.notify_all()
                # step THETA
                for s in range(S):
                    old_theta = self.param.theta.clone()
                    self.update_parameter_by_step(StepType.THETA)
                    theta_change = (self.param.theta - old_theta).norm()
                    if self.logger is not None:
                        self.logger.info(f"theta:{self.param.theta.tolist()}")
                    self.theta_logger.info(f"theta:{self.param.theta.tolist()}")
                    if theta_change < inner_precision:
                        if self.logger is not None:
                            self.logger.info(f"Early stopping at step {s} due to small changes")
                        self.param.index = (t+2, StepType.MU_SIGMA, 0)
                        param=Parameter(theta=self.param.theta,index=self.param.index)
                        with self.condition_send:
                            self.params.append(param)
                            self.condition_send.notify_all()
                        break
                    else:
                        if s<S-1:
                            self.param.index=(t+1,StepType.THETA,s+1)  
                        else: 
                            self.param.index=(t+2,StepType.MU_SIGMA,0)
                        param=Parameter(theta=self.param.theta,index=self.param.index) # just broadcast the updated parts   
                        with self.condition_send:
                            self.params.append(param)
                            self.condition_send.notify_all()
            times_elapsed.append(time.time() - start_time_g)
            params_GDT.append(Parameter(None,None,self.param.gamma,self.param.delta,self.param.theta))
        
        self.times_elapsed = times_elapsed
        self.params_GDT=params_GDT
        self.stop_flag.set() 
        with self.condition_send:
            self.condition_send.notify_all()
          
    def start(self,T:int,S:int,initialization_server_path:str,ratio=1,inner_precision=1e-5):
        try:
            self.initialization(initialization_server_path)
            #receive_thread=threading.Thread(target=self.receive_local_quantity)
            time_start=time.time()
            receive_threads:List[threading.Thread]=[]
            for work_rank in self.worker_ranks:
                receive_thread=threading.Thread(target=self.receive_local_quantity_from_worker,args=(work_rank,))
                receive_threads.append(receive_thread)
            update_thread=threading.Thread(target=self.update_parameter,args=(T,S,ratio,inner_precision))
            send_threads:List[threading.Thread]=[]
            for work_rank in self.worker_ranks:
                send_thread=threading.Thread(target=self.send_param_to_worker,args=(work_rank,))
                send_threads.append(send_thread)
            for receive_thread in receive_threads:
                receive_thread.start()
            update_thread.start()
            for send_thread in send_threads:
                send_thread.start()
            
            for receive_thread in receive_threads:
                receive_thread.join()
            update_thread.join()
            for send_thread in send_threads:
                send_thread.join()
            time_end=time.time()
            if self.logger is not None:
                self.logger.info(f"Server finishes the later iterations in {time_end-time_start} seconds")
        except Exception as e:
            logging.error(f"Error in server execution: {e}")
            logging.error("Detailed traceback:\n" + traceback.format_exc())
            self.stop_flag.set()  
            
    def save_params_GDT(self, filename: str):
        # Save self.params using pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.params_GDT, f)
    def save_times_elapsed(self, filename: str):
        # Save self.params using pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.times_elapsed, f)


class NonDisEstimation():
    def  __init__(self,full_data:torch.Tensor,knots:torch.Tensor,\
        kernel:Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],type_LR='F'):
        self.__computation=LocalComputation(full_data,knots,kernel,type_LR)
    def get_minimizer(self,param0:Parameter,tol=None):
        param=self.__computation.get_local_minimizer(param0,tol)
        return param
        
    
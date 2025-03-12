import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from collections import deque
from typing import Deque,Dict,List,Tuple,Callable
from enum import Enum
import torch
from scipy.optimize import minimize
from src.utils import softplus, inv_softplus
from warnings import warn
import math
import pickle
import psutil
import logging
#from torch.autograd.functional import hessian
#from torch.autograd import grad


class StepType(Enum):
    MU_SIGMA = 1
    GAMMA = 2
    DELTA = 3
    THETA = 4 # The step for updating theta
    #CHECK_THETA = 5 # The step for checking theta


class Parameter():
    def __init__(self, mu:torch.Tensor,sigma:torch.Tensor,gamma:torch.Tensor,delta:torch.Tensor,theta:torch.Tensor,index:Tuple[int,StepType,int]=None):
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
    
    def get_by_worker(self,worker_id:int)->Dict[int,LocalQuantity]:
        return self.local_quantities_dict.get(worker_id,{})
    
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
        kernel:Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]):
        self.local_data=local_data
        self.knots=knots
        self.kernel=kernel
        
    def K_f(self,theta: torch.Tensor)->torch.Tensor:
        K=self.kernel(self.knots, self.knots, theta)
        return K
    
    def local_B(self,theta: torch.Tensor)->torch.Tensor:
        K=self.K_f(theta)
        invK:torch.Tensor = torch.linalg.inv(K)
        local_locs = self.local_data[:, :2]
        local_B = self.kernel(local_locs, self.knots, theta) @ invK
        return local_B
    
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
    def compute_local_for_mu_sigma(self,para:Parameter)->Dict[str,torch.Tensor]:
        
        theta=para.theta
        gamma=para.gamma   
        local_B = self.local_B(theta)
        local_for_sigma=local_B.T@local_B
        local_errorv = self.local_errorv(gamma)
        local_for_mu=local_B.T@local_errorv
        
        value={}
        value['mu']=local_for_mu
        value['sigma']=local_for_sigma
        
        return value
    
    def compute_local_for_gamma(self,para:Parameter)->Dict[str,torch.Tensor]:
        theta=para.theta
        mu=para.mu
        local_X =self.local_data[:, 3:]
        local_B = self.local_B(theta)
        local_for_gamma=local_X.T @ local_B @ mu  
        value={}
        value['gamma']=local_for_gamma
        return value
    
    def compute_local_for_delta(self,para:Parameter)->Dict[str,torch.Tensor]:
        theta=para.theta
        gamma=para.gamma   
        mu=para.mu
        sigma=para.sigma
        local_B =self.local_B(theta) 
        local_errorv =self.local_errorv(gamma) 
        term1 = torch.trace(local_B.T @ local_B @
                                    (sigma + mu @ mu.T))
        term2 = 2 * local_errorv.T @ local_B @ mu
        term3 = local_errorv.T @ local_errorv
        local_for_delta=term1+term2+term3
        value={}
        value['delta']=local_for_delta

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

        local_B =self.local_B(theta) 
        local_errorv=self.local_errorv(gamma)
        
        
        f_value:torch.Tensor = -n * torch.log(delta) + delta * (
            torch.trace(local_B.T @ local_B @ (sigma + mu @ mu.T))
            + 2 * local_errorv.T @ local_B @ mu
            + local_errorv.T @ local_errorv
        )
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
        
        return value  # Return both gradient and Hessian  
   
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
        K=self.K_f(theta)
        invK=torch.linalg.inv(K)
        local_B=self.local_B(theta)
        tempM = invK+delta*local_B.T@local_B
        local_errorv=self.local_errorv(gamma)
        f_value:torch.Tensor = torch.logdet(tempM)+torch.logdet(K)-n*torch.log(delta)+delta*(
            local_errorv.T)@local_errorv-delta**2*(local_errorv.T@local_B@torch.linalg.inv(tempM)@local_B.T@local_errorv)
        f_value = f_value/n
        
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
        else:
            result = minimize(fun=nllikf,
                            x0=free_vector_GDT0,
                            method="BFGS",
                            jac=nllikgf)
        minimizer_lik = torch.tensor(result.x, dtype=torch.float64)
        param= self.__FreeVector2Parameter_GDT(minimizer_lik)
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
        kernel:Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],step_size_theta:float):
        self.knots=knots
        self.kernel=kernel
        self.step_size_theta=step_size_theta
        self.global_para_ind_quantities:Dict[str,torch.Tensor]={}
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
        grad = theta.grad.clone()
        hessian = torch.zeros((theta.numel(), theta.numel()), dtype=torch.float64)
        for i in range(theta.numel()):
            g2 = torch.autograd.grad(
                grad[i], 
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
        global_XX=self.global_para_ind_quantities['XX']
        global_Xz=self.global_para_ind_quantities['Xz']
        gamma=torch.linalg.inv(global_XX)@(global_Xz-global_for_gamma)
        param=Parameter(param.mu,param.sigma,gamma,param.delta,param.theta)
        return param
    
    def update_delta(self,param: Parameter,global_quantity_delta:Dict[str,torch.Tensor]):
        global_for_delta=global_quantity_delta['delta']
        n=self.global_para_ind_quantities['n']
        delta=n/global_for_delta
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
        step_size=self.step_size_theta
        
        invh_m_grad = torch.linalg.inv(modified_hess)@gradient
        theta = param.theta-step_size*invh_m_grad
        param=Parameter(param.mu,param.sigma,param.gamma,param.delta,theta)
        return param
 


class Worker():
    def __init__(self, worker_id:int, local_computation: LocalComputation, time_dict:Dict[StepType,float]):
        self.worker_id = worker_id
        self.time_dict = time_dict #the time needed for finishing the C&C for the upcoming parameter
        self.cur_param=None
        self.local_params: Deque[Parameter]= deque()  # is a queue of local parameters, in the principal of "first in and first out"
        self.coming_time=None
        self.pre_local_quantity_dict:Dict[StepType,LocalQuantity]={}
        self.local_computation=local_computation
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
        
    def get_refresh_coming_time(self):
        #the function that sets the C&C time for the upcoming parameter, which is just for simulation, in real world, it should be unknown until the C&C is finished
        if self.cur_param: 
            param=self.cur_param 
            time=self.time_dict[param.index[1]] # maybe plus some random part and the seed should be fixed 
        else:
            time=np.inf
        return time
    def receive_param(self, param:Parameter):
        #when a new parameter is received, add it to the local parameters
        id=None
        if self.local_params is not None and len(self.local_params)>0:
            for i, local_param in enumerate(self.local_params):
                if local_param.index[1]==param.index[1]:
                    id=i
        if id is None:
            self.local_params.append(param)
        else:
            self.local_params[id]=param
        #if the current parameter is None, set the current parameter as the first parameter in the deque
        if self.cur_param is None:
            self.set_cur_param() 
            self.coming_time=self.get_refresh_coming_time()  
            
    def set_cur_param(self):
        if self.local_params:
            self.cur_param=self.local_params.popleft()
        else:
            self.cur_param=None
            
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
        return LocalQuantity(self.worker_id, value, param.index)
    
    def compute_new_local_quantity(self)->LocalQuantity:
        if self.cur_param:
            local_quantity=self.compute_local_quantity(self.cur_param)
            return local_quantity
    def compute_para_ind_local_quantity(self)->Dict[str,torch.Tensor]: 
        #compute local quantity that is independent of the parameter
        para_ind_local_quantity={}
        para_ind_local_quantity['XX']=self.local_computation.local_XX()
        para_ind_local_quantity['Xz']=self.local_computation.local_Xz()
        para_ind_local_quantity['n']=self.local_computation.local_n()
        return para_ind_local_quantity
    
    def compute_dif_and_update_pre(self)->LocalQuantity:
        local_quantity=self.compute_new_local_quantity()
        step=local_quantity.index[1]
        dif_value={}
        for key in local_quantity.value.keys():
            dif_value[key]=local_quantity.value[key]-self.pre_local_quantity_dict[step].value[key]
        dif_quantity=LocalQuantity(local_quantity.worker_id,dif_value,local_quantity.index)
        self.set_pre_local_quantity(local_quantity)
        return dif_quantity         
        
class  Server():
    def __init__(self,workers:List[Worker],concurrency:int,global_computation:GlobalComputation):
        self.param:Parameter = None # The global parameter for the model
        self.params_GDT:List[Parameter]=[] # The global parameters for the model
        for w in workers:
            w.reset()
        self.workers=workers # The workers in the system
        self.local_quantities:LocalQuantities=LocalQuantities() # The local quantities (in difference form) received from the workers and group by the 
        self.global_quantities:Dict[StepType,Dict[str,torch.Tensor]]={} # The global quantities that are computed from the local quantities
        self.concurrency=concurrency # The number of workers that are needed to compute the global quantities
        global_computation.reset()
        self.global_computation=global_computation
        self.time:float=0 # record the accumulated time
        self.times_elapsed:List[float]=[] # record the time for each iteration 
        #self.nums_local_quantities:Dict[int,Dict[StepType,int]]={} # record the number of local quantities that the server used for each step
        #self.cur_num_local_quantities:Dict[StepType,int]={}
        #self.recording_delay:Dict[Tuple[int,StepType,int],Tuple[Dict[int,Tuple[int,int]],Dict[int,Tuple[int,int]]]]={}
        #{(t,step,s):(new, old)} with new={worker_id:(t,s)} and old={worker_id:(t,s)}
        
    def initial_step(self,step: StepType=None):
        if step is None:
            step=self.param.index[1]
        elif self.param.index[1] is None:
            self.param.index[1]=step
        elif step!=self.param.index[1]:
            warn("the step is not consistent with the step in parameter", category=None, stacklevel=1)
        local_quantities_list = []
        for w in self.workers:
            local_quantity = w.compute_local_quantity(self.param, step)
            w.set_pre_local_quantity(local_quantity)
            local_quantities_list.append(local_quantity)

        local_quantities = LocalQuantities(local_quantities_list)
        local_quantities_step = local_quantities.get_by_step(step)
        step_keys_dict={StepType.MU_SIGMA:['mu','sigma'],StepType.GAMMA:['gamma'],StepType.DELTA:['delta'],StepType.THETA:['grad','hessian']}        
        sum_dict = {key: sum(lq.value[key] for lq in local_quantities_step.values()) 
                    for key in step_keys_dict[step]}
        n = len(local_quantities_step)
        global_quantity_step = {key: value / n for key, value in sum_dict.items()}
        self.global_quantities[step]=global_quantity_step
        # Update parameters
        step_func_dict={StepType.MU_SIGMA:self.global_computation.update_mu_Sigma,StepType.GAMMA:self.global_computation.update_gamma,StepType.DELTA:self.global_computation.update_delta,StepType.THETA:self.global_computation.update_theta}
        self.param=step_func_dict[step](self.param,global_quantity_step)
        
    def initialization(self,param0: Parameter,params_path:str=None):
        #initialize the global parameter and the local quantities
        if params_path: 
            if os.path.exists(params_path):
                with open(params_path, 'rb') as f:
                    params = pickle.load(f)
            else:
                params = [w.get_initial_param(param0) for w in self.workers]
                with open(params_path, 'wb') as f:
                    pickle.dump(params, f)
        self.param =compute_average_param(params)
        self.param.index=(0,StepType.MU_SIGMA,0)
        self.params_GDT.append(Parameter(None,None,self.param.gamma,self.param.delta,self.param.theta))
        #initialize the global quantities that are independent of the parameter
        keys={'XX','Xz','n'} #n is the average local sample size
        sum_dict={key:sum(worker.compute_para_ind_local_quantity()[key] for worker in self.workers) for key in keys}
        global_para_ind_quantities={key: value / len(self.workers) for key, value in sum_dict.items()}
        global_para_ind_quantities['J']=len(self.workers)
        self.global_computation.set_global_para_ind_quantities(global_para_ind_quantities)
        
        #update mu and sigma
        self.initial_step(StepType.MU_SIGMA)
        self.param.index=(0,StepType.GAMMA,0) 
        #update gamma
        self.initial_step(StepType.GAMMA)
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
        self.broadcast_param(self.param)
        
        
          
    def get_local_quantities_by_step(self,step:StepType):
        #update the global parameter by the local quantities
        #Find the corresponding local quantities that stored in local_quantities, if there aren't any or the number is not enough, wait for the until the condition is satisfied
        count_by_step=self.local_quantities.get_count_by_step(step)
        
        if count_by_step>=self.concurrency:
            # the condition is satisfied already
            local_quantities_by_step=self.local_quantities.get_by_step(step) 
            self.local_quantities.remove_by_step(step)
            
        else:
            # if not, we should wait for the condition to be satisfied by get from local workers
            #time_min_index=self.time_min_index
            #coming_times=self.coming_times # the time needed for finishing the C&C for the upcoming parameter for all workers        
            while count_by_step<self.concurrency:
                coming_times=torch.Tensor([worker.coming_time for worker in self.workers])
                time_min_index:int=torch.argmin(coming_times)
                LocalQuantity=self.workers[time_min_index].compute_dif_and_update_pre()
                self.local_quantities.merge_new_dif(LocalQuantity)
                count_by_step=self.local_quantities.get_count_by_step(step)

                time_min:float=coming_times[time_min_index]
                self.time=self.time+time_min
                
                for i  in range(len(self.workers)):
                    if i==time_min_index:
                        self.workers[i].set_cur_param()
                        self.workers[i].coming_time=self.workers[i].get_refresh_coming_time()
                    else:
                        self.workers[i].coming_time=self.workers[i].coming_time-time_min
                        
            local_quantities_by_step=self.local_quantities.get_by_step(step)
            self.local_quantities.remove_by_step(step)
            
        return local_quantities_by_step
    
    
    
    def update_parameter_by_step(self,step:StepType):
        local_quantities_by_step=self.get_local_quantities_by_step(step)
        # t=self.param.index[0]
        # s=self.param.index[2]
        
        # self.recording_delay[(t,step,s)]=(dict(),dict()) 
        #previous+local_quantities_by_step, if step is not theta, then previous is 
        
        
        
        #record the number of local quantities that the server used for each step
        # num=len(local_quantities_by_step)
        # #round=self.param.index[0]
        # #self.num_local_quantities[round][step]=num
        # logging.info(f"num_local_quantities:{num}")
        # index_max=None
        # round_itera_sum=0
        # for work_id in local_quantities_by_step.keys():
        #     local_quantity = local_quantities_by_step[work_id]
        #     index=local_quantity.index
        #     if index[0]+index[2]>round_itera_sum:
        #         round_itera_sum=index[0]+index[2]
        #         index_max=index
        # logging.info(f"index_max:{index_max}")
        
        
        #take a summation of local_quantities_by_step, then get the global quantity
        step_keys_dict = {
            StepType.MU_SIGMA: ['mu', 'sigma'],
            StepType.GAMMA: ['gamma'],
            StepType.DELTA: ['delta'],
            StepType.THETA: ['grad', 'hessian']
        }
        # Initialize sum dictionary
        sum_dict = {key: 0 for key in step_keys_dict[step]}
    
        # Sum up all values for each key
        for work_id in local_quantities_by_step.keys():
            local_quantity = local_quantities_by_step[work_id]
            for key in step_keys_dict[step]:
                sum_dict[key] += local_quantity.value[key]
        J=len(self.workers)
        
        for key in step_keys_dict[step]:
            self.global_quantities[step][key]=self.global_quantities[step][key]+sum_dict[key]/J 
        
        step_func_dict={StepType.MU_SIGMA:self.global_computation.update_mu_Sigma,StepType.GAMMA:self.global_computation.update_gamma,StepType.DELTA:self.global_computation.update_delta,StepType.THETA:self.global_computation.update_theta}
        self.param=step_func_dict[step](self.param,self.global_quantities[step])
         
        
         
    def broadcast_param(self,param):
        #broadcast the parameter to all the workers
        for worker in self.workers:
            worker.receive_param(param)
            
        
    def run(self,param0: Parameter,T,S,local_params_path:str=None):
        self.initialization(param0,local_params_path)
        self.params_GDT.append(Parameter(None,None,self.param.gamma,self.param.delta,self.param.theta))
        for t in range(T):
            logging.info(f"iteration:{t}")
            time_start=self.time
            # step MU_SIGMA
            self.update_parameter_by_step(StepType.MU_SIGMA) 
            self.param.index=(t+1,StepType.GAMMA,0)    
            self.broadcast_param(self.param)      
            # step GAMMA
            self.update_parameter_by_step(StepType.GAMMA)
            self.param.index=(t+1,StepType.DELTA,0)
            self.broadcast_param(self.param) 
            # step DELTA
            self.update_parameter_by_step(StepType.DELTA)
            self.param.index=(t+1,StepType.THETA,0)
            self.broadcast_param(self.param) 
            # step THETA
            for s in range(S):
                self.update_parameter_by_step(StepType.THETA)
                logging.info(f"theta:{self.param.theta.tolist()}")
                if s<S-1:
                    self.param.index=(t+1,StepType.THETA,s+1)  
                else: 
                    self.param.index=(t+2,StepType.MU_SIGMA,0)
                self.broadcast_param(self.param)
            
            time_end=self.time
            time_elapsed=time_end-time_start
            #logging.info(f"time elapsed:{time_elapsed}")
            self.times_elapsed.append(time_elapsed)
            
            #logging.info(f"theta:{self.param.theta.tolist()}")
            self.params_GDT.append(Parameter(None,None,self.param.gamma,self.param.delta,self.param.theta))
            
    def save_params_GDT(self, filename: str):
        # Save self.params using pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.params_GDT, f)


class NonDisEstimation():
    def  __init__(self,full_data:torch.Tensor,knots:torch.Tensor,\
        kernel:Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]):
        self.__computation=LocalComputation(full_data,knots,kernel)
    def get_minimizer(self,param0:Parameter,tol=None):
        param=self.__computation.get_local_minimizer(param0,tol)
        return param
        
    
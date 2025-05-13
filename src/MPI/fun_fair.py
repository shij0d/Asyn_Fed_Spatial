#for fair comparison, we should use the optimal step size and inner iterations,
#function: optimal_step_size compute the optimal step size
#function: multiple_inner_iterations compute the results with different number of inner iterations

import subprocess
from pathlib import Path
import pickle
import os
import sys
import time
from typing import List
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import generate_exp_slurm_script,generate_exp_slurm_script_multiple_inner_iterations
from src.MPI.estimation import Parameter



def optimal_step_size(type_LR:str,
                     concurrency: int,
                     worker_numbers: int, 
                     data_dir: str, 
                     result_dir: str,
                     script_dir: str,
                     oracle_estimator_path: str, 
                     iterations: int = 30, 
                     sub_iterations: int = 10,
                     inner_precision: float = 1e-5):
    """
    Find the optimal step size for Newton method based on the oracle estimator.
    
    Args:
        type_LR: type of low-rank models: 'F' for full low-rank, "P" for partial low-rank, "H" for hybrid low-rank
        concurrency: Number of concurrent workers
        worker_numbers: Total number of workers
        data_dir: Directory containing input data
        result_dir: Directory to store results
        script_dir: Directory for SLURM scripts
        oracle_estimator_path: Path to oracle estimator pickle file
        iterations: Number of main iterations
        sub_iterations: Number of sub-iterations per main iteration
        inner_precision: Precision for Newton method
    """
    
    # Load the oracle estimator (ground truth) for comparison
    with open(oracle_estimator_path, 'rb') as f:
        oracle_estimator_list = pickle.load(f)
        
    # SLURM job configuration
    sbatch_dict = {
        'account': 'k10164',
        'partition': 'workq',
        'job_name': 'optimal_step_size',
        'nodes': worker_numbers + 1,  # +1 for the master node
        'ntasks': worker_numbers + 1,
        'cpus_per_task': 192,
        'time': '01:00:00',
        'output': f"{result_dir}/optimal_step_size_exp_%j.out",
        'error': f"{result_dir}/optimal_step_size_exp_%j.err"
    }
    
    # Experiment configuration
    exp_dict = {
        'threads': 192,
        'concurrency': concurrency,
        'data_dir': data_dir,
        'result_dir': result_dir,
        'iterations': iterations,
        'sub_iterations': sub_iterations,
        'type_LR': type_LR,
        'inner_precision': inner_precision
    }
    
    def run(step_size: float):
        """Run a single experiment with given step size and return results."""
        exp_dict['step_size'] = step_size
        exp_dict['result_params_GDT_name'] = f"params_GDT_step_size_{step_size}_concurrency_{concurrency}_type_LR_{type_LR}.pkl"
        exp_dict['result_time_elapsed_name'] = f"times_elapsed_step_size_{step_size}_concurrency_{concurrency}_type_LR_{type_LR}.pkl"
        
        # Generate and submit SLURM job
        slurm_script = generate_exp_slurm_script(sbatch_dict, exp_dict)        
        #check if the script_dir exists
        if not os.path.exists(script_dir):
            os.makedirs(script_dir)
        script_path = Path(script_dir) / "optimal_step_size.slurm"
        script_path.write_text(slurm_script)
        script_path.chmod(0o755)
        print(f"Submitting job with step size {step_size}")
        #check if the result_path exists
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
        print("STDOUT:", result.stdout.strip())
        # Load and return results
        job_id = result.stdout.strip().split()[-1]
        file_path = os.path.join(result_dir, exp_dict['result_params_GDT_name'])
        return job_id, file_path
    
    def distance(params_GDT_list:List[Parameter],oracle_estimator_list:List[Parameter])->float:
        """
        Calculate the distance between the estimated parameters and the oracle estimator.
        """
        # L2 error of the last 5 iterations
        NUM=10
        oracle_estimator:Parameter=oracle_estimator_list[-1]
        dist=0
        for i in range(len(params_GDT_list)-NUM,len(params_GDT_list)):
            param_GDT:Parameter=params_GDT_list[i]
            dist+=torch.sqrt(torch.norm(param_GDT.gamma - oracle_estimator.gamma)**2 + \
                            torch.norm(param_GDT.delta - oracle_estimator.delta)**2 + \
                            torch.norm(param_GDT.theta - oracle_estimator.theta)**2)
        dist=dist/NUM
        dist=float(dist)
        return dist
    def load_params_GDT(params_GDT_path) -> List[Parameter]:
        """Load parameters from a completed job's output file."""
        with open(params_GDT_path, 'rb') as f:
            return pickle.load(f)
    
    # Initial step size based on concurrency and worker numbers
    step_size_cur = round(concurrency/worker_numbers * 0.5, 4)
    step_size_min = round(step_size_cur * 0.5, 4)  # Minimum step size
    step_size_max = round(step_size_cur * 2.0, 4)  # Maximum step size
    GOLDEN_RATIO = (1 + 5**0.5) / 2  # Golden ratio constant
    
    # Initialize dictionaries for storing job and file information
    job_dict = dict()
    file_path_dict = dict()
    dist_dict = dict()
    
    # Run initial points
    step_size_a = round(step_size_max - (step_size_max - step_size_min) / GOLDEN_RATIO, 4)
    step_size_b = round(step_size_min + (step_size_max - step_size_min) / GOLDEN_RATIO, 4)
    
    job_dict['a'], file_path_dict['a'] = run(step_size_a)
    job_dict['b'], file_path_dict['b'] = run(step_size_b)
    
    # Wait for initial jobs to finish
    while True:
        cs = []
        for job_id in job_dict.values():
            result = subprocess.run(["sacct", "-j", job_id, "--format=State", "--noheader"], capture_output=True, text=True)
            state = result.stdout.strip().split('\n')[-1].strip()
            if state in ['COMPLETED', 'COMPLETED+']:
                cs.append(1)
        if len(cs) == len(job_dict):
            break
        time.sleep(10)
    
    # Load initial results
    params_GDT_temp = load_params_GDT(file_path_dict['a'])
    dist_dict['a'] = distance(params_GDT_temp, oracle_estimator_list)
    params_GDT_temp = load_params_GDT(file_path_dict['b'])
    dist_dict['b'] = distance(params_GDT_temp, oracle_estimator_list)
    MAX_ITERATIONS=10
    # Golden ratio search
    for i in range(MAX_ITERATIONS):  # Maximum 10 iterations
        if dist_dict['a'] < dist_dict['b']:
            step_size_max = step_size_b
            step_size_b = step_size_a
            dist_dict['b'] = dist_dict['a']
            step_size_a = round(step_size_max - (step_size_max - step_size_min) / GOLDEN_RATIO, 4)
            job_dict['a'], file_path_dict['a'] = run(step_size_a)
            
            # Wait for job to finish
            while True:
                result = subprocess.run(["sacct", "-j", job_dict['a'], "--format=State", "--noheader"], capture_output=True, text=True)
                state = result.stdout.strip().split('\n')[-1].strip()
                if state in ['COMPLETED', 'COMPLETED+']:
                    break
                time.sleep(10)
            
            params_GDT_temp = load_params_GDT(file_path_dict['a'])
            dist_dict['a'] = distance(params_GDT_temp, oracle_estimator_list)
        else:
            step_size_min = step_size_a
            step_size_a = step_size_b
            dist_dict['a'] = dist_dict['b']
            step_size_b = round(step_size_min + (step_size_max - step_size_min) / GOLDEN_RATIO, 4)
            job_dict['b'], file_path_dict['b'] = run(step_size_b)
            
            # Wait for job to finish
            while True:
                result = subprocess.run(["sacct", "-j", job_dict['b'], "--format=State", "--noheader"], capture_output=True, text=True)
                state = result.stdout.strip().split('\n')[-1].strip()
                if state in ['COMPLETED', 'COMPLETED+']:
                    break
                time.sleep(10)
            
            params_GDT_temp = load_params_GDT(file_path_dict['b'])
            dist_dict['b'] = distance(params_GDT_temp, oracle_estimator_list)
        
        # Check convergence
        if abs(step_size_max - step_size_min) < 0.05:  # Convergence threshold
            optimal_step_size = round((step_size_a + step_size_b) / 2, 4)
            print(f"Optimal step size found: {optimal_step_size}")
            break
    
    if i == MAX_ITERATIONS-1:  # If we reached max iterations
        optimal_step_size = round((step_size_a + step_size_b) / 2, 4)
        print(f"Maximum iterations reached, the optimal step size is: {optimal_step_size}")
    return optimal_step_size
        
      
def multiple_inner_iterations(type_LR:str,
                              step_size:float,
                              sub_iterations_list: List[int],
                              iterations_list: List[int],
                              concurrency:int,
                              worker_numbers: int, 
                              data_dir: str, 
                              result_dir: str,
                              script_dir:str,
                              inner_precision: float = 1e-10): 
    """
    Run with different number of inner iterations for Newtont method with optimal step size
    """
    # SLURM job configuration
    sbatch_dict = {
        'account': 'k10164',
        'partition': 'workq',
        'job_name': 'multiple_inner_iterations',
        'nodes': worker_numbers + 1,  # +1 for the master node
        'ntasks': worker_numbers + 1,
        'cpus_per_task': 192,
        'time': '01:00:00',
        'output': f"{result_dir}/multiple_inner_iterations_%j.out",
        'error': f"{result_dir}/multiple_inner_iterations_%j.err"
    }
    result_params_GDT_name_list=[]
    result_time_elapsed_name_list=[]
    for sub_iteration in sub_iterations_list:
        result_params_GDT_name_list.append(f"params_GDT_sub_iterations_{sub_iteration}_concurrency_{concurrency}_type_LR_{type_LR}.pkl")
        result_time_elapsed_name_list.append(f"times_elapsed_sub_iterations_{sub_iteration}_concurrency_{concurrency}_type_LR_{type_LR}.pkl")
    # Experiment configuration
    exp_dict = {
        'threads': 192,
        'concurrency': concurrency,
        'data_dir': data_dir,
        'result_dir': result_dir,
        'iterations_list': iterations_list,
        'type_LR': type_LR,
        'step_size': step_size,
        'inner_precision': inner_precision,
        'result_params_GDT_name_list': result_params_GDT_name_list,
        'result_time_elapsed_name_list': result_time_elapsed_name_list
    }
        
    exp_dict['sub_iterations_list']=sub_iterations_list 
    slurm_script=generate_exp_slurm_script_multiple_inner_iterations(sbatch_dict,exp_dict)
    script_path = Path(script_dir) / "multiple_inner_iterations_job.slurm"
    #check if the script_dir exists
    if not os.path.exists(script_dir):
        os.makedirs(script_dir)
    script_path.write_text(slurm_script)
    script_path.chmod(0o755)
    print(f"SLURM script written to: {script_path}")
    #check if the result_dir exists
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result = subprocess.run(
        ["sbatch", str(script_path)], 
        capture_output=True, 
        text=True
    )
    print("STDOUT:", result.stdout.strip())
    #return the job_id
    return result.stdout.strip().split()[-1]
        
        
    

import time

start_time=time.time()
import os
from typing import Dict

from src.MPI.estimation import Server,Worker,LocalComputation,GlobalComputation
import pickle
import logging
from mpi4py import MPI
import argparse
import psutil
import socket
from src.utils import get_logical_cpus_for_physical_cores
end_time=time.time()

comm: MPI.Intracomm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
rank_suffix_map = {rank: i-1 for i, rank in enumerate(range(size))}

server_rank=0
default_num_cpus_list=[192 for _ in range(size)]

def setup_logging(result_dir:str,name:str, file_suffix=None):
    """
    Set up a logger with a dedicated log file
    
    Args:
        name: Name for the logger
        file_suffix: Suffix for the log file name (defaults to name if not provided)
    """
    if file_suffix is None:
        file_suffix = name
        
    log_file = os.path.join(result_dir,'logs', f'{file_suffix}.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create a new logger (don't use basicConfig which affects the root logger)
    new_logger = logging.getLogger(f'Logger-{name}')
    new_logger.setLevel(logging.INFO)
    
    # Remove existing handlers if any
    for handler in new_logger.handlers[:]:
        new_logger.removeHandler(handler)
    
    # Create and add file handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    new_logger.addHandler(file_handler)
    
    # # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    new_logger.addHandler(console_handler)
    
    return new_logger

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run MPI-based federated spatial regression')
    parser.add_argument('--step_size', type=float, default=0.1, 
                        help='Step size for parameter updates')
    parser.add_argument('--concurrency', type=int, default=2, 
                        help='Number of workers to wait for before updates')
    parser.add_argument('--iterations', type=int, default=50, 
                        help='Number of iterations (T)')
    parser.add_argument('--sub_iterations', type=int, default=10, 
                        help='Number of sub-iterations for theta (S)')
    parser.add_argument('--inner_precision', type=float, default=1e-5, 
                        help='Inner precision for theta')
    parser.add_argument('--data_dir', type=str, 
                        help='the directory of the data')
    parser.add_argument('--result_dir', type=str,
                        help='the directory to store the results')
    parser.add_argument('--result_params_GDT_name', type=str, default='None',
                        help='the name of the result file for parameters GDT')
    parser.add_argument('--result_time_elapsed_name', type=str, default='None',
                        help='the name of the result file for tume elapsed')
    parser.add_argument('--type_LR', type=str, default='P',
                        help='the type of the local-rank models')
    #the number of cpus for each rank, it is a list of integers
    parser.add_argument('--num_cpus_list', type=str, default=str(default_num_cpus_list),
                        help='the number of physical cores for each rank as a string list, e.g. "[192,192,192]"')
    
    # Parse arguments on rank 0 only to avoid conflicts
    if MPI.COMM_WORLD.Get_rank() == 0:
        args = parser.parse_args()
    else:
        args = None
    # Broadcast arguments from rank 0 to all processes
    args = MPI.COMM_WORLD.bcast(args, root=0)
    return args

# read the arguments from the command line
args=parse_arguments()
step_size = args.step_size
concurrency = args.concurrency
T = args.iterations
S = args.sub_iterations
inner_precision = args.inner_precision
data_dir = args.data_dir
result_dir = args.result_dir
type_LR = args.type_LR
num_cpus_list = args.num_cpus_list
num_cpus_list = eval(num_cpus_list)  # Convert string to list
num_cpus_list = [int(num_cpus) for num_cpus in num_cpus_list]
# the name of the result file for parameters GDT and time elapsed
result_params_GDT_name = args.result_params_GDT_name
result_time_elapsed_name = args.result_time_elapsed_name
if result_params_GDT_name == 'None':
    result_params_GDT_name = f'params_GDT_step_size_{step_size:g}_concurrency_{concurrency}_type_LR_{type_LR}.pkl'
if result_time_elapsed_name == 'None':
    result_time_elapsed_name = f'times_elapsed_step_size_{step_size:g}_concurrency_{concurrency}_type_LR_{type_LR}.pkl'


if type_LR == 'P' or type_LR == 'H':
    dl_th_together_default=True
else:
    dl_th_together_default=False

num_cpus_physical=num_cpus_list[rank]
process = psutil.Process()
logical_cpus=get_logical_cpus_for_physical_cores(list(range(num_cpus_physical)))
'''
process.cpu_affinity(logical_cpus) 
num_cpus_logical=len(logical_cpus)
os.environ['OMP_NUM_THREADS'] = str(num_cpus_logical)
os.environ['MKL_NUM_THREADS'] = str(num_cpus_logical)
os.environ['NUMEXPR_NUM_THREADS'] = str(num_cpus_logical)
os.environ['TORCH_NUM_THREADS'] = str(num_cpus_logical)
'''
if rank==server_rank:
    #logger.info("Initializing server process")
    #read the following variables from the file
    logger = setup_logging(result_dir, 'server', f'rank_{rank}')
    theta_logger = setup_logging(result_dir, 'theta', 'theta_values')
    #logger.disabled=True
    # Get node name mac address
    hostname = socket.gethostname()
    logger.info(f"rank {rank} has {num_cpus_physical} physical cores on node {hostname}")
    
    logger.info(f"******************Step size: {step_size}, Concurrency: {concurrency}, sub_iterations: {S}******************")
    theta_logger.info(f"******************Step size: {step_size}, Concurrency: {concurrency}, sub_iterations: {S}******************")
    
    with open(os.path.join(data_dir, 'data_server.pkl'),'rb') as f:
        data_dict:Dict=pickle.load(f)
    knots=data_dict["knots"]
    kernel_est=data_dict["kernel_est"]
    param0=data_dict["param0"]
    global_computation=GlobalComputation(knots,kernel_est,step_size_inner=step_size)
    server=Server(comm,concurrency=concurrency,global_computation=global_computation,logger=logger,theta_logger=theta_logger,type_LR=type_LR,dl_th_together=dl_th_together_default,iflog=False)
    os.makedirs(result_dir, exist_ok=True)
    params_path=os.path.join(result_dir, 'local_params.pkl')
    server.start(param0=param0,T=T,S=S,local_params_path=params_path,inner_precision=inner_precision,cpu_affinity_cores=logical_cpus)
    server.save_params_GDT(os.path.join(result_dir, result_params_GDT_name))
    server.save_times_elapsed(os.path.join(result_dir, result_time_elapsed_name))
else:
    logger = setup_logging(result_dir, 'worker', f'rank_{rank}')
    hostname = socket.gethostname()
    logger.info(f"rank {rank} has {num_cpus_physical} physical cores on node {hostname}")
    
    #logger.disabled=True
    logger.info(f"******************Step size:{step_size}, Concurrency: {concurrency}******************")
    
    #read the following variables from the file
    worker_file_suffix=rank_suffix_map[rank]
    #logger.info(f"Initializing worker process with worker_id={rank}")
    with open(os.path.join(data_dir, f'data_worker_{worker_file_suffix}.pkl'),'rb') as f:
        data_dict=pickle.load(f)
    local_data=data_dict["local_data"]
    knots=data_dict["knots"]
    kernel_est=data_dict["kernel_est"]
    local_computation=LocalComputation(local_data,knots,kernel_est)
    worker=Worker(comm=comm,server_rank=server_rank,local_computation=local_computation,logger=logger,iflog=True)
    worker.start(cpu_affinity_cores=logical_cpus)




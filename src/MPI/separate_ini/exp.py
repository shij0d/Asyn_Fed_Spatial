import os
from typing import Dict
import subprocess
import pickle
import logging
from mpi4py import MPI
import argparse
import psutil
import socket

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
    parser.add_argument('--initialization_dir', type=str,
                        help='the directory of the initialization')
    parser.add_argument('--result_dir', type=str,
                        help='the directory to store the results')
    parser.add_argument('--result_params_GDT_name', type=str, default='None',
                        help='the name of the result file for parameters GDT')
    parser.add_argument('--result_time_elapsed_name', type=str, default='None',
                        help='the name of the result file for tume elapsed')
    parser.add_argument('--type_LR', type=str, default='P',
                        help='the type of the local-rank models')
    #the number of cpus for each rank, it is a list of integers
    parser.add_argument('--num_cpus_list', type=str, default=str([192 for _ in range(11)]),
                        help='the number of physical cores for each rank as a string list, e.g. "[192,192,192]"')
    parser.add_argument('--threads_setting', type=int, default=0,
                        help='the setting of the threads, each number corresponds to a setting')
    parser.add_argument('--binding_setting', type=int, default=0,
                        help='CPU binding setting: 0=no binding, 1=bind to physical cores, 2=bind to logical cores')
    # Parse arguments on rank 0 only to avoid conflicts
    if MPI.COMM_WORLD.Get_rank() == 0:
        args = parser.parse_args()
    else:
        args = None
    # Broadcast arguments from rank 0 to all processes
    args = MPI.COMM_WORLD.bcast(args, root=0)
    return args

def get_logical_cpus_for_physical_cores(core_ids):
    """
    Given a list of physical core IDs, return a list of logical CPU IDs
    corresponding to all hyperthreads of those cores.
    """
    result = subprocess.run(['lscpu', '-e=CPU,Core'], stdout=subprocess.PIPE, text=True)
    lines = result.stdout.strip().split('\n')[1:]  # Skip header
    core_to_logical = {}

    for line in lines:
        cpu, core = map(int, line.strip().split())
        core_to_logical.setdefault(core, []).append(cpu)

    logical_cpus = []
    for core_id in core_ids:
        logical_cpus.extend(core_to_logical.get(core_id, []))

    return sorted(logical_cpus)


comm: MPI.Intracomm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
rank_suffix_map = {rank: i-1 for i, rank in enumerate(range(size))}

server_rank=0
default_num_cpus_list=[192 for _ in range(size)]
# read the arguments from the command line
args=parse_arguments()
step_size = args.step_size
concurrency = args.concurrency
T = args.iterations
S = args.sub_iterations
inner_precision = args.inner_precision
data_dir = args.data_dir
initialization_dir = args.initialization_dir
result_dir = args.result_dir
type_LR = args.type_LR
num_cpus_list = args.num_cpus_list
num_cpus_list = eval(num_cpus_list)  # Convert string to list
num_cpus_list = [int(num_cpus) for num_cpus in num_cpus_list]
binding_setting = args.binding_setting
# the name of the result file for parameters GDT and time elapsed
result_params_GDT_name = args.result_params_GDT_name
result_time_elapsed_name = args.result_time_elapsed_name
threads_setting = args.threads_setting
if result_params_GDT_name == 'None':
    result_params_GDT_name = f'params_GDT_step_size_{step_size:g}_concurrency_{concurrency}_type_LR_{type_LR}_threads_setting_{threads_setting}.pkl'
if result_time_elapsed_name == 'None':
    result_time_elapsed_name = f'times_elapsed_step_size_{step_size:g}_concurrency_{concurrency}_type_LR_{type_LR}_threads_setting_{threads_setting}.pkl'


if type_LR == 'P' or type_LR == 'H':
    dl_th_together_default=True
else:
    dl_th_together_default=False

num_cpus_physical=num_cpus_list[rank]

if binding_setting == 1:
    #binding to the physical cores
    process = psutil.Process()
    process.cpu_affinity(list(range(num_cpus_physical))) 
elif binding_setting == 2:
    #binding to the logical cores
    process = psutil.Process()
    logical_cpus=get_logical_cpus_for_physical_cores(list(range(num_cpus_physical)))
    #process.cpu_affinity(list(range(num_cpus_physical))) 
    process.cpu_affinity(logical_cpus) 
    #num_cpus_logical=len(logical_cpus)


'''
The setting of the threads where -, represents the default setting., G1 is for the group of more cores, G2 is for the group of less cores.
188 and 5 is use all the cores, 94 and 2 is use half of the cores, 1 and 1 is use 1 thread for each core.
#OMP_NUM_THREADS		MKL_NUM_THREADS		NUMEXPR_MAX_THREADS
G1	G2	G1	G2	-
188	5	-	-	188
188	5	188	5	188
-	-	188	5	188
188	5	1	1	188
1	1	188	5	188
94	2	94	2	188
188	5	-	-	-
188	5	188	5	-
-	-	188	5	-
188	5	1	1	-
1	1	188	5	-
94	2	94	2	-
'''

if threads_setting == 0:
    os.environ['NUMEXPR_MAX_THREADS'] = str(num_cpus_physical)
    os.environ['OMP_NUM_THREADS'] = str(num_cpus_physical)
elif threads_setting == 1:
    os.environ['NUMEXPR_MAX_THREADS'] = str(num_cpus_physical)
    os.environ['OMP_NUM_THREADS'] = str(num_cpus_physical)
    os.environ['MKL_NUM_THREADS'] = str(num_cpus_physical)
elif threads_setting == 2:
    os.environ['NUMEXPR_MAX_THREADS'] = str(num_cpus_physical)
    os.environ['MKL_NUM_THREADS'] = str(num_cpus_physical)
elif threads_setting == 3:
    os.environ['NUMEXPR_MAX_THREADS'] = str(num_cpus_physical)
    os.environ['OMP_NUM_THREADS'] = str(num_cpus_physical)
    os.environ['MKL_NUM_THREADS'] = '1'
elif threads_setting == 4:
    os.environ['NUMEXPR_MAX_THREADS'] = str(num_cpus_physical)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = str(num_cpus_physical)
elif threads_setting == 5:
    os.environ['NUMEXPR_MAX_THREADS'] = str(num_cpus_physical)
    os.environ['OMP_NUM_THREADS'] = str(num_cpus_physical // 2)
    os.environ['MKL_NUM_THREADS'] = str(num_cpus_physical // 2)
    
elif threads_setting == 6:
    os.environ['OMP_NUM_THREADS'] = str(num_cpus_physical)
elif threads_setting == 7:
    os.environ['OMP_NUM_THREADS'] = str(num_cpus_physical)
    os.environ['MKL_NUM_THREADS'] = str(num_cpus_physical)
elif threads_setting == 8:
    os.environ['MKL_NUM_THREADS'] = str(num_cpus_physical)
elif threads_setting == 9:
    os.environ['OMP_NUM_THREADS'] = str(num_cpus_physical)
    os.environ['MKL_NUM_THREADS'] = '1'
elif threads_setting == 10:
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = str(num_cpus_physical)
elif threads_setting == 11:
    os.environ['OMP_NUM_THREADS'] = str(num_cpus_physical // 2)
    os.environ['MKL_NUM_THREADS'] = str(num_cpus_physical // 2)
    


#load the torch and numpy after the process.cpu_affinity
from src.MPI.separate_ini.estimation import Server,Worker,LocalComputation,GlobalComputation

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
    initialization_server_path=os.path.join(initialization_dir, f'initialization_server.pkl')
    server.start(T=T,S=S,initialization_server_path=initialization_server_path,inner_precision=inner_precision)
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
    local_computation=LocalComputation(local_data,knots,kernel_est,type_LR=type_LR)
    worker=Worker(comm=comm,server_rank=server_rank,local_computation=local_computation,logger=logger,iflog=False)
    initialization_worker_path=os.path.join(initialization_dir, f'initialization_worker_{worker_file_suffix}.pkl')
    worker.start(initialization_worker_path=initialization_worker_path)




import time
import psutil
import os
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#set affinity and number of threads
process = psutil.Process()
process.cpu_affinity([]) 
num_cpus_physical=psutil.cpu_count(logical=False)
os.environ['OMP_NUM_THREADS'] = str(1)
os.environ['MKL_NUM_THREADS'] = str(1)
os.environ['NUMEXPR_NUM_THREADS'] = str(1)



start_time=time.time()

from typing import Dict

from src.MPI.separate_ini.estimation import InitializationServer,InitializationWorker,LocalComputation,GlobalComputation,torch
import pickle
from mpi4py import MPI
import argparse

torch.set_num_threads(num_cpus_physical)


end_time=time.time()

comm: MPI.Intracomm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
rank_suffix_map = {rank: i-1 for i, rank in enumerate(range(size))}

server_rank=0
default_num_cpus_list=[192 for _ in range(size)]


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run MPI-based federated spatial regression')
    parser.add_argument('--step_size', type=float, default=0.5, 
                        help='Step size for parameter updates')
    parser.add_argument('--data_dir', type=str, 
                        help='the directory of the data')
    parser.add_argument('--result_dir', type=str,
                        help='the directory to store the results')
    parser.add_argument('--type_LR', type=str, default='P',
                        help='the type of the local-rank models')
    parser.add_argument('--method', type=str, default='loc_opt',
                        help='the method to initialize the parameters')
    
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
data_dir = args.data_dir
result_dir = args.result_dir
type_LR = args.type_LR
method = args.method

if type_LR == 'P' or type_LR == 'H':
    dl_th_together_default=True
else:
    dl_th_together_default=False

if rank==server_rank:
    
    with open(os.path.join(data_dir, 'data_server.pkl'),'rb') as f:
        data_dict:Dict=pickle.load(f)
    knots=data_dict["knots"]
    kernel_est=data_dict["kernel_est"]
    param0=data_dict["param0"]
    global_computation=GlobalComputation(knots,kernel_est,step_size_inner=step_size)
    server=InitializationServer(comm,global_computation=global_computation,type_LR=type_LR,dl_th_together=dl_th_together_default)
    os.makedirs(result_dir, exist_ok=True)
    #the directory of the result is already created
    logger.info(f"The directory of the result is {result_dir}")
    
    logger.info(f"Initializing server with {server.size} workers")
    params_path=os.path.join(result_dir, 'local_params.pkl')
    server.initialization(param0,params_path,method=method)
    server.save(os.path.join(result_dir, 'initialization_server.pkl'))
    logger.info(f"Server initialization complete")
else:
    
    #read the following variables from the file
    worker_file_suffix=rank_suffix_map[rank]
    #logger.info(f"Initializing worker process with worker_id={rank}")
    with open(os.path.join(data_dir, f'data_worker_{worker_file_suffix}.pkl'),'rb') as f:
        data_dict=pickle.load(f)
    local_data=data_dict["local_data"]
    knots=data_dict["knots"]
    kernel_est=data_dict["kernel_est"]
    local_computation=LocalComputation(local_data,knots,kernel_est)
    worker=InitializationWorker(comm=comm,server_rank=server_rank,local_computation=local_computation)
    worker.initialization()
    worker.save(os.path.join(result_dir, f'initialization_worker_{worker_file_suffix}.pkl'))
    print(f"Worker {rank} initialization complete")



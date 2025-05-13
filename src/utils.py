import torch
from typing import List,Dict
import subprocess
import numpy as np

def softplus(x: torch.Tensor) -> torch.Tensor:
    value=torch.logaddexp(torch.tensor(0), x)
    return value 

def inv_softplus(x: torch.Tensor) -> torch.Tensor:
    # each element of x is required to be positive

    value=x+torch.log(1-torch.exp(-x))
    return value
def inv_softplus(x: torch.Tensor) -> torch.Tensor:
    # each element of x is required to be positive

    value=x+torch.log(1-torch.exp(-x))
    return value
def replace_negative_eigenvalues_with_zero(matrix:torch.Tensor):
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    
    # Replace negative eigenvalues with zero
    eigenvalues = torch.clamp(eigenvalues, min=0)
    
    # Reconstruct the matrix with the modified eigenvalues
    matrix_modified = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
    
    return matrix_modified


def set_times(time_mean_dict:Dict,J)->List[Dict]:
    #absolute value of normal distribution,
    # J is the number of samples
    # mean is the mean of the normal distribution
    time_dicts=[]
    torch.manual_seed(0)
    for j in range(J):
        time_dict={}
        for key in time_mean_dict.keys():
            mean=time_mean_dict[key]
            time_dict[key]=torch.abs(torch.normal(mean,mean/2,(1,)))
        time_dicts.append(time_dict)
    return time_dicts




def generate_exp_slurm_script(sbatch_dict:Dict,exp_dict:Dict) -> str:
    """
    Generate a SLURM script for running distributed experiments.
    
    Args:
        sbatch_dict: Dictionary containing SLURM job parameters
            - account: Account name (default: 'k10164')
            - partition: Partition name (default: 'workq')
            - job_name: Name of the job
            - nodes: Number of nodes (default: 1)
            - ntasks: Number of tasks
            - cpus_per_task: CPUs per task
            - time: Job time limit (default: '01:00:00')
            - output: Output file path
            - error: Error file path
            
        exp_dict: Dictionary containing experiment parameters
            - threads: Number of threads for OMP/MKL
            - concurrency: Number of concurrent workers
            - data_dir: Input data directory
            - result_dir: Output result directory
            - result_params_GDT_name: Name for parameters output file
            - result_time_elapsed_name: Name for timing output file
            - iterations: Number of main iterations
            - sub_iterations: Number of sub-iterations
            - inner_precision: Precision for Newton method  
            - step_size: Step size parameter
            - type_LR: Type of low-rank model ('F', 'P', or 'H')
            
    Returns:
        str: Generated SLURM script
    """
    # Validate required parameters
    required_sbatch = ['job_name', 'ntasks', 'cpus_per_task']
    required_exp = ['threads', 'concurrency', 'data_dir', 'result_dir', 
                   'result_params_GDT_name', 'result_time_elapsed_name',
                   'iterations', 'sub_iterations', 'step_size', 'type_LR',
                   'inner_precision']
    
    for param in required_sbatch:
        if param not in sbatch_dict:
            raise ValueError(f"Missing required sbatch parameter: {param}")
    for param in required_exp:
        if param not in exp_dict:
            raise ValueError(f"Missing required experiment parameter: {param}")
    
    # Extract and validate SLURM parameters with defaults
    account = sbatch_dict.get('account', 'k10164')
    partition = sbatch_dict.get('partition', 'workq')
    job_name = sbatch_dict['job_name']
    nodes = sbatch_dict.get('nodes', 1)
    ntasks = sbatch_dict['ntasks']
    cpus_per_task = sbatch_dict['cpus_per_task']
    time = sbatch_dict.get('time', '01:00:00')
    output = sbatch_dict.get('output', f"{job_name}_%j.out")
    error = sbatch_dict.get('error', f"{job_name}_%j.err")
    
    # Extract experiment parameters
    threads = exp_dict['threads']
    concurrency = exp_dict['concurrency']
    data_dir = exp_dict['data_dir']
    result_dir = exp_dict['result_dir']
    result_params_GDT_name = exp_dict['result_params_GDT_name']
    result_time_elapsed_name = exp_dict['result_time_elapsed_name']
    iterations = exp_dict['iterations']
    sub_iterations = exp_dict['sub_iterations']
    inner_precision = exp_dict['inner_precision']
    step_size = exp_dict['step_size']
    type_LR = exp_dict['type_LR']
    
    # Validate type_LR
    if type_LR not in ['F', 'P', 'H']:
        raise ValueError("type_LR must be one of: 'F', 'P', 'H'")
    
    # Generate SLURM script
    slurm_script = f"""#!/bin/bash
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time}
#SBATCH --output={output}
#SBATCH --error={error}

# Load required modules
module load python/3.10.13
source /scratch/shij0d/venvs/asy_fed_spatial_env/bin/activate
module load libfabric #for mpi

export PYTHONPATH=/scratch/shij0d/projects/Asyn_Fed_Spatial:$PYTHONPATH

# Set thread configuration
export OMP_NUM_THREADS={threads}
export MKL_NUM_THREADS={threads}
export NUMEXPR_NUM_THREADS={threads}

# Print job information
echo "Job ${{SLURM_JOB_ID}} is starting on ${{SLURM_JOB_NODELIST}}"
echo "==============================================="
echo "Experiment Configuration:"
echo "  Step Size: {step_size}"
echo "  Concurrency: {concurrency}"
echo "  Iterations: {iterations}"
echo "  Sub-iterations: {sub_iterations}"
echo "  Type LR: {type_LR}"
echo "==============================================="

# Run the experiment
srun python3 "/scratch/shij0d/projects/Asyn_Fed_Spatial/src/MPI/exp.py" \\
    --step_size "{step_size}" \\
    --concurrency "{concurrency}" \\
    --iterations "{iterations}" \\
    --sub_iterations "{sub_iterations}" \\
    --inner_precision "{inner_precision}" \\
    --data_dir "{data_dir}" \\
    --result_dir "{result_dir}" \\
    --result_params_GDT_name "{result_params_GDT_name}" \\
    --result_time_elapsed_name "{result_time_elapsed_name}" \\
    --type_LR "{type_LR}"
"""
    return slurm_script

#generate_exp_slurm_script with multiple inner iterations
def generate_exp_slurm_script_multiple_inner_iterations(sbatch_dict:Dict,exp_dict:Dict) -> str:
    """
    Generate a SLURM script for running distributed experiments.
    
    Args:
        sbatch_dict: Dictionary containing SLURM job parameters
            - account: Account name (default: 'k10164')
            - partition: Partition name (default: 'workq')
            - job_name: Name of the job
            - nodes: Number of nodes (default: 1)
            - ntasks: Number of tasks
            - cpus_per_task: CPUs per task
            - time: Job time limit (default: '01:00:00')
            - output: Output file path
            - error: Error file path
            
        exp_dict: Dictionary containing experiment parameters
            - threads: Number of threads for OMP/MKL
            - concurrency: Number of concurrent workers
            - data_dir: Input data directory
            - result_dir: Output result directory
            - result_params_GDT_name: Name for parameters output file
            - result_time_elapsed_name: Name for timing output file
            - iterations_list: a list of number of main iterations
            - sub_iterations_list: a list of number of inner iterations
            - inner_precision: Precision for Newton method  
            - step_size: Step size parameter
            - type_LR: Type of low-rank model ('F', 'P', or 'H')
            
    Returns:
        str: Generated SLURM script
    """
    # Validate required parameters
    required_sbatch = ['job_name', 'ntasks', 'cpus_per_task']
    required_exp = ['threads', 'concurrency', 'data_dir', 'result_dir', 
                   'result_params_GDT_name_list', 'result_time_elapsed_name_list',
                   'iterations_list', 'sub_iterations_list', 'step_size', 'type_LR',
                   'inner_precision']
    
    for param in required_sbatch:
        if param not in sbatch_dict:
            raise ValueError(f"Missing required sbatch parameter: {param}")
    for param in required_exp:
        if param not in exp_dict:
            raise ValueError(f"Missing required experiment parameter: {param}")
    
    # Extract and validate SLURM parameters with defaults
    account = sbatch_dict.get('account', 'k10164')
    partition = sbatch_dict.get('partition', 'workq')
    job_name = sbatch_dict['job_name']
    nodes = sbatch_dict.get('nodes', 1)
    ntasks = sbatch_dict['ntasks']
    cpus_per_task = sbatch_dict['cpus_per_task']
    time = sbatch_dict.get('time', '01:00:00')
    output = sbatch_dict.get('output', f"{job_name}_%j.out")
    error = sbatch_dict.get('error', f"{job_name}_%j.err")
    
    # Extract experiment parameters
    threads = exp_dict['threads']
    concurrency = exp_dict['concurrency']
    data_dir = exp_dict['data_dir']
    result_dir = exp_dict['result_dir']
    result_params_GDT_name_list = exp_dict['result_params_GDT_name_list']
    result_time_elapsed_name_list = exp_dict['result_time_elapsed_name_list']
    iterations_list = exp_dict['iterations_list']
    sub_iterations_list = exp_dict['sub_iterations_list']
    inner_precision = exp_dict['inner_precision']
    step_size = exp_dict['step_size']
    type_LR = exp_dict['type_LR']
    
    # Validate type_LR
    if type_LR not in ['F', 'P', 'H']:
        raise ValueError("type_LR must be one of: 'F', 'P', 'H'")
    
    # Generate SLURM script
    slurm_script = f"""#!/bin/bash
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time}
#SBATCH --output={output}
#SBATCH --error={error}

# Load required modules
module load python/3.10.13
source /scratch/shij0d/venvs/asy_fed_spatial_env/bin/activate
module load libfabric #for mpi

export PYTHONPATH=/scratch/shij0d/projects/Asyn_Fed_Spatial:$PYTHONPATH

# Set thread configuration
export OMP_NUM_THREADS={threads}
export MKL_NUM_THREADS={threads}
export NUMEXPR_NUM_THREADS={threads}

# Print job information
echo "Job ${{SLURM_JOB_ID}} is starting on ${{SLURM_JOB_NODELIST}}"
echo "==============================================="
echo "Experiment Configuration:"
echo "  Step Size: {step_size}"
echo "  Concurrency: {concurrency}"
echo "  Iterations_list: {iterations_list}"
echo "  Sub_iterations_list: {sub_iterations_list}"
echo "  Type LR: {type_LR}"
echo "==============================================="

# Convert Python list to bash array of integers
sub_iterations_array=({' '.join(map(str, sub_iterations_list))})
iterations_array=({' '.join(map(str, iterations_list))})
result_params_GDT_name_array=({' '.join(map(str, result_params_GDT_name_list))})
result_time_elapsed_name_array=({' '.join(map(str, result_time_elapsed_name_list))})
# Run experiments for each sub_iterations value
for i in "${{!sub_iterations_array[@]}}"; do
    sub_iterations="${{sub_iterations_array[$i]}}"
    iterations="${{iterations_array[$i]}}"
    result_params_GDT_name="${{result_params_GDT_name_array[$i]}}"
    result_time_elapsed_name="${{result_time_elapsed_name_array[$i]}}"
    echo "Running with sub_iterations: $sub_iterations"
    srun python3 "/scratch/shij0d/projects/Asyn_Fed_Spatial/src/MPI/exp.py" \\
        --step_size "{step_size}" \\
        --concurrency "{concurrency}" \\
        --iterations "$iterations" \\
        --sub_iterations "$sub_iterations" \\
        --inner_precision "{inner_precision}" \\
        --data_dir "{data_dir}" \\
        --result_dir "{result_dir}" \\
        --result_params_GDT_name "$result_params_GDT_name" \\
        --result_time_elapsed_name "$result_time_elapsed_name" \\
        --type_LR "{type_LR}"
done
"""
    return slurm_script



def generate_exp_slurm_script_varying_ncpu(sbatch_dict:Dict,exp_dict:Dict) -> str:
    """
    Generate a SLURM script for running distributed experiments with varying cpus_per_task.
    
    Args:
        sbatch_dict: Dictionary containing SLURM job parameters
            - account: Account name (default: 'k10164')
            - partition: Partition name (default: 'workq')
            - job_name: Name of the job
            - cpus_per_task_list: a list of CPUs (number of cpus) per task
            - time: Job time limit (default: '01:00:00')
            - output: Output file path
            - error: Error file path
            
        exp_dict: Dictionary containing experiment parameters
            - concurrency: Number of concurrent workers
            - data_dir: Input data directory
            - result_dir: Output result directory
            - result_params_GDT_name: Name for parameters output file
            - result_time_elapsed_name: Name for timing output file
            - iterations: Number of main iterations
            - sub_iterations: Number of sub-iterations
            - inner_precision: Precision for Newton method  
            - step_size: Step size parameter
            - type_LR: Type of low-rank model ('F', 'P', or 'H')
            
    Returns:
        str: Generated SLURM script
    """
    # Validate required parameters
    required_sbatch = ['job_name', 'cpus_per_task_list']
    required_exp = ['concurrency', 'data_dir', 'result_dir', 
                   'result_params_GDT_name', 'result_time_elapsed_name',
                   'iterations', 'sub_iterations', 'step_size', 'type_LR',
                   'inner_precision']
    
    for param in required_sbatch:
        if param not in sbatch_dict:
            raise ValueError(f"Missing required sbatch parameter: {param}")
    for param in required_exp:
        if param not in exp_dict:
            raise ValueError(f"Missing required experiment parameter: {param}")
    
    # Extract and validate SLURM parameters with defaults
    account = sbatch_dict.get('account', 'k10164')
    partition = sbatch_dict.get('partition', 'workq')
    job_name = sbatch_dict['job_name']
    cpus_per_task_list = sbatch_dict['cpus_per_task_list']
    time = sbatch_dict.get('time', '01:00:00')
    output = sbatch_dict.get('output', f"{job_name}_%j.out")
    error = sbatch_dict.get('error', f"{job_name}_%j.err")
    
    # Extract experiment parameters
    concurrency = exp_dict['concurrency']
    data_dir = exp_dict['data_dir']
    result_dir = exp_dict['result_dir']
    result_params_GDT_name = exp_dict['result_params_GDT_name']
    result_time_elapsed_name = exp_dict['result_time_elapsed_name']
    iterations = exp_dict['iterations']
    sub_iterations = exp_dict['sub_iterations']
    inner_precision = exp_dict['inner_precision']
    step_size = exp_dict['step_size']
    type_LR = exp_dict['type_LR']
    
    # Validate type_LR
    if type_LR not in ['F', 'P', 'H']:
        raise ValueError("type_LR must be one of: 'F', 'P', 'H'")
    
    # Generate heterogeneous job components
    heterogeneous_components = []
    for i, cpus in enumerate(cpus_per_task_list):
        component = f"#SBATCH hetjob:{i+1} --nodes=1 --ntasks=1 --cpus-per-task={cpus}"
        heterogeneous_components.append(component)
    
    # Generate SLURM script
    slurm_script = f"""#!/bin/bash
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --job-name={job_name}
#SBATCH --time={time}
#SBATCH --output={output}
#SBATCH --error={error}

# Heterogeneous job components
{chr(10).join(heterogeneous_components)} #add a new line between each component, chr(10) is the newline character

# Load required modules
module load python/3.10.13
source /scratch/shij0d/venvs/asy_fed_spatial_env/bin/activate
module load libfabric #for mpi

export PYTHONPATH=/scratch/shij0d/projects/Asyn_Fed_Spatial:$PYTHONPATH

# Print job information
echo "Job ${{SLURM_JOB_ID}} is starting on ${{SLURM_JOB_NODELIST}}"
echo "==============================================="
echo "Experiment Configuration:"
echo "  Step Size: {step_size}"
echo "  Concurrency: {concurrency}"
echo "  Iterations: {iterations}"
echo "  Sub-iterations: {sub_iterations}"
echo "  Type LR: {type_LR}"
echo "  CPU Configuration: {cpus_per_task_list}"
echo "==============================================="

# Get the number of heterogeneous components
num_components=${{#SLURM_JOB_CPUS_PER_NODE[@]}}

# Run the experiment with heterogeneous components
srun --het-group=0-$((num_components-1)) bash -c '
    # Set thread configuration based on CPU count
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
    
    # Run the Python script
    python3 "/scratch/shij0d/projects/Asyn_Fed_Spatial/src/MPI/exp.py" \\
        --step_size "{step_size}" \\
        --concurrency "{concurrency}" \\
        --iterations "{iterations}" \\
        --sub_iterations "{sub_iterations}" \\
        --inner_precision "{inner_precision}" \\
        --data_dir "{data_dir}" \\
        --result_dir "{result_dir}" \\
        --result_params_GDT_name "{result_params_GDT_name}" \\
        --result_time_elapsed_name "{result_time_elapsed_name}" \\
        --type_LR "{type_LR}"
'
"""
    return slurm_script


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

def generate_cpu_allocation(J, heterogeneity=0.5, min_cpu=1, max_cpu=192, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    if heterogeneity == 0:
        return np.full(J, max_cpu)
    
    # Sample from a distribution (uniform for now)
    raw = np.random.rand(J)
    
    # Convert to CPU counts with max deviation controlled by heterogeneity
    cpus = max_cpu - (heterogeneity * (max_cpu - min_cpu) * raw)
    
    # Round and clip to ensure valid CPU counts
    cpus = np.clip(np.round(cpus), min_cpu, max_cpu).astype(int)
    
    return cpus

import subprocess
from pathlib import Path
from src.utils import generate_exp_slurm_script

def compute_oracle_estimator(
    type_LR: str,
    worker_numbers: int, 
    data_dir: str, 
    result_dir: str,
    script_dir: str,
    iterations: int = 150, 
    sub_iterations: int = 10, 
    step_size: float = 0.1
) -> None:
    """
    Generate and submit a SLURM job for oracle estimator computation.
    
    Args:
        type_LR: Type of low-rank model ('F', 'P', or 'H')
        worker_numbers: Number of worker nodes
        data_dir: Directory containing input data
        result_dir: Directory for storing results
        script_dir: Directory for SLURM script
        iterations: Number of main iterations
        sub_iterations: Number of sub-iterations
        step_size: Initial step size for optimization
        
    Returns:
        None
    """
    # Validate inputs
    if type_LR not in ['F', 'P', 'H']:
        raise ValueError("type_LR must be one of: 'F', 'P', 'H'")
    if worker_numbers < 1:
        raise ValueError("worker_numbers must be at least 1")
    
    # Create directories if they don't exist
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    Path(script_dir).mkdir(parents=True, exist_ok=True)
    
    # Define script path
    script_path = Path(script_dir) / "oracle_estimator.slurm"
    
    # SLURM job configuration
    sbatch_dict = {
        'account': 'k10164',
        'partition': 'workq',
        'job_name': 'oracle_estimator',
        'nodes': worker_numbers + 1,  # +1 for server node
        'ntasks': worker_numbers + 1,
        'cpus_per_task': 192,
        'time': '01:00:00',
        'output': f"{result_dir}/oracle_estimator_exp_%j.out",
        'error': f"{result_dir}/oracle_estimator_exp_%j.err"
    }
    
    # Experiment configuration
    exp_dict = {
        'threads': 192,
        'concurrency': worker_numbers,
        'data_dir': data_dir,
        'result_dir': result_dir,
        'result_params_GDT_name': f"params_GDT_step_size_{step_size}_concurrency_{worker_numbers}_type_LR_{type_LR}_oracle.pkl",
        'result_time_elapsed_name': f"times_elapsed_step_size_{step_size}_concurrency_{worker_numbers}_type_LR_{type_LR}_oracle.pkl",
        'iterations': iterations,
        'sub_iterations': sub_iterations,
        'step_size': step_size,
        'type_LR': type_LR
    }
    
    try:
        # Generate SLURM script
        slurm_script = generate_exp_slurm_script(sbatch_dict, exp_dict)
        
        # Write and make executable
        script_path.write_text(slurm_script)
        script_path.chmod(0o755)
        
        print(f"SLURM script written to: {script_path}")
        
        # Submit job
        result = subprocess.run(
            ["sbatch", str(script_path)], 
            capture_output=True, 
            text=True,
            check=True  # Raises CalledProcessError if return code is non-zero
        )
        
        print("Job submission result:")
        print("STDOUT:", result.stdout.strip())
        if result.stderr:
            print("STDERR:", result.stderr.strip())
            
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.stderr}")
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise



import subprocess
import os

# Define the paths to the scenario scripts
scenario_1_path = os.path.join(os.path.dirname(__file__), 'scenario_1.py')
scenario_2_path = os.path.join(os.path.dirname(__file__), 'scenario_2.py')
scenario_3_path = os.path.join(os.path.dirname(__file__), 'scenario_3.py')

# Run scenario 1
subprocess.run(['python3', scenario_1_path])

# Run scenario 2
subprocess.run(['python3', scenario_2_path])

# Run scenario 3
subprocess.run(['python3', scenario_3_path])

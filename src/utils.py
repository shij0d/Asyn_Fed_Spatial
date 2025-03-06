import torch
from typing import List,Dict
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
    
    
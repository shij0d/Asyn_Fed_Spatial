from __future__ import annotations
import numpy as np
from sklearn.gaussian_process.kernels import Matern,Kernel
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import gamma, kv
import math
import torch

from abc import ABC, abstractmethod

class Kernel_with_Grad(Kernel):
    # Abstract base class for kernels with gradient computation
    @abstractmethod
    def kernel_and_gradient(self, X, Y=None):
        pass

class Matern_with_Grad(Kernel_with_Grad,Matern):
   
   # Initialize the Matern_with_Grad class with length_scale, length_scale_bounds, and nu
   def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=1.5):
        super().__init__(length_scale, length_scale_bounds)
        self.nu = nu

   def kernel_and_gradient(self, X, Y=None):

      X = np.atleast_2d(X)
      length_scale =self.length.length_scale
      if Y is None:
         dists = pdist(X / length_scale, metric="euclidean")
      else:

         dists = cdist(X / length_scale, Y / length_scale, metric="euclidean")
      # Compute the kernel matrix based on the value of nu
      if self.nu == 0.5:
         K = np.exp(-dists)
      elif self.nu == 1.5:
         K = dists * math.sqrt(3)
         K = (1.0 + K) * np.exp(-K)
      elif self.nu == 2.5:
         K = dists * math.sqrt(5)
         K = (1.0 + K + K**2 / 3.0) * np.exp(-K)
      elif self.nu == np.inf:
         K = np.exp(-(dists**2) / 2.0)
      else:  # general case; expensive to evaluate
         K = dists
         K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
         tmp = math.sqrt(2 * self.nu) * K
         K.fill((2 ** (1.0 - self.nu)) / gamma(self.nu))
         K *= tmp**self.nu
         K *= kv(self.nu, tmp)

      if Y is None:
         # convert from upper-triangular matrix to square matrix
         K = squareform(K)
         np.fill_diagonal(K, 1)
         D=squareform(dists)
       
      # Compute the gradient of the kernel matrix with respect to the length scale
      if self.nu == 0.5:
         K_grad=K*D/length_scale
      elif self.nu == 1.5:
         K_grad=3*K*D**2/length_scale
      elif self.nu == 2.5:
         K_grad=K*(5/3*D**2+5*math.sqrt(5)/3*D**3)/length_scale
      elif self.nu == np.inf:
         K_grad=K*D**2/length_scale
      else:
         raise NotImplementedError("Gradient for nu={} is not implemented".format(self.nu))
      K_grad=K_grad[:, :, np.newaxis]
      return (K, K_grad)
       
def Matern_1_5_kernel(X:torch.Tensor|np.ndarray, Y:torch.Tensor|np.ndarray|None, theta:torch.Tensor|np.ndarray):
        """
        Computes the Matern1.5 kernel between X and Y.
        
        Parameters:
        X (torch.Tensor): Input tensor of shape (n_samples_X, n_features).
        Y (torch.Tensor): Input tensor of shape (n_samples_Y, n_features).
        theta: parameter vector.
        
        Returns:
        torch.Tensor: Kernel matrix of shape (n_samples_X, n_samples_Y).
        """
        if not isinstance(theta,torch.Tensor):
            theta=torch.tensor(theta)
        alpha = theta[0]
        length_scale = theta[1]
        # Ensure inputs are tensors
        if not isinstance(X,torch.Tensor):
            X = torch.tensor(X, dtype=torch.float64)
        if Y is None:
            dist= torch.pdist(X, p=2) # Calculate the squared Euclidean distance
        else:
            if not isinstance(Y,torch.Tensor):
               Y = torch.tensor(Y, dtype=torch.float64)
            dist= torch.cdist(X, Y, p=2)# Calculate the squared Euclidean distance
         
        dist_scaled=dist * math.sqrt(3)/length_scale
        # Compute the kernel
        K =alpha * (1.0 + dist_scaled) * torch.exp(-dist_scaled)
        
        return K

def Matern_2_5_kernel(X:torch.Tensor|np.ndarray, Y:torch.Tensor|np.ndarray|None, theta:torch.Tensor|np.ndarray):
        """
        Computes the Matern2.5 kernel between X and Y.
        
        Parameters:
        X (torch.Tensor): Input tensor of shape (n_samples_X, n_features).
        Y (torch.Tensor): Input tensor of shape (n_samples_Y, n_features).
        theta: parameter vector.
        
        Returns:
        torch.Tensor: Kernel matrix of shape (n_samples_X, n_samples_Y).
        """
        if not isinstance(theta,torch.Tensor):
            theta=torch.tensor(theta)
        alpha = theta[0]
        length_scale = theta[1]
        # Ensure inputs are tensors
        if not isinstance(X,torch.Tensor):
            X = torch.tensor(X, dtype=torch.float64)
        if Y is None:
            dist= torch.pdist(X , p=2) # Calculate the squared Euclidean distance
        else:
            if not isinstance(Y,torch.Tensor):
               Y = torch.tensor(Y, dtype=torch.float64)
            dist= torch.cdist(X , Y , p=2)# Calculate the squared Euclidean distance
         
           
        # Compute the kernel
        dist_scaled = dist * math.sqrt(5)/length_scale
        K = alpha*(1.0 + dist_scaled + dist_scaled**2 / 3.0) * torch.exp(-dist_scaled)
        
        return K
   
def exponential_kernel(X:torch.Tensor|np.ndarray, Y:torch.Tensor|np.ndarray|None, theta:torch.Tensor|np.ndarray,type="Euclidean"):
        """
        Computes the exponential kernel between X and Y.
        
        Parameters:
        X (torch.Tensor): Input tensor of shape (n_samples_X, n_features).
        Y (torch.Tensor): Input tensor of shape (n_samples_Y, n_features).
        theta: parameter vector.
        
        Returns:
        torch.Tensor: Kernel matrix of shape (n_samples_X, n_samples_Y).
        """
        
        if not isinstance(theta,torch.Tensor):
            theta=torch.tensor(theta)
        alpha = theta[0]
        length_scale = theta[1]
        
        if type=="Euclidean":
            # Ensure inputs are tensors
            if not isinstance(X,torch.Tensor):
                X = torch.tensor(X, dtype=torch.float64)
            if Y is None:
                dist= torch.pdist(X, p=2) # Calculate the squared Euclidean distance
            else:
                if not isinstance(Y,torch.Tensor):
                    Y = torch.tensor(Y, dtype=torch.float64)
                dist= torch.cdist(X, Y, p=2)# Calculate the squared Euclidean distance
        elif type=="chordal":
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float64)
            if Y is not None and not isinstance(Y, torch.Tensor):
                Y = torch.tensor(Y, dtype=torch.float64)
            # Convert lat/lon to radians
            X_rad = torch.deg2rad(X)
            if Y is not None:
                Y_rad = torch.deg2rad(Y)
            else:
                Y_rad = X_rad
            def lat_lon_to_cartesian(lat_lon, radius):
                lat, lon = lat_lon[:, 0], lat_lon[:, 1]
                x = radius * torch.cos(lat) * torch.cos(lon)
                y = radius * torch.cos(lat) * torch.sin(lon)
                z = radius * torch.sin(lat)
                return torch.stack([x, y, z], dim=1)
            X_cart = lat_lon_to_cartesian(X_rad, 1)
            Y_cart = lat_lon_to_cartesian(Y_rad, 1)
            if Y is None:
                dist = torch.cdist(X_cart, X_cart, p=2)  # Pairwise Euclidean distance in Cartesian space
            else:
                dist = torch.cdist(X_cart, Y_cart, p=2)


        dist_scaled=dist/length_scale        
        # Compute the kernel
        K = alpha * torch.exp(-dist_scaled)
        
        return K

def onedif_kernel(X:torch.Tensor|np.ndarray, Y:torch.Tensor|np.ndarray|None, theta:torch.Tensor|np.ndarray,type="Euclidean"):
        """
        Computes the matern kernel when nu=1.5 between X and Y.
        
        Parameters:
        X (torch.Tensor): Input tensor of shape (n_samples_X, n_features).
        Y (torch.Tensor): Input tensor of shape (n_samples_Y, n_features).
        theta: parameter vector.
        
        Returns:
        torch.Tensor: Kernel matrix of shape (n_samples_X, n_samples_Y).
        """
        if not isinstance(theta,torch.Tensor):
            theta=torch.tensor(theta)
        alpha = theta[0]
        length_scale = theta[1]
        # Ensure inputs are tensors
        if type=="Euclidean":
            # Ensure inputs are tensors
            if not isinstance(X,torch.Tensor):
                X = torch.tensor(X, dtype=torch.float64)
            if Y is None:
                dist= torch.pdist(X, p=2) # Calculate the squared Euclidean distance
            else:
                if not isinstance(Y,torch.Tensor):
                    Y = torch.tensor(Y, dtype=torch.float64)
                dist= torch.cdist(X, Y, p=2)# Calculate the squared Euclidean distance
        elif type=="chordal":
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float64)
            if Y is not None and not isinstance(Y, torch.Tensor):
                Y = torch.tensor(Y, dtype=torch.float64)
            # Convert lat/lon to radians
            X_rad = torch.deg2rad(X)
            if Y is not None:
                Y_rad = torch.deg2rad(Y)
            else:
                Y_rad = X_rad
            def lat_lon_to_cartesian(lat_lon, radius):
                lat, lon = lat_lon[:, 0], lat_lon[:, 1]
                x = radius * torch.cos(lat) * torch.cos(lon)
                y = radius * torch.cos(lat) * torch.sin(lon)
                z = radius * torch.sin(lat)
                return torch.stack([x, y, z], dim=1)
            X_cart = lat_lon_to_cartesian(X_rad, 1)
            Y_cart = lat_lon_to_cartesian(Y_rad, 1)
            if Y is None:
                dist = torch.cdist(X_cart, X_cart, p=2)  # Pairwise Euclidean distance in Cartesian space
            else:
                dist = torch.cdist(X_cart, Y_cart, p=2)
         
        dist_scaled=dist*math.sqrt(3)/length_scale        
        # Compute the kernel
        K = alpha *(1+dist_scaled)*torch.exp(-dist_scaled)
        
        return K


def exponential_kernel_tfed(X:torch.Tensor|np.ndarray, Y:torch.Tensor|np.ndarray|None, theta:torch.Tensor|np.ndarray):
        """
        Computes the exponential kernel between X and Y.
        
        Parameters:
        X (torch.Tensor): Input tensor of shape (n_samples_X, n_features).
        Y (torch.Tensor): Input tensor of shape (n_samples_Y, n_features).
        theta: parameter vector.
        
        Returns:
        torch.Tensor: Kernel matrix of shape (n_samples_X, n_samples_Y).
        """
        if not isinstance(theta,torch.Tensor):
            theta=torch.tensor(theta)
        alpha = theta[0]
        length_scale = theta[1]
        # Ensure inputs are tensors
        if not isinstance(X,torch.Tensor):
            X = torch.tensor(X, dtype=torch.float64)
        if Y is None:
            dist= torch.pdist(X, p=2) # Calculate the squared Euclidean distance
        else:
            if not isinstance(Y,torch.Tensor):
               Y = torch.tensor(Y, dtype=torch.float64)
            dist= torch.cdist(X, Y, p=2)# Calculate the squared Euclidean distance
         
                
        # Compute the kernel
        K = alpha * torch.exp(-0.5 * dist*length_scale)
        
        return K

def squared_exponential_kernel(X:torch.Tensor|np.ndarray, Y:torch.Tensor|np.ndarray|None, theta:torch.Tensor|np.ndarray):
        """
        Computes the squared exponential (Gaussian) kernel between X and Y.
        
        Parameters:
        X (torch.Tensor): Input tensor of shape (n_samples_X, n_features).
        Y (torch.Tensor): Input tensor of shape (n_samples_Y, n_features).
        theta: parameter vector.
        
        Returns:
        torch.Tensor: Kernel matrix of shape (n_samples_X, n_samples_Y).
        """
        theta=torch.tensor(theta)
        alpha = theta[0]
        length_scale = theta[1]
        # Ensure inputs are tensors
        if not isinstance(X,torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if Y is None:
            dist_sq = torch.pdist(X, p=2).pow(2) # Calculate the squared Euclidean distance
        else:
            if not isinstance(Y,torch.Tensor):
               Y = torch.tensor(Y, dtype=torch.float32)
            dist_sq = torch.cdist(X , Y, p=2).pow(2) # Calculate the squared Euclidean distance
         
        
        # Compute the kernel
        K = alpha * torch.exp(-0.5 * dist_sq/(length_scale**2))
        
        return K


class scaled_besselKv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x:torch.Tensor, v:float):
        ctx.save_for_backward(x)
        ctx._v = v
        if v>0:
            c=2**(1-v)/gamma(v)
        else:
            c=1
        x_numpy=x.numpy()
        values = np.ones_like(x_numpy)/c
        non_zero_indices = x_numpy != 0
        values[non_zero_indices] = (x_numpy[non_zero_indices] ** v) * kv(v, x_numpy[non_zero_indices])
        return torch.from_numpy(values)
    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        v = ctx._v
        grad_x=torch.zeros_like(x)
        non_zero_indices = x != 0
        if v<1: 
            grad_x[non_zero_indices]=-(x[non_zero_indices]**(2*v-1))*scaled_besselKv.apply(x[non_zero_indices],1-v)
        elif v>=1:
            grad_x[non_zero_indices]=-x[non_zero_indices]*scaled_besselKv.apply(x[non_zero_indices],v-1)
        
        return grad_out * grad_x, None


def matern_kernel(X: torch.Tensor | np.ndarray, 
                  Y: torch.Tensor | np.ndarray | None, 
                  theta: torch.Tensor | np.ndarray, 
                  nu: float = 1.5,type="Euclidean") -> torch.Tensor:
    """
    Computes the Matérn kernel (for nu=1.5) between X and Y.

    Parameters:
    X (torch.Tensor): Input tensor of shape (n_samples_X, n_features).
    Y (torch.Tensor): Input tensor of shape (n_samples_Y, n_features) or None.
    theta: parameter vector (length 2), where:
        - theta[0] is the scaling parameter (alpha)
        - theta[1] is the length scale.
    nu: Smoothness parameter (default is 1.5).

    Returns:
    torch.Tensor: Kernel matrix of shape (n_samples_X, n_samples_Y).
    """
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=torch.float64)

    alpha = theta[0]
    length_scale = theta[1]
    
    if type=="Euclidean":
            # Ensure inputs are tensors
            if not isinstance(X,torch.Tensor):
                X = torch.tensor(X, dtype=torch.float64)
            if Y is None:
                dist= torch.pdist(X, p=2) # Calculate the squared Euclidean distance
            else:
                if not isinstance(Y,torch.Tensor):
                    Y = torch.tensor(Y, dtype=torch.float64)
                dist= torch.cdist(X, Y, p=2)# Calculate the squared Euclidean distance
    elif type=="chordal":
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float64)
            if Y is not None and not isinstance(Y, torch.Tensor):
                Y = torch.tensor(Y, dtype=torch.float64)
            # Convert lat/lon to radians
            X_rad = torch.deg2rad(X)
            if Y is not None:
                Y_rad = torch.deg2rad(Y)
            else:
                Y_rad = X_rad
            def lat_lon_to_cartesian(lat_lon, radius):
                lat, lon = lat_lon[:, 0], lat_lon[:, 1]
                x = radius * torch.cos(lat) * torch.cos(lon)
                y = radius * torch.cos(lat) * torch.sin(lon)
                z = radius * torch.sin(lat)
                return torch.stack([x, y, z], dim=1)
            X_cart = lat_lon_to_cartesian(X_rad, 1)
            Y_cart = lat_lon_to_cartesian(Y_rad, 1)
            if Y is None:
                dist = torch.cdist(X_cart, X_cart, p=2)  # Pairwise Euclidean distance in Cartesian space
            else:
                dist = torch.cdist(X_cart, Y_cart, p=2)
        
    # Ensure inputs are tensors
    
    # Calculate squared Euclidean distance
    dist = torch.cdist(X, Y, p=2)
    
    dist_scaled=dist*math.sqrt(2*nu) / length_scale 
    
    # Compute the Matérn kernel for nu = 1.5
    coeff = (2 ** (1 - nu)) / gamma(nu)
    kernel_matrix = coeff * scaled_besselKv.apply(dist_scaled,nu)

    # Set the diagonal entries to the scaling factor
    kernel_matrix[torch.isnan(kernel_matrix)] = 0  # Handle NaNs for distance 0 case

    return alpha * kernel_matrix


def matern_kernel_factory(nu: float,type):
    def matern_kernel_nu(X: torch.Tensor | np.ndarray, 
                  Y: torch.Tensor | np.ndarray | None, 
                  theta: torch.Tensor | np.ndarray):
        return matern_kernel(X,Y,theta,nu)
    return matern_kernel_nu

#from __future__ import annotations
import numpy as np
import random
from sklearn.gaussian_process.kernels import Kernel

import numpy as np
import random
import math
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI plotting
import matplotlib.pyplot as plt
from typing import Tuple,List
import torch
#a large number
M=10*8

class GPPSampleGenerator:
    def __init__(self,num: int, min_dis: float, extent: Tuple[float, float, float, float],kernel: Kernel,coefficients: List[float], noise: float,seed=2024):
        """
        Initializes a SampleGenerator object with parameters for generating synthetic data.

        Parameters:
        - num (int): The number of random points to generate.
        - min_dis (float): The minimum distance between any two points.
        - extent (tuple): A tuple containing the extent of the area to generate points.
            Should be in the format (x_min, x_max, y_min, y_max).
        - kernel (Kernel): A kernel function for covariance calculation.
        - coefficients (list): A list of coefficients for generating X values.
        - noise (float): The standard deviation of noise in the generated data.
        - seed (int, optional): Seed for random number generation (default is 2024).
        """
        self.num=num
        self.min_dis=min_dis
        self.extent=extent
        self.kernel=kernel
        self.noise=noise
        self.coefficients=coefficients
        self.seed=seed
        

    def random_points(self) -> List[Tuple[float, float]]:
        """
        Generates a list of random points within the specified extent, 
        with a minimum distance between points along both x and y axes.

        Parameters:
        - num (int): The number of random points to generate.
        - min_dis (float): The minimum distance between any two points.
        - extent (tuple): A tuple containing the extent of the area to generate points.
            Should be in the format (x_min, x_max, y_min, y_max).

        Returns:
        - list: A list of tuples, each representing a random point (x, y) within the extent.
        """
        x_min, x_max, y_min, y_max = self.extent  # Unpack extent tuple

        # Generate grid of points with minimum distance min_dis
        xs = np.arange(x_min, x_max, self.min_dis)
        if xs[-1] + self.min_dis <= x_max:
            xs = np.append(xs, x_max)

        ys = np.arange(y_min, y_max, self.min_dis)
        if ys[-1] + self.min_dis <= y_max:
            ys = np.append(ys, y_max)

        random.seed(self.seed)
        np.random.seed(self.seed)
        # Create list of all possible points with added noise
        noise_scale = 0.4 * self.min_dis  # Adjust this value to control noise level
        points = [(x + np.random.uniform(-noise_scale, noise_scale), y + np.random.uniform(-noise_scale, noise_scale)) for x in xs for y in ys]

        # Randomly sample num points from the list
        
        ran_points = random.sample(points, self.num)

        return ran_points
    
    def get_knots_random(self, locations, m):
        """
        Randomly selects m knot points from given locations.

        Parameters:
        - locations (list): List of tuples representing locations.
        - m (int): Number of knot points to select.

        Returns:
        - list: List of tuples representing randomly selected knot points.
        """
        random.seed(self.seed)
        knots = random.sample(locations, m)
        return knots
    
    def get_knots_grid(self, m):
        """
        Generates knot points on a grid within the extent.

        Parameters:
        - m (int): Number of knot points to generate.

        Returns:
        - list: List of tuples representing grid-generated knot points.
        """
        l = int(math.sqrt(m))
        x_min, x_max, y_min, y_max = self.extent  # Unpack extent tuple

        # Generate grid of points with minimum distance min_dis
        xs = np.linspace(x_min, x_max, l)
        ys = np.linspace(y_min, y_max, l)
        knots = [(x, y) for x in xs for y in ys]

        return knots
        
    def generate_x_epsilon(self):
        """
        Generates linear term with added noise.

        Returns:
        - numpy.ndarray: Array of X values adjusted with noise.
        """

        p = len(self.coefficients)  # Number of coefficients

        # Generate design matrix X with normal distribution
        np.random.seed(self.seed)
        X = np.random.normal(size=(self.num, p))

        # Generate noise term epsilon with normal distribution
        epsilon = np.random.normal(scale=self.noise, size=self.num)
        epsilon=epsilon.reshape(-1,1)
        # Convert coefficients to NumPy array and reshape for matrix multiplication
        coefficients_array = np.array(self.coefficients).reshape(p, 1)

        value= X @ coefficients_array + epsilon

        return value,X
    
    def generate_obs_gp(self,m,method):
        """
        Generates observations following Gaussian Process.

        Returns:
        - numpy.ndarray: Array of generated observations.
        """

        locations=self.random_points()
        if method=="random":
            knots=self.get_knots_random(locations,m)
        elif method=="grid":
            knots=self.get_knots_grid(m)
        else:  
            raise("Invalid choice. Please select from 'random' or 'grid'.")
        # Compute the covariance matrix using the kernel function
        cov = self.kernel(np.array(locations))

        # Initialize the mean vector as zeros
        mean = np.zeros(self.num)

        # Generate random samples (y) from a multivariate normal distribution
        np.random.seed(self.seed)
        y = np.random.multivariate_normal(mean, cov)
        y=y.reshape(-1,1)
        # Generate observations based on linear model: z = X @ coefficients + y + epsilon
        value,X=self.generate_x_epsilon()
        z = y+value
        data=np.hstack((locations,z,X))
        return data,knots
    
    
    def generate_obs_gpp(self,m,method):

        """
        Generates observations following Gaussian Predictive Process(GPP).

        Parameters:
        - m (int): Number of knot points.
        - method (str): Method for selecting knot points ("random" or "grid").

        Returns:
        - numpy.ndarray: Array of generated observations.
        """

        locations=self.random_points()    

        if method=="random":
            knots=self.get_knots_random(locations,m)
        elif method=="grid":
            knots=self.get_knots_grid(m)
        else:  
            raise("Invalid choice. Please select from 'random' or 'grid'.")  

        # Generate random eta values from a multivariate normal distribution
        np.random.seed(self.seed)
        mean_eta = np.zeros(np.array(knots).shape[0])
        cov_eta = self.kernel(np.array(knots))
        eta = np.random.multivariate_normal(mean_eta, cov_eta)
        eta=eta.reshape(-1,1)
        # Compute the product B = K(locations, knots) @ inv(cov_eta)
        B = self.kernel(np.array(locations), np.array(knots)) @ np.linalg.inv(cov_eta)
        
        

        # Generate y using the GPP model: y = B @ eta
        y = B @ eta

        # res1=[]
        # for i in range(100):  
        #     eta = np.random.multivariate_normal(mean_eta, cov_eta)
        #     eta=eta.reshape(-1,1)
        #     y1 = B @ eta
        #     res1.append((y1.T@y1).item())
        
        
        
        # mean=np.zeros(y.shape[0])
        # cov=B@cov_eta@B.T
        # avg=0
        # res=[]
        # for i in range(100):
        #     y2=np.random.multivariate_normal(mean=mean,cov=cov).reshape(-1,1)
        #     res.append(y2.T@y2)
        # avg=avg/100
        # print(avg)

        value,X=self.generate_x_epsilon()
        z = y+value
        data=np.hstack((locations,z,X))
        return data,knots
    def generate_obs_gpp_est_pre(self,N_pre,m,method):

        """
        Generates observations following Gaussian Predictive Process(GPP) and divide into two data set for estimation and prediction,respetively.

        Parameters:
        - m (int): Number of knot points.
        - method (str): Method for selecting knot points ("random" or "grid").

        Returns:
        - numpy.ndarray: Array of generated observations.
        """

        locations=self.random_points()    
        N=len(locations)
        N_est=N-N_pre
        locations_est=locations[0:N_est]
        locations_pre=locations[N_est:]
        # Compute the number of knots (m)
        if method=="random":
            knots=self.get_knots_random(locations_est,m)
        elif method=="grid":
            knots=self.get_knots_grid(m)
        else:  
            raise("Invalid choice. Please select from 'random' or 'grid'.")  
        m = len(knots)

        # Generate random eta values from a multivariate normal distribution
        np.random.seed(self.seed)
        mean_eta = np.zeros(m)
        cov_eta = self.kernel(np.array(knots))
        eta = np.random.multivariate_normal(mean_eta, cov_eta)
        eta=eta.reshape(-1,1)
        # Compute the product B = K(locations, knots) @ inv(cov_eta)
        B = self.kernel(np.array(locations), np.array(knots)) @ np.linalg.inv(cov_eta)
        
        

        # Generate y using the GPP model: y = B @ eta
        y = B @ eta


        value,X=self.generate_x_epsilon()
        z = y+value
        data=np.hstack((locations,z,X))
        
        return data,knots
    
    
    def data_split(self,data,J,method='random',neighbours=None):
        '''
        method: random, by area, random_nearest
        '''
        if method=='random':
            dis_data=np.array_split(data,J,axis=0)
        if method=='by_area':
            sqrt_J=int(math.sqrt(J))
            if int(sqrt_J**2)!=J:
                Warning("it is not a perfect square")
            locations=data[:,:2]
            x_min, x_max = locations[:, 0].min(), locations[:, 0].max()
            y_min, y_max = locations[:, 1].min(), locations[:, 1].max()
            
            # Define the partition edges
            x_bins = np.linspace(x_min, x_max, sqrt_J + 1)
            y_bins = np.linspace(y_min, y_max, sqrt_J + 1)
            
            dis_data = []
            
            # Loop over all partitions and collect data
            for i in range(sqrt_J):
                for j in range(sqrt_J):
                    # Find locations that belong to this partition
                    x_in_bin = (locations[:, 0] >= x_bins[i]) & (locations[:, 0] < x_bins[i+1])
                    y_in_bin = (locations[:, 1] >= y_bins[j]) & (locations[:, 1] < y_bins[j+1])
                    in_bin = x_in_bin & y_in_bin
                    
                    # Get the corresponding data for this partition
                    partition_data = data[in_bin]
                    dis_data.append(partition_data)
        if method=='random_nearest':
            if neighbours==None:
                raise("please specify the number of nearest locations")
            N = data.shape[0]
            n = int(N / J)  # Size of each partition
            N_random = J * int(n / (1 + neighbours))  # Number of random locations

            np.random.seed(self.seed)
            data=np.random.permutation(data)
            locations = data[:, :2]  # Only use the first two columns for locations
            random_locations = locations[0:N_random, :]  # Get random locations

            # Use scipy.spatial.distance.cdist for efficient distance calculation
            rest_locations=locations[N_random:,:]
            dists = cdist(rest_locations, random_locations)  # Calculate pairwise distances

            # Initialize neighbour indices and nums
            neighbours_idx = np.full((N_random, neighbours), -1, dtype=int)  # Stores the neighbor indices
            nums = np.zeros(N_random, dtype=int)  # Keeps track of neighbors per random location

            # Assign each point to its nearest random location (up to 'neighbours' per random location)
            for i, dist_row in enumerate(dists):
                # Set large distance for locations with full neighbors
                dist_row[nums >= neighbours] = np.inf
                idx = np.argmin(dist_row)  # Find the nearest random location
                neighbours_idx[idx, nums[idx]] = i+N_random  # Assign the point index
                nums[idx] += 1  # Increment the neighbor count

            # Partition the data based on the neighbor indices
            dis_data = []
            for j in range(J):
                partition_indices = []
                for i in range(int(n / (1 + neighbours))):
                    # Collect neighbor indices for this partition
                    partition_indices.append(j * int(n / (1 + neighbours)) + i)
                    partition_indices.extend(neighbours_idx[j * int(n / (1 + neighbours)) + i, :].tolist())
                
                partition_indices = [idx for idx in partition_indices if idx >= 0]  # Remove invalid (-1) indices
                partition_data = data[partition_indices, :]  # Get the partitioned data
                dis_data.append(partition_data)
        
        return dis_data
    def visual_locations(dis_data:List[np.ndarray],save_file=None,title=None):
        J=len(dis_data)
        colors = plt.cm.get_cmap('tab10', J)  # Choose a colormap for the partitions
        plt.figure(figsize=(8, 6))
        for i, partition in enumerate(dis_data):
            partition_locations = partition[:, :2]  # Only the 2D locations
            plt.scatter(partition_locations[:, 0], partition_locations[:, 1], color=colors(i),alpha=0.5,s=10)
        plt.title(title,fontsize=20)
        plt.xlabel('X Coordinate',fontsize=18)
        plt.ylabel('Y Coordinate',fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend()
        plt.grid(True)
        if save_file!=None:
            plt.savefig(save_file)
        plt.show()
    

class GPPSampleGeneratorUnitSquare:        
    def __init__(self,num: int,kernel: Kernel,coefficients: List[float], noise_level: float,seed=2024):
        """
        Initializes a SampleGenerator object with parameters for generating synthetic data.

        Parameters:
        - num (int): The number of random points to generate.
        - kernel (Kernel): A kernel function for covariance calculation.
        - coefficients (list): A list of coefficients for generating X values.
        - noise_level (float): The standard deviation of noise in the generated data.
        - seed (int, optional): Seed for random number generation (default is 2024).
        """
        self.num=num
        self.kernel=kernel
        self.noise_level=noise_level
        self.coefficients=coefficients
        self.seed=seed
    def random_points(self) -> torch.Tensor:
        """
        Generates a list of random spatial location points, 

        Returns:
        - list: A list of tuples, each representing a random point (x, y).
        """
        torch.manual_seed(self.seed)
        # Step 1: Create grid from 1 to n_sqrt
        n_sqrt=math.ceil(math.sqrt(self.num))
        n=n_sqrt**2
        xs=torch.arange(1, n_sqrt+1)
        ys=torch.arange(1, n_sqrt+1)
        xx,yy=torch.meshgrid(xs,ys,indexing='ij')
        grid_points = torch.column_stack([xx.ravel(), yy.ravel()]) # shape: (n, 2)
        # Step 2: Add uniform jitter in [-0.4, 0.4]
        jitter = torch.empty(n, 2).uniform_(-0.4, 0.4)
        
        # Step 3: Apply transformation (grid - 0.5 + jitter) / nx
        jittered_points = (grid_points - 0.5 + jitter) / n_sqrt
        #Step 4: Randomly sample num points from the jittered points
        indices = torch.randperm(n)[:self.num]
        sampled_points = jittered_points[indices]
        return sampled_points

    
    def get_knots_random(self, locations:torch.Tensor, m:int,n_samples=1)->torch.Tensor:
        """
        Randomly selects m knot points from given locations.

        Parameters:
        - locations (torch.Tensor): locations.
        - m (int): Number of knot points to select.

        Returns:
        - torch.Tensor: randomly selected knot points.
        """
        torch.manual_seed(self.seed)  # for reproducibility
        n = locations.shape[0]

        # Use a list comprehension to generate n_samples of m knots each
        knots_list = [locations[torch.randperm(n)[:m]] for _ in range(n_samples)]

        # stack them into a single tensor of shape (n_samples, m, location_dim)
        knots_tensor = torch.stack(knots_list)

        return knots_tensor
           
    def get_knots_grid(self, m:int,n_samples=1)->torch.Tensor:
        """
        Generates knot points on a grid within the unit square.
        Parameters:
        - m (int): Number of knot points to generate.
        Returns:
        - torch.Tensor: Grid-generated knot points.
        """
        torch.manual_seed(self.seed)
        m_sqrt = math.ceil(math.sqrt(m))
        total_num = m_sqrt ** 2
        xs= torch.arange(1, m_sqrt+1)
        ys=torch.arange(1, m_sqrt+1)
        xx,yy=torch.meshgrid(xs,ys,indexing='ij')
        grid_points = torch.column_stack([xx.ravel(), yy.ravel()])
        
        knots_list = []

        for _ in range(n_samples):
            # For each sample: create new jitter
            jitter = torch.empty(total_num, 2).uniform_(-0.4, 0.4)
            jittered_points = (grid_points - 0.5 + jitter) / m_sqrt

            # Then select m points
            indices = torch.randperm(total_num)[:m]
            knots = jittered_points[indices]

            knots_list.append(knots)

        knots_tensor = torch.stack(knots_list)  # shape (n_samples, m, 2)
        
        return knots_tensor
    
    
    def generate_x_epsilon(self,n_samples)->Tuple[torch.Tensor,torch.Tensor]:
        """
        Generates linear term with added noise.

        Returns:
        - numpy.ndarray: Array of X values adjusted with noise.
        """

        p = len(self.coefficients)  # Number of coefficients

        # Generate design matrix X with normal distribution
        torch.manual_seed(self.seed)
        X = torch.normal(mean=0, std=1, size=(n_samples,self.num, p))
       

        # Generate noise term epsilon with normal distribution
        epsilon= torch.normal(mean=0, std=self.noise_level, size=(n_samples,self.num,1))
        
        # Convert coefficients to torch tensor for matrix multiplication
        coefficients_array = torch.tensor(self.coefficients,dtype=torch.float32).reshape(p, 1)

        value= X @ coefficients_array + epsilon
        
        #turn value to be 2D
        value = value.reshape(n_samples,self.num)

        return value,X
    def generate_obs_gp(self,m:int,method:str,n_samples=1)-> List[Tuple[torch.Tensor, torch.Tensor]]: 
        """
        Generates observations following Gaussian Process.

        Returns:
        - torch.Tensor: Array of generated observations.
        """
        torch.manual_seed(self.seed)
                
        locations=self.random_points()
        if method=="random":
            knots=self.get_knots_random(locations,m,n_samples)
        elif method=="grid":
            knots=self.get_knots_grid(m,n_samples)
        else:  
            raise("Invalid choice. Please select from 'random' or 'grid'.")
        # Compute the covariance matrix using the kernel function
        cov = self.kernel(locations,locations)
        # Convert the covariance matrix to a PyTorch tensor
        if not isinstance(cov, torch.Tensor):
            cov = torch.tensor(cov, dtype=torch.float32)  # Convert to tensor
        elif cov.dtype != torch.float32:
            cov = cov.to(torch.float32)


        # Generate random samples (y) from a multivariate normal distribution
        L = torch.linalg.cholesky(cov)
        z_normal = torch.randn(n_samples, cov.shape[0], dtype=torch.float32)
        y=z_normal @ L.T 
        
        # Generate observations based on linear model: z = X @ coefficients + y + epsilon
        value,X=self.generate_x_epsilon(n_samples)
        z:torch.Tensor = y+value
        res=[]
        for i in range(n_samples):
            data = torch.hstack((locations,z[i].reshape(-1,1),X[i]))
            tup=(data,knots[i])
            res.append(tup)
        return res
    def data_split(self,data:torch.Tensor,J:int,method:str='random',neighbours:int=None)->List[torch.Tensor]:
        '''
        method: random, by area, random_nearest
        '''
        if method=='random':
            dis_data=list(torch.chunk(data,J,dim=0))
            
        if method=='by_area':
            sqrt_J=int(math.sqrt(J))
            if int(sqrt_J**2)!=J:
                Warning("J is not a perfect square, partitioning may be uneven.")
                
            locations=data[:,:2]
            x_min, x_max = locations[:, 0].min(), locations[:, 0].max()
            y_min, y_max = locations[:, 1].min(), locations[:, 1].max()
            
            # Define the partition edges
            x_bins = torch.linspace(x_min, x_max, sqrt_J + 1)
            y_bins = torch.linspace(y_min, y_max, sqrt_J + 1)
            
            dis_data = []
            
            # Loop over all partitions and collect data
            for i in range(sqrt_J):
                for j in range(sqrt_J):
                    # Find locations that belong to this partition
                    x_in_bin = (locations[:, 0] >= x_bins[i]) & (locations[:, 0] < x_bins[i+1])
                    y_in_bin = (locations[:, 1] >= y_bins[j]) & (locations[:, 1] < y_bins[j+1])
                    in_bin = x_in_bin & y_in_bin
                    
                    # Get the corresponding data for this partition
                    partition_data = data[in_bin]
                    dis_data.append(partition_data)
        if method=='random_nearest':
            if neighbours==None:
                raise ValueError("please specify the number of nearest locations")
            N = data.shape[0]
            n = int(N / J)  # Size of each partition
            N_random = J * int(n / (1 + neighbours))  # Number of random locations
            
            torch.manual_seed(self.seed)
            perm=torch.randperm(N)
            data=data[perm]
            locations = data[:, :2]  # Only use the first two columns for locations
            random_locations = locations[0:N_random, :]  # Get random locations

            # Use scipy.spatial.distance.cdist for efficient distance calculation
            rest_locations=locations[N_random:,:]
            dists =torch.cdist(rest_locations, random_locations,p=2)  # Calculate pairwise distances

            # Initialize neighbour indices and nums
            neighbours_idx=torch.full((N_random, neighbours), -1, dtype=torch.long)  # Stores the neighbor indices
            nums = torch.zeros(N_random, dtype=torch.int)  # Keeps track of neighbors per random location
            # Assign each point to its nearest random location (up to 'neighbours' per random location)
            for i, dist_row in enumerate(dists):
                dist_row = dist_row.clone()
                # Set large distance for locations with full neighbors
                dist_row[nums >= neighbours] = torch.inf
                idx = torch.argmin(dist_row)  # Find the nearest random location
                neighbours_idx[idx, nums[idx]] = i+N_random  # Assign the point index
                nums[idx] += 1  # Increment the neighbor count

            # Partition the data based on the neighbor indices
            dis_data = []
            for j in range(J):
                partition_indices = []
                for i in range(int(n / (1 + neighbours))):
                    # Collect neighbor indices for this partition
                    partition_indices.append(j * int(n / (1 + neighbours)) + i)
                    partition_indices.extend(neighbours_idx[j * int(n / (1 + neighbours)) + i, :].tolist())
                
                partition_indices = [idx for idx in partition_indices if idx >= 0]  # Remove invalid (-1) indices
                partition_data = data[partition_indices, :]  # Get the partitioned data
                dis_data.append(partition_data)
        
        return dis_data
    def visual_locations(dis_data:List[torch.Tensor],save_file=None,title=None):
        J=len(dis_data)
        colors = plt.cm.get_cmap('tab10', J)  # Choose a colormap for the partitions
        plt.figure(figsize=(8, 6))
        for i, partition in enumerate(dis_data):
            partition_locations = partition[:, :2]  # Only the 2D locations
            plt.scatter(partition_locations[:, 0], partition_locations[:, 1], color=colors(i),alpha=0.5,s=10)
        plt.title(title,fontsize=20)
        plt.xlabel('X Coordinate',fontsize=18)
        plt.ylabel('Y Coordinate',fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend()
        plt.grid(True)
        if save_file!=None:
            plt.savefig(save_file)
        plt.show()
    
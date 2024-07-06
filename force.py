import torch
import torch.nn as nn
from utility import *
import matplotlib.pyplot as plt

class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return torch.nn.functional.softplus(x) - self.shift
    
class SchNet(nn.Module):
    def __init__(self, z, cell, n_gaussians, hidden_channels, n_filters, num_interactions, r_cut):
        super(SchNet, self).__init__()
        self.z = z.long()
        self.cell = cell.to(torch.float32)
       
        
        num_atom_types = 200
        self.embedding = nn.Embedding(num_atom_types, hidden_channels)



        self.GS_layer = GaussianSmearing(start=0.0, stop=r_cut, n_gaussians = n_gaussians)

        self.interactions = nn.ModuleList([InteractionBlock(n_gaussians, hidden_channels, n_filters, r_cut) for _ in range(num_interactions)])
        
        self.output_layer1 = nn.Linear(hidden_channels, hidden_channels//2)
        self.output_layer1_activation = ShiftedSoftplus()
        self.output_layer2 = nn.Linear(hidden_channels//2, 1)
        
        self.r_cut = r_cut
        
        
    def forward(self, pos):
        z = self.z

        # Generate graph data
        neighbor_indices, offsets, distances, unit_vectors = generate_nbr_list(pos, self.cell, self.r_cut)
        
        
        x = self.embedding(z)


        edge_index = neighbor_indices.t().contiguous()
        edge_weight = distances  # or any other edge features
        edge_attr = self.GS_layer(edge_weight)
        

        
        for interaction in self.interactions:
            x = x + interaction(x, edge_index, edge_weight, edge_attr)
        
        x = self.output_layer1(x)
        x = self.output_layer1_activation(x)
        x = self.output_layer2(x)

        return x.sum()
    
class InteractionBlock(nn.Module):
    def __init__(self, n_gaussians, hidden_channels, n_filters, r_cut):
        super(InteractionBlock, self).__init__()
        

        ## convolution
        self.mlp = nn.Sequential(
            nn.Linear(n_gaussians, n_filters),
            ShiftedSoftplus(),
            nn.Linear(n_filters, n_filters),
            )
        
        self.conv = SchConv(hidden_channels, hidden_channels, n_filters, self.mlp, r_cut)

        self.act = ShiftedSoftplus()

        self.lin = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class SchConv(nn.Module):
    def __init__(self, channel_in, channel_out, n_filters, mlp, r_cut) -> None:
        super(SchConv, self).__init__()

        self.mlp = mlp
        self.lin1 = nn.Linear(channel_in, n_filters, bias=False)
        self.lin2 = nn.Linear(n_filters, channel_out)
        self.r_cut = r_cut

        

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * torch.pi / self.r_cut) + 1.0)
    
        ## node info processing
        x = self.lin1(x)

        # edge info processing (convolution)
        row, col = edge_index
        edge_messages1 = (self.mlp(edge_attr)) * x[row]
        edge_messages2 = (self.mlp(edge_attr)) * x[col]

        # message 
        aggr_messages = torch.zeros_like(x).scatter_add(0, col.unsqueeze(-1).expand_as(edge_messages1), edge_messages1)
        aggr_messages += torch.zeros_like(x).scatter_add(0, row.unsqueeze(-1).expand_as(edge_messages2), edge_messages2)


        
        return self.lin2(aggr_messages)
    


class GaussianSmearing(nn.Module):


    def __init__(self, start, stop, n_gaussians, width=None, centered=False, trainable=False):
        super().__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        if width is None:
            widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        else:
            widths = torch.FloatTensor(width * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer('width', widths)
            self.register_buffer('offsets', offset)
        self.centered = centered

    def forward(self, distances):

        coeff = -0.5 / torch.pow(self.width, 2)
        diff = distances - self.offsets
        gauss = torch.exp(coeff * torch.pow(diff, 2))
        
        return gauss


#%%

class Tabulated(torch.nn.Module):
    def __init__(self, cell, Tabulated_data):
        super(Tabulated, self).__init__()
        self.cell = cell
        
        from scipy.interpolate import interp1d
        #f = interp1d(Tabulated_data[:,1], Tabulated_data[:,3])
        f = interp1d(Tabulated_data[:,1], Tabulated_data[:,3], kind='cubic')
        
        self.force_magnitude = f
        
    def forward(self, q):
        nbr_list, offsets, pdist, unit_vector = generate_nbr_list(q, self.cell)
        
        force = torch.zeros_like(q, device=q.device)
        force_magnitude = torch.tensor(self.force_magnitude(pdist.detach().cpu().numpy())).to(q.device).to(q.dtype)
        force_vector = force_magnitude*unit_vector
                
        force.index_add_(0, nbr_list[:,0], force_vector)
        force.index_add_(0, nbr_list[:,1], -force_vector)
        return force.detach()

class LJ(torch.nn.Module):
    def __init__(self, cell, sigma=3.405 , epsilon=0.0103, Trainable = False):
        super(LJ, self).__init__()
        self.cell = cell # This is box size --> used for force calculation in PBC environment
        if Trainable==True:
            self.sigma = torch.nn.Parameter(torch.Tensor([sigma]))
            self.epsilon = torch.nn.Parameter(torch.Tensor([epsilon]))
        else:
            self.sigma = sigma
            self.epsilon = epsilon
        
    def LJ(self, r):
        return 4 *  torch.exp(self.epsilon) * ((torch.exp(self.sigma)/r)**12 - (torch.exp(self.sigma)/r)**6 )


    def forward(self, q):
        nbr_list, offsets, pdist, unit_vector = generate_nbr_list(q, self.cell)
        return self.LJ(pdist).sum()
    
class LJ_repulsive(torch.nn.Module):
    def __init__(self, cell, sigma=3.405 , epsilon=0.0103, Trainable = False):
        super(LJ_repulsive, self).__init__()
        self.cell = cell # This is box size --> used for force calculation in PBC environment
        if Trainable==True:
            self.log_sigma = torch.nn.Parameter(torch.Tensor([sigma]))
            self.log_epsilon = torch.nn.Parameter(torch.Tensor([epsilon]))
        else:
            self.log_sigma = sigma
            self.log_epsilon = epsilon
        
    def LJ_repulsive(self, r):
        return 4 *  torch.exp(self.log_epsilon) * (torch.exp(self.log_sigma)/r)**12


    def forward(self, q):
        nbr_list, offsets, pdist, unit_vector = generate_nbr_list(q, self.cell)
        return self.LJ_repulsive(pdist)

class Langevin_TS(torch.nn.Module):
    def __init__(self, gamma=0.1, BOLTZMAN =  0.001987191, Trainable=False):
        super(Langevin_TS, self).__init__()
        if Trainable==False:
            self.gamma_log = torch.log(torch.tensor([gamma]))
        else:
            self.gamma_log = torch.nn.Parameter(torch.log(torch.Tensor([gamma])))
            
        self.BOLTZMAN = BOLTZMAN

    def forward(self, v, T, dt, mass):
        friction = -v*torch.exp(self.gamma_log)
        Langevin_coeff = torch.sqrt(2.0 * torch.exp(self.gamma_log) / mass * self.BOLTZMAN * T * dt)
        random = torch.randn_like(v, device=v.device) * Langevin_coeff/dt
        delta_v_langevin = friction + random
        return delta_v_langevin
    

class DPD(torch.nn.Module):
    def __init__(self, cell, gamma_parallel=0.1, gamma_transverse=0.1, BOLTZMAN =  0.001987191):
        super(DPD, self).__init__()
        self.cell = cell
        
        #self.logit_gamma_parallel = torch.nn.Parameter(torch.Tensor([gamma]))
        #self.logit_gamma_transverse = torch.nn.Parameter(torch.Tensor([gamma]))
        #self.gamma_parallel = torch.sigmoid(self.logit_gamma_parallel) / 
        #self.gamma_transverse = torch.sigmoid(self.logit_gamma_transverse)

        
        self.gamma_parallel = torch.nn.Parameter(torch.Tensor([gamma_parallel]))
        self.gamma_transverse = torch.nn.Parameter(torch.Tensor([gamma_transverse]))

        self.BOLTZMAN = BOLTZMAN

    def forward(self, q, v, T, dt, mass):
        #self.gamma_parallel = torch.sigmoid(self.logit_gamma_parallel)
        #self.gamma_transverse = torch.sigmoid(self.logit_gamma_transverse)
        
        # velocity decomposition into v_pair and v_transverse
        nbr_list, offsets, pdist, unit_vector = generate_nbr_list(q, self.cell)
        
        v_ij = v[nbr_list[:,0]] - v[nbr_list[:,1]]
        
        v_ij_parallel = torch.sum(v_ij * unit_vector, axis=1).unsqueeze(1) * unit_vector
        v_ij_transverse = v_ij - v_ij_parallel
        
        norms = torch.norm(v_ij_transverse, p=2, dim=1, keepdim=True)
        v_ij_unit = v_ij_transverse/norms
        
        

        
        
        friction_parallel = -v_ij_parallel*self.gamma_parallel
        Langevin_coeff_parallel = torch.sqrt(2.0 * self.gamma_parallel / mass[0] * self.BOLTZMAN * T * dt)
        random_parallel = torch.randn(len(v_ij_parallel), device=v.device).unsqueeze(1)* Langevin_coeff_parallel/dt * unit_vector
        
        
        friction_transverse = -v_ij_transverse*self.gamma_transverse
        Langevin_coeff_transverse = 2**0.5 * torch.sqrt(2.0 * self.gamma_transverse / mass[0] * self.BOLTZMAN * T * dt)
        random_transverse = torch.randn(len(v_ij_transverse), device=v.device).unsqueeze(1) * Langevin_coeff_transverse/dt * v_ij_unit
        
        friction_force = friction_parallel + friction_transverse
        random_force = random_parallel + random_transverse
        dpd_force = friction_force + random_force
        
        delta_v_langevin = torch.zeros_like(q, device=q.device)
        delta_v_langevin.index_add_(0, nbr_list[:,0], dpd_force)
        delta_v_langevin.index_add_(0, nbr_list[:,1], -dpd_force )
        
        return delta_v_langevin

class LSTM_memory(torch.nn.Module):
    def __init__(self, Natom, BOLTZMAN =  0.001987191):
        super(LSTM_memory, self).__init__()
        self.Natom = Natom
        self.BOLTZMAN =  BOLTZMAN
        self.lstm = torch.nn.LSTM(input_size=self.Natom*3, hidden_size=self.Natom*3, num_layers=3, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
            )
        
    def forward(self, v, T, dt, mass):
        v_reshaped = v.flatten().unsqueeze(1) # Example reshape, might need to be adjustedv
        lstm_output = self.mlp(torch.zeros_like(v_reshaped).to(v.device))
        
        # Reshape the output from LSTM to match the dimensions of `v`
        lstm_output = lstm_output.reshape(v.shape) # Adjust as needed
        
           
        # Using LSTM output instead of -v*self.gamma
        friction = -v * (lstm_output)
           
        gamma = (lstm_output)
        Langevin_coeff = torch.sqrt(2.0 * gamma / mass * self.BOLTZMAN * T * dt)
        random = torch.randn_like(v, device=v.device) * Langevin_coeff/dt
        delta_v_langevin = friction + random
        return delta_v_langevin


class Langevin_FF(torch.nn.Module):
    def __init__(self):
        super(Langevin_FF, self).__init__()
        self.log_friction_coeff = torch.nn.Parameter(torch.Tensor([-4.0]))
        self.log_random_coeff = torch.nn.Parameter(torch.Tensor([-4.0]))
        

    def forward(self, v):
        friction = -v*torch.exp(self.log_friction_coeff)
        
        random_coeff = torch.exp(self.log_random_coeff)
        random = random_coeff  * torch.randn_like(v, device=v.device)
        delta_v_langevin = friction + random
        return delta_v_langevin
    
    
    
    
class Friction_force(torch.nn.Module):
    def __init__(self):
        super(Friction_force, self).__init__()
        self.log_friction_coeff = torch.nn.Parameter(torch.Tensor([-1.0]))
        

    def forward(self, v):
        friction = -v*torch.sigmoid(self.log_friction_coeff)
        return friction
    
class Random_force(torch.nn.Module):
    def __init__(self):
        super(Random_force, self).__init__()
        self.log_random_coeff = torch.nn.Parameter(torch.Tensor([-2.0]))
        

    def forward(self, v):
        random_coeff = torch.exp(self.log_random_coeff)
        random = random_coeff  * torch.randn_like(v, device=v.device)
        return random


class Random_force_coeff(torch.nn.Module):
    def __init__(self):
        super(Random_force_coeff, self).__init__()
        self.log_random_coeff = torch.nn.Parameter(torch.Tensor([-3.0]))
        

    def forward(self, v):
        random_coeff = torch.exp(self.log_random_coeff) * torch.ones_like(v, device=v.device)
        #random_coeff = self.log_random_coeff * torch.ones_like(v, device=v.device)
        return random_coeff





class NNP(nn.Module):
    def __init__(self, cell, r=None,RDF=None, BOLTZMAN=None, Temp_target=None):
        super(NNP, self).__init__()
        self.fc1 = nn.Linear(1, 100).to(RDF.device)   # Input layer (1 input feature)
        self.fc2 = nn.Linear(100, 1).to(RDF.device)
        self.activation = nn.LeakyReLU()
        
        self.cell = cell
        
        if r!=None and RDF!=None and BOLTZMAN!=None and Temp_target!=None: #if all of them are given
            self.r = r
            self.PE_GT = - BOLTZMAN * Temp_target * torch.log(RDF)
            nan_mask = torch.isnan(self.PE_GT)
            inf_mask = torch.isinf(self.PE_GT)
            mask = nan_mask + inf_mask
            self.r = self.r[~mask]
            self.PE_GT = self.PE_GT[~mask]
            
            #self.force_GT = -torch.diff(self.PE_GT)/torch.diff(r[:2])
            #self.force_GT = torch.cat((self.force_GT, self.force_GT[-1].unsqueeze(0)))
            #nan_mask = torch.isnan(self.force_GT)
            #inf_mask = torch.isinf(self.force_GT)
            #mask = nan_mask + inf_mask
            #self.force_GT[mask] = torch.max(self.force_GT[~mask])
        
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)


        print("Pretraining...")
        for i in range(500000):
            if i%1000 ==0:
                print(i, "/", 500000)
            
            x = self.r.reshape(-1,1)
            x = self.activation(self.fc1(x))
            y_ = self.fc2(x).flatten()
            
            y = self.PE_GT
            loss = (y-y_).pow(2).sum()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
        
        print("Done")
        
        plt.plot(self.r.detach().cpu().numpy(), self.PE_GT.cpu().numpy(), 'k')
        plt.plot(self.r.detach().cpu().numpy(), y_.detach().cpu().numpy(), 'r')
        plt.show()

        

    def forward(self, x):
        nbr_list, offsets, pdist, unit_vector = generate_nbr_list(x, self.cell)

        #LJ_potential = self.LJ_function(pdist, self.sigma, self.epsilon)
        
        NNP = self.fc2(self.activation(self.fc1(pdist))) #+ self.prior_A*(1/pdist)**12
        
        return NNP
    
    
    
    

class NNP_REmin(nn.Module):
    def __init__(self, cell, r, PE, Force):
        super(NNP_REmin, self).__init__()
        self.fc1 = nn.Linear(1, 200).to(r.device)   # Input layer (1 input feature)
        self.fc2 = nn.Linear(200, 1).to(r.device)
        self.activation = nn.ELU()
        
        self.cell = cell
        
        self.r = r
        self.PE = PE
        self.Force = Force
        
        self.pre_train()
    
    def pre_train(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)


        print("Pretraining...")
        for i in range(20000):
            if i%1000 ==0:
                print(i, "/", 20000)
            
            x = self.r.reshape(-1,1)
            x = self.activation(self.fc1(x))
            y_ = self.fc2(x).flatten()
            
            y = self.PE
            loss = (y-y_).pow(2).sum()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
        
        print("Done")
        
        plt.plot(self.r.detach().cpu().numpy(), self.PE.cpu().numpy(), 'k')
        plt.plot(self.r.detach().cpu().numpy(), y_.detach().cpu().numpy(), 'r')
        plt.show()

        

    def forward(self, x):
        nbr_list, offsets, pdist, unit_vector = generate_nbr_list(x, self.cell)

        #LJ_potential = self.LJ_function(pdist, self.sigma, self.epsilon)
        
        NNP = self.fc2(self.activation(self.fc1(pdist))) #+ self.prior_A*(1/pdist)**12
        
        return NNP
    

class NNM(nn.Module):
    def __init__(self):
        super(NNM, self).__init__()
        self.gamma = torch.nn.Parameter(torch.tensor([0.1]))
        self.activation = torch.nn.functional.sigmoid
        
        self.log_std = torch.nn.Parameter(torch.tensor([0.1]))
        
    def forward(self, v):
        return -self.activation(self.gamma)*v + torch.randn_like(v, device=v.device) * torch.exp(self.log_std)

# %%

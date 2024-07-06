#%%  True SDE with LJ, complete
import os
#os.chdir('/workspace/jinu/trchmd')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn


#from torchmd.systems import System
#from torchmd.integrator import maxwell_boltzmann

material_number = 1
material_list = ["CH4", "CO2", "Water"]
mass_list = [16.04, 44.1, 18.01540]


## Tabulated potential
r_GT = torch.load("./"+material_list[material_number]+"/r_GT.pt")
RDF_GT = torch.load("./"+material_list[material_number]+"/RDF_GT.pt")
MSD_GT = torch.load("./"+material_list[material_number]+"/MSD_GT.pt")
y0 = torch.from_numpy(np.load("./"+material_list[material_number]+"/y0.npy"))

global Natom, device, mass, TIMEFACTOR, dt, Temp_target
Natom = len(y0)
device = "cuda:0"
dtype = torch.float32
mass = torch.full((Natom,1), mass_list[material_number]).to(device)
TIMEFACTOR = 48.88821
BOLTZMAN = 0.001987191
timestep = 1 #fs
Temp_target = 300 # 50 for argon, 300 for water

cell = torch.tensor(np.diag(np.array([40.0, 40.0, 40.0]))).to(device)

#%% Select force and load utility functions
from force import *

prior = LJ_repulsive(cell, sigma=torch.log(torch.tensor([3.5660])).to(device), epsilon=torch.log(torch.tensor([0.0698635])).to(device), Trainable = False).to(device)
schnet = SchNet(torch.tensor([0]*Natom).to(device), cell, n_gaussians=32, hidden_channels=64, n_filters=64, num_interactions=4, r_cut=8.0).to(device)
potential_model = CombinedNN(prior, schnet)

Thermostat = Langevin_TS(gamma = (1/PICOSEC2TIMEU), Trainable=True).to(device)

from utility import *

#%% Define differential equation

class f(torch.nn.Module):

    def __init__(self, potential, Thermostat, Temp_target, timestep, TIMEFACTOR, mass, non_integrand_mask, saver=False, append_dict=None):
        super().__init__()
        self.potential = potential
        self.Thermostat = Thermostat

        self.Temp_target = Temp_target
        self.dt = timestep / TIMEFACTOR
        
        self.mass = mass

        self.non_integrand_mask = non_integrand_mask
        
        self.force_mode = False
        
        if saver==True:
            self.saver = True
            self.pos_saver = []
            self.force_saver = []
        else:
            self.saver = False
        
        if append_dict != None:
            self.append_dict = append_dict
        else:
            self.append_dict = None

        self.Temperature_log = []
            

    def forward(self, t, state):
        original_state_shape = state.shape
        
        state = state.view(-1,3)
        Natom = len(state)//2
        
        with torch.set_grad_enabled(True):
            #3N
            v0 = state[:Natom].requires_grad_(True)            
            q0 = state[Natom:].requires_grad_(True)
            
            
            ######## Thermostat
            
            if self.Thermostat!=None:
                dv_dt_thermostat = self.Thermostat(v0, self.Temp_target, self.dt, self.mass)
            else:
                dv_dt_thermostat = 0
            
            ########
                        
            ######### first VV
            if self.force_mode==False:
                u0 = self.potential(q0).sum()
                f0 = -compute_grad(inputs=q0, output=u0)
            else:
                f0 = self.potential(q0)
            a0 = f0/self.mass
            
            if self.saver==True:
                self.pos_saver.append(q0.clone().detach().cpu())
                self.force_saver.append(f0.clone().detach().cpu())
            
            if self.append_dict!=None:
                write_xyz_dump(self.append_dict["file_name"], self.append_dict["atom_types"], self.append_dict["atom_types_map"], q0.unsqueeze(0), self.potential.cell, mode = "a")
            
            dqdt = v0 + 0.5*a0*self.dt
            q1 = q0 + dqdt *self.dt
            

            
            ######### second VV
            if self.force_mode==False:
                u1 = self.potential(q1).sum()
                f1 = -compute_grad(inputs=q1, output=u1)
            else:
                f1 = self.potential(q1)

            a1 = f1/self.mass
            dvdt = 0.5*(a0 + a1) + dv_dt_thermostat
            
            
            ######### Integrate only selected DOFs
            
        if self.non_integrand_mask!=None:
            dvdt[self.non_integrand_mask] = 0
            dqdt[self.non_integrand_mask] = 0
            
            
        
        KE = (self.mass*(v0.pow(2))).sum()/2
        T = kinetic_to_temp(KE, Natom)
        self.Temperature_log.append(T.item())

        return torch.concatenate((dvdt, dqdt)).view(original_state_shape)

#%%

func= f(potential_model, Thermostat=Thermostat, Temp_target=300, timestep=1, TIMEFACTOR=TIMEFACTOR, mass=mass, non_integrand_mask=None, saver=False)
func.force_mode=False

#%%
q, v = y0.to(device).to(dtype), maxwell_boltzmann(mass, T=Temp_target, replicas=1)[0].to(device)
y0 = torch.concatenate((v,q))
f0 = func(0., y0)

#%% Simulation prep
T = 2000

from torchdiffeq import odeint_adjoint
y0 = torch.concatenate((v,q))
t = (torch.tensor(np.arange(0, timestep/TIMEFACTOR*T, timestep/TIMEFACTOR))).to(device)

from pp import *
RDF = RDF_computer(cell, device)
MSD = MSD_computer(1000)


#%% Equilibration
with torch.no_grad():
    y = odeint_adjoint(func, y0, t, method="euler")

#%% compute RDF and MSD

r_curr, RDF_curr = RDF(y[::10,Natom:])
r_curr = r_curr.detach(); RDF_curr = RDF_curr.detach()
MSD_curr  = MSD(y[:,Natom:]).detach()

#%%
fix, ax = plt.subplots(2,1)
ax[0].plot(r_curr.detach().cpu().numpy(), RDF_curr.detach().cpu().numpy(), 'r')
ax[0].plot(r_GT.detach().cpu().numpy(), RDF_GT.detach().cpu().numpy(), 'k--')

ax[1].plot(MSD_curr.detach().cpu().numpy()[:2000], 'r')
ax[1].plot(MSD_GT.detach().cpu().numpy()[:2000], 'k')


plt.show()

KE = (mass*(y[:,:Natom].pow(2))).mean(0).sum()/2
Temp = kinetic_to_temp(KE, Natom)
print(Temp.item())




#%% MSD optimization
T = 50
MSD = MSD_computer(T)
y0 = torch.concatenate((v,q))
#%% 

optimizer_structure = torch.optim.AdamW(func.potential.parameters(), lr=0.001)
optimizer_dynamics = torch.optim.RMSprop(func.Thermostat.parameters(), lr=0.01)

for iter in range(250):
    # Forward pass: Compute predicted y by passing x to the model
    t = (torch.tensor(np.arange(0, timestep/TIMEFACTOR*T, timestep/TIMEFACTOR))).to(device)
    y = odeint_adjoint(func, y0, t, method="euler")
    y0 = y[1].detach()
    
    # Compute properties    
    r_curr, RDF_curr = RDF(y[::len(y)//5,Natom:])
    MSD_curr = (y[:,Natom:] -y[0,Natom:]).pow(2).mean((1,2))

    loss = 0
    if iter%2 ==0:
        loss += (RDF_curr[:min([len(RDF_curr), 200])]-RDF_GT.detach().to(device)[:min([len(RDF_curr), len(RDF_GT)])]).pow(2).sum()
        optimizer = optimizer_structure

        if loss < 0.1:
            T = min([T+50, len(MSD_GT)])

    else:
        loss += (MSD_curr[T-1] - MSD_GT[T-1].to(device)).pow(2).sum()
        optimizer = optimizer_dynamics
    
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
    
    
    fix, ax = plt.subplots(2)
    ax[0].plot(r_curr.detach().cpu().numpy(), RDF_curr.detach().cpu().numpy(), 'r')
    ax[0].plot(r_GT.detach().cpu().numpy(), RDF_GT.detach().cpu().numpy(), 'k--')

    ax[1].plot(MSD_curr[:T].detach().cpu().numpy(), 'r')
    ax[1].plot(MSD_GT[:T].detach().cpu().numpy(), 'k')
    
    #ax[2].plot(PACF_curr.detach().cpu().numpy())
    plt.show()
    

    #plt.plot(MSD.detach().cpu().numpy())
    #plt.show()
    #print(iter,  loss.item(), func.potential.sigma.item(), func.potential.epsilon.item(), func.Thermostat.gamma.item())
    print(iter,  loss.item(), np.exp(func.Thermostat.gamma_log.item()))
    print("")
# %%
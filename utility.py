import numpy as np
import torch

TIMEFACTOR = 48.88821
BOLTZMAN = 0.001987191
PICOSEC2TIMEU = 1000.0 / TIMEFACTOR


def kinetic_energy(masses, vel):
    Ekin = torch.sum(0.5 * torch.sum(vel * vel, dim=2,
                     keepdim=True) * masses, dim=1)
    return Ekin


def maxwell_boltzmann(masses, T, replicas=1):
    natoms = len(masses)
    velocities = []
    for i in range(replicas):
        velocities.append(
            torch.sqrt(T * BOLTZMAN / masses) *
            torch.randn((natoms, 3)).type_as(masses)
        )

    return torch.stack(velocities, dim=0)


def kinetic_to_temp(Ekin, natoms):
    return 2.0 / (3.0 * natoms * BOLTZMAN) * Ekin


def _first_VV(pos, vel, force, mass, dt):
    accel = force / mass
    pos = pos + vel * dt + 0.5 * accel * dt * dt
    vel = vel + 0.5 * dt * accel


def _second_VV(vel, force, mass, dt):
    accel = force / mass
    vel += 0.5 * dt * accel


def langevin(vel, mass, dt, T, device, gamma=0.1):

    coeff = torch.sqrt(2.0 * gamma / mass * BOLTZMAN * T * dt).to(device)
    csi = torch.randn_like(vel, device=device) * coeff
    delta_vel = -gamma * vel * dt + csi
    return delta_vel


def nose_hoover_thermostat(velocities, xi, Q, target_temperature, dt):
    # Assuming reduced units where Boltzmann's constant is 1. Adjust if needed.
    kB = 1.0
    N = velocities.size(0)
    kinetic_energy = 0.5 * torch.sum(velocities**2)

    # Update xi using its equation of motion
    dxi = dt * (kinetic_energy / N - kB * target_temperature) / Q
    xi += dxi

    # Update velocities considering the xi variable
    scaling_factor = torch.exp(-xi * dt)
    velocities *= scaling_factor

    return velocities, xi


class Integrator:
    def __init__(self, system, potential, memory, mass, timestep, device, T):
        self.dt = timestep / TIMEFACTOR
        self.system = system
        self.potential = potential
        self.memory = memory
        self.mass = mass
        self.device = device
        self.T = T
        self.gamma = 1 / PICOSEC2TIMEU

    def step(self, niter=1):

        natoms = len(self.mass)
        for _ in range(niter):
            q, cell = self.system.pos[0], self.system.box[0]
            #

            v = self.system.vel[0]
            v = v + langevin(v, self.mass, self.dt, self.T,
                             self.device, self.gamma)
            nbr_list, offsets, pdist, unit_vector = generate_nbr_list(q, cell)
            with torch.no_grad():
                u = self.potential(pdist)

            f = torch.zeros_like(q).to(self.device)
            f.index_add_(0, nbr_list[:, 0], u*unit_vector)
            f.index_add_(0, nbr_list[:, 1], -u*unit_vector)
            accel = f / self.mass
            q1 = q + v * self.dt + 0.5 * accel * self.dt * self.dt
            v1 = v + 0.5 * self.dt * accel

            nbr_list, offsets, pdist, unit_vector = generate_nbr_list(q1, cell)
            with torch.no_grad():
                u = self.potential(pdist)
            f = torch.zeros_like(q1).to(self.device)
            f.index_add_(0, nbr_list[:, 0], u*unit_vector)
            f.index_add_(0, nbr_list[:, 1], -u*unit_vector)

            accel = f / self.mass
            v2 = v + 0.5 * self.dt * accel

            self.system.pos = q1[None, :]
            self.system.vel = v2[None, :]

        Ekin = np.array([v.item()
                        for v in kinetic_energy(self.mass, self.system.vel)])
        T = kinetic_to_temp(Ekin, natoms)
        return Ekin, u.sum().item(), T


def compute_grad(inputs, output, create_graph=True, retain_graph=True):
    """Compute gradient of the scalar output with respect to inputs.

    Args:
        inputs (torch.Tensor): torch tensor, requires_grad=True
        output (torch.Tensor): scalar output 

    Returns:
        torch.Tensor: gradients with respect to each input component 
    """

    assert inputs.requires_grad

    gradspred, = torch.autograd.grad(output, inputs, grad_outputs=output.data.new(output.shape).fill_(1),
                                     create_graph=create_graph, retain_graph=retain_graph)

    return gradspred


def gaussian_smearing(centered, sigma):
    return 1/(sigma*(2*np.pi)**0.5)*torch.exp(-0.5*(centered/sigma)**2)







def generate_nbr_list(coordinates, lattice_matrix, cutoff=9.0):
    
    lattice_matrix_diag = torch.diag(lattice_matrix).view(1, 1, -1)
    
    device = coordinates.device
    displacement = (
        coordinates[..., None, :, :] - coordinates[..., :, None, :])

    # Transform distance using lattice matrix inverse
    offsets = ((displacement+lattice_matrix_diag/2) // lattice_matrix_diag).detach()
    
    
    # Apply periodic boundary conditions
    displacement = displacement - offsets*(lattice_matrix_diag)

    # Compute squared distances and create mask for cutoff
    squared_displacement = torch.triu(displacement.pow(2).sum(-1))
    
    within_cutoff = (squared_displacement < cutoff **2) & (squared_displacement != 0)
    neighbor_indices = torch.nonzero(within_cutoff.to(torch.long), as_tuple=False)
    
    
    offsets = offsets[neighbor_indices[:, 0], neighbor_indices[:, 1], :]

    # Compute unit vectors and actual distances    
    unit_vectors = displacement[neighbor_indices[:,0], neighbor_indices[:,1]]
    magnitudes = squared_displacement[neighbor_indices[:,0], neighbor_indices[:,1]].sqrt()
    
    
    
    unit_vectors = unit_vectors / magnitudes.view(-1, 1)
    
    actual_distances = magnitudes[:, None]

    return neighbor_indices.detach(), offsets, actual_distances, -unit_vectors



def compute_RDF_new(Traj, cell, device):
    L_max = cell[0, 0]//2
    dr = 0.1
    r_list = torch.arange(0.5*dr, L_max - dr*2, dr, device=device)

    Hist = []
    for t, q in enumerate(Traj):
        nbr_list, offsets, pdist, unit_vector = generate_nbr_list(
            q, cell, cutoff=9.0)

        pdist_gaussian = gaussian_smearing(pdist-r_list, 0.3).sum(0)*dr
        if t == 0:
            Pdist_gaussian = pdist_gaussian
        else:
            Pdist_gaussian += pdist_gaussian

    Pdist_gaussian /= (t+1)

    v = 4 * np.pi / 3 * ((r_list+0.5*dr)**3 - (r_list-0.5*dr)**3)
    natom = len(Traj[0])
    bulk_density = (natom-1)/(torch.det(cell))
    gr = Pdist_gaussian/v * (torch.det(cell))/(natom-1)/natom*2

    return r_list, gr


class RDF_computer(torch.nn.Module):
    def __init__(self, cell, device, L_max=None):
        super(RDF_computer, self).__init__()
        self.cell = cell

        if L_max == None:
            self.L_max = self.cell[0, 0]//2
        else:
            self.L_max = L_max
        self.dr = 0.1
        self.device = device
        self.r_list = torch.arange(
            0.5*self.dr, self.L_max - self.dr*2, self.dr, device=self.device)

    def forward(self, Traj):
        Hist = []
        for t, q in enumerate(Traj):
            nbr_list, offsets, pdist, unit_vector = generate_nbr_list(
                q, self.cell, cutoff=self.L_max)

            pdist_gaussian = gaussian_smearing(
                pdist-self.r_list, self.dr).sum(0)*self.dr
            if t == 0:
                Pdist_gaussian = pdist_gaussian
            else:
                Pdist_gaussian += pdist_gaussian

        Pdist_gaussian /= (t+1)

        v = 4 * np.pi / 3 * ((self.r_list+0.5*self.dr) **
                             3 - (self.r_list-0.5*self.dr)**3)
        natom = len(Traj[0])
        bulk_density = (natom-1)/(torch.det(self.cell))
        gr = Pdist_gaussian/v * (torch.det(self.cell))/(natom-1)/natom*2

        return self.r_list, gr



class MSD_computer(torch.nn.Module):
    def __init__(self, td_max, ensemble_average=False):
        super(MSD_computer, self).__init__()
        self.td_max = td_max
        self.ensemble_average = ensemble_average


    def forward(self, x):
        if len(x.shape) == 2:
            x = x.view(len(x), -1, 3)

        if self.ensemble_average==True:
            t0_list = list(range(0, len(x)-self.td_max))
        else:
            t0_list = [0]
        MSD = torch.zeros(self.td_max, device=x.device)

        for t0 in t0_list:
            MSD += (x[t0:t0+self.td_max] - x[t0]).pow(2).mean((1, 2))
        MSD /= len(t0_list)
        return MSD



class ACF_computer(torch.nn.Module):
    def __init__(self, td_max, ensemble_average=False):
        super(ACF_computer, self).__init__()
        self.td_max = td_max
        self.ensemble_average = ensemble_average
        

    def forward(self, v):
        
        if self.ensemble_average==True:
            t0_list = list(range(0, len(v)-self.td_max))
        else:
            t0_list = [0]

        VACF = torch.zeros(self.td_max, device=v.device)

        for t0 in t0_list:
            if len(v.shape) == 3:
                VACF += (v[t0:t0+self.td_max] * v[t0].detach()).mean((1, 2))
            elif len(v.shape)==2:
                VACF += (v[t0:t0+self.td_max] * v[t0].detach()).mean((1))
        VACF /= len(t0_list)
        return VACF


class ACF_sum_computer(torch.nn.Module):
    def __init__(self, td_max, ensemble_average=False):
        super(ACF_sum_computer, self).__init__()
        self.td_max = td_max
        self.ensemble_average = ensemble_average
        

    def forward(self, v):
        
        if self.ensemble_average==True:
            t0_list = list(range(0, len(v)-self.td_max))
        else:
            t0_list = [0]

        ACF_sum = 0

        for t0 in t0_list:
            if len(v.shape) == 3:
                ACF_sum += (v[t0:t0+self.td_max] * v[t0].detach()).mean((1, 2)).sum()
            elif len(v.shape)==2:
                ACF_sum += (v[t0:t0+self.td_max] * v[t0].detach()).mean((1)).sum()
        ACF_sum /= len(t0_list)
        return ACF_sum


class pressure_computer(torch.nn.Module):
    def __init__(self, potential):
        super(pressure_computer, self).__init__()
        self.potential = potential
        

    def forward(self, mass, y, cell, KE_only=False):
        Natom = y.shape[1] // 2
        
        Q = y[:,Natom:]
        V = y[:,:Natom]
        
        
        # KE contribution
        assert len(V.shape)==3
        vol = cell.det().item() * 1e-30
        unit_conversion = 1/0.001987191 * 1.380649 * 1e-23 # one over kB_torchmd multiplied by kB_SI unit
        2.99 * 1e-26 * (1e-10 / (0.02045482949774598 * 1e-15))**2 * 1e30
        
        
        p_xy = (V[:,:,0]*V[:,:,1] * mass.unsqueeze(0).squeeze(2)).sum(1) / vol *unit_conversion
        p_xz = (V[:,:,0]*V[:,:,2] * mass.unsqueeze(0).squeeze(2)).sum(1) / vol *unit_conversion
        p_yz = (V[:,:,1]*V[:,:,2] * mass.unsqueeze(0).squeeze(2)).sum(1) / vol *unit_conversion
        
        
        if KE_only==True:
            return torch.cat([p_xy.unsqueeze(1), p_xz.unsqueeze(1), p_yz.unsqueeze(1)], dim=1)
        
        # PE contribution
        virial_xy = torch.zeros_like(p_xy, device=p_xy.device)
        virial_xz = torch.zeros_like(p_xz, device=p_xz.device)
        virial_yz = torch.zeros_like(p_yz, device=p_yz.device)
        with torch.no_grad():
            for t, q in enumerate(Q):
                nbr_list, offsets, pdist, unit_vector = generate_nbr_list(q, self.potential.cell)
                r_k = pdist * unit_vector
                r_k_x, r_k_y, r_k_z = r_k[:,0], r_k[:,1], r_k[:,2]
                
                force_magnitude = torch.tensor(self.potential.force_magnitude(pdist.detach().cpu().numpy())).to(q.device).to(q.dtype)
                f_k= force_magnitude*unit_vector
                
                f_k_x, f_k_y, f_k_z = f_k[:,0], f_k[:,1], f_k[:,2]
                
                virial_xy[t] = (r_k_x*f_k_y).sum()*2 / vol * 6.946704300182635e-24# real energy unit to SI unit
                virial_xz[t] = (r_k_x*f_k_z).sum()*2 / vol * 6.946704300182635e-24# real energy unit to SI unit
                virial_yz[t] = (r_k_y*f_k_z).sum()*2 / vol * 6.946704300182635e-24# real energy unit to SI unit
        
        p_xy += virial_xy.detach()
        p_xz += virial_xz.detach()
        p_yz += virial_yz.detach()
        
        
        return torch.cat([p_xy.unsqueeze(1), p_xz.unsqueeze(1), p_yz.unsqueeze(1)], dim=1)



#%%


def distribute_particles_in_cubic_space(L, Natom):
    # Calculate the number of particles per dimension (assuming a cubic grid)
    num_per_dimension = int(np.ceil(Natom**(1/3)))

    # Calculate the spacing between particles
    spacing = L / num_per_dimension

    # Generate particle coordinates
    particle_coordinates = []
    for x in range(num_per_dimension):
        for y in range(num_per_dimension):
            for z in range(num_per_dimension):
                x_coord = x * spacing
                y_coord = y * spacing
                z_coord = z * spacing
                particle_coordinates.append([x_coord, y_coord, z_coord])

    # Convert the list of coordinates to a NumPy array
    particle_coordinates = np.array(particle_coordinates)

    return particle_coordinates[:Natom]




# Model construction
class CombinedNN(torch.nn.Module):
    def __init__(self, nn1, nn2):
        super(CombinedNN, self).__init__()
        self.nn1 = nn1
        self.nn2 = nn2

    def forward(self, x):
        return self.nn1(x).sum() + self.nn2(x).sum()



'''
def compute_force(system, potential):
    q, cell = system.pos, system.box[0]
    q.requires_grad_(True)
    nbr_list, offsets, pdist = generate_nbr_list(q, cell)
    u = potential(pdist)
    f = -compute_grad(inputs=q, output=u.sum(-1))
    return f


def compute_memory(system, NNM):
    v = system.vel
    
    u = potential(pdist)
    f = -compute_grad(inputs=q, output=u.sum(-1))
    return f
'''

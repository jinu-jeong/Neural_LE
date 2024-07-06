import os
import numpy as np
import matplotlib.pyplot as plt

def my_position2distance(pos, L):
    R_hyp = np.reshape(pos,(1,-1,3)) # hyper 2d matrix (1,Natom,Ndim), which is regarded as
    R_hyp_t = np.transpose(R_hyp,(1,0,2)) # hyper 2d matrix being transposed (Natom,1,Ndim)
    rij = R_hyp - R_hyp_t
    rij = my_disp_in_box(rij, L)
    dij = np.sum(rij*rij,axis=2)**0.5
    return dij

def my_pos_in_box(pos, lbox): # I worngly named this function. It should be "PCBwrap", but I have been using this in all of my code since my first year, so I didn't change it.
    a = pos
    L = lbox
    return (a+L/2)%L-L/2

def my_disp_in_box(drij, lbox):
    drij_pbc = my_pos_in_box(drij, lbox)
    return drij_pbc

def my_histogram_distances(dists, nbins, dr):
    counts = np.zeros(nbins, dtype=int)
    for i in range(nbins):
        r0_local = i*dr
        r1_local = (i+1)*dr
        dists_True = np.sum((dists>r0_local) & (dists<=r1_local)) # np.sum( dists btw r0 to
        counts[i] = dists_True
    return counts/2

def my_pair_correlation(dists, natom, nbins, dr, lbox):
    counts = my_histogram_distances(dists, nbins, dr)
    r = np.linspace(0,dr*nbins,num=nbins+1)[:-1]+0.5*dr
    v = 4 * np.pi / 3 *((r+0.5*dr)**3 - (r-0.5*dr)**3)
    bulk_density = (natom-1)/(lbox**3)
    gr = counts/v * (lbox**3)/(natom-1)/natom*2
    return r, gr

def compute_RDF(COORD, lbox, N_oxygen_atoms):
    RDF = []
    for i in range(len(COORD)):
        dist_local = my_position2distance(COORD[i], lbox)
        dist_local = my_disp_in_box(dist_local, lbox)
        r, gr = my_pair_correlation(dist_local, N_oxygen_atoms, 300, 0.1, lbox)
        RDF.append(gr)
    RDF_final = np.mean(RDF, axis=0)
    return r, RDF_final

#def compute_differentiable_pair_distribution(y, box, s = 1.0):
#    nbr_list, offsets, pdist, unit_vector = generate_nbr_list(y, cell)
#    Histogram_gaussian = pdist
    
    
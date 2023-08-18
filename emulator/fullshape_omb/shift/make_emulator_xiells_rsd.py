import numpy as np
import json
import sys
import os
from mpi4py import MPI

from compute_fid_dists import compute_fid_dists
from taylor_approximation import compute_derivatives
from make_pkclass import make_pkclass_dists

# np.seterr(over='raise')
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()
if mpi_rank==0:
    print(sys.argv[0]+" running on {:d} processes.".format(mpi_size))
#print( "Hello I am process %d of %d." %(mpi_rank, mpi_size) )

basedir = sys.argv[1] +'/'
z = float(sys.argv[2])
Omfid = float(sys.argv[3])

# Compute fiducial distances
_,fid_dists = make_pkclass_dists(z=z)
# h = 0.6766
# speed_of_light = 2.99792458e5
# Hz_fid = fid_dist_class.Hubble(z) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
# chiz_fid = fid_dist_class.angular_distance(z) * (1.+z) * h # this is the comoving radius in units of Mpc/h 
# fid_dists = (Hz_fid, chiz_fid)

# Set up the output k vector:
from compute_xiell_tables_rsd import compute_xiell_tables, rout

output_shape = (len(rout),19) # two multipoles and 19 types of terms


# First construct the grid

order = 4
# these are omega_b,omega_cdm, h, sigma8
x0s = [0.02237, 0.1200,0.68, 0.8]; Nparams = len(x0s) # these are chosen to be roughly at the BOSS best fit value
dxs = [0.0001, 0.01,0.01, 0.05]

template = np.arange(-order,order+1,1)
Npoints = 2*order + 1
grid_axes = [ dx*template + x0 for x0, dx in zip(x0s,dxs)]

Inds   = np.meshgrid( * (np.arange(Npoints),)*Nparams, indexing='ij')
Inds = [ind.flatten() for ind in Inds]
center_ii = (order,)*Nparams
Coords = np.meshgrid( *grid_axes, indexing='ij')

Xi0grid = np.zeros( (Npoints,)*Nparams+ output_shape)
Xi2grid = np.zeros( (Npoints,)*Nparams+ output_shape)
Xi4grid = np.zeros( (Npoints,)*Nparams+ output_shape)

Xi0gridii = np.zeros( (Npoints,)*Nparams+ output_shape)
Xi2gridii = np.zeros( (Npoints,)*Nparams+ output_shape)
Xi4gridii = np.zeros( (Npoints,)*Nparams+ output_shape)

for nn, iis in enumerate(zip(*Inds)):
    if nn%mpi_size == mpi_rank:
        coord = [Coords[i][iis] for i in range(Nparams)]
        print(coord,iis)
        # try:
        rout,xi0, xi2, xi4 = compute_xiell_tables(coord,z=z,fid_dists=fid_dists)
        # except RuntimeWarning as r:
        #     print("Overflow error happened")
        #     print("coords were", coord)
            
        Xi0gridii[iis] = xi0
        Xi2gridii[iis] = xi2
        Xi4gridii[iis] = xi4
        
comm.Allreduce(Xi0gridii, Xi0grid, op=MPI.SUM)
comm.Allreduce(Xi2gridii, Xi2grid, op=MPI.SUM)
comm.Allreduce(Xi4gridii, Xi4grid, op=MPI.SUM)

del(Xi0gridii, Xi2gridii, Xi4gridii)

# Now compute the derivatives
derivs0 = compute_derivatives(Xi0grid, dxs, center_ii, 5)
derivs2 = compute_derivatives(Xi2grid, dxs, center_ii, 5)
derivs4 = compute_derivatives(Xi4grid, dxs, center_ii, 5)

if mpi_rank == 0:
    # Make the emulator (emu) directory if it
    # doesn't already exist.
    fb = basedir+'emu'
    if not os.path.isdir(fb):
        print("Making directory ",fb)
        os.mkdir(fb)
    else:
        print("Found directory ",fb)
    #
comm.Barrier()

# Now save:
outfile = basedir + 'emu/rsd_z_%.2f_xiells.json'%(z)

list0 = [ dd.tolist() for dd in derivs0 ]
list2 = [ dd.tolist() for dd in derivs2 ]
list4 = [ dd.tolist() for dd in derivs4 ]

outdict = {'params': ['omega_b','omega_cdm', 'h', 'sigma8'],\
           'x0': x0s,\
           'rout': rout.tolist(),\
           'derivs0': list0,\
           'derivs2': list2,\
           'derivs4': list4}

if mpi_rank == 0:
    json_file = open(outfile, 'w')
    json.dump(outdict, json_file)
    json_file.close()


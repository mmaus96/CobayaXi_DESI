#!/bin/bash -l
#SBATCH -J Fit_fs_mock
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -o Mock_fit.out
#SBATCH -e Mock_fit.err
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -A desi

date
#
module load python
conda activate cobaya
#

export PYTHONPATH=${PYTHONPATH}:/pscratch/sd/m/mmaus/CobayaXi_DESI
export PYTHONPATH=${PYTHONPATH}:/pscratch/sd/m/mmaus/CobayaXi_DESI/rsd_likelihood
# export PYTHONPATH=${PYTHONPATH}:/global/cscratch1/sd/mmaus/new_template/Cobaya_template/emulator/template/emu

echo "Setup done.  Starting to run code ..."

mpirun -n 8 -c 8 cobaya-run --debug fs_abacus_am_omb.yaml 
# mpirun -n 8 -c 8 cobaya-run --debug fs_abacus_am_omb-b3.yaml 
# mpirun -n 8 -c 8 cobaya-run fs_abacus_am_omb_ELG.yaml --resume
# mpirun -n 8 -c 8 cobaya-run fs_abacus_am_omb_joint.yaml -r
# mpirun -n 8 -c 8 cobaya-run fs_abacus_am_omb-xplanck.yaml 
# mpirun -n 8 -c 8 cobaya-run fs_abacus_rsd_bao_LRG_am_omb.yaml 
# srun -n 8 -c 8 --unbuffered cobaya-run --debug /global/cscratch1/sd/mmaus/new_template/Cobaya_template/cosmo_chains/mocks/LRG/abacus_fid/mean/pk

#!/bin/bash -l
#SBATCH -J MakePreal
#SBATCH -t 0:30:00
#SBATCH -N 1
#SBATCH -o MakePreal.out
#SBATCH -e MakePreal.err
#SBATCH -p debug
#SBATCH -C haswell
#SBATCH -A desi
#

date
#
module load python
conda activate cobaya
#
export PYTHONPATH=${PYTHONPATH}:/pscratch/sd/m/mmaus/CobayaXi_DESI/emulator/fullshape_omb/shift
export PYTHONPATH=${PYTHONPATH}:/global/homes/m/mmaus/Python/velocileptors
export PYTHONPATH=${PYTHONPATH}:/pscratch/sd/m/mmaus/Cobaya/Packages/code
export OMP_NUM_THREADS=2
#
echo "Setup done.  Starting to run code ..."
#
# Now run the code -- we do Preal and Phalofit here.
#
mpirun -np 32 python make_emulator_xiells_rsd.py  $PWD 0.8 0.315

#
date
#

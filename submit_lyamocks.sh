#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=128 # tasks out of 128  
#SBATCH -C cpu             
#SBATCH -q regular               
#SBATCH -J mk2v8.8
#SBATCH -o lyatestlog       
#SBATCH --mail-user=fsin_ext@iac.es     
#SBATCH --mail-type=ALL   
#SBATCH -t 03:00:00                                                                                                       
#SBATCH --mem=475G

#export OMP_NUM_THREADS=16
#module load spack
#module load gnu/8.4.0
#module load gsl/2.6--gnu--8.4.0
#spack load gsl
#module load fftw/3.3.8--gnu--8.4.0
#module load gsl/2.7
#module load cray-fftw/3.3.10.3 

module load python
export NUMBA_NUM_THREADS=256

cp auxarr/dcomOM0.314OL0.686.DAT .
cp auxarr/zarr.DAT .

# QSO catalog
python3 make_catalog_lightcone.py
python3 write_catalog.py
python3 rewrite_qso_cat_for_xcf.py

# DLA catalog
conda run -n pyigm python3 make_catalog_DLA.py
python3 write_catalog_DLA.py

# Lya catalog
#python3 make_mocks_cells.py         #in all the other cases

cp auxarr/dcomOM0.314OL0.686.DAT .
cp auxarr/zarr.DAT .

python3 extract_and_regrid_gaussian.py

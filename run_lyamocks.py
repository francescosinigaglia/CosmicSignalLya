import os

#module load python
#export NUMBA_NUM_THREADS=256

os.system('cp auxarr/dcomOM0.314OL0.686.DAT .')
os.system('cp auxarr/zarr.DAT .')

# QSO catalog                                                                                                                                                
os.system('python3 make_catalog_lightcone.py')
os.system('python3 write_catalog.py')
os.system('python3 rewrite_qso_cat_for_xcf.py')

# DLA catalog                                                                                                                                                
os.system('conda run -n pyigm python3 make_catalog_DLA.py')
os.system('python3 write_catalog_DLA.py')

# Lya catalog                                                                                                                                                
os.system('python3 make_mocks_cells.py')         #in all the other cases                 

os.system('cp auxarr/dcomOM0.314OL0.686.DAT .')
os.system('cp auxarr/zarr.DAT .')

os.system('python3 extract_and_regrid_gaussian.py')

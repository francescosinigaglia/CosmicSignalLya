import numpy as np

# RSD parameters
#bbpars = [1.,1.,1., 1., 1.]
#betapars = [1.2,1.2,1.2,1.2, 1.2]
#bvpars = [1.,1.,1.,1., 1.]

# Bias parameters
zzarrbias = [0., 1.1, 1.4, 1.7, 2.5, 4.]

#meandensarr = [0.6, 0.6, 0.6, 0.6, 0.5, 0.4] 
meandensarr = [0.8, 0.7, 0.5, 0.3, 0.25, 0.2]

def make_pars_list():
    nmean_arr = np.zeros((len(zzarrbias),4,4))
    alpha_arr = np.zeros((len(zzarrbias),4,4))
    beta_arr = np.zeros((len(zzarrbias),4,4))
    dth_arr = np.zeros((len(zzarrbias),4,4))
    rhoeps_arr = np.zeros((len(zzarrbias),4,4))
    eps_arr = np.zeros((len(zzarrbias),4,4))

    bb_arr = np.zeros((len(zzarrbias),4))
    betarsd_arr = np.zeros((len(zzarrbias),4))
    bv_arr = np.zeros((len(zzarrbias),4))
    gamma_arr = np.zeros((len(zzarrbias),4))

    # Knots
    wnmean11=[]; walpha11=[]; wbeta11=[]; wdth11=[]; wrhoeps11=[]; weps11=[]
    wnmean12=[]; walpha12=[]; wbeta12=[]; wdth12=[]; wrhoeps12=[]; weps12=[]
    wnmean13=[]; walpha13=[]; wbeta13=[]; wdth13=[]; wrhoeps13=[]; weps13=[]
    wnmean14=[]; walpha14=[]; wbeta14=[]; wdth14=[]; wrhoeps14=[]; weps14=[]
    
    wnmean21=[]; walpha21=[]; wbeta21=[]; wdth21=[]; wrhoeps21=[]; weps21=[]
    wnmean22=[]; walpha22=[]; wbeta22=[]; wdth22=[]; wrhoeps22=[]; weps22=[]
    wnmean23=[]; walpha23=[]; wbeta23=[]; wdth23=[]; wrhoeps23=[]; weps23=[]
    wnmean24=[]; walpha24=[]; wbeta24=[]; wdth24=[]; wrhoeps24=[]; weps24=[]
    
    wnmean31=[]; walpha31=[]; wbeta31=[]; wdth31=[]; wrhoeps31=[]; weps31=[]
    wnmean32=[]; walpha32=[]; wbeta32=[]; wdth32=[]; wrhoeps32=[]; weps32=[]
    wnmean33=[]; walpha33=[]; wbeta33=[]; wdth33=[]; wrhoeps33=[]; weps33=[]
    wnmean34=[]; walpha34=[]; wbeta34=[]; wdth34=[]; wrhoeps34=[]; weps34=[]
    
    wnmean41=[]; walpha41=[]; wbeta41=[]; wdth41=[]; wrhoeps41=[]; weps41=[]
    wnmean42=[]; walpha42=[]; wbeta42=[]; wdth42=[]; wrhoeps42=[]; weps42=[]
    wnmean43=[]; walpha43=[]; wbeta43=[]; wdth43=[]; wrhoeps43=[]; weps43=[]
    wnmean44=[]; walpha44=[]; wbeta44=[]; wdth44=[]; wrhoeps44=[]; weps44=[]
    
    # ******************************************************
    # ********* z=1.1 **************************************
    # Delta-web

    dth = -1.

    nmean11=2.21; alpha11 = 0.6;   beta11 = 1.;  dth11 = -0.4; rho_eps11 = 0.83; eps11 = 0.42 #KK                                                                  \
                                                                                                                                                                    
    nmean12=0.43; alpha12 = 0.5;   beta12 = 5.8; dth12 = -0.8; rho_eps12 = 0.12; eps12 = -0.17 #KF                                                                 \
                                                                                                                                                                    
    nmean13=0.30; alpha13 = 0.01;  beta13 = 4.5; dth13 = -0.3; rho_eps13 = 0.2;  eps13 = -0.5 #KS                                                                  \
                                                                                                                                                                    
    nmean14=0.0;  alpha14 = 2.8;   beta14 = 8.7; dth14 = -0.9; rho_eps14 = 0.7;  eps14 = 2. #KV                                                                    \
                                                                                                                                                                    

    nmean21=0.68; alpha21 = 0.8;  beta21 = 5.5; dth21 = -0.2;  rho_eps21 = 1.4; eps21 = 0.5  #FK                                                                   \
                                                                                                                                                                    
    nmean22=0.34; alpha22 = 0.28; beta22 = 5.2; dth22 = -0.8;  rho_eps22 = 1.1; eps22 = -0.45 #FF                                                                  \
                                                                                                                                                                    
    nmean23=0.13; alpha23 = 0.6;  beta23 = 2.9; dth23 = -0.38; rho_eps23 = 0.5; eps23 = -0.42 #FS                                                                  \
                                                                                                                                                                    
    nmean24=0.02; alpha24 = 2.8;  beta24 = 5.2; dth24 = -1.;   rho_eps24 = 2.5; eps24 = 20. #FV                                                                    \
                                                                                                                                                                    

    nmean31=0.16; alpha31 = 0.3; beta31 = 3.1;  dth31 = -0.2; rho_eps31 = 1.6; eps31 = -0.2 #SK                                                                    \
                                                                                                                                                                    
    nmean32=0.16; alpha32 = 0.3; beta32 = 3.1;  dth32 = -0.2; rho_eps32 = 1.6; eps32 = -0.2 #SF                                                                    \
                                                                                                                                                                    
    nmean33=0.16; alpha33 = 0.3; beta33 = 3.1;  dth33 = -0.2; rho_eps33 = 1.6; eps33 = -0.2 #SS                                                                    \
                                                                                                                                                                    
    nmean34=0.16; alpha34 = 0.3; beta34 = 3.1;  dth34 = -0.2; rho_eps34 = 1.6; eps34 = -0.2 #SV                                                                    \
                                                                                                                                                                    

    nmean41=0.02; alpha41 = 4.; beta41 = 15.3; dth41 = -1.; rho_eps41 = 8.5; eps41 = 30. #VK                                                                       \
                                                                                                                                                                    
    nmean42=0.02; alpha42 = 4.; beta42 = 15.3; dth42 = -1.; rho_eps42 = 8.5; eps42 = 30.   #VF                                                                     \
                                                                                                                                                                    
    nmean43=0.02; alpha43 = 4.; beta43 = 15.3; dth43 = -1.; rho_eps43 = 8.5; eps43 = 30. #VS                                                                       \
                                                                                                                                                                    
    nmean44=0.02; alpha44 = 4.; beta44 = 15.3; dth44 = -1.; rho_eps44 = 8.5; eps44 = 30. #VV  

    # Do it one first time for the lowest z interpolation node
    wnmean11.append(nmean11); walpha11.append(alpha11); wbeta11.append(beta11); wdth11.append(dth11); wrhoeps11.append(rho_eps11); weps11.append(eps11)
    wnmean12.append(nmean12); walpha12.append(alpha12); wbeta12.append(beta12); wdth12.append(dth12); wrhoeps12.append(rho_eps12); weps12.append(eps12)
    wnmean13.append(nmean13); walpha13.append(alpha13); wbeta13.append(beta13); wdth13.append(dth13); wrhoeps13.append(rho_eps13); weps13.append(eps13)
    wnmean14.append(nmean14); walpha14.append(alpha14); wbeta14.append(beta14); wdth14.append(dth14); wrhoeps14.append(rho_eps14); weps14.append(eps14)

    wnmean21.append(nmean21); walpha21.append(alpha21); wbeta21.append(beta21); wdth21.append(dth21); wrhoeps21.append(rho_eps21); weps21.append(eps21)
    wnmean22.append(nmean22); walpha22.append(alpha22); wbeta22.append(beta22); wdth22.append(dth22); wrhoeps22.append(rho_eps22); weps22.append(eps22)
    wnmean23.append(nmean23); walpha23.append(alpha23); wbeta23.append(beta23); wdth23.append(dth23); wrhoeps23.append(rho_eps23); weps23.append(eps23)
    wnmean24.append(nmean24); walpha24.append(alpha24); wbeta24.append(beta24); wdth24.append(dth24); wrhoeps24.append(rho_eps24); weps24.append(eps24)

    wnmean31.append(nmean31); walpha31.append(alpha31); wbeta31.append(beta31); wdth31.append(dth31); wrhoeps31.append(rho_eps31); weps31.append(eps31)
    wnmean32.append(nmean32); walpha32.append(alpha32); wbeta32.append(beta32); wdth32.append(dth32); wrhoeps32.append(rho_eps32); weps32.append(eps32)
    wnmean33.append(nmean33); walpha33.append(alpha33); wbeta33.append(beta33); wdth33.append(dth33); wrhoeps33.append(rho_eps33); weps33.append(eps33)
    wnmean34.append(nmean34); walpha34.append(alpha34); wbeta34.append(beta34); wdth34.append(dth34); wrhoeps34.append(rho_eps34); weps34.append(eps34)

    wnmean41.append(nmean41); walpha41.append(alpha41); wbeta41.append(beta41); wdth41.append(dth41); wrhoeps41.append(rho_eps41); weps41.append(eps41)
    wnmean42.append(nmean42); walpha42.append(alpha42); wbeta42.append(beta42); wdth42.append(dth42); wrhoeps42.append(rho_eps42); weps42.append(eps42)
    wnmean43.append(nmean43); walpha43.append(alpha43); wbeta43.append(beta43); wdth43.append(dth43); wrhoeps43.append(rho_eps43); weps43.append(eps43)
    wnmean44.append(nmean44); walpha44.append(alpha44); wbeta44.append(beta44); wdth44.append(dth44); wrhoeps44.append(rho_eps44); weps44.append(eps44)


    # First real redshift
    wnmean11.append(nmean11); walpha11.append(alpha11); wbeta11.append(beta11); wdth11.append(dth11); wrhoeps11.append(rho_eps11); weps11.append(eps11)
    wnmean12.append(nmean12); walpha12.append(alpha12); wbeta12.append(beta12); wdth12.append(dth12); wrhoeps12.append(rho_eps12); weps12.append(eps12)
    wnmean13.append(nmean13); walpha13.append(alpha13); wbeta13.append(beta13); wdth13.append(dth13); wrhoeps13.append(rho_eps13); weps13.append(eps13)
    wnmean14.append(nmean14); walpha14.append(alpha14); wbeta14.append(beta14); wdth14.append(dth14); wrhoeps14.append(rho_eps14); weps14.append(eps14)
    
    wnmean21.append(nmean21); walpha21.append(alpha21); wbeta21.append(beta21); wdth21.append(dth21); wrhoeps21.append(rho_eps21); weps21.append(eps21)
    wnmean22.append(nmean22); walpha22.append(alpha22); wbeta22.append(beta22); wdth22.append(dth22); wrhoeps22.append(rho_eps22); weps22.append(eps22)
    wnmean23.append(nmean23); walpha23.append(alpha23); wbeta23.append(beta23); wdth23.append(dth23); wrhoeps23.append(rho_eps23); weps23.append(eps23)
    wnmean24.append(nmean24); walpha24.append(alpha24); wbeta24.append(beta24); wdth24.append(dth24); wrhoeps24.append(rho_eps24); weps24.append(eps24)
    
    wnmean31.append(nmean31); walpha31.append(alpha31); wbeta31.append(beta31); wdth31.append(dth31); wrhoeps31.append(rho_eps31); weps31.append(eps31)
    wnmean32.append(nmean32); walpha32.append(alpha32); wbeta32.append(beta32); wdth32.append(dth32); wrhoeps32.append(rho_eps32); weps32.append(eps32)
    wnmean33.append(nmean33); walpha33.append(alpha33); wbeta33.append(beta33); wdth33.append(dth33); wrhoeps33.append(rho_eps33); weps33.append(eps33)
    wnmean34.append(nmean34); walpha34.append(alpha34); wbeta34.append(beta34); wdth34.append(dth34); wrhoeps34.append(rho_eps34); weps34.append(eps34)
    
    wnmean41.append(nmean41); walpha41.append(alpha41); wbeta41.append(beta41); wdth41.append(dth41); wrhoeps41.append(rho_eps41); weps41.append(eps41)
    wnmean42.append(nmean42); walpha42.append(alpha42); wbeta42.append(beta42); wdth42.append(dth42); wrhoeps42.append(rho_eps42); weps42.append(eps42)
    wnmean43.append(nmean43); walpha43.append(alpha43); wbeta43.append(beta43); wdth43.append(dth43); wrhoeps43.append(rho_eps43); weps43.append(eps43)
    wnmean44.append(nmean44); walpha44.append(alpha44); wbeta44.append(beta44); wdth44.append(dth44); wrhoeps44.append(rho_eps44); weps44.append(eps44)

    # ******************************************************
    # ********* z=1.4 **************************************
    # Delta-web

    dth = -1.

    nmean11=2.21; alpha11 = 0.6;   beta11 = 1.;  dth11 = -0.4; rho_eps11 = 0.83; eps11 = 0.42 #KK                                                                  \
                                                                                                                                                                    
    nmean12=0.43; alpha12 = 0.5;   beta12 = 5.8; dth12 = -0.8; rho_eps12 = 0.12; eps12 = -0.17 #KF                                                                 \
                                                                                                                                                                    
    nmean13=0.30; alpha13 = 0.01;  beta13 = 4.5; dth13 = -0.3; rho_eps13 = 0.2;  eps13 = -0.5 #KS                                                                  \
                                                                                                                                                                    
    nmean14=0.0;  alpha14 = 2.8;   beta14 = 8.7; dth14 = -0.9; rho_eps14 = 0.7;  eps14 = 2. #KV                                                                    \
                                                                                                                                                                    

    nmean21=0.68; alpha21 = 0.8;  beta21 = 5.5; dth21 = -0.2;  rho_eps21 = 1.4; eps21 = 0.5  #FK                                                                   \
                                                                                                                                                                    
    nmean22=0.34; alpha22 = 0.28; beta22 = 5.2; dth22 = -0.8;  rho_eps22 = 1.1; eps22 = -0.45 #FF                                                                  \
                                                                                                                                                                    
    nmean23=0.13; alpha23 = 0.6;  beta23 = 2.9; dth23 = -0.38; rho_eps23 = 0.5; eps23 = -0.42 #FS                                                                  \
                                                                                                                                                                    
    nmean24=0.02; alpha24 = 2.8;  beta24 = 5.2; dth24 = -1.;   rho_eps24 = 2.5; eps24 = 20. #FV                                                                    \
                                                                                                                                                                    

    nmean31=0.16; alpha31 = 0.3; beta31 = 3.1;  dth31 = -0.2; rho_eps31 = 1.6; eps31 = -0.2 #SK                                                                    \
                                                                                                                                                                    
    nmean32=0.16; alpha32 = 0.3; beta32 = 3.1;  dth32 = -0.2; rho_eps32 = 1.6; eps32 = -0.2 #SF                                                                    \
                                                                                                                                                                    
    nmean33=0.16; alpha33 = 0.3; beta33 = 3.1;  dth33 = -0.2; rho_eps33 = 1.6; eps33 = -0.2 #SS                                                                    \
                                                                                                                                                                    
    nmean34=0.16; alpha34 = 0.3; beta34 = 3.1;  dth34 = -0.2; rho_eps34 = 1.6; eps34 = -0.2 #SV                                                                    \
                                                                                                                                                                    

    nmean41=0.02; alpha41 = 4.; beta41 = 15.3; dth41 = -1.; rho_eps41 = 8.5; eps41 = 30. #VK                                                                       \
                                                                                                                                                                    
    nmean42=0.02; alpha42 = 4.; beta42 = 15.3; dth42 = -1.; rho_eps42 = 8.5; eps42 = 30.   #VF                                                                     \
                                                                                                                                                                    
    nmean43=0.02; alpha43 = 4.; beta43 = 15.3; dth43 = -1.; rho_eps43 = 8.5; eps43 = 30. #VS                                                                       \
                                                                                                                                                                    
    nmean44=0.02; alpha44 = 4.; beta44 = 15.3; dth44 = -1.; rho_eps44 = 8.5; eps44 = 30. #VV  
    
    wnmean11.append(nmean11); walpha11.append(alpha11); wbeta11.append(beta11); wdth11.append(dth11); wrhoeps11.append(rho_eps11); weps11.append(eps11)
    wnmean12.append(nmean12); walpha12.append(alpha12); wbeta12.append(beta12); wdth12.append(dth12); wrhoeps12.append(rho_eps12); weps12.append(eps12)
    wnmean13.append(nmean13); walpha13.append(alpha13); wbeta13.append(beta13); wdth13.append(dth13); wrhoeps13.append(rho_eps13); weps13.append(eps13)
    wnmean14.append(nmean14); walpha14.append(alpha14); wbeta14.append(beta14); wdth14.append(dth14); wrhoeps14.append(rho_eps14); weps14.append(eps14)
    
    wnmean21.append(nmean21); walpha21.append(alpha21); wbeta21.append(beta21); wdth21.append(dth21); wrhoeps21.append(rho_eps21); weps21.append(eps21)
    wnmean22.append(nmean22); walpha22.append(alpha22); wbeta22.append(beta22); wdth22.append(dth22); wrhoeps22.append(rho_eps22); weps22.append(eps22)
    wnmean23.append(nmean23); walpha23.append(alpha23); wbeta23.append(beta23); wdth23.append(dth23); wrhoeps23.append(rho_eps23); weps23.append(eps23)
    wnmean24.append(nmean24); walpha24.append(alpha24); wbeta24.append(beta24); wdth24.append(dth24); wrhoeps24.append(rho_eps24); weps24.append(eps24)
    
    wnmean31.append(nmean31); walpha31.append(alpha31); wbeta31.append(beta31); wdth31.append(dth31); wrhoeps31.append(rho_eps31); weps31.append(eps31)
    wnmean32.append(nmean32); walpha32.append(alpha32); wbeta32.append(beta32); wdth32.append(dth32); wrhoeps32.append(rho_eps32); weps32.append(eps32)
    wnmean33.append(nmean33); walpha33.append(alpha33); wbeta33.append(beta33); wdth33.append(dth33); wrhoeps33.append(rho_eps33); weps33.append(eps33)
    wnmean34.append(nmean34); walpha34.append(alpha34); wbeta34.append(beta34); wdth34.append(dth34); wrhoeps34.append(rho_eps34); weps34.append(eps34)
    
    wnmean41.append(nmean41); walpha41.append(alpha41); wbeta41.append(beta41); wdth41.append(dth41); wrhoeps41.append(rho_eps41); weps41.append(eps41)
    wnmean42.append(nmean42); walpha42.append(alpha42); wbeta42.append(beta42); wdth42.append(dth42); wrhoeps42.append(rho_eps42); weps42.append(eps42)
    wnmean43.append(nmean43); walpha43.append(alpha43); wbeta43.append(beta43); wdth43.append(dth43); wrhoeps43.append(rho_eps43); weps43.append(eps43)
    wnmean44.append(nmean44); walpha44.append(alpha44); wbeta44.append(beta44); wdth44.append(dth44); wrhoeps44.append(rho_eps44); weps44.append(eps44)

    # ******************************************************
    # ********* z=1.7 **************************************
    # Delta-web
    dth = -1.

    nmean11=2.21; alpha11 = 0.6;   beta11 = 1.;  dth11 = -0.4; rho_eps11 = 0.83; eps11 = 0.42 #KK                                                                  \
                                                                                                                                                                    
    nmean12=0.43; alpha12 = 0.5;   beta12 = 5.8; dth12 = -0.8; rho_eps12 = 0.12; eps12 = -0.17 #KF                                                                 \
                                                                                                                                                                    
    nmean13=0.30; alpha13 = 0.01;  beta13 = 4.5; dth13 = -0.3; rho_eps13 = 0.2;  eps13 = -0.5 #KS                                                                  \
                                                                                                                                                                    
    nmean14=0.0;  alpha14 = 2.8;   beta14 = 8.7; dth14 = -0.9; rho_eps14 = 0.7;  eps14 = 2. #KV                                                                    \
                                                                                                                                                                    

    nmean21=0.68; alpha21 = 0.8;  beta21 = 5.5; dth21 = -0.2;  rho_eps21 = 1.4; eps21 = 0.5  #FK                                                                   \
                                                                                                                                                                    
    nmean22=0.34; alpha22 = 0.28; beta22 = 5.2; dth22 = -0.8;  rho_eps22 = 1.1; eps22 = -0.45 #FF                                                                  \
                                                                                                                                                                    
    nmean23=0.13; alpha23 = 0.6;  beta23 = 2.9; dth23 = -0.38; rho_eps23 = 0.5; eps23 = -0.42 #FS                                                                  \
                                                                                                                                                                    
    nmean24=0.02; alpha24 = 2.8;  beta24 = 5.2; dth24 = -1.;   rho_eps24 = 2.5; eps24 = 20. #FV                                                                    \
                                                                                                                                                                    

    nmean31=0.16; alpha31 = 0.3; beta31 = 3.1;  dth31 = -0.2; rho_eps31 = 1.6; eps31 = -0.2 #SK                                                                    \
                                                                                                                                                                    
    nmean32=0.16; alpha32 = 0.3; beta32 = 3.1;  dth32 = -0.2; rho_eps32 = 1.6; eps32 = -0.2 #SF                                                                    \
                                                                                                                                                                    
    nmean33=0.16; alpha33 = 0.3; beta33 = 3.1;  dth33 = -0.2; rho_eps33 = 1.6; eps33 = -0.2 #SS                                                                    \
                                                                                                                                                                    
    nmean34=0.16; alpha34 = 0.3; beta34 = 3.1;  dth34 = -0.2; rho_eps34 = 1.6; eps34 = -0.2 #SV                                                                    \
                                                                                                                                                                    

    nmean41=0.02; alpha41 = 4.; beta41 = 15.3; dth41 = -1.; rho_eps41 = 8.5; eps41 = 30. #VK                                                                       \
                                                                                                                                                                    
    nmean42=0.02; alpha42 = 4.; beta42 = 15.3; dth42 = -1.; rho_eps42 = 8.5; eps42 = 30.   #VF                                                                     \
                                                                                                                                                                    
    nmean43=0.02; alpha43 = 4.; beta43 = 15.3; dth43 = -1.; rho_eps43 = 8.5; eps43 = 30. #VS                                                                       \
                                                                                                                                                                    
    nmean44=0.02; alpha44 = 4.; beta44 = 15.3; dth44 = -1.; rho_eps44 = 8.5; eps44 = 30. #VV  

    wnmean11.append(nmean11); walpha11.append(alpha11); wbeta11.append(beta11); wdth11.append(dth11); wrhoeps11.append(rho_eps11); weps11.append(eps11)
    wnmean12.append(nmean12); walpha12.append(alpha12); wbeta12.append(beta12); wdth12.append(dth12); wrhoeps12.append(rho_eps12); weps12.append(eps12)
    wnmean13.append(nmean13); walpha13.append(alpha13); wbeta13.append(beta13); wdth13.append(dth13); wrhoeps13.append(rho_eps13); weps13.append(eps13)
    wnmean14.append(nmean14); walpha14.append(alpha14); wbeta14.append(beta14); wdth14.append(dth14); wrhoeps14.append(rho_eps14); weps14.append(eps14)
    
    wnmean21.append(nmean21); walpha21.append(alpha21); wbeta21.append(beta21); wdth21.append(dth21); wrhoeps21.append(rho_eps21); weps21.append(eps21)
    wnmean22.append(nmean22); walpha22.append(alpha22); wbeta22.append(beta22); wdth22.append(dth22); wrhoeps22.append(rho_eps22); weps22.append(eps22)
    wnmean23.append(nmean23); walpha23.append(alpha23); wbeta23.append(beta23); wdth23.append(dth23); wrhoeps23.append(rho_eps23); weps23.append(eps23)
    wnmean24.append(nmean24); walpha24.append(alpha24); wbeta24.append(beta24); wdth24.append(dth24); wrhoeps24.append(rho_eps24); weps24.append(eps24)
    
    wnmean31.append(nmean31); walpha31.append(alpha31); wbeta31.append(beta31); wdth31.append(dth31); wrhoeps31.append(rho_eps31); weps31.append(eps31)
    wnmean32.append(nmean32); walpha32.append(alpha32); wbeta32.append(beta32); wdth32.append(dth32); wrhoeps32.append(rho_eps32); weps32.append(eps32)
    wnmean33.append(nmean33); walpha33.append(alpha33); wbeta33.append(beta33); wdth33.append(dth33); wrhoeps33.append(rho_eps33); weps33.append(eps33)
    wnmean34.append(nmean34); walpha34.append(alpha34); wbeta34.append(beta34); wdth34.append(dth34); wrhoeps34.append(rho_eps34); weps34.append(eps34)
    
    wnmean41.append(nmean41); walpha41.append(alpha41); wbeta41.append(beta41); wdth41.append(dth41); wrhoeps41.append(rho_eps41); weps41.append(eps41)
    wnmean42.append(nmean42); walpha42.append(alpha42); wbeta42.append(beta42); wdth42.append(dth42); wrhoeps42.append(rho_eps42); weps42.append(eps42)
    wnmean43.append(nmean43); walpha43.append(alpha43); wbeta43.append(beta43); wdth43.append(dth43); wrhoeps43.append(rho_eps43); weps43.append(eps43)
    wnmean44.append(nmean44); walpha44.append(alpha44); wbeta44.append(beta44); wdth44.append(dth44); wrhoeps44.append(rho_eps44); weps44.append(eps44)

    # ******************************************************
    # ********* z=2.5 **************************************
    # Delta-web
    
    dth = -1.
    
    nmean11=2.21; alpha11 = 0.6;   beta11 = 1.;  dth11 = -0.4; rho_eps11 = 0.83; eps11 = 0.42 #KK                                                                                     
    nmean12=0.43; alpha12 = 0.5;   beta12 = 5.8; dth12 = -0.8; rho_eps12 = 0.12; eps12 = -0.17 #KF                                                                                   
    nmean13=0.30; alpha13 = 0.01;  beta13 = 4.5; dth13 = -0.3; rho_eps13 = 0.2;  eps13 = -0.5 #KS                                                                                    
    nmean14=0.0;  alpha14 = 2.8;   beta14 = 8.7; dth14 = -0.9; rho_eps14 = 0.7;  eps14 = 2. #KV                                                                                       

    nmean21=0.68; alpha21 = 0.8;  beta21 = 5.5; dth21 = -0.2;  rho_eps21 = 1.4; eps21 = 0.5  #FK                                                                                      
    nmean22=0.34; alpha22 = 0.28; beta22 = 5.2; dth22 = -0.8;  rho_eps22 = 1.1; eps22 = -0.45 #FF                                                                                    
    nmean23=0.13; alpha23 = 0.6;  beta23 = 2.9; dth23 = -0.38; rho_eps23 = 0.5; eps23 = -0.42 #FS                                                                                    
    nmean24=0.02; alpha24 = 2.8;  beta24 = 5.2; dth24 = -1.;   rho_eps24 = 2.5; eps24 = 20. #FV                                                                                           

    nmean31=0.16; alpha31 = 0.3; beta31 = 3.1;  dth31 = -0.2; rho_eps31 = 1.6; eps31 = -0.2 #SK                                                                                      
    nmean32=0.16; alpha32 = 0.3; beta32 = 3.1;  dth32 = -0.2; rho_eps32 = 1.6; eps32 = -0.2 #SF                                                                                    
    nmean33=0.16; alpha33 = 0.3; beta33 = 3.1;  dth33 = -0.2; rho_eps33 = 1.6; eps33 = -0.2 #SS                                                                                        
    nmean34=0.16; alpha34 = 0.3; beta34 = 3.1;  dth34 = -0.2; rho_eps34 = 1.6; eps34 = -0.2 #SV                                                                                         

    nmean41=0.02; alpha41 = 4.; beta41 = 15.3; dth41 = -1.; rho_eps41 = 8.5; eps41 = 30. #VK                                                                                          
    nmean42=0.02; alpha42 = 4.; beta42 = 15.3; dth42 = -1.; rho_eps42 = 8.5; eps42 = 30.   #VF                                                                                       
    nmean43=0.02; alpha43 = 4.; beta43 = 15.3; dth43 = -1.; rho_eps43 = 8.5; eps43 = 30. #VS                                                                                           
    nmean44=0.02; alpha44 = 4.; beta44 = 15.3; dth44 = -1.; rho_eps44 = 8.5; eps44 = 30. #VV  
    
    wnmean11.append(nmean11); walpha11.append(alpha11); wbeta11.append(beta11); wdth11.append(dth11); wrhoeps11.append(rho_eps11); weps11.append(eps11)
    wnmean12.append(nmean12); walpha12.append(alpha12); wbeta12.append(beta12); wdth12.append(dth12); wrhoeps12.append(rho_eps12); weps12.append(eps12)
    wnmean13.append(nmean13); walpha13.append(alpha13); wbeta13.append(beta13); wdth13.append(dth13); wrhoeps13.append(rho_eps13); weps13.append(eps13)
    wnmean14.append(nmean14); walpha14.append(alpha14); wbeta14.append(beta14); wdth14.append(dth14); wrhoeps14.append(rho_eps14); weps14.append(eps14)
    
    wnmean21.append(nmean21); walpha21.append(alpha21); wbeta21.append(beta21); wdth21.append(dth21); wrhoeps21.append(rho_eps21); weps21.append(eps21)
    wnmean22.append(nmean22); walpha22.append(alpha22); wbeta22.append(beta22); wdth22.append(dth22); wrhoeps22.append(rho_eps22); weps22.append(eps22)
    wnmean23.append(nmean23); walpha23.append(alpha23); wbeta23.append(beta23); wdth23.append(dth23); wrhoeps23.append(rho_eps23); weps23.append(eps23)
    wnmean24.append(nmean24); walpha24.append(alpha24); wbeta24.append(beta24); wdth24.append(dth24); wrhoeps24.append(rho_eps24); weps24.append(eps24)
    
    wnmean31.append(nmean31); walpha31.append(alpha31); wbeta31.append(beta31); wdth31.append(dth31); wrhoeps31.append(rho_eps31); weps31.append(eps31)
    wnmean32.append(nmean32); walpha32.append(alpha32); wbeta32.append(beta32); wdth32.append(dth32); wrhoeps32.append(rho_eps32); weps32.append(eps32)
    wnmean33.append(nmean33); walpha33.append(alpha33); wbeta33.append(beta33); wdth33.append(dth33); wrhoeps33.append(rho_eps33); weps33.append(eps33)
    wnmean34.append(nmean34); walpha34.append(alpha34); wbeta34.append(beta34); wdth34.append(dth34); wrhoeps34.append(rho_eps34); weps34.append(eps34)
    
    wnmean41.append(nmean41); walpha41.append(alpha41); wbeta41.append(beta41); wdth41.append(dth41); wrhoeps41.append(rho_eps41); weps41.append(eps41)
    wnmean42.append(nmean42); walpha42.append(alpha42); wbeta42.append(beta42); wdth42.append(dth42); wrhoeps42.append(rho_eps42); weps42.append(eps42)
    wnmean43.append(nmean43); walpha43.append(alpha43); wbeta43.append(beta43); wdth43.append(dth43); wrhoeps43.append(rho_eps43); weps43.append(eps43)
    wnmean44.append(nmean44); walpha44.append(alpha44); wbeta44.append(beta44); wdth44.append(dth44); wrhoeps44.append(rho_eps44); weps44.append(eps44)

    # Do it one last time for the largest z interpoation node
    wnmean11.append(nmean11); walpha11.append(alpha11); wbeta11.append(beta11); wdth11.append(dth11); wrhoeps11.append(rho_eps11); weps11.append(eps11)
    wnmean12.append(nmean12); walpha12.append(alpha12); wbeta12.append(beta12); wdth12.append(dth12); wrhoeps12.append(rho_eps12); weps12.append(eps12)
    wnmean13.append(nmean13); walpha13.append(alpha13); wbeta13.append(beta13); wdth13.append(dth13); wrhoeps13.append(rho_eps13); weps13.append(eps13)
    wnmean14.append(nmean14); walpha14.append(alpha14); wbeta14.append(beta14); wdth14.append(dth14); wrhoeps14.append(rho_eps14); weps14.append(eps14)

    wnmean21.append(nmean21); walpha21.append(alpha21); wbeta21.append(beta21); wdth21.append(dth21); wrhoeps21.append(rho_eps21); weps21.append(eps21)
    wnmean22.append(nmean22); walpha22.append(alpha22); wbeta22.append(beta22); wdth22.append(dth22); wrhoeps22.append(rho_eps22); weps22.append(eps22)
    wnmean23.append(nmean23); walpha23.append(alpha23); wbeta23.append(beta23); wdth23.append(dth23); wrhoeps23.append(rho_eps23); weps23.append(eps23)
    wnmean24.append(nmean24); walpha24.append(alpha24); wbeta24.append(beta24); wdth24.append(dth24); wrhoeps24.append(rho_eps24); weps24.append(eps24)

    wnmean31.append(nmean31); walpha31.append(alpha31); wbeta31.append(beta31); wdth31.append(dth31); wrhoeps31.append(rho_eps31); weps31.append(eps31)
    wnmean32.append(nmean32); walpha32.append(alpha32); wbeta32.append(beta32); wdth32.append(dth32); wrhoeps32.append(rho_eps32); weps32.append(eps32)
    wnmean33.append(nmean33); walpha33.append(alpha33); wbeta33.append(beta33); wdth33.append(dth33); wrhoeps33.append(rho_eps33); weps33.append(eps33)
    wnmean34.append(nmean34); walpha34.append(alpha34); wbeta34.append(beta34); wdth34.append(dth34); wrhoeps34.append(rho_eps34); weps34.append(eps34)

    wnmean41.append(nmean41); walpha41.append(alpha41); wbeta41.append(beta41); wdth41.append(dth41); wrhoeps41.append(rho_eps41); weps41.append(eps41)
    wnmean42.append(nmean42); walpha42.append(alpha42); wbeta42.append(beta42); wdth42.append(dth42); wrhoeps42.append(rho_eps42); weps42.append(eps42)
    wnmean43.append(nmean43); walpha43.append(alpha43); wbeta43.append(beta43); wdth43.append(dth43); wrhoeps43.append(rho_eps43); weps43.append(eps43)
    wnmean44.append(nmean44); walpha44.append(alpha44); wbeta44.append(beta44); wdth44.append(dth44); wrhoeps44.append(rho_eps44); weps44.append(eps44)

    # Now put the lists into the final array
    nmean_arr[:,0,0] = wnmean11[:]
    nmean_arr[:,0,1] = wnmean12[:]
    nmean_arr[:,0,2] = wnmean13[:]
    nmean_arr[:,0,3] = wnmean14[:]

    nmean_arr[:,1,0] = wnmean21[:]
    nmean_arr[:,1,1] = wnmean22[:]
    nmean_arr[:,1,2] = wnmean23[:]
    nmean_arr[:,1,3] = wnmean24[:]
    
    nmean_arr[:,2,0] = wnmean31[:]
    nmean_arr[:,2,1] = wnmean32[:]
    nmean_arr[:,2,2] = wnmean33[:]
    nmean_arr[:,2,3] = wnmean34[:]
    
    nmean_arr[:,3,0] = wnmean41[:]
    nmean_arr[:,3,1] = wnmean42[:]
    nmean_arr[:,3,2] = wnmean43[:]
    nmean_arr[:,3,3] = wnmean44[:]

    # ***************************

    alpha_arr[:,0,0] = walpha11[:]
    alpha_arr[:,0,1] = walpha12[:]
    alpha_arr[:,0,2] = walpha13[:]
    alpha_arr[:,0,3] = walpha14[:]

    alpha_arr[:,1,0] = walpha21[:]
    alpha_arr[:,1,1] = walpha22[:]
    alpha_arr[:,1,2] = walpha23[:]
    alpha_arr[:,1,3] = walpha24[:]
    
    alpha_arr[:,2,0] = walpha31[:]
    alpha_arr[:,2,1] = walpha32[:]
    alpha_arr[:,2,2] = walpha33[:]
    alpha_arr[:,2,3] = walpha34[:]
    
    alpha_arr[:,3,0] = walpha41[:]
    alpha_arr[:,3,1] = walpha42[:]
    alpha_arr[:,3,2] = walpha43[:]
    alpha_arr[:,3,3] = walpha44[:]

    # ***********************************

    beta_arr[:,0,0] = wbeta11[:]
    beta_arr[:,0,1] = wbeta12[:]
    beta_arr[:,0,2] = wbeta13[:]
    beta_arr[:,0,3] = wbeta14[:]

    beta_arr[:,1,0] = wbeta21[:]
    beta_arr[:,1,1] = wbeta22[:]
    beta_arr[:,1,2] = wbeta23[:]
    beta_arr[:,1,3] = wbeta24[:]
    
    beta_arr[:,2,0] = wbeta31[:]
    beta_arr[:,2,1] = wbeta32[:]
    beta_arr[:,2,2] = wbeta33[:]
    beta_arr[:,2,3] = wbeta34[:]
    
    beta_arr[:,3,0] = wbeta41[:]
    beta_arr[:,3,1] = wbeta42[:]
    beta_arr[:,3,2] = wbeta43[:]
    beta_arr[:,3,3] = wbeta44[:]

    # ****************************************

    dth_arr[:,0,0] = wdth11[:]
    dth_arr[:,0,1] = wdth12[:]
    dth_arr[:,0,2] = wdth13[:]
    dth_arr[:,0,3] = wdth14[:]

    dth_arr[:,1,0] = wdth21[:]
    dth_arr[:,1,1] = wdth22[:]
    dth_arr[:,1,2] = wdth23[:]
    dth_arr[:,1,3] = wdth24[:]
    
    dth_arr[:,2,0] = wdth31[:]
    dth_arr[:,2,1] = wdth32[:]
    dth_arr[:,2,2] = wdth33[:]
    dth_arr[:,2,3] = wdth34[:]
    
    dth_arr[:,3,0] = wdth41[:]
    dth_arr[:,3,1] = wdth42[:]
    dth_arr[:,3,2] = wdth43[:]
    dth_arr[:,3,3] = wdth44[:]

    # *******************************************
    
    rhoeps_arr[:,0,0] = wrhoeps11[:]
    rhoeps_arr[:,0,1] = wrhoeps12[:]
    rhoeps_arr[:,0,2] = wrhoeps13[:]
    rhoeps_arr[:,0,3] = wrhoeps14[:]

    rhoeps_arr[:,1,0] = wrhoeps21[:]
    rhoeps_arr[:,1,1] = wrhoeps22[:]
    rhoeps_arr[:,1,2] = wrhoeps23[:]
    rhoeps_arr[:,1,3] = wrhoeps24[:]
    
    rhoeps_arr[:,2,0] = wrhoeps31[:]
    rhoeps_arr[:,2,1] = wrhoeps32[:]
    rhoeps_arr[:,2,2] = wrhoeps33[:]
    rhoeps_arr[:,2,3] = wrhoeps34[:]
    
    rhoeps_arr[:,3,0] = wrhoeps41[:]
    rhoeps_arr[:,3,1] = wrhoeps42[:]
    rhoeps_arr[:,3,2] = wrhoeps43[:]
    rhoeps_arr[:,3,3] = wrhoeps44[:]

    # *********************************************

    eps_arr[:,0,0] = weps11[:]
    eps_arr[:,0,1] = weps12[:]
    eps_arr[:,0,2] = weps13[:]
    eps_arr[:,0,3] = weps14[:]

    eps_arr[:,1,0] = weps21[:]
    eps_arr[:,1,1] = weps22[:]
    eps_arr[:,1,2] = weps23[:]
    eps_arr[:,1,3] = weps24[:]
    
    eps_arr[:,2,0] = weps31[:]
    eps_arr[:,2,1] = weps32[:]
    eps_arr[:,2,2] = weps33[:]
    eps_arr[:,2,3] = weps34[:]
    
    eps_arr[:,3,0] = weps41[:]
    eps_arr[:,3,1] = weps42[:]
    eps_arr[:,3,2] = weps43[:]
    eps_arr[:,3,3] = weps44[:]

    # ***********************************
    # RSD parameters
    bv1 = np.array([1.56, 1.56, 1.56, 1.56, 1.56, 1.56])
    bb1 = np.array([0.11, 0.11, 0.11, 0.11, 0.11, 0.11])
    betarsd1 = np.array([0.11,0.11,0.11,0.11,0.11,0.11])
    gamma1 = np.array([2.01,2.01,2.01,2.01,2.1,2.01])

    bv2 = np.array([0.81,0.81,0.81,0.81,0.81,0.81])
    bb2 = np.array([0.09, 0.09,0.09,0.09,0.09,0.09])
    betarsd2 = np.array([0.39,0.39,0.39,0.39,0.39,0.39])
    gamma2 = np.array([2.08,2.08,2.08,2.08,2.08,2.08])

    bv3 = np.array([0.59, 0.59,0.59, 0.59, 0.59, 0.59])
    bb3 = np.array([0.84,0.84,0.84,0.84,0.84,0.84])
    betarsd3 = np.array([0.22,0.22,0.22,0.22,0.22,0.22])
    gamma3 = np.array([1.16,1.16,1.16,1.16,1.16,1.16])

    bv4 = np.array([0.6,0.6,0.6,0.6,0.6,0.6])
    bb4 = np.array([0.32,0.32,0.32,0.32,0.32,0.32])
    betarsd4 = np.array([1.5,1.5,1.5,1.5,1.5,1.5])
    gamma4 = np.array([0.34,0.34,0.34,0.34,0.34,0.34])

    bv_arr[:,0] = bv1[:]
    bv_arr[:,1] = bv2[:]
    bv_arr[:,2] = bv3[:]
    bv_arr[:,3] = bv4[:]

    bb_arr[:,0] = bb1[:]
    bb_arr[:,1] = bb2[:]
    bb_arr[:,2] = bb3[:]
    bb_arr[:,3] = bb4[:]

    betarsd_arr[:,0] = betarsd1[:]
    betarsd_arr[:,1] = betarsd2[:]
    betarsd_arr[:,2] = betarsd3[:]
    betarsd_arr[:,3] = betarsd4[:]

    gamma_arr[:,0] = gamma1[:]
    gamma_arr[:,1] = gamma2[:]
    gamma_arr[:,2] = gamma3[:]
    gamma_arr[:,3] = gamma4[:]

    return nmean_arr, alpha_arr, beta_arr, dth_arr, rhoeps_arr, eps_arr, bv_arr, bb_arr, betarsd_arr, gamma_arr
    #return wnmean11,walpha11,wbeta11,wdth11,wrhoeps11,weps11,wnmean12,walpha12,wbeta12,wdth12,wrhoeps12,weps12,wnmean13,walpha13,wbeta13,wdth13,wrhoeps13,weps13,wnmean14,walpha14,wbeta14,wdth14,wrhoeps14,weps14,wnmean21,walpha21,wbeta21,wdth21,wrhoeps21,weps21,wnmean22,walpha22,wbeta22,wdth22,wrhoeps22,weps22,wnmean23,walpha23,wbeta23,wdth23,wrhoeps23,weps23,wnmean24,walpha24,wbeta24,wdth24,wrhoeps24,weps24,wnmean31,walpha31,wbeta31,wdth31,wrhoeps31,weps31,wnmean32,walpha32,wbeta32,wdth32,wrhoeps32,weps32,wnmean33,walpha33,wbeta33,wdth33,wrhoeps33,weps33,wnmean34,walpha34,wbeta34,wdth34,wrhoeps34,weps34,wnmean41,walpha41,wbeta41,wdth41,wrhoeps41,weps41,wnmean42,walpha42,wbeta42,wdth42,wrhoeps42,weps42,wnmean43,walpha43,wbeta43,wdth43,wrhoeps43,weps43,wnmean44,walpha44,wbeta44,wdth44,wrhoeps44,weps44

#wnmean11,walpha11,wbeta11,wdth11,wrhoeps11,weps11,wnmean12,walpha12,wbeta12,wdth12,wrhoeps12,weps12,wnmean13,walpha13,wbeta13,wdth13,wrhoeps13,weps13,wnmean14,walpha14,wbeta14,wdth14,wrhoeps14,weps14,wnmean21,walpha21,wbeta21,wdth21,wrhoeps21,weps21,wnmean22,walpha22,wbeta22,wdth22,wrhoeps22,weps22,wnmean23,walpha23,wbeta23,wdth23,wrhoeps23,weps23,wnmean24,walpha24,wbeta24,wdth24,wrhoeps24,weps24,wnmean31,walpha31,wbeta31,wdth31,wrhoeps31,weps31,wnmean32,walpha32,wbeta32,wdth32,wrhoeps32,weps32,wnmean33,walpha33,wbeta33,wdth33,wrhoeps33,weps33,wnmean34,walpha34,wbeta34,wdth34,wrhoeps34,weps34,wnmean41,walpha41,wbeta41,wdth41,wrhoeps41,weps41,wnmean42,walpha42,wbeta42,wdth42,wrhoeps42,weps42,wnmean43,walpha43,wbeta43,wdth43,wrhoeps43,weps43,wnmean44,walpha44,wbeta44,wdth44,wrhoeps44,weps44 = make_pars_list() 
nmean_arr, alpha_arr, beta_arr, dth_arr, rhoeps_arr, eps_arr, bv_arr, bb_arr, betarsd_arr, gamma_arr = make_pars_list() 

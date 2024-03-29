import numpy as np
import healpy as hp
import pytempura
from pytempura import norm_general, noise_spec
import os
from falafel import utils as futils, qe
from orphics import maps
from pixell import lensing, curvedsky
import matplotlib.pyplot as plt
from os.path import join as opj
import pickle
import solenspipe

try:
    WEBSKY_DIR=os.environ["WEBSKY_DIR"]
except KeyError:
    WEBSKY_DIR="/global/project/projectdirs/act/data/maccrann/websky"
try:
    SEHGAL_DIR=os.environ["SEHGAL_DIR"]
except KeyError:
    SEHGAL_DIR="/global/project/projectdirs/act/data/maccrann/sehgal"
    

class ClBinner(object):
    """
    Class for binning Cls in equal
    width bins, weighting by 2L+1
    """
    def __init__(self, lmin=100, lmax=1000, nbin=20):
        self.lmin=lmin
        self.lmax=lmax
        self.nbin=nbin
        self.bin_lims = np.ceil(np.linspace(
            self.lmin, self.lmax+1, self.nbin+1
        )).astype(int)
        self.deltal = np.diff(self.bin_lims)
        self.bin_mids = 0.5*(self.bin_lims[:-1]
                             +self.bin_lims[1:])
        
    def __call__(self, cl):
        L = np.arange(len(cl)).astype(int)
        w = 2*L+1
        cl_binned = np.zeros(self.nbin)
        for i in range(self.nbin):
            use = (L>=self.bin_lims[i])*(L<self.bin_lims[i+1])
            cl_binned[i] = np.average(cl[use], weights=w[use])
        return cl_binned

def filter_T(T_alm, cltot, lmin, lmax):
    """
    filter by 1/cltot within lmin<=l<=lmax
    return zero otherwise
    """
    mlmax=qe.get_mlmax(T_alm)
    filt = np.zeros_like(cltot)
    ls = np.arange(filt.size)
    assert lmax<=ls.max()
    assert lmin<lmax
    filt[2:] = 1./cltot[2:]
    filt[ls<lmin] = 0.
    filt[ls>lmax] = 0.
    return curvedsky.almxfl(T_alm.copy(), filt)

def dummy_teb(alms):
    return [alms, np.zeros_like(alms), np.zeros_like(alms)]

def setup_sym_estimator(px, lmin, lmax, mlmax,
                        cltot_X, cltot_Y, cltot_XY):
    """
    Setup quadratic estimators for case of a
    pair of temperature maps X and Y. In particular,
    return 'qfuncs' for the two asymmetric estimators
    (i.e. Q[X,Y] and Q[Y,X]) and also the symmetrized
    estimator from https://arxiv.org/abs/2004.01139.
    These estimators are normalized. Also return 
    theory N0s assuming the total Cls used in filters
    are appropriate for the maps.
    """

    #get cls for gradient filter                                                                                                                                            
    ucls,_ = futils.get_theory_dicts(grad=True, lmax=mlmax)

    cltot_X, cltot_Y, cltot_XY = (
        cltot_X[:lmax+1], cltot_Y[:lmax+1], cltot_XY[:lmax+1]
    )
    
    #Now get the norms
    norm_tt_YX = norm_general.qtt_asym(
        "lens", mlmax, lmin, lmax, lmin, 
        lmax, lmax, ucls['TT'][:lmax+1], cltot_X,
        cltot_Y)
    norm_tt_XY = norm_general.qtt_asym(
        "lens", mlmax, lmin, lmax, lmin, lmax, lmax,
        ucls['TT'][:lmax+1], cltot_Y,
        cltot_X)
    
    #Now get the N0s (need these for constructing
    #symmetrized estimator)
    #Note these are the noise on the unnormalized
    #estimators
    #First the aba'b' = XYXY case
    #wx0 = 1./cltot_X
    N0_XYXY_phi_nonorm = noise_spec.qtt_asym(
        'lens',mlmax,lmin,lmax,
         1./cltot_X, ucls['TT'][:lmax+1]/cltot_Y,
         1./cltot_X, ucls['TT'][:lmax+1]/cltot_Y,
         cltot_X,cltot_Y,cltot_XY,cltot_XY)
    #Normalize the N0
    N0_XYXY_phi = (N0_XYXY_phi_nonorm[0]*norm_tt_XY[0]**2,
                  N0_XYXY_phi_nonorm[1]*norm_tt_XY[1]**2)

    N0_XYYX_phi_nonorm = noise_spec.qtt_asym(
        'lens',mlmax,lmin,lmax,
         1./cltot_X, ucls['TT'][:lmax+1]/cltot_Y,
         1./cltot_Y, ucls['TT'][:lmax+1]/cltot_X,
         cltot_XY, cltot_XY, cltot_X, cltot_Y)
    N0_XYYX_phi = (N0_XYYX_phi_nonorm[0]*norm_tt_XY[0]*norm_tt_YX[0],
                   N0_XYYX_phi_nonorm[1]*norm_tt_XY[1]*norm_tt_YX[1]
                  )

    N0_YXYX_phi_nonorm = noise_spec.qtt_asym(
        'lens',mlmax,lmin,lmax,
         1./cltot_Y, ucls['TT'][:lmax+1] / cltot_X,
         1./cltot_Y, ucls['TT'][:lmax+1] / cltot_X,
         cltot_Y,cltot_X,
         cltot_XY,cltot_XY)
    N0_YXYX_phi = (N0_YXYX_phi_nonorm[0]*norm_tt_YX[0]**2,
                  N0_YXYX_phi_nonorm[1]*norm_tt_YX[1]**2)


    #Also will be useful to define here functions to get the
    #tripsectrum N0 for foregrounds. 
    def get_fg_trispectrum_N0_XYXY(clfg_X, clfg_Y, clfg_XY):
        N0_XYXY_fg_phi_nonorm = noise_spec.qtt_asym(
            'lens', mlmax, lmin, lmax,
            1./cltot_X, ucls['TT'][:lmax+1]/cltot_Y,
            1./cltot_X, ucls['TT'][:lmax+1]/cltot_Y,
            clfg_X[:lmax+1], clfg_Y[:lmax+1],
            clfg_XY[:lmax+1], clfg_XY[:lmax+1])
        N0_XYXY_fg_phi = (N0_XYXY_fg_phi_nonorm[0]*norm_tt_XY[0]**2,
                  N0_XYXY_fg_phi_nonorm[1]*norm_tt_XY[1]**2)
        return N0_XYXY_fg_phi

    def get_fg_trispectrum_N0_XYYX(clfg_X, clfg_Y, clfg_XY):
        N0_XYYX_fg_phi_nonorm = noise_spec.qtt_asym(
            'lens',mlmax,lmin,lmax,
            1./cltot_X, ucls['TT'][:lmax+1]/cltot_Y,
            1./cltot_Y, ucls['TT'][:lmax+1]/cltot_X,
            clfg_XY[:lmax+1], clfg_XY[:lmax+1],
            clfg_X[:lmax+1], clfg_Y[:lmax+1])
        N0_XYYX_fg_phi = (N0_XYYX_fg_phi_nonorm[0]*norm_tt_XY[0]*norm_tt_YX[0],
                       N0_XYYX_fg_phi_nonorm[1]*norm_tt_XY[1]*norm_tt_YX[1]
        )
        return N0_XYYX_fg_phi

    def get_fg_trispectrum_N0_YXYX(clfg_X, clfg_Y, clfg_XY):
        N0_YXYX_fg_phi_nonorm = noise_spec.qtt_asym(
            'lens',mlmax,lmin,lmax,
             1./cltot_Y, ucls['TT'][:lmax+1] / cltot_X,
             1./cltot_Y, ucls['TT'][:lmax+1] / cltot_X,
             clfg_Y[:lmax+1], clfg_X[:lmax+1],
             clfg_XY[:lmax+1], clfg_XY[:lmax+1])
        N0_YXYX_fg_phi = (N0_YXYX_fg_phi_nonorm[0]*norm_tt_YX[0]**2,
                      N0_YXYX_fg_phi_nonorm[1]*norm_tt_YX[1]**2)
        return N0_YXYX_fg_phi
    
    #Ok, so we have norms and N0s
    #Now the qfuncs
    def qfunc_XY(X, Y):
        #Filter X and Y
        X_filtered = filter_T(X, cltot_X, lmin, lmax)
        Y_filtered = filter_T(Y, cltot_Y, lmin, lmax)
        
        phi_nonorm = qe.qe_all(px,ucls,mlmax,
                                fTalm=X_filtered,fEalm=None,fBalm=None,
                                estimators=['TT'],
                                xfTalm=Y_filtered,xfEalm=None,xfBalm=None)['TT']
        print("phi_nonorm lmax:", hp.Alm.getlmax(len(phi_nonorm[0])))
        print("len(norm_tt_XY[0]):", len(norm_tt_XY[0]))
        #normalize and return
        return (curvedsky.almxfl(phi_nonorm[0], norm_tt_XY[0]),
                curvedsky.almxfl(phi_nonorm[1], norm_tt_XY[1]))
    
    def qfunc_YX(X, Y):
        #Filter X and Y
        X_filtered = filter_T(X, cltot_X, lmin, lmax)
        Y_filtered = filter_T(Y, cltot_Y, lmin, lmax)
                
        phi_nonorm = qe.qe_all(px,ucls,mlmax,
                                fTalm=Y_filtered,fEalm=None,fBalm=None,
                                estimators=['TT'],
                                xfTalm=X_filtered,xfEalm=None,xfBalm=None)['TT']
        #normalize and return
        return (curvedsky.almxfl(phi_nonorm[0], norm_tt_YX[0]),
                curvedsky.almxfl(phi_nonorm[1], norm_tt_YX[1]))
    
    #Now get weights etc. for symmetric estimator
    #The symmetrized version is a linear combination of these two,
    #weighted by the inverse covariance
    #i.e. phi_sym(L) = (C)^-1_L [phi_XY(L) phi_YX(L)]^T
    #So at each L, we need to compute the covariance 
    #C = [[N0^XYXY, N0^XYYX],[N0^XYYX, N0^YXYX]],
    #invert and sum
    Cov_g = np.zeros((mlmax+1, 2, 2))
    Cov_g[:,0,0] = N0_XYXY_phi[0]
    Cov_g[:,0,1] = N0_XYYX_phi[0]
    Cov_g[:,1,1] = N0_YXYX_phi[0]
    Cov_g[:,1,0] = Cov_g[:,0,1]

    Cov_c = np.zeros_like(Cov_g)
    Cov_c[:,0,0] = N0_XYXY_phi[1]
    Cov_c[:,0,1] = N0_XYYX_phi[1]
    Cov_c[:,1,1] = N0_YXYX_phi[1]
    Cov_c[:,1,0] = Cov_c[:,0,1]        

    w_XY_g = np.zeros(mlmax+1)
    w_YX_g = np.zeros(mlmax+1)
    w_XY_c = np.zeros(mlmax+1)
    w_YX_c = np.zeros(mlmax+1)
    for l in range(2, mlmax+1):
    #for l in range(lmin,lmax+1):
        #gradient
        try:
            inv_cov_g = np.linalg.inv(Cov_g[l])
        except LinAlgError as e:
            print(l,Cov_g[l])
            raise(e)
        w_XY_g[l] = inv_cov_g[0,0] + inv_cov_g[0,1]
        w_YX_g[l] = inv_cov_g[1,0] + inv_cov_g[1,1]
        #curl
        try:
            inv_cov_c = np.linalg.inv(Cov_c[l])
        except LinAlgError as e:
            print(l,Cov_c[l])
            raise(e)
        w_XY_c[l] = inv_cov_c[0,0] + inv_cov_c[0,1]
        w_YX_c[l] = inv_cov_c[1,0] + inv_cov_c[1,1]      

    w_sum_g = w_XY_g+w_YX_g
    w_sum_c = w_XY_c+w_YX_c
    #w_XY_g[lmin:lmax+1] = w_XY_g[lmin:lmax+1]/w_sum_g[lmin:lmax+1]
    #w_YX_g[lmin:lmax+1] = w_YX_g[lmin:lmax+1]/w_sum_g[lmin:lmax+1]
    #w_XY_c[lmin:lmax+1] = w_XY_c[lmin:lmax+1]/w_sum_c[lmin:lmax+1]
    #w_YX_c[lmin:lmax+1] = w_YX_c[lmin:lmax+1]/w_sum_c[lmin:lmax+1]
    w_XY_g[2:mlmax+1] = w_XY_g[2:mlmax+1]/w_sum_g[2:mlmax+1]
    w_YX_g[2:mlmax+1] = w_YX_g[2:mlmax+1]/w_sum_g[2:mlmax+1]
    w_XY_c[2:mlmax+1] = w_XY_c[2:mlmax+1]/w_sum_c[2:mlmax+1]
    w_YX_c[2:mlmax+1] = w_YX_c[2:mlmax+1]/w_sum_c[2:mlmax+1]
    
    
    def qfunc_sym(X, Y, phi_XY=None, phi_YX=None):

        #By default, we calculate the asymmetric estimates
        #phi_XY and phi_YX here, but if you've already 
        #calculated them, you can provide them as optional
        #arguments here
        if phi_XY is None:
            phi_XY = qfunc_XY(X, Y)
        if phi_YX is None:
            phi_YX = qfunc_YX(Y, X)
        
        phi_sym_grad = (curvedsky.almxfl(phi_XY[0], w_XY_g)
                   +curvedsky.almxfl(phi_YX[0], w_YX_g)
                  )
        phi_sym_curl = (curvedsky.almxfl(phi_XY[1], w_XY_c)
                   +curvedsky.almxfl(phi_YX[1], w_YX_c)
                  )
        return (phi_sym_grad, phi_sym_curl)

    def get_fg_trispectrum_N0_sym(clfg_X, clfg_Y, clfg_XY):
        """
        The N0 for the foreground trispectrum when using
        the symmetrized estimator
        The symmetrized estimator is 
        kappa_sym = w_XY*kappa_XY + w_YX*kappa_YX
        so N0^sym = <kappa_sym kappa_sym>
                  = w_XY^2 * N0^XY + w_YX^2 * N0^YX
                    + 2*w_XY*w_YX*N0^XYYX
        """
        N0_fg_XYXY = get_fg_trispectrum_N0_XYXY(
            clfg_X, clfg_Y, clfg_XY)
        N0_fg_XYYX = get_fg_trispectrum_N0_XYYX(
            clfg_X, clfg_Y, clfg_XY)
        N0_fg_YXYX = get_fg_trispectrum_N0_YXYX(
            clfg_X, clfg_Y, clfg_XY)
        N0_fg_sym = (
            w_XY_g**2 * N0_fg_XYXY[0]
            +w_YX_g**2 * N0_fg_YXYX[0]
            +2*w_XY_g*w_YX_g * N0_fg_XYYX[0],
            w_XY_c**2 * N0_fg_XYXY[1]
            +w_YX_c**2 * N0_fg_YXYX[1]
            +2*w_XY_c*w_YX_c * N0_fg_XYYX[1],
        )
        return N0_fg_sym            
    
    #Return useful stuff
    #- qfuncs to apply to maps
    #- normalizations and N0s for the asymmetric estimators 
    # for theory noise calculations (note these N0s are for 
    # unnormalized estimators) 
    #- For the symmetric estimator, also return the sum of 
    # the weight functions used to combine the asymmetric
    # estimators. I think this should be 1/N0 for the symmetric
    # estimator? 
    return {"qfunc_XY_incfilter" : qfunc_XY, "qfunc_YX_incfilter": qfunc_YX,
            "qfunc_sym_incfilter" : qfunc_sym, "norm_tt_XY" : norm_tt_XY,
            "norm_tt_YX" : norm_tt_YX, "N0_XYXY_phi" : N0_XYXY_phi,
            "N0_XYYX_phi" : N0_XYYX_phi, "N0_YXYX_phi" : N0_YXYX_phi,
            "w_sum_g" : w_sum_g, "w_sum_c" : w_sum_c,
            "w_XY" : (w_XY_g, w_XY_c), "w_YX" : (w_YX_g, w_YX_c),
            "get_fg_trispectrum_N0_XYXY": get_fg_trispectrum_N0_XYXY,
            "get_fg_trispectrum_N0_YXYX": get_fg_trispectrum_N0_YXYX,
            "get_fg_trispectrum_N0_XYYX": get_fg_trispectrum_N0_XYYX,
            "get_fg_trispectrum_N0_sym": get_fg_trispectrum_N0_sym}

def setup_recon(px, lmin, lmax, mlmax,
                tcls_X, tcls_Y=None,
                tcls_XY=None, do_pol=False,
                do_Tpol=False,
                do_psh=False, do_prh=False,
                profile=None):
    """
    Setup needed for reconstruction and foreground
    bias estimation. If cltot_Y and cltot_XY are not 
    None, setup symmetrized TT estimator
    """
    ucls,_ = futils.get_theory_dicts(grad=True, lmax=mlmax)

    if do_pol:
        pols = ["TE", "TB", "EE", "EB", "BB"]
    elif do_Tpol:
        pols = ["TE", "TB"]
    else:
        pols = []
    for pol in pols:
        assert pol in tcls_X
        
    recon_stuff = {}

    norms_X = pytempura.get_norms(
        ['TT']+pols, ucls, tcls_X,
        lmin, lmax,
	k_ellmax=mlmax)
    norm_lens_X = norms_X['TT']

    def filter_alms_X(alms):
        if len(alms)!=3:
            alms = dummy_teb(alms)
        alms_filtered = futils.isotropic_filter(alms,
                tcls_X, lmin, lmax, ignore_te=True)
        return alms_filtered

    """
    def filter_alms_X(alms):
        return filter_T(alms, tcls_X['TT'], lmin, lmax)
    """
    recon_stuff["filter_alms_X"] = filter_alms_X

    if tcls_Y is None:
        qfunc_tt_X = solenspipe.get_qfunc(px, ucls, mlmax, "TT", Al1=norms_X['TT'])
        recon_stuff["qfunc_tt"] = qfunc_tt_X
        recon_stuff["qfunc_tt_incfilter"] = lambda X,Y: qfunc_tt_X(filter_alms_X(X),
                                                                    filter_alms_X(Y))

        #Get the N0
        recon_stuff["N0_phi"] = norm_lens_X

        #Also define functions here for getting the
        #trispectrum N0
        def get_fg_trispectrum_phi_N0(cl_fg):
            #N0 is (A^phi)^2 / A_fg (see eqn. 9 of 1310.7023
            #for the normal estimator, and a bit more complex for
            #the bias hardened case
            Ctot = tcls_X['TT']**2 / cl_fg
            norm_fg = pytempura.norm_lens.qtt(
                mlmax, lmin,
                lmax, ucls['TT'],
                Ctot,gtype='')

            return (norm_lens_X[0]**2 / norm_fg[0],
                    norm_lens_X[1]**2 / norm_fg[1])
        
        recon_stuff["get_fg_trispectrum_phi_N0"] = get_fg_trispectrum_phi_N0

        if do_psh:
            R_src_tt = pytempura.get_cross(
                'SRC','TT',ucls,tcls_X,lmin,lmax,
                k_ellmax=mlmax)
            norm_src = pytempura.get_norms(
                    ['src'], ucls, tcls_X,
                    lmin, lmax,
                    k_ellmax=mlmax)['src']

            #Get N0
            recon_stuff["N0_phi_psh"] = (
                norm_lens_X[0] / (1 - norm_lens_X[0]*norm_src
                                  *R_src_tt**2),
                norm_lens_X[1] / (1 - norm_lens_X[1]*norm_src
                                  *R_src_tt**2),

                )

            qfunc_tt_psh_X = solenspipe.get_qfunc(
                px, ucls, mlmax, "TT", est2='SRC', Al1=norms_X['TT'],
                Al2=norm_src, R12=R_src_tt)
            recon_stuff["qfunc_tt_psh"] = qfunc_tt_psh_X
            recon_stuff["qfunc_tt_psh_incfilter"] = lambda X,Y: qfunc_tt_psh_X(filter_alms_X(X),
                                                                    filter_alms_X(Y))
            def get_fg_trispectrum_phi_N0_psh(cl_fg):
                Ctot = tcls['TT']**2 / cl_fg
                norm_lens = norm_lens_X
                norm_fg = pytempura.norm_lens.qtt(
                    mlmax, lmin,
                    lmax, ucls['TT'],
                    Ctot,gtype='')
                norm_src_fg = pytempura.get_norms(
                    ['TT','src'], ucls, {'TT':Ctot},
                    lmin, lmax,
                    k_ellmax=mlmax)['src']
                R_src_fg = pytempura.get_cross(
                    'SRC','TT', ucls, {'TT':Ctot},
                    lmin, lmax,
                    k_ellmax=mlmax)
                N0_tris = []
                #gradient and curl:
                for i in [0,1]:
                    N0 = norm_lens[i]**2/norm_fg[i]
                    N0_s = norm_src**2/norm_src_fg
                    N0_tri = (N0 + R_src_tt**2 * norm_lens[i]**2 * N0_s
                              - 2 * R_src_tt * norm_lens[i]**2 * norm_src * R_src_fg)
                    N0_tri /= (1 - norm_lens[i]*norm_src*R_src_tt**2)**2
                    N0_tris.append(N0_tri)
                return tuple(N0_tris)
            
            recon_stuff["get_fg_trispectrum_phi_N0_psh"] = get_fg_trispectrum_phi_N0_psh

        if do_prh:
            R_prof_tt = pytempura.get_cross(
                'SRC', 'TT', ucls, tcls_X, lmin,
                lmax, k_ellmax=mlmax,
                profile=profile)
            R_prof_te = pytempura.get_cross(
                'SRC', 'TE', ucls, tcls_X, lmin,
                lmax, k_ellmax=mlmax,
                profile=profile)

            norm_prof = pytempura.get_norms(
                ['TT','src'], ucls, tcls_X,
                lmin, lmax,
                k_ellmax=mlmax, profile=profile)['src']


            recon_stuff["N0_phi_prh"] = (
                norm_lens_X[0] / (1 - norm_lens_X[0]*
                               norm_prof*
                               R_prof_tt**2),
                norm_lens_X[1] / (1 - norm_lens_X[1]*
                               norm_prof*
                               R_prof_tt**2)
            )
                
            qfunc_ph = solenspipe.get_qfunc(
                px, ucls, mlmax, "TT",
                Al1=norms_X['TT'], est2='PH', Al2=norm_prof,
                R12=R_prof_tt, profile=profile)

            recon_stuff["profile"] = profile
            recon_stuff["qfunc_tt_prh"] = qfunc_ph
            recon_stuff["qfunc_tt_prh_incfilter"] = lambda X,Y: qfunc_tt_prh_X(filter_alms_X(X),
                                                                    filter_alms_X(Y))
            recon_stuff["R_prof_tt"] = R_prof_tt
            recon_stuff["norm_prof"] = norm_prof

            def get_fg_trispectrum_phi_N0_prh(cl_fg):
                Ctot = tcls_X['TT']**2 / cl_fg
                norm_lens = norm_lens_X
                norm_fg = pytempura.norm_lens.qtt(
                    mlmax, lmin,
                    lmax, ucls['TT'],
                    Ctot,gtype='')

                norm_src_fg = pytempura.get_norms(
                    ['TT','src'], recon_stuff['ucls'], {'TT':Ctot},
                    kappa_lmin, kappa_lmax,
                    k_ellmax=mlmax, profile=profile)['src']
                R_src_fg = pytempura.get_cross(
                    'SRC','TT', recon_stuff['ucls'], {'TT':Ctot},
                    kappa_lmin, kappa_lmax,
                    k_ellmax=mlmax, profile=profile)

                #gradient and curl
                N0_tris=[]
                for i in [0,1]:
                    N0 = norm_lens[i]**2/norm_fg[i]
                    N0_s = norm_prof**2/norm_src_fg
                    N0_tri = (N0 + R_prof_tt**2 * norm_lens[i]**2 * N0_s
                              - 2 * R_prof_tt * norm_lens[i]**2 * norm_prof * R_src_fg)
                    N0_tri /= (1 - norm_lens[i]*norm_prof*R_prof_tt**2)**2
                    N0_tris.append(N0_tri)
                return tuple(N0_tris)
            
            recon_stuff["get_fg_trispectrum_phi_N0_prh"] = get_fg_trispectrum_phi_N0_prh


        if do_Tpol:
            qfunc_te_X = solenspipe.get_qfunc(px, ucls, mlmax, "TE", Al1=norms_X['TE'])
            recon_stuff["qfunc_te"] = qfunc_te_X
            recon_stuff["qfunc_te_incfilter"] = lambda X,Y: qfunc_te_X(filter_alms_X(X),
                                                                    filter_alms_X(Y))
            if do_psh:
                R_src_te = pytempura.get_cross(
                    'SRC','TE',ucls,tcls,lmin,lmax,
                    k_ellmax=mlmax)
                qfunc_te_psh_X = solenspipe.get_qfunc(
                    px, ucls, mlmax,"TE", est2='SRC', Al1=norms_X['TE'],
                    Al2=norm_src['src'], R12=R_src_te)
                recon_stuff["qfunc_te_psh"] = qfunc_te_psh_X
                recon_stuff["qfunc_te_psh_incfilter"] = lambda X,Y: qfunc_te_psh_X(
                    filter_alms_X(X), filter_alms(Y))

            if do_prh:
                R_prof_te = pytempura.get_cross(
                    'SRC','TE',ucls,tcls,lmin,lmax,
                    k_ellmax=mlmax, profile=profile)
                qfunc_te_prh_X = solenspipe.get_qfunc(
                    px, ucls, mlmax,"TE", est2='PH', Al1=norms_X['TE'],
                    Al2=norm_prof['src'], R12=R_prof_te)
                recon_stuff["qfunc_te_psh"] = qfunc_te_prh_X
                recon_stuff["qfunc_te_psh_incfilter"] = lambda X,Y: qfunc_te_prh_X(
                    filter_alms_X(X), filter_alms(Y))

    else:
        """Setup symmetrized TT estimator"""
        assert (tcls_XY is not None)
        recon_stuff_sym = setup_sym_estimator(
                px, lmin, lmax, mlmax,
                tcls_X['TT'], tcls_Y['TT'],  tcls_XY['TT']
                )
        assert set(recon_stuff_sym.keys()).isdisjoint(recon_stuff.keys())
        recon_stuff.update(recon_stuff_sym)

    return recon_stuff


def test_signal(nsim=10, use_mpi=False, from_pkl=False):
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank,size = comm.Get_rank(), comm.Get_size()
    else:
        comm = None
        rank,size = 0,1
    if rank==0:
        outdir="sym_estimator_test_output"
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

    if not from_pkl:
    
        px=qe.pixelization(nside=4096)
        mlmax=4000
        lmin=100
        lmax=3000
        binner=ClBinner(lmin=lmin, lmax=lmax, nbin=20)

        noise_sigma_X = 10.
        noise_sigma_Y = 100.
        #cross-correlation coefficient 
        r = 0.5

        beam_fwhm=2.
        ells = np.arange(mlmax+1)
        beam = maps.gauss_beam(ells, beam_fwhm)
        Nl_tt_X = (noise_sigma_X*np.pi/180./60.)**2./beam**2
        Nl_tt_Y = (noise_sigma_Y*np.pi/180./60.)**2./beam**2
        nells_X = {"TT":Nl_tt_X, "EE":2*Nl_tt_X, "BB":2*Nl_tt_X}
        nells_Y = {"TT":Nl_tt_Y, "EE":2*Nl_tt_Y, "BB":2*Nl_tt_Y}

        _,tcls_X = futils.get_theory_dicts(grad=True, nells=nells_X, lmax=mlmax)
        _,tcls_Y = futils.get_theory_dicts(grad=True, nells=nells_Y, lmax=mlmax)
        _,tcls_nonoise = futils.get_theory_dicts(grad=True, lmax=mlmax)

        cltot_X = tcls_X['TT']
        cltot_Y = tcls_Y['TT']
        cltot_XY = tcls_nonoise['TT'] + r*np.sqrt(Nl_tt_X*Nl_tt_Y)
        #We need N_l for generating uncorrelated component of Y,
        #call this map Z
        #We have noise_Y = a*noise_X + noise_Z
        #then N_l^Y = a^2 N_l^X + N_l^Z
        #and <XY> = a * N_l^X = r_l * sqrt(N_l^X N_l^Y) (from defintion of r_l)
        #then we have a = r_l*sqrt(N_l^X N_l^Y) / N_l^X
        #and N_l^Z = N_l^Y - a^2 N_l^X 
        a = r * np.sqrt(Nl_tt_X*Nl_tt_Y)/Nl_tt_X
        Nl_tt_Z = Nl_tt_Y - a**2 * Nl_tt_X

        """
        # Test the noise is working as expected
        noise_alm_X = curvedsky.rand_alm(Nl_tt_X)
        noise_alm_Z = curvedsky.rand_alm(Nl_tt_Z)
        noise_alm_Y = curvedsky.almxfl(noise_alm_X,a) + noise_alm_Z

        cl_XX = curvedsky.alm2cl(noise_alm_X)
        cl_YY = curvedsky.alm2cl(noise_alm_Y)
        cl_XY = curvedsky.alm2cl(noise_alm_X, noise_alm_Y)

        fig,ax=plt.subplots()

        ax.plot(cl_XX/Nl_tt_X)
        ax.plot(cl_YY/Nl_tt_Y)
        ax.plot(cl_XY / (r*np.sqrt(Nl_tt_X*Nl_tt_Y)))
        ax.set_ylim([0.8,1.2])
        """
        # Setup the symmetrized estimator
        sym_setup = setup_sym_estimator(px, lmin, lmax, mlmax,
                                cltot_X, cltot_Y, cltot_XY)
        qfunc_XY = sym_setup["qfunc_XY_incfilter"]
        qfunc_YX = sym_setup["qfunc_YX_incfilter"]
        qfunc_sym = sym_setup["qfunc_sym_incfilter"]
        
        #Loop through sims 
        #- getting cmb alms
        #- adding noise
        #- running lensing estimators 
        #- cross-correlating with input
        #– getting auto cls
        cl_dict = {"kk_XY" : [],
                   "kxi_XY" : [],
                   "kk_YX" : [],
                   "kxi_YX" : [],
                   "kk_sym" : [],
                   "kxi_sym" : [],
                   "ii" : [] #input cl_kappa,
        }

        for isim in range(nsim):
            if isim%size != rank:
                continue
            print("rank %d doing sim %d"%(rank,isim))
            print("reading cmb and kappa alm")
            cmb_alm = futils.get_cmb_alm(isim,0)[0]
            cmb_alm = futils.change_alm_lmax(cmb_alm, mlmax)
            kappa_alm = futils.get_kappa_alm(isim)
            kappa_alm = futils.change_alm_lmax(kappa_alm, mlmax)
            ells = np.arange(mlmax+1)
            cl_kk_binned = binner(curvedsky.alm2cl(kappa_alm))
            cl_dict['ii'].append(cl_kk_binned)

            print("generating noise")
            noise_alm_X = curvedsky.rand_alm(Nl_tt_X, seed=isim*nsim)
            noise_alm_Z = curvedsky.rand_alm(Nl_tt_Z, seed=isim*nsim+1)
            noise_alm_Y = curvedsky.almxfl(noise_alm_X,a) + noise_alm_Z

            sky_alm_X = cmb_alm+noise_alm_X
            sky_alm_Y = cmb_alm+noise_alm_Y

            print("running phi estimators")
            phi_XY = qfunc_XY(sky_alm_X, sky_alm_Y)
            phi_YX = qfunc_YX(sky_alm_X, sky_alm_Y)
            phi_sym = qfunc_sym(
                None, None, phi_XY=phi_XY, phi_YX=phi_YX
            )

            kappa_XY = lensing.phi_to_kappa(phi_XY[0])
            kappa_YX = lensing.phi_to_kappa(phi_YX[0])
            kappa_sym = lensing.phi_to_kappa(phi_sym[0])

            print("getting Cls")
            #cross with input
            cl_dict["kxi_XY"].append(binner(
                curvedsky.alm2cl(kappa_XY, kappa_alm)
            ))
            cl_dict["kxi_YX"].append(binner(
                curvedsky.alm2cl(kappa_YX, kappa_alm)
            ))
            cl_dict["kxi_sym"].append(binner(
                curvedsky.alm2cl(kappa_sym, kappa_alm)
            ))

            #auto
            cl_dict["kk_XY"].append(binner(
                curvedsky.alm2cl(kappa_XY)
            ))
            cl_dict["kk_YX"].append(binner(
                curvedsky.alm2cl(kappa_YX)
            ))              
            cl_dict["kk_sym"].append(binner(
                curvedsky.alm2cl(kappa_sym)
            ))


        #rank 0 collects and plots
        if rank==0:
            #collect and plot
            n_collected=1
            while n_collected<size:
                cls_to_add = comm.recv(source=MPI.ANY_SOURCE)
                for key in cl_dict.keys():
                    cl_dict[key] += cls_to_add[key]
                n_collected+=1
            #convert to arrays
            for key in cl_dict:
                cl_dict[key] = np.array(cl_dict[key])
                
            #also save binner info
            cl_dict["ell_mids"] = binner.bin_mids
            cl_dict["lmin"] = binner.lmin
            cl_dict["lmax"] = binner.lmax
            cl_dict["nbin"] = binner.nbin
            #and N0s
            L = np.arange(lmax+1)
            N0_XYXY = binner(sym_setup["N0_XYXY_phi"][0] * (L*(L+1)/2)**2)
            N0_YXYX = binner(sym_setup["N0_YXYX_phi"][0] * (L*(L+1)/2)**2)
            N0_sym = binner((1./sym_setup["w_sum_g"]) * (L*(L+1)/2)**2)
            cl_dict["N0_XYXY"] = N0_XYXY
            cl_dict["N0_YXYX"] = N0_YXYX
            cl_dict["N0_sym"] = N0_sym
                
            with open(opj(outdir,"cls.pkl"), 'wb') as f:
                pickle.dump(cl_dict, f)
        else:
            comm.send(cl_dict, dest=0)
            return 0

    else:
        if rank==0:
            with open(opj(outdir, "cls.pkl"),"rb") as f:
                cl_dict = pickle.load(f)
                
    if rank==0:
        #get means and plot
        #first do x input
        ell_mids = cl_dict["ell_mids"]
        cl_iis = cl_dict["ii"]
        nsim = cl_dict["ii"].shape[0]
        print("nsim:",nsim)
        cl_kxi_XY_fracdiff_mean = (cl_dict["kxi_XY"]/cl_iis-1).mean(axis=0)
        cl_kxi_XY_err = (cl_dict["kxi_XY"]/cl_iis-1).std(axis=0)/np.sqrt(nsim-1)

        cl_kxi_YX_fracdiff_mean = (cl_dict["kxi_YX"]/cl_iis-1).mean(axis=0)
        cl_kxi_YX_err = (cl_dict["kxi_YX"]/cl_iis-1).std(axis=0)/np.sqrt(nsim-1)

        cl_kxi_sym_fracdiff_mean = (cl_dict["kxi_sym"]/cl_iis-1).mean(axis=0)
        cl_kxi_sym_err = (cl_dict["kxi_sym"]/cl_iis-1).std(axis=0)/np.sqrt(nsim-1)

        #do some plotting
        fig,ax=plt.subplots(figsize=(5,4))

        ax.errorbar(ell_mids, cl_kxi_XY_fracdiff_mean, yerr=cl_kxi_XY_err, label='XY')
        ax.errorbar(ell_mids-10, cl_kxi_YX_fracdiff_mean, yerr=cl_kxi_YX_err, label='YX')
        ax.errorbar(ell_mids+10, cl_kxi_sym_fracdiff_mean, yerr=cl_kxi_sym_err, label='sym')

        ax.legend()
        ax.set_title("x input")
        ax.set_xlabel(r"$L$")
        ax.set_ylabel(r"$C_l^{\hat{\kappa},\kappa} / C_l^{\kappa, \kappa}-1$")
        ax.set_ylim([-0.03,0.03])
        fig.tight_layout()
        fig.savefig(opj(outdir, "clkxinput_fracdiff.png"), dpi=200)

        #Now auto
        for i in range(nsim):
            cl_dict["kk_XY"][i] -= cl_dict['N0_XYXY']
            cl_dict["kk_YX"][i] -= cl_dict['N0_YXYX']
            cl_dict["kk_sym"][i] -= cl_dict['N0_sym']

        #absolute
        fig,ax=plt.subplots(figsize=(5,4))
        cl_kk_XY_mean = (cl_dict["kk_XY"]).mean(axis=0)
        cl_kk_XY_err = (cl_dict["kk_XY"]).std(axis=0)/np.sqrt(nsim-1)
        cl_kk_YX_mean = (cl_dict["kk_YX"]).mean(axis=0)
        cl_kk_YX_err = (cl_dict["kk_YX"]).std(axis=0)/np.sqrt(nsim-1)
        cl_kk_sym_mean = (cl_dict["kk_sym"]).mean(axis=0)
        cl_kk_sym_err = (cl_dict["kk_sym"]).std(axis=0)/np.sqrt(nsim-1)
        ax.plot(ell_mids, cl_dict['ii'].mean(axis=0), 'k', label='input')
        ax.errorbar(ell_mids, cl_kk_XY_mean, yerr=cl_kk_XY_err, label='XY')
        ax.errorbar(ell_mids-10, cl_kk_YX_mean, yerr=cl_kk_YX_err, label='YX')
        ax.errorbar(ell_mids+10, cl_kk_sym_mean, yerr=cl_kk_sym_err, label='sym')

        ax.legend()
        ax.set_yscale('log')
        ax.set_ylim([1.e-9, ax.get_ylim()[1]])
        ax.set_title("auto")
        ax.set_xlabel(r"$L$")
        ax.set_ylabel(r"$C_l^{\kappa\kappa}$")
        fig.tight_layout()
        fig.savefig(opj(outdir, "clkk_auto.png"), dpi=200)
            
        #fractional difference
        fig,ax=plt.subplots(figsize=(5,4))
        cl_kk_XY_fracdiff_mean = (cl_dict["kk_XY"]/cl_iis-1).mean(axis=0)
        cl_kk_XY_fracdiff_err = (cl_dict["kk_XY"]/cl_iis-1).std(axis=0)/np.sqrt(nsim-1)
        cl_kk_YX_fracdiff_mean = (cl_dict["kk_YX"]/cl_iis-1).mean(axis=0)
        cl_kk_YX_fracdiff_err = (cl_dict["kk_YX"]/cl_iis-1).std(axis=0)/np.sqrt(nsim-1)
        cl_kk_sym_fracdiff_mean = (cl_dict["kk_sym"]/cl_iis-1).mean(axis=0)
        cl_kk_sym_fracdiff_err = (cl_dict["kk_sym"]/cl_iis-1).std(axis=0)/np.sqrt(nsim-1)

        ax.errorbar(ell_mids, cl_kk_XY_fracdiff_mean, yerr=cl_kk_XY_err, label='XY')
        ax.errorbar(ell_mids-10, cl_kk_YX_fracdiff_mean, yerr=cl_kk_YX_err, label='YX')
        ax.errorbar(ell_mids+10, cl_kk_sym_fracdiff_mean, yerr=cl_kk_sym_err, label='sym')

        ax.legend()
        #ax.set_ylim([-0.5,0.5])
        ax.set_title("auto")
        ax.set_xlabel(r"$L$")
        ax.set_ylabel(r"$C_l^{\hat{\kappa}\hat{\kappa}} / C_l^{\kappa, \kappa}-1$")
        fig.tight_layout()
        fig.savefig(opj(outdir, "clkk_auto_fracdiff.png"), dpi=200)

        #Also the N0s
        fig,ax=plt.subplots(figsize=(5,4))
        ax.plot(ell_mids, cl_dict['N0_XYXY'], label='<Q[XY]Q[XY]>')
        ax.plot(ell_mids, cl_dict['N0_YXYX'], label='<Q[YX]Q[YX]>')
        ax.plot(ell_mids, cl_dict['N0_sym'], label='sym')
        ax.set_yscale('log')
        ax.set_xlabel(r"$L$")
        ax.set_ylabel(r"$N^0$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(opj(outdir, "N0.png"), dpi=200)


def test_N0(use_mpi=False, nsim=10, from_pkl=False):
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank,size = comm.Get_rank(), comm.Get_size()
    else:
        comm = None
        rank,size = 0,1
    if rank==0:
        outdir="sym_estimator_test_output"
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

    if not from_pkl:
    
        px=qe.pixelization(nside=4096)
        mlmax=2000
        lmin=100
        lmax=1500
        binner=ClBinner(lmin=lmin, lmax=lmax, nbin=20)

        noise_sigma_X = 10.
        noise_sigma_Y = 100.
        #cross-correlation coefficient 
        r = 0.5

        beam_fwhm=2.
        ells = np.arange(mlmax+1)
        beam = maps.gauss_beam(ells, beam_fwhm)
        Nl_tt_X = (noise_sigma_X*np.pi/180./60.)**2./beam**2
        Nl_tt_Y = (noise_sigma_Y*np.pi/180./60.)**2./beam**2
        nells_X = {"TT":Nl_tt_X, "EE":2*Nl_tt_X, "BB":2*Nl_tt_X}
        nells_Y = {"TT":Nl_tt_Y, "EE":2*Nl_tt_Y, "BB":2*Nl_tt_Y}

        _,tcls_X = futils.get_theory_dicts(grad=True, nells=nells_X, lmax=mlmax)
        _,tcls_Y = futils.get_theory_dicts(grad=True, nells=nells_Y, lmax=mlmax)
        _,tcls_nonoise = futils.get_theory_dicts(grad=True, lmax=mlmax)

        cltot_X = tcls_X['TT']
        cltot_Y = tcls_Y['TT']
        cltot_XY = tcls_nonoise['TT'] + r*np.sqrt(Nl_tt_X*Nl_tt_Y)
        #We need N_l for generating uncorrelated component of Y,
        #call this map Z
        #We have noise_Y = a*noise_X + noise_Z
        #then N_l^Y = a^2 N_l^X + N_l^Z
        #and <XY> = a * N_l^X = r_l * sqrt(N_l^X N_l^Y) (from defintion of r_l)
        #then we have a = r_l*sqrt(N_l^X N_l^Y) / N_l^X
        #and N_l^Z = N_l^Y - a^2 N_l^X 
        a = r * np.sqrt(Nl_tt_X*Nl_tt_Y)/Nl_tt_X
        Nl_tt_Z = Nl_tt_Y - a**2 * Nl_tt_X

        # Setup the symmetrized estimator
        sym_setup = setup_sym_estimator(px, lmin, lmax, mlmax,
                                cltot_X, cltot_Y, cltot_XY)
        qfunc_XY = sym_setup["qfunc_XY_incfilter"]
        qfunc_YX = sym_setup["qfunc_YX_incfilter"]
        qfunc_sym = sym_setup["qfunc_sym_incfilter"]

        #Also want to test foreground trispectrum
        #Let's assume map Y has 1/10th of the foreground
        #contamination
        cl_fg_X = np.loadtxt("cl_tsz_websky_0093.txt")[:mlmax+1]
        cl_fg_Y = 0.1**2 * cl_fg_X
        cl_fg_XY = 0.1*cl_fg_X

        N0_tri_phi_XY = sym_setup["get_fg_trispectrum_N0_XYXY"](
            cl_fg_X, cl_fg_Y, cl_fg_XY)
        N0_tri_phi_YX = sym_setup["get_fg_trispectrum_N0_YXYX"](
            cl_fg_X, cl_fg_Y, cl_fg_XY)
        N0_tri_phi_XYYX = sym_setup["get_fg_trispectrum_N0_XYYX"](
            cl_fg_X, cl_fg_Y, cl_fg_XY)
        N0_tri_phi_sym = sym_setup["get_fg_trispectrum_N0_sym"](
            cl_fg_X, cl_fg_Y, cl_fg_XY)
        
        #Loop through sims 
        #generating Gaussian sims,
        #running reconstruction,
        #and comparing to theory N0
        #Run cmb+noise Gaussian sims,
        #as well as foreground only sims,
        #to tests trispectrum N0
        cl_dict = {"kk_XY" : [],
                   "kk_YX" : [],
                   "kk_sym" : [],
                   "fgfg_XY" : [],
                   "fgfg_YX" : [],
                   "fgfg_XYYX" : [],
                   "fgfg_sym" : []
        }

        for isim in range(nsim):
            if isim%size != rank:
                continue
            print("rank %d doing sim %d"%(rank,isim))

            print("generating noise")
            noise_alm_X = curvedsky.rand_alm(Nl_tt_X, seed=isim*nsim)
            noise_alm_Z = curvedsky.rand_alm(Nl_tt_Z, seed=isim*nsim+1)
            noise_alm_Y = curvedsky.almxfl(noise_alm_X,a) + noise_alm_Z

            cmb_alm = curvedsky.rand_alm(tcls_nonoise['TT'],
                                          seed=isim*nsim+2)
            gaussian_alm_X = cmb_alm+noise_alm_X
            gaussian_alm_Y = cmb_alm+noise_alm_Y

            gaussian_fg_alm_X = curvedsky.rand_alm(cl_fg_X, seed=isim*nsim)
            gaussian_fg_alm_Y = 0.1*gaussian_fg_alm_X
            
            print("running phi estimators")
            phi_XY = qfunc_XY(gaussian_alm_X, gaussian_alm_Y)
            phi_YX = qfunc_YX(gaussian_alm_X, gaussian_alm_Y)
            phi_sym = qfunc_sym(
                None, None, phi_XY=phi_XY, phi_YX=phi_YX)

            kappa_XY = lensing.phi_to_kappa(phi_XY[0])
            kappa_YX = lensing.phi_to_kappa(phi_YX[0])
            kappa_sym = lensing.phi_to_kappa(phi_sym[0])

            #run on foreground sim to test trispectrum N0
            phi_fg_XY = qfunc_XY(gaussian_fg_alm_X, gaussian_fg_alm_Y)
            phi_fg_YX = qfunc_YX(gaussian_fg_alm_X, gaussian_fg_alm_Y)
            phi_fg_sym = qfunc_sym(
                None, None, phi_XY=phi_fg_XY, phi_YX=phi_fg_YX)

            kappa_fg_XY = lensing.phi_to_kappa(phi_fg_XY[0])
            kappa_fg_YX = lensing.phi_to_kappa(phi_fg_YX[0])
            kappa_fg_sym = lensing.phi_to_kappa(phi_fg_sym[0])

            #get cls
            cl_dict["kk_XY"].append(binner(
                curvedsky.alm2cl(kappa_XY)
            ))
            cl_dict["kk_YX"].append(binner(
                curvedsky.alm2cl(kappa_YX)
            ))
            cl_dict["kk_sym"].append(binner(
                curvedsky.alm2cl(kappa_sym)
            ))

            cl_dict["fgfg_XY"].append(binner(
                curvedsky.alm2cl(kappa_fg_XY)
            ))
            cl_dict["fgfg_YX"].append(binner(
                curvedsky.alm2cl(kappa_fg_YX)
            ))
            cl_dict["fgfg_XYYX"].append(binner(
                curvedsky.alm2cl(kappa_fg_XY,
                                 kappa_fg_YX)
            ))
            cl_dict["fgfg_sym"].append(binner(
                curvedsky.alm2cl(kappa_fg_sym)
            ))

        #rank 0 collects and plots
        if rank==0:
            #collect and plot
            n_collected=1
            while n_collected<size:
                cls_to_add = comm.recv(source=MPI.ANY_SOURCE)
                for key in cl_dict.keys():
                    cl_dict[key] += cls_to_add[key]
                n_collected+=1
            #convert to arrays
            for key in cl_dict:
                cl_dict[key] = np.array(cl_dict[key])
                
            #also save binner info
            cl_dict["ell_mids"] = binner.bin_mids
            cl_dict["lmin"] = binner.lmin
            cl_dict["lmax"] = binner.lmax
            cl_dict["nbin"] = binner.nbin
            #and N0s
            L = np.arange(mlmax+1)
            N0_XYXY = binner(sym_setup["N0_XYXY_phi"][0] * (L*(L+1)/2)**2)
            N0_YXYX = binner(sym_setup["N0_YXYX_phi"][0] * (L*(L+1)/2)**2)
            N0_sym = binner((1./sym_setup["w_sum_g"]) * (L*(L+1)/2)**2)
            cl_dict["N0_XYXY"] = N0_XYXY
            cl_dict["N0_YXYX"] = N0_YXYX
            cl_dict["N0_sym"] = N0_sym

            #trispectrum N0
            cl_dict["N0_fg_XYXY"] = binner(
                N0_tri_phi_XY[0] * (L*(L+1)/2)**2)
            cl_dict["N0_fg_YXYX"] = binner(
                N0_tri_phi_YX[0] * (L*(L+1)/2)**2)
            cl_dict["N0_fg_XYYX"] = binner(
                N0_tri_phi_XYYX[0] * (L*(L+1)/2)**2)
            cl_dict["N0_fg_sym"] = binner(
                N0_tri_phi_sym[0] * (L*(L+1)/2)**2)
                
            with open(opj(outdir,"cls_N0.pkl"), 'wb') as f:
                pickle.dump(cl_dict, f)
        else:
            comm.send(cl_dict, dest=0)
            return 0

    else:
        if rank==0:
            with open(opj(outdir, "cls_N0.pkl"),"rb") as f:
                cl_dict = pickle.load(f)

    if rank==0:
        #get means and plot
        #first do x input
        ell_mids = cl_dict["ell_mids"]
        nsim = cl_dict["kk_sym"].shape[0]
        print("nsim:",nsim)

        #mean cls of gaussian sims should be equal to N0
        cl_kk_XY_mean = cl_dict["kk_XY"].mean(axis=0)
        cl_kk_XY_err = (cl_dict["kk_XY"]).std(axis=0)/np.sqrt(nsim-1)
        cl_kk_YX_mean = cl_dict["kk_YX"].mean(axis=0)
        cl_kk_YX_err = (cl_dict["kk_YX"]).std(axis=0)/np.sqrt(nsim-1)
        cl_kk_sym_mean = cl_dict["kk_sym"].mean(axis=0)
        cl_kk_sym_err = (cl_dict["kk_sym"]).std(axis=0)/np.sqrt(nsim-1)
        
        #do some plotting
        fig,ax=plt.subplots(figsize=(5,4))

        ax.errorbar(ell_mids, cl_kk_XY_mean/cl_dict["N0_XYXY"],
                    yerr=cl_kk_XY_err/cl_dict["N0_XYXY"], label="XYXY")
        ax.errorbar(ell_mids, cl_kk_YX_mean/cl_dict["N0_YXYX"],
                    yerr=cl_kk_YX_err/cl_dict["N0_YXYX"], label="YXYX")
        ax.errorbar(ell_mids, cl_kk_sym_mean/cl_dict["N0_sym"],
                    yerr=cl_kk_sym_err/cl_dict["N0_sym"], label="sym")
        ax.legend()

        ax.set_xlabel(r"$L$")
        ax.set_ylabel(r"$N^0$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(opj(outdir, "N0_test.png"), dpi=200)

        #foreground trispectrum N0 plot
        cl_fgfg_XY_mean = cl_dict["fgfg_XY"].mean(axis=0)
        cl_fgfg_XY_err = (cl_dict["fgfg_XY"]).std(axis=0)/np.sqrt(nsim-1)
        cl_fgfg_YX_mean = cl_dict["fgfg_YX"].mean(axis=0)
        cl_fgfg_YX_err = (cl_dict["fgfg_YX"]).std(axis=0)/np.sqrt(nsim-1)
        cl_fgfg_XYYX_mean = cl_dict["fgfg_XYYX"].mean(axis=0)
        cl_fgfg_XYYX_err = cl_dict["fgfg_XYYX"].std(axis=0)/np.sqrt(nsim-1)
        cl_fgfg_sym_mean = cl_dict["fgfg_sym"].mean(axis=0)
        cl_fgfg_sym_err = (cl_dict["fgfg_sym"]).std(axis=0)/np.sqrt(nsim-1)
        fig,ax=plt.subplots(figsize=(5,4))

        ax.errorbar(ell_mids, cl_fgfg_XY_mean/cl_dict["N0_fg_XYXY"],
                    yerr=cl_fgfg_XY_err/cl_dict["N0_fg_XYXY"], label="XYXY")
        ax.errorbar(ell_mids, cl_fgfg_YX_mean/cl_dict["N0_fg_YXYX"],
                    yerr=cl_fgfg_YX_err/cl_dict["N0_fg_YXYX"], label="YXYX")
        ax.errorbar(ell_mids, cl_fgfg_XYYX_mean/cl_dict["N0_fg_XYYX"],
                    yerr=cl_fgfg_XYYX_err/cl_dict["N0_fg_XYYX"], label="XYYX")
        ax.errorbar(ell_mids, cl_fgfg_sym_mean/cl_dict["N0_fg_sym"],
                    yerr=cl_fgfg_sym_err/cl_dict["N0_fg_sym"], label="sym")
        ax.legend()

        ax.set_title("foreground trispectrum N0")
        ax.set_xlabel(r"$L$")
        ax.set_ylabel(r"$N^0_{\mathrm{sim}} / N^0_{\mathrm{theory}}$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(opj(outdir, "N0_fg_trispectrum_test.png"), dpi=200)

def get_TT_secondary(qfunc_incfilter, Tf1,
                     Tcmb, Tcmb_prime, Tf2=None):
    #Secondary is 
    #<(Q[Tcmb, Tf_2]+Q[Tf_1, Tcmb])(Q[Tcmb, Tf_2]+Q[Tf_1, Tcmb])>
    #to remove noise bias we need to subtract
    #(Q[Tcmb_prime, Tf_2]+Q[Tf_1, Tcmb_prime]) from both
    #sides of the correlator, where Tcmb_prime is a cmb
    #map with the same unlensed CMB as T_cmb, but lensed
    #by an independent kappa
    if Tf2 is None:
        Tf2 = Tf1
    phi_Tcmb_Tf2 = qfunc_incfilter(Tcmb, Tf2)
    phi_Tf1_Tcmb = qfunc_incfilter(Tf1, Tcmb)
    phi_Tcmbp_Tf2 = qfunc_incfilter(Tcmb_prime, Tf2)
    phi_Tf1_Tcmbp = qfunc_incfilter(Tf1, Tcmb_prime)
    kappa = lensing.phi_to_kappa(
        phi_Tcmb_Tf2[0]+phi_Tf1_Tcmb[0])
    kappap = lensing.phi_to_kappa(
        phi_Tcmbp_Tf2[0]+phi_Tf1_Tcmbp[0])
    S = curvedsky.alm2cl(kappa)-curvedsky.alm2cl(kappap)
    return S

def get_TT_secondary(qfunc_incfilter, Tf1,
                     Tcmb, Tcmb_prime, Tf2=None):
    #Secondary is 
    #<(Q[Tcmb, Tf_2]+Q[Tf_1, Tcmb])(Q[Tcmb, Tf_2]+Q[Tf_1, Tcmb])>
    #to remove noise bias we need to subtract
    #(Q[Tcmb_prime, Tf_2]+Q[Tf_1, Tcmb_prime]) from both
    #sides of the correlator, where Tcmb_prime is a cmb
    #map with the same unlensed CMB as T_cmb, but lensed
    #by an independent kappa
    if Tf2 is None:
        Tf2 = Tf1
    phi_Tcmb_Tf2 = qfunc_incfilter(Tcmb, Tf2)
    phi_Tf1_Tcmb = qfunc_incfilter(Tf1, Tcmb)
    phi_Tcmbp_Tf2 = qfunc_incfilter(Tcmb_prime, Tf2)
    phi_Tf1_Tcmbp = qfunc_incfilter(Tf1, Tcmb_prime)
    kappa = lensing.phi_to_kappa(
        phi_Tcmb_Tf2[0]+phi_Tf1_Tcmb[0])
    kappap = lensing.phi_to_kappa(
        phi_Tcmbp_Tf2[0]+phi_Tf1_Tcmbp[0])
    S = curvedsky.alm2cl(kappa)-curvedsky.alm2cl(kappap)
    return S

def get_all_secondary_terms(qfunc_tt_incfilter,
                      qfunc_te_incfilter,
                      qfunc_tb_incfilter,
                      Tf1_alm, cmb_alm, cmb_prime_alm,
                      Tf2_alm=None):
    #Get all secondary terms
    #i.e. TTTT, TTTE, TETE, TTTB, TBTB
    #if Tf2_alm != Tf1_alm, there would be more
    #potentially. But let's assume for now
    #that T1 is used for the T-pol estimators
    equal_Tf = False
    if Tf2_alm is None:
        equal_Tf=True
        Tf2_alm=Tf1_alm
        
    #make sure cmb alms are the right format
    for x in [cmb_alm, cmb_prime_alm]:
        assert isinstance(x, tuple)
        assert len(x)==3

    #First do TTTT
    Tcmb, Tcmb_prime = cmb_alm[0]
    phi_Tcmb_Tf2 = qfunc_TT_incfilter(Tcmb, Tf2)
    phi_Tf1_Tcmb = qfunc_TT_incfilter(Tf1, Tcmb)
    phi_Tcmbp_Tf2 = qfunc_TT_incfilter(Tcmb_prime, Tf2)
    phi_Tf1_Tcmbp = qfunc_TT_incfilter(Tf1, Tcmb_prime)

    kappa_TT = lensing.phi_to_kappa(
        phi_Tcmb_Tf2[0]+phi_Tf1_Tcmb[0])
    kappa_TTp = lensing.phi_to_kappa(
        phi_Tcmbp_Tf2[0]+phi_Tf1_Tcmbp[0])

    S_TTTT = curvedsky.alm2cl(kappa_TT)-curvedsky.alm2cl(kappa_TTp)

    #Now pol
    E, E_prime, B, B_prime = (cmb_alm[1], cmb_prime_alm[1],
                              cmb_alm[2], cmb_prime_alm[2])
    kappa_Tf1_Ecmb = lensing.phi_to_kappa(
        qfunc_te_incfilter(Tf1, E)[0]
        )
    kappa_Tf1_Ecmbp = lensing.phi_to_kappa(
        qfunc_te_incfilter(Tf1, E_prime)[0]
        )
    kappa_Tf1_Bcmb = lensing.phi_to_kappa(
        qfunc_tb_incfilter(Tf1, B)[0]
        )
    kappa_Tf1_Bcmbp = lensing.phi_to_kappa(
        qfunc_tb_incfilter(Tf1, B_prime)[0]
        )
    S_TTTE = (
        curvedsky.alm2cl(kappa_TT, kappa_Tf1_Ecmb)
        - curvedsky.alm2cl(kappa_TTp, kappa_Tf1_Ecmbp)
        )
    S_TTTB = (
        curvedsky.alm2cl(kappa_TT, kappa_Tf1_Bcmb)
        - curvedsky.alm2cl(kappa_TTp, kappa_Tf1_Bcmbp)
        )
    S_TETE = (curvedsky.alm2cl(kappa_Tf1_Ecmb)
              - curvedsky.alm2cl(kappa_Tf1_Ecmbp)
              )
    S_TBTB = (curvedsky.alm2cl(kappa_Tf1_Bcmb)
              - curvedsky.alm2cl(kappa_Tf1_Bcmbp)
              )
    #let's return a dictionary here
    #becuase there's more than a couple of
    #things to return
    S = {"TTTT" : S_TTTT,
         "TTTE" : S_TTTE,
         "TTTB" : S_TTTB,
         "TETE" : S_TETE,
         "TBTB" : S_TBTB}
    return S
        
def get_TT_secondary_T0T1(qfunc_incfilter, Tf1,
                     T0_cmb, T1_cmb, Tf2=None):
    #For non symmetric estimators Q[T_1,T_2], we have
    #S = <(Q[Tf_1, Tcmb_0] + Q[Tcmb_0, Tf_2])(Q[Tf_1, Tcmb_1]+Q[Tcmb_1, Tf_2])>
    #   +<(Q[Tf_1, Tcmb_1] + Q[Tcmb_1, Tf_2])(Q[Tf_1, Tcmb_0]+Q[Tcmb_0, Tf_2])>  
    if Tf2 is None:
        Tf2 = Tf1
    Tf1_Tcmb0 = lensing.phi_to_kappa(
        qfunc_incfilter(Tf1, T0_cmb)[0]
    )
    Tcmb0_Tf2 = lensing.phi_to_kappa(
        qfunc_incfilter(T0_cmb, Tf2)[0]
    )
    Tf1_Tcmb1 = lensing.phi_to_kappa(
    qfunc_incfilter(Tf1, T1_cmb)[0]
    )
    Tcmb1_Tf2 = lensing.phi_to_kappa(
    qfunc_incfilter(T1_cmb, Tf2)[0]
    )
    
    S = 2*curvedsky.alm2cl(
    (Tf1_Tcmb0+Tcmb0_Tf2), (Tf1_Tcmb1+Tcmb1_Tf2)
    )
    return S

def plot_secondary_terms():
    outdir="sym_estimator_test_output"
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    px=qe.pixelization(nside=1024)
    mlmax=2000
    lmin=100
    lmax=1500
    binner=ClBinner(lmin=lmin, lmax=lmax, nbin=15)
    ell_mids = binner.bin_mids
    
    noise_sigma_X = 10.
    beam_fwhm=2.
    ells = np.arange(mlmax+1)
    beam = maps.gauss_beam(ells, beam_fwhm)
    Nl_tt_X = (noise_sigma_X*np.pi/180./60.)**2./beam**2
    nells_X = {"TT":Nl_tt_X, "EE":2*Nl_tt_X, "BB":2*Nl_tt_X}

    _,tcls_X = futils.get_theory_dicts(grad=True, nells=nells_X, lmax=mlmax)
    _,tcls_nonoise = futils.get_theory_dicts(grad=True, lmax=mlmax)

    recon_setup_XX = setup_recon(px, lmin, lmax, mlmax,
                                 tcls_X)
    qfunc_XX = recon_setup_XX["qfunc_tt_incfilter"]
    #Also want to test foreground trispectrum
    #Let's assume map Y has 1/10th of the foreground
    #contamination
    fg_alm_X = futils.change_alm_lmax(
        hp.read_alm("fg_nonoise_alms_0093.fits"),
        mlmax)
    cl_fg_X = np.loadtxt("cl_tsz_websky_0093.txt")[:mlmax+1]

    #also get secondary the other way
    websky_dir="/global/project/projectdirs/act/data/maccrann/websky"
    cmb0_alms = hp.fitsfunc.read_alm(
        opj(websky_dir, 'unlensed_alm.fits'))
    cmb1_alms = hp.fitsfunc.read_alm(
        opj(websky_dir, 'T1alm_websky_lmax6000_nside4096.fits'))
    T0_alms = futils.change_alm_lmax(cmb0_alms, mlmax)
    T1_alms = futils.change_alm_lmax(cmb1_alms, mlmax)

    #get each term in secondary (see A10 of Omar's paper)
    kappa_Tf_Tcmb0 = lensing.phi_to_kappa(
        qfunc_XX(fg_alm_X, T0_alms)[0])
    kappa_Tcmb0_Tf = lensing.phi_to_kappa(
        qfunc_XX(T0_alms, fg_alm_X)[0])
    kappa_Tf_Tcmb1 = lensing.phi_to_kappa(
        qfunc_XX(fg_alm_X, T1_alms)[0])
    kappa_Tcmb1_Tf = lensing.phi_to_kappa(
        qfunc_XX(T1_alms, fg_alm_X)[0])

    cl_Tf_Tcmb0_Tf_Tcmb1 = binner(curvedsky.alm2cl(
        kappa_Tf_Tcmb0, kappa_Tf_Tcmb1
        ))
    cl_Tf_Tcmb0_Tcmb1_Tf = binner(curvedsky.alm2cl(
        kappa_Tf_Tcmb0, kappa_Tcmb1_Tf)
                                  )
    cl_Tcmb0_Tf_Tf_Tcmb1 = binner(curvedsky.alm2cl(
        kappa_Tcmb0_Tf, kappa_Tf_Tcmb1)
                                  )
    cl_Tcmb0_Tf_Tcmb1_Tf = binner(curvedsky.alm2cl(
        kappa_Tcmb0_Tf, kappa_Tcmb1_Tf)
                                  )

    fig,ax=plt.subplots()

    ax.plot(ell_mids, cl_Tf_Tcmb0_Tf_Tcmb1,
            label=r"$<Q[T_f, T_{CMB,0}] Q[T_f, T_{CMB,1}]>$"
            )
    ax.plot(ell_mids, cl_Tf_Tcmb0_Tcmb1_Tf,
            label=r"$<Q[T_f, T_{CMB,0}] Q[T_{CMB,1}, T_f]>$"
            )
    ax.plot(ell_mids, cl_Tcmb0_Tf_Tf_Tcmb1,
            label=r"$<Q[T_{CMB,0}, T_f] Q[T_f, T_{CMB,1}]>$"
            )
    ax.plot(ell_mids, cl_Tcmb0_Tf_Tcmb1_Tf,
            label=r"$<Q[T_{CMB,0}, T_f] Q[T_{CMB,1}, T_f]>$"
            )

    #also plot total
    total_secondary = (cl_Tf_Tcmb0_Tf_Tcmb1
                       +cl_Tf_Tcmb0_Tcmb1_Tf
                       +cl_Tcmb0_Tf_Tf_Tcmb1
                       +cl_Tcmb0_Tf_Tcmb1_Tf)
    ax.plot(ell_mids, total_secondary,
            label="total")
    
    ax.set_xlabel(r"$L$")
    ax.set_title("websky tsz qe secondary terms")
    ax.legend()
    fig.tight_layout()
    fig.savefig(opj(outdir, "secondary_terms.png"), dpi=200)  

def get_websky_lensed_cmb(cmb_seed):
    f = opj(WEBSKY_DIR, "lensed_cmb", "lensed_cmb_alm_websky_cmb%d_lmax6000.fits"%cmb_seed)
    return hp.fitsfunc.read_alm(f, hdu=(1,2,3))

def get_sehgal_lensed_cmb(cmb_seed):
    f = opj(SEHGAL_DIR, "lensed_cmb", "lensed_cmb_alm_sehgal_cmb%d_lmax6000.fits"%cmb_seed)
    return hp.fitsfunc.read_alm(f, hdu=(1,2,3))
    
def test_secondary(use_mpi=False, nsim=10, from_pkl=False):
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank,size = comm.Get_rank(), comm.Get_size()
    else:
        comm = None
        rank,size = 0,1
    if rank==0:
        outdir="sym_estimator_test_output"
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

    if not from_pkl:
    
        px=qe.pixelization(nside=1024)
        mlmax=2000
        lmin=100
        lmax=1500
        binner=ClBinner(lmin=lmin, lmax=lmax, nbin=15)

        noise_sigma = 10.
        beam_fwhm=2.
        ells = np.arange(mlmax+1)
        beam = maps.gauss_beam(ells, beam_fwhm)
        Nl_tt = (noise_sigma*np.pi/180./60.)**2./beam**2
        nells = {"TT":Nl_tt, "EE":2*Nl_tt, "BB":2*Nl_tt}

        _,tcls = futils.get_theory_dicts(grad=True, nells=nells, lmax=mlmax)
        _,tcls_nonoise = futils.get_theory_dicts(grad=True, lmax=mlmax)

        recon_setup = setup_recon(px, lmin, lmax, mlmax,
                                     tcls)
        qfunc = recon_setup["qfunc_tt_incfilter"]
        #Also want to test foreground trispectrum
        #Let's assume map Y has 1/10th of the foreground
        #contamination
        fg_alm = futils.change_alm_lmax(
            hp.read_alm("fg_nonoise_alms_0093.fits"),
            mlmax)
        cl_fg = np.loadtxt("cl_tsz_websky_0093.txt")[:mlmax+1]
        cl_dict = {"S_raw" : [],
                   "S_gaussian" : [],
                   "Tf_Tcmb_Tf_Tcmb" : [],
                   "Tf_Tcmb_Tcmb_Tf" : [],
                   "Tcmb_Tf_Tcmb_Tf" : [],
                   "Tf_Tcmb_Tf_Tcmb_gaussian" : [],
                   "Tf_Tcmb_Tcmb_Tf_gaussian" : [],
                   "Tcmb_Tf_Tcmb_Tf_gaussian" : []
        }
        for isim in range(nsim):
            if isim%size != rank:
                continue
            print("rank %d doing sim %d"%(rank,isim))
            print("reading cmb and kappa alm")
            cmb_alm = get_websky_lensed_cmb(1999-isim)
            #cmb_alm = futils.get_cmb_alm(isim,0)[0]
            cmb_alm = futils.change_alm_lmax(cmb_alm, mlmax)

            print("generating noise")
            noise_alm = curvedsky.rand_alm(Nl_tt, seed=isim*nsim)

            print("Gaussian cmb alm")
            cmb_gaussian_alm = curvedsky.rand_alm(tcls_nonoise['TT'],
                                          seed=isim*nsim+2)

            cmb_and_noise_alm = cmb_alm+noise_alm
            gaussian_alm = cmb_gaussian_alm+noise_alm
            #gaussian_fg_alm_X = curvedsky.rand_alm(cl_fg_X, seed=isim*nsim)

            #Get the secondary.
            #For <Q[T_1,T_2]Q[T_1,T_2]>
            #   =<Q[(Tcmb+Tf_1), (Tcmb, Tf_2)]Q[(Tcmb+Tf_1), (Tcmb, Tf_2)]>
            #this is
            #<(Q[Tcmb, Tf_2]+Q[Tf_1, Tcmb])(Q[Tcmb, Tf_2]+Q[Tf_1, Tcmb])>
            phi_cmb_f2 = qfunc(cmb_alm, fg_alm)
            phi_f1_cmb = qfunc(fg_alm,  cmb_alm)

            kappa_cmb_f2 = lensing.phi_to_kappa(
                phi_cmb_f2[0])
            kappa_f1_cmb = lensing.phi_to_kappa(
                phi_f1_cmb[0])
            kappa = kappa_cmb_f2+kappa_f1_cmb


            cl_dict["S_raw"].append(binner(curvedsky.alm2cl(
                kappa)))

            cl_dict["Tf_Tcmb_Tf_Tcmb"].append(
                binner(curvedsky.alm2cl(
                    kappa_f1_cmb)
                       ))
            cl_dict["Tf_Tcmb_Tcmb_Tf"].append(
                binner(curvedsky.alm2cl(
                    kappa_f1_cmb, kappa_cmb_f2
                    )))
            cl_dict["Tcmb_Tf_Tcmb_Tf"].append(
                binner(curvedsky.alm2cl(
                    kappa_cmb_f2
                    )))

            
            
            #we need the same for Gaussian sims
            phi_cmb_f2_gaussian = qfunc(cmb_gaussian_alm, fg_alm)
            phi_f1_cmb_gaussian = qfunc(fg_alm,  cmb_gaussian_alm)
            kappa_cmb_f2_gaussian = lensing.phi_to_kappa(
                phi_cmb_f2_gaussian[0])
            kappa_f1_cmb_gaussian = lensing.phi_to_kappa(
                phi_f1_cmb_gaussian[0])
            cl_dict["Tf_Tcmb_Tf_Tcmb_gaussian"].append(
                binner(curvedsky.alm2cl(
                    kappa_f1_cmb_gaussian)
                       ))
            cl_dict["Tf_Tcmb_Tcmb_Tf_gaussian"].append(
                binner(curvedsky.alm2cl(
                    kappa_f1_cmb_gaussian, kappa_cmb_f2_gaussian
                    )))
            cl_dict["Tcmb_Tf_Tcmb_Tf_gaussian"].append(
                binner(curvedsky.alm2cl(
                    kappa_cmb_f2_gaussian
                    )))

            cl_dict["S_gaussian"].append(binner(
                curvedsky.alm2cl(
                    lensing.phi_to_kappa(
                        phi_cmb_f2_gaussian[0]+phi_f1_cmb_gaussian[0]
                        )
                    )
                ))
            
        if rank==0:
            #also get secondary the other way
            websky_dir="/global/project/projectdirs/act/data/maccrann/websky"
            """
            cmb0_alms = hp.fitsfunc.read_alm(
                opj(websky_dir, 'unlensed_alm.fits'))
            cmb1_alms = hp.fitsfunc.read_alm(
                opj(websky_dir, 'T1alm_websky_lmax6000_nside4096.fits'))
            T0_alms = futils.change_alm_lmax(cmb0_alms, mlmax)
            T1_alms = futils.change_alm_lmax(cmb1_alms, mlmax)

            S = get_TT_secondary_T0T1(qfunc, fg_alm,
                                    T0_alms, T1_alms, Tf2=None)
            """
            T_alm = get_websky_lensed_cmb(1999)
            T_p_alm = futils.get_cmb_alm(1999,0)[0]
            S = get_TT_secondary(qfunc, fg_alms, T_alm, T_p_alm)
            
            #collect and plot
            n_collected=1
            while n_collected<size:
                cls_to_add = comm.recv(source=MPI.ANY_SOURCE)
                for key in cl_dict.keys():
                    cl_dict[key] += cls_to_add[key]
                n_collected+=1
            #convert to arrays
            for key in cl_dict:
                cl_dict[key] = np.array(cl_dict[key])
                
            #also save binner info
            cl_dict["ell_mids"] = binner.bin_mids
            cl_dict["lmin"] = binner.lmin
            cl_dict["lmax"] = binner.lmax
            cl_dict["nbin"] = binner.nbin

            #and fast secondary calc
            cl_dict["S_fast"] = binner(S)
            print(cl_dict)
            with open(opj(outdir,"cls_secondary.pkl"), 'wb') as f:
                pickle.dump(cl_dict, f)

        else:
            comm.send(cl_dict, dest=0)
            return 0

    else:
        if rank==0:
            with open(opj(outdir, "cls_secondary.pkl"),"rb") as f:
                cl_dict = pickle.load(f)
            
    if rank==0:
        #get means and plot
        ell_mids = cl_dict["ell_mids"]
        nsim = cl_dict["S_raw"].shape[0]
        print("nsim:",nsim)

        #get mean secondary
        S_raw_mean = cl_dict["S_raw"].mean(axis=0)
        S_raw_err = cl_dict["S_raw"].std(axis=0)/np.sqrt(nsim-1)
        S_gaussian_mean = cl_dict["S_gaussian"].mean(axis=0)
        S_gaussian_err = cl_dict["S_gaussian"].std(axis=0)/np.sqrt(nsim-1)

        #do some plotting
        fig,ax=plt.subplots(figsize=(5,4))
        
        ax.errorbar(0.95*ell_mids, S_raw_mean-S_gaussian_mean,
                    yerr=np.sqrt(S_raw_err**2+S_gaussian_err**2),
                    linestyle='-', color='C0', label="brute force")
        
        ax.plot(0.95*ell_mids, cl_dict["S_fast"], '--', color='C0', label="fast")

        ax.set_xlabel(r"$L$")
        ax.set_title("websky tsz secondary")
        ax.set_ylabel(r"$secondary$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(opj(outdir, "secondary_test.png"), dpi=200)  

        #frac diff
        fig,ax=plt.subplots(figsize=(5,4))
        
        ax.errorbar(ell_mids, (S_raw_mean-S_gaussian_mean)/cl_dict["S_fast"]-1,
                    yerr=np.sqrt(S_raw_err**2+S_gaussian_err**2)/cl_dict["S_fast"],
                    linestyle='-')

        ax.set_xlabel(r"$L$")
        ax.set_ylabel("Brute force / fast method")
        ax.set_title("websky tsz secondary")
        ax.legend()
        fig.tight_layout()
        fig.savefig(opj(outdir, "secondary_test_fracdiff.png"), dpi=200)


        #I also want to look at the terms in the secondary as calculated
        #using the brute force method
        Tf_Tcmb_Tf_Tcmb_mean = (cl_dict["Tf_Tcmb_Tf_Tcmb"].mean(axis=0)
                                -cl_dict["Tf_Tcmb_Tf_Tcmb_gaussian"].mean(axis=0))
        Tf_Tcmb_Tcmb_Tf_mean = (cl_dict["Tf_Tcmb_Tcmb_Tf"].mean(axis=0)
                                -cl_dict["Tf_Tcmb_Tcmb_Tf_gaussian"].mean(axis=0))
        Tcmb_Tf_Tcmb_Tf_mean = (cl_dict["Tcmb_Tf_Tcmb_Tf"].mean(axis=0)
                                -cl_dict["Tcmb_Tf_Tcmb_Tf_gaussian"].mean(axis=0))

        print(Tf_Tcmb_Tf_Tcmb_mean.shape)
        
        fig,ax=plt.subplots(figsize=(5,4))

        ax.plot(ell_mids, Tf_Tcmb_Tf_Tcmb_mean,
                label = r"$<Q[Tf1, Tcmb]Q[Tf1, Tcmb]>$")
        ax.plot(ell_mids, 2*Tf_Tcmb_Tcmb_Tf_mean,
                label = r"$2*<Q[Tf1, Tcmb]Q[Tcmb, Tf2]>$")
        ax.plot(ell_mids, Tcmb_Tf_Tcmb_Tf_mean,
                label = r"$<Q[Tcmb, Tf2]Q[Tcmb, Tf2]>$")
        ax.set_xlabel(r"$L$")
        ax.set_ylabel("secondary terms")
        ax.set_title("websky tsz secondary terms (note Tf1=Tf2 for this case)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(opj(outdir, "secondary_terms_brute_force.png"), dpi=200)


def test_sym_secondary(use_mpi=False, nsim=10, from_pkl=False):
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank,size = comm.Get_rank(), comm.Get_size()
    else:
        comm = None
        rank,size = 0,1
    if rank==0:
        outdir="sym_estimator_test_output"
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

    if not from_pkl:
    
        px=qe.pixelization(nside=1024)
        mlmax=2000
        lmin=100
        lmax=1500
        binner=ClBinner(lmin=lmin, lmax=lmax, nbin=15)

        noise_sigma_X = 10.
        noise_sigma_Y = 100.
        #cross-correlation coefficient 
        r = 0.5

        beam_fwhm=2.
        ells = np.arange(mlmax+1)
        beam = maps.gauss_beam(ells, beam_fwhm)
        Nl_tt_X = (noise_sigma_X*np.pi/180./60.)**2./beam**2
        Nl_tt_Y = (noise_sigma_Y*np.pi/180./60.)**2./beam**2
        nells_X = {"TT":Nl_tt_X, "EE":2*Nl_tt_X, "BB":2*Nl_tt_X}
        nells_Y = {"TT":Nl_tt_Y, "EE":2*Nl_tt_Y, "BB":2*Nl_tt_Y}

        _,tcls_X = futils.get_theory_dicts(grad=True, nells=nells_X, lmax=mlmax)
        _,tcls_Y = futils.get_theory_dicts(grad=True, nells=nells_Y, lmax=mlmax)
        _,tcls_nonoise = futils.get_theory_dicts(grad=True, lmax=mlmax)

        cltot_X = tcls_X['TT']
        cltot_Y = tcls_Y['TT']
        cltot_XY = tcls_nonoise['TT'] + r*np.sqrt(Nl_tt_X*Nl_tt_Y)
        #We need N_l for generating uncorrelated component of Y,
        #call this map Z
        #We have noise_Y = a*noise_X + noise_Z
        #then N_l^Y = a^2 N_l^X + N_l^Z
        #and <XY> = a * N_l^X = r_l * sqrt(N_l^X N_l^Y) (from defintion of r_l)
        #then we have a = r_l*sqrt(N_l^X N_l^Y) / N_l^X
        #and N_l^Z = N_l^Y - a^2 N_l^X 
        a = r * np.sqrt(Nl_tt_X*Nl_tt_Y)/Nl_tt_X
        Nl_tt_Z = Nl_tt_Y - a**2 * Nl_tt_X

        # Setup the symmetrized estimator
        sym_setup = setup_sym_estimator(px, lmin, lmax, mlmax,
                                cltot_X, cltot_Y, cltot_XY)

        qfunc_XY = sym_setup["qfunc_XY_incfilter"]
        qfunc_YX = sym_setup["qfunc_YX_incfilter"]
        qfunc_sym = sym_setup["qfunc_sym_incfilter"]

        #Also want to test foreground trispectrum
        #Let's assume map Y has 1/10th of the foreground
        #contamination
        fg_alm_X = futils.change_alm_lmax(
            hp.read_alm("fg_nonoise_alms_0093.fits"),
            mlmax)
        cl_fg_X = np.loadtxt("cl_tsz_websky_0093.txt")[:mlmax+1]
        fg_alm_Y = 0.1*fg_alm_X
        cl_fg_Y = 0.1**2 * cl_fg_X
        cl_fg_XY = 0.1*cl_fg_X

        #Loop through sims 
        #generating Gaussian sims,
        #running reconstruction,
        #and comparing to theory N0
        #Run cmb+noise Gaussian sims,
        #as well as foreground only sims,
        #to tests trispectrum N0
        cl_dict = {"S_XY_raw" : [],
                   "S_YX_raw" : [],
                   "S_sym_raw" : [],
                   "S_XY_gaussian" : [],
                   "S_YX_gaussian" : [],
                   "S_sym_gaussian" : [],
        }

        for isim in range(nsim):
            if isim%size != rank:
                continue
            print("rank %d doing sim %d"%(rank,isim))
            print("reading cmb and kappa alm")
            cmb_alm = futils.get_cmb_alm(isim,0)[0]
            cmb_alm = futils.change_alm_lmax(cmb_alm, mlmax)

            print("generating noise")
            noise_alm_X = curvedsky.rand_alm(Nl_tt_X, seed=isim*nsim)
            noise_alm_Z = curvedsky.rand_alm(Nl_tt_Z, seed=isim*nsim+1)
            noise_alm_Y = curvedsky.almxfl(noise_alm_X,a) + noise_alm_Z

            print("Gaussian cmb alm")
            cmb_gaussian_alm = curvedsky.rand_alm(tcls_nonoise['TT'],
                                          seed=isim*nsim+2)

            cmb_and_noise_alm_X = cmb_alm+noise_alm_X
            cmb_and_noise_alm_Y = cmb_alm+noise_alm_Y
            gaussian_alm_X = cmb_gaussian_alm+noise_alm_X
            gaussian_alm_Y = cmb_gaussian_alm+noise_alm_Y

            gaussian_fg_alm_X = curvedsky.rand_alm(cl_fg_X, seed=isim*nsim)
            gaussian_fg_alm_Y = 0.1*gaussian_fg_alm_X

            #Get the secondary.
            #For <Q[T_1,T_2]Q[T_1,T_2]>
            #   =<Q[(Tcmb+Tf_1), (Tcmb, Tf_2)]Q[(Tcmb+Tf_1), (Tcmb, Tf_2)]>
            #this is
            #<(Q[Tcmb, Tf_2]+Q[Tf_1, Tcmb])(Q[Tcmb, Tf_2]+Q[Tf_1, Tcmb])>
            phi_cmb_f2_XY = qfunc_XY(cmb_alm, fg_alm_Y)
            phi_f1_cmb_XY = qfunc_XY(fg_alm_X,  cmb_alm)

            phi_cmb_f2_YX = qfunc_YX(cmb_alm, fg_alm_Y)
            phi_f1_cmb_YX = qfunc_YX(fg_alm_X,  cmb_alm)

            phi_cmb_f2_sym = qfunc_sym(None, None, phi_XY=phi_cmb_f2_XY,
                                       phi_YX=phi_cmb_f2_YX)
            phi_f1_cmb_sym = qfunc_sym(None, None, phi_XY=phi_f1_cmb_XY,
                                       phi_YX=phi_f1_cmb_YX)

            kappa_XY = lensing.phi_to_kappa(
                phi_cmb_f2_XY[0] + phi_f1_cmb_XY[0])
            kappa_YX = lensing.phi_to_kappa(
                phi_cmb_f2_YX[0] + phi_f1_cmb_YX[0])
            kappa_sym = lensing.phi_to_kappa(
                phi_cmb_f2_sym[0] + phi_f1_cmb_sym[0])

            cl_dict["S_XY_raw"].append(binner(curvedsky.alm2cl(
                kappa_XY)))
            cl_dict["S_YX_raw"].append(binner(curvedsky.alm2cl(
                kappa_YX)))
            cl_dict["S_sym_raw"].append(binner(curvedsky.alm2cl(
		kappa_sym)))
            
            #we need the same for Gaussian sims
            phi_cmb_f2_XY_gaussian = qfunc_XY(cmb_gaussian_alm, fg_alm_Y)
            phi_f1_cmb_XY_gaussian = qfunc_XY(fg_alm_X,  cmb_gaussian_alm)

            phi_cmb_f2_YX_gaussian = qfunc_YX(cmb_gaussian_alm, fg_alm_Y)
            phi_f1_cmb_YX_gaussian = qfunc_YX(fg_alm_X,  cmb_gaussian_alm)

            phi_cmb_f2_sym_gaussian = qfunc_sym(None, None, phi_XY=phi_cmb_f2_XY_gaussian,
                                                phi_YX=phi_cmb_f2_YX_gaussian)
            phi_f1_cmb_sym_gaussian = qfunc_sym(None, None, phi_XY=phi_f1_cmb_XY_gaussian,
                                                phi_YX=phi_f1_cmb_YX_gaussian)
            cl_dict["S_XY_gaussian"].append(binner(
                curvedsky.alm2cl(
                    lensing.phi_to_kappa(
                        phi_cmb_f2_XY_gaussian[0]+phi_f1_cmb_XY_gaussian[0]
                        )
                    )
                ))
            cl_dict["S_YX_gaussian"].append(binner(
                curvedsky.alm2cl(
                    lensing.phi_to_kappa(
                        phi_cmb_f2_YX_gaussian[0]+phi_f1_cmb_YX_gaussian[0]
                        )
                    )
                ))
            cl_dict["S_sym_gaussian"].append(binner(
                curvedsky.alm2cl(
                    lensing.phi_to_kappa(
                        phi_cmb_f2_sym_gaussian[0]+phi_f1_cmb_sym_gaussian[0]
                        )
                    )
                ))
            
        if rank==0:
            #also get secondary the other way
            websky_dir="/global/project/projectdirs/act/data/maccrann/websky"
            cmb0_alms = hp.fitsfunc.read_alm(
                opj(websky_dir, 'unlensed_alm.fits'))
            cmb1_alms = hp.fitsfunc.read_alm(
                opj(websky_dir, 'T1alm_websky_lmax6000_nside4096.fits'))
            T0_alms = futils.change_alm_lmax(cmb0_alms, mlmax)
            T1_alms = futils.change_alm_lmax(cmb1_alms, mlmax)

            
            S_XY = get_TT_secondary(qfunc_XY, fg_alm_X,
                                    T0_alms, T1_alms, Tf2=fg_alm_Y)
            S_YX = get_TT_secondary(qfunc_YX, fg_alm_X,
                                    T0_alms, T1_alms, Tf2=fg_alm_Y)
            S_sym = get_TT_secondary(qfunc_sym, fg_alm_X,
                                    T0_alms, T1_alms, Tf2=fg_alm_Y)

            #collect and plot
            n_collected=1
            while n_collected<size:
                cls_to_add = comm.recv(source=MPI.ANY_SOURCE)
                for key in cl_dict.keys():
                    cl_dict[key] += cls_to_add[key]
                n_collected+=1
            #convert to arrays
            for key in cl_dict:
                cl_dict[key] = np.array(cl_dict[key])
                
            #also save binner info
            cl_dict["ell_mids"] = binner.bin_mids
            cl_dict["lmin"] = binner.lmin
            cl_dict["lmax"] = binner.lmax
            cl_dict["nbin"] = binner.nbin

            #and fast secondary calc
            cl_dict["S_XY_fast"] = binner(S_XY)
            cl_dict["S_YX_fast"] = binner(S_YX)
            cl_dict["S_sym_fast"] = binner(S_sym)

            print(cl_dict)
            with open(opj(outdir,"cls_secondary.pkl"), 'wb') as f:
                pickle.dump(cl_dict, f)

        else:
            comm.send(cl_dict, dest=0)
            return 0

    else:
        if rank==0:
            with open(opj(outdir, "cls_secondary.pkl"),"rb") as f:
                cl_dict = pickle.load(f)
            
    if rank==0:
        #get means and plot
        #first do x input
        ell_mids = cl_dict["ell_mids"]
        nsim = cl_dict["S_XY_raw"].shape[0]
        print("nsim:",nsim)

        #get mean secondary
        S_XY_raw_mean = cl_dict["S_XY_raw"].mean(axis=0)
        S_YX_raw_mean = cl_dict["S_YX_raw"].mean(axis=0)
        S_sym_raw_mean = cl_dict["S_sym_raw"].mean(axis=0)

        S_XY_raw_err = cl_dict["S_XY_raw"].std(axis=0)/np.sqrt(nsim-1)
        S_YX_raw_err = cl_dict["S_YX_raw"].std(axis=0)/np.sqrt(nsim-1)
        S_sym_raw_err = cl_dict["S_sym_raw"].std(axis=0)/np.sqrt(nsim-1)

        S_XY_gaussian_mean = cl_dict["S_XY_gaussian"].mean(axis=0)
        S_YX_gaussian_mean = cl_dict["S_YX_gaussian"].mean(axis=0)
        S_sym_gaussian_mean = cl_dict["S_sym_gaussian"].mean(axis=0)
        S_XY_gaussian_err = cl_dict["S_XY_gaussian"].std(axis=0)/np.sqrt(nsim-1)
        S_YX_gaussian_err = cl_dict["S_YX_gaussian"].std(axis=0)/np.sqrt(nsim-1)
        S_sym_gaussian_err = cl_dict["S_sym_gaussian"].std(axis=0)/np.sqrt(nsim-1)
        
        #do some plotting
        fig,ax=plt.subplots(figsize=(5,4))
        
        ax.errorbar(0.95*ell_mids, S_XY_raw_mean-S_XY_gaussian_mean,
                    yerr=np.sqrt(S_XY_raw_err**2+S_XY_gaussian_err**2),
                    linestyle='-', color='C0', label="XY brute force")
        
        ax.plot(0.95*ell_mids, cl_dict["S_XY_fast"], '--', color='C0', label="XY fast")

        ax.errorbar(ell_mids, S_YX_raw_mean-S_YX_gaussian_mean,
                yerr=np.sqrt(S_YX_raw_err**2+S_YX_gaussian_err**2),
                linestyle='-', color='C1', label="YX brute force")

        ax.plot(ell_mids, cl_dict["S_YX_fast"], '--', color='C1', label="YX fast")

        ax.errorbar(1.05*ell_mids, S_sym_raw_mean-S_sym_gaussian_mean,
                yerr=np.sqrt(S_sym_raw_err**2+S_sym_gaussian_err**2),
                linestyle='-', color='C2', label="sym brute force")

        ax.plot(1.05*ell_mids, cl_dict["S_sym_fast"], '--', color='C2', label="sym fast")

        ax.set_xlabel(r"$L$")
        ax.set_title("websky tsz secondary")
        ax.set_ylabel(r"$secondary$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(opj(outdir, "sym_secondary_test.png"), dpi=200)  

        #frac diff
        fig,ax=plt.subplots(figsize=(5,4))
        
        ax.errorbar(0.95*ell_mids, (S_XY_raw_mean-S_XY_gaussian_mean)/cl_dict["S_XY_fast"]-1,
                    yerr=np.sqrt(S_XY_raw_err**2+S_XY_gaussian_err**2)/cl_dict["S_XY_fast"],
                    linestyle='-', color='C0', label="XY")

        ax.errorbar(ell_mids, (S_YX_raw_mean-S_YX_gaussian_mean)/cl_dict["S_YX_fast"]-1,
                yerr=np.sqrt(S_YX_raw_err**2+S_YX_gaussian_err**2)/cl_dict["S_YX_fast"],
                linestyle='-', color='C1', label="YX")

        ax.errorbar(1.05*ell_mids, (S_sym_raw_mean-S_sym_gaussian_mean)/cl_dict["S_sym_fast"]-1,
                yerr=np.sqrt(S_sym_raw_err**2+S_sym_gaussian_err**2),
                linestyle='-', color='C2', label="sym")

        ax.set_xlabel(r"$L$")
        ax.set_ylabel(r"$Brute force / fast method$")
        ax.set_title("websky tsz secondary")
        ax.legend()
        fig.tight_layout()
        fig.savefig(opj(outdir, "sym_secondary_test_fracdiff.png"), dpi=200)  

        
        
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="run tests")
    parser.add_argument("-m", "--mpi", action='store_true',
                        help="use_mpi")
    parser.add_argument("--from_pkl", action="store_true",
                        help="read cls from pkl")
    parser.add_argument("-n", "--nsim", type=int, default=2)
    args = parser.parse_args()

    #test_N0(use_mpi=args.mpi, nsim=args.nsim, from_pkl=args.from_pkl)
    test_secondary(use_mpi=args.mpi, nsim=args.nsim, from_pkl=args.from_pkl)
    #plot_secondary_terms()

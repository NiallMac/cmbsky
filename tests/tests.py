from cmbsky.reconstruction import *
import os
from os.path import join as opj

websky_dir=os.environ["WEBSKY_DIR"]
TSZ_ALM_FILE = opj(websky_dir, "tsz_alms_0093_lmax6000.fits")
CL_TSZ_FILE = opj(websky_dir, "cl_tsz_websky_0093.txt")

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

        #Read in foreground cl
        cl_fg_X = np.loadtxt(CL_TSZ_FILE)[:mlmax+1]
        cl_fg_Y = 0.1**2 * cl_fg_X
        cl_fg_XY = 0.1*cl_fg_X

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
        print("setting up estimators")
        sym_setup = setup_sym_estimator(px, lmin, lmax, mlmax,
                                        cltot_X, cltot_Y, cltot_XY,
                                        do_psh=True)
        
        qfunc_XY = sym_setup["qfunc_XY_incfilter"]
        qfunc_YX = sym_setup["qfunc_YX_incfilter"]
        qfunc_sym = sym_setup["qfunc_sym_incfilter"]
        qfunc_XY_psh = sym_setup["qfunc_XY_psh_incfilter"]
        qfunc_YX_psh = sym_setup["qfunc_YX_psh_incfilter"]
        qfunc_sym_psh = sym_setup["qfunc_sym_psh_incfilter"]
        
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
                   "ii" : [], #input cl_kappa,
                   "kk_gaussian_XY" : [],
                   "kk_gaussian_YX" : [],
                   "kk_gaussian_sym" : [],
        }
        for key in list(cl_dict.keys()):
            cl_dict[key+"_psh"] = []

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
            cl_dict["ells"] = ells
            cl_kk_binned = binner(curvedsky.alm2cl(kappa_alm))
            cl_dict['ii'].append(cl_kk_binned)

            print("generating noise")
            noise_alm_X = curvedsky.rand_alm(Nl_tt_X, seed=isim*(10*nsim))
            noise_alm_Z = curvedsky.rand_alm(Nl_tt_Z, seed=isim*(10*nsim)+1)
            noise_alm_Y = curvedsky.almxfl(noise_alm_X,a) + noise_alm_Z

            X = cmb_alm+noise_alm_X
            Y = cmb_alm+noise_alm_Y

            cmb_gaussian = curvedsky.rand_alm(tcls_nonoise['TT'],
                                          seed=isim*(10*nsim)+2)
            X_gaussian = cmb_alm+noise_alm_X
            Y_gaussian = cmb_alm+noise_alm_Y

            X_fg_gaussian = curvedsky.rand_alm(cl_fg_X, seed=isim*(10*nsim)+3)
            Y_fg_gaussian = 0.1*X_fg_gaussian
        
            print("running phi estimators")
            phi_XY = qfunc_XY(X, Y)
            phi_YX = qfunc_YX(X, Y)
            phi_sym = qfunc_sym(
                None, None, phi_XY=phi_XY, phi_YX=phi_YX
            )

            kappa_XY = lensing.phi_to_kappa(phi_XY[0])
            kappa_YX = lensing.phi_to_kappa(phi_YX[0])
            kappa_sym = lensing.phi_to_kappa(phi_sym[0])

            print("running gaussian case")
            phi_XY_gaussian = qfunc_XY(X_gaussian, Y_gaussian)
            phi_YX_gaussian = qfunc_YX(X_gaussian, Y_gaussian)
            phi_sym_gaussian = qfunc_sym(
                None, None, phi_XY=phi_XY_gaussian, phi_YX=phi_YX_gaussian
            )

            kappa_XY_gaussian = lensing.phi_to_kappa(phi_XY_gaussian[0])
            kappa_YX_gaussian = lensing.phi_to_kappa(phi_YX_gaussian[0])
            kappa_sym_gaussian = lensing.phi_to_kappa(phi_sym_gaussian[0])

            print("psh case")
            phi_XY_psh = qfunc_XY_psh(X, Y)
            phi_YX_psh = qfunc_YX_psh(X, Y)
            phi_sym_psh = qfunc_sym_psh(
                None, None, phi_XY=phi_XY_psh, phi_YX=phi_YX_psh
            )

            kappa_XY_psh = lensing.phi_to_kappa(phi_XY_psh[0])
            kappa_YX_psh = lensing.phi_to_kappa(phi_YX_psh[0])
            kappa_sym_psh = lensing.phi_to_kappa(phi_sym_psh[0])

            print("running psh gaussian case")
            phi_XY_psh_gaussian = qfunc_XY_psh(X_gaussian, Y_gaussian)
            phi_YX_psh_gaussian = qfunc_YX_psh(X_gaussian, Y_gaussian)
            phi_sym_psh_gaussian = qfunc_sym_psh(
                None, None, phi_XY=phi_XY_psh_gaussian, phi_YX=phi_YX_psh_gaussian
            )

            kappa_XY_psh_gaussian = lensing.phi_to_kappa(phi_XY_psh_gaussian[0])
            kappa_YX_psh_gaussian = lensing.phi_to_kappa(phi_YX_psh_gaussian[0])
            kappa_sym_psh_gaussian = lensing.phi_to_kappa(phi_sym_psh_gaussian[0])

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
            cl_dict["kxi_XY_psh"].append(binner(
                curvedsky.alm2cl(kappa_XY_psh, kappa_alm)
            ))
            cl_dict["kxi_YX_psh"].append(binner(
                curvedsky.alm2cl(kappa_YX_psh, kappa_alm)
            ))
            cl_dict["kxi_sym_psh"].append(binner(
                curvedsky.alm2cl(kappa_sym_psh, kappa_alm)
            ))

            #auto of gaussians
            cl_dict["kk_gaussian_XY"].append(binner(
                curvedsky.alm2cl(kappa_XY_gaussian)
            ))
            cl_dict["kk_gaussian_YX"].append(binner(
                curvedsky.alm2cl(kappa_YX_gaussian)
            ))              
            cl_dict["kk_gaussian_sym"].append(binner(
                curvedsky.alm2cl(kappa_sym_gaussian)
            ))
            cl_dict["kk_gaussian_XY_psh"].append(binner(
                curvedsky.alm2cl(kappa_XY_psh_gaussian)
            ))
            cl_dict["kk_gaussian_YX_psh"].append(binner(
                curvedsky.alm2cl(kappa_YX_psh_gaussian)
            ))              
            cl_dict["kk_gaussian_sym_psh"].append(binner(
                curvedsky.alm2cl(kappa_sym_psh_gaussian)
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
            L = np.arange(mlmax+1)
            N0_XYXY = binner(sym_setup["N0_XYXY_phi"][0] * (L*(L+1)/2)**2)
            N0_YXYX = binner(sym_setup["N0_YXYX_phi"][0] * (L*(L+1)/2)**2)
            N0_sym = binner(sym_setup["N0_sym_phi"][0] * (L*(L+1)/2)**2)
            cl_dict["N0_XYXY"] = N0_XYXY
            cl_dict["N0_YXYX"] = N0_YXYX
            cl_dict["N0_sym"] = N0_sym
            N0_XYXY_psh = binner(sym_setup["N0_XYXY_phi_psh"][0] * (L*(L+1)/2)**2)
            N0_YXYX_psh = binner(sym_setup["N0_YXYX_phi_psh"][0] * (L*(L+1)/2)**2)
            N0_sym_psh = binner(sym_setup["N0_sym_phi_psh"][0] * (L*(L+1)/2)**2)
            cl_dict["N0_XYXY_psh"] = N0_XYXY_psh
            cl_dict["N0_YXYX_psh"] = N0_YXYX_psh
            cl_dict["N0_sym_psh"] = N0_sym_psh
                
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
        binner = ClBinner(lmin=cl_dict["lmin"], lmax=cl_dict["lmax"],
                          nbin=cl_dict["nbin"])
        ell_mids = binner.bin_mids
        def plot_kxi(cl_iis, cl_kxi_XYs, cl_kxi_YXs, cl_kxi_syms,
                     filename):
            cl_iis = cl_dict["ii"]
            nsim = cl_dict["ii"].shape[0]
            print("nsim:",nsim)
            cl_kxi_XY_fracdiff_mean = (cl_kxi_XYs/cl_iis-1).mean(axis=0)
            cl_kxi_XY_err = (cl_dict["kxi_XY"]/cl_iis-1).std(axis=0)/np.sqrt(nsim-1)

            cl_kxi_YX_fracdiff_mean = (cl_kxi_YXs/cl_iis-1).mean(axis=0)
            cl_kxi_YX_err = (cl_kxi_YXs/cl_iis-1).std(axis=0)/np.sqrt(nsim-1)

            cl_kxi_sym_fracdiff_mean = (cl_kxi_syms/cl_iis-1).mean(axis=0)
            cl_kxi_sym_err = (cl_kxi_syms/cl_iis-1).std(axis=0)/np.sqrt(nsim-1)

            #do some plotting
            fig,ax=plt.subplots(figsize=(5,4))

            ax.errorbar(ell_mids, cl_kxi_XY_fracdiff_mean, yerr=cl_kxi_XY_err, label='XY')
            ax.errorbar(ell_mids-10, cl_kxi_YX_fracdiff_mean, yerr=cl_kxi_YX_err, label='YX')
            ax.errorbar(ell_mids+10, cl_kxi_sym_fracdiff_mean, yerr=cl_kxi_sym_err, label='sym')

            ax.legend()
            ax.set_title("x input")
            ax.set_xlabel(r"$L$")
            ax.set_ylabel(r"$C_l^{\hat{\kappa},\kappa} / C_l^{\kappa, \kappa}-1$")
            ax.set_ylim([-0.05,0.05])
            fig.tight_layout()
            #fig.savefig(opj(outdir, "clkxinput_fracdiff.png"), dpi=200)
            fig.savefig(filename, dpi=200)

        def plot_N0(N0s_theory, N0s_sim,
                    labels, filename):

            nsim = N0s_sim[0].shape[0]
            print("nsim:",nsim)
            #do some plotting
            fig,ax=plt.subplots(figsize=(5,4))

            for N0_theory, N0_sim, label in zip(N0s_theory, N0s_sim, labels):
                m = N0_sim.mean(axis=0)
                err = np.std(N0_sim, axis=0)/np.sqrt(nsim)
                ax.errorbar(ell_mids, m/N0_theory-1, yerr=err/N0_theory, label=label)

            ax.legend()
            ax.set_title("x input")
            ax.set_xlabel(r"$L$")
            ax.set_ylabel(r"$N0_{sim} / N0_{data} - 1$")
            #ax.set_ylim([-0.05,0.05])
            fig.tight_layout()
            #fig.savefig(opj(outdir, "clkxinput_fracdiff.png"), dpi=200)
            fig.savefig(filename, dpi=200)

            
        plot_kxi(cl_dict["ii"], cl_dict["kxi_XY"],
                 cl_dict["kxi_YX"], cl_dict["kxi_sym"],
                 opj(outdir, "clkxinput_fracdiff.png"))
            
        #same for psh
        plot_kxi(cl_dict["ii"], cl_dict["kxi_XY_psh"],
                 cl_dict["kxi_YX_psh"], cl_dict["kxi_sym_psh"],
                 opj(outdir, "clkxinput_psh_fracdiff.png"))

        #Now N0s
        L = cl_dict["ells"]
        N0s_theory = [binner(recon_setup[key][0]*(L*(L+1)/2)**2)
                      for key in ["N0_XYXY_phi", "N0_sym_phi"]]
        N0s_sim = [cl_dict[key] for key in
                   ["kk_gaussian_XY", "kk_gaussian_sym"]]
        labels=["XYXY", "sym"]
        filename = opj(outdir, "N0_fracdiff_qe.png")
        plot_N0(N0s_theory, N0s_sim,
                    labels, filename)
        
        #Now auto
        #Not that useful to plot auto because of N1 bias, so
        #I'm going to comment out for now
        """
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
        """


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
        cl_fg_X = np.loadtxt(CL_TSZ_FILE)[:mlmax+1]
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


def plot_secondary_terms():
    outdir = "secondary_plots"
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    px=qe.pixelization(nside=1024)
    mlmax=2000
    lmin=100
    lmax=1500
    binner=ClBinner(lmin=lmin, lmax=lmax, nbin=15)
    ell_mids = binner.bin_mids
    
    noise_sigma = 10.
    beam_fwhm=2.
    ells = np.arange(mlmax+1)
    beam = maps.gauss_beam(ells, beam_fwhm)
    Nl_tt = (noise_sigma*np.pi/180./60.)**2./beam**2
    nells = {"TT":Nl_tt, "EE":2*Nl_tt, "BB":2*Nl_tt}

    ucls,tcls = futils.get_theory_dicts(grad=True, nells=nells, lmax=mlmax)
    _,tcls_nonoise = futils.get_theory_dicts(grad=True, lmax=mlmax)

    recon_setup = setup_recon(px, lmin, lmax, mlmax,
                                 tcls)

    #read foreground
    fg_alm = futils.change_alm_lmax(
        hp.read_alm(TSZ_ALM_FILE),
        mlmax)

    cmb_alm = futils.change_alm_lmax(
        get_websky_lensed_cmb(1999), mlmax)
    cmb_prime_alm = futils.change_alm_lmax(
        futils.get_cmb_alm(1999,0), mlmax)

    #get secondary 
    S_all = get_all_secondary(
        recon_setup["qfunc_tt_incfilter"],
        recon_setup["qfunc_te_incfilter"],
        recon_setup["qfunc_tb_incfilter"],
        fg_alm, cmb_alm, cmb_prime_alm
    )

    fig,ax=plt.subplots()
    ax.plot(ell_mids, binner(S_all["TTTT"]), label="TTTT")
    ax.plot(ell_mids, binner(S_all["TTTE"]), label="TTTE")
    ax.plot(ell_mids, binner(S_all["TTTB"]), label="TTTB")
    ax.plot(ell_mids, binner(S_all["TETE"]), label="TETE")
    ax.plot(ell_mids, binner(S_all["TBTB"]), label="TBTB")

    #also plots cl_kk
    cl_kk = ucls['kk']
    ax.plot(ell_mids, binner(cl_kk), color='k', label=r"$C_L^{\kappa\kappa}$")
    ax.set_xlabel(r"$L$")
    fig.tight_layout()
    ax.set_title("Websky tSZ secondary")
    fig.savefig(opj(outdir, "websky_tsz_all_secondary.png"), dpi=200)
    
        
def plot_tt_secondary_terms():
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
    fg_alm_X = futils.change_alm_lmax(
        hp.read_alm(TSZ_ALM_FILE),
        mlmax)

    #also get secondary the other way
    #websky_dir="/global/project/projectdirs/act/data/maccrann/websky"
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
        #read tsz alms and cl
        fg_alm = futils.change_alm_lmax(
            hp.read_alm(TSZ_ALM_FILE),
            mlmax)
        cl_fg = np.loadtxt(CL_TSZ_FILE)[:mlmax+1]

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
            #websky_dir="/global/project/projectdirs/act/data/maccrann/websky"
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
            T_alm = futils.change_alm_lmax(T_alm, mlmax)
            T_p_alm = futils.get_cmb_alm(1999,0)[0]
            T_p_alm = futils.change_alm_lmax(T_p_alm, mlmax)
            S = get_TT_secondary(qfunc, fg_alm, T_alm, T_p_alm)
            
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

    test_signal(nsim=args.nsim, use_mpi=args.mpi, from_pkl=args.from_pkl)
    #test_N0(use_mpi=args.mpi, nsim=args.nsim, from_pkl=args.from_pkl)
    #test_secondary(use_mpi=args.mpi, nsim=args.nsim, from_pkl=args.from_pkl)
    #plot_secondary_terms()

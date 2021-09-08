from __future__ import print_function
import os,sys
from os.path import join as opj
import numpy as np
import healpy as hp
from pixell import curvedsky
from orphics import maps,io,cosmology,stats,pixcov
from pixell import enmap,curvedsky,utils,enplot,lensing
from falafel import utils as futils
from falafel.utils import change_alm_lmax
import astropy.io.fits as afits

NERSC_ORIG_DATA_DIR="/global/project/projectdirs/act/data/sehgal_et_al_sims/cosmo_sim_maps/July2009/inputs/microwaveSky"
NERSC_DATA_DIR="/global/project/projectdirs/act/data/maccrann/sehgal/"

CONVERSION_FACTORS = {"CIB" : 
                      {"0093" : 4.6831e3, "0100" : 4.1877e3, "0145" : 2.6320e3, "0545" : 1.7508e4},
                      "Y" : 
                      {"0093" : -4.2840e6, "0100": -4.1103e6, "0145" : -2.8355e6},
}

def get_cib_conversion_factor(freq, T_cmb=2.7255):
    #get factor for converting delta flux density in MJy
    #to delta T in CMB units
    freq = float(freq)
    x = freq / 56.8
    return (1.05e3 * (np.exp(x)-1)**2 *
            np.exp(-x) * (freq / 100)**-4)

def flux_density_to_temp(freq):
    #get factor for converting delta flux density in MJy
    #to delta T in CMB units
    freq = float(freq)
    x = freq / 56.8
    return (1.05e3 * (np.exp(x)-1)**2 *
            np.exp(-x) * (freq / 100)**-4)


class CMBSky(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.halodata = None

    def get_nemo_source_mask(self, nemo_file,
                             mask_radius,
                             snr_min=None):
        nemo_data = afits.open(cib_cat_file)[1].data

        #cut on snr
        if snr_min is not None:
            use = nemo_data['SNR']>=snr_min
            nemo_data = nemo_data[use]
        print("masking %d nemo sources"%len(nemo_data))
        
        ra_deg, dec_deg = nemo_data['RADeg'], nemo_data['decDeg']
        dec,ra = np.radians(dec_deg),np.radians(ra_deg)

        r = halo_mask_fgs_config['mask_radius']*utils.arcmin

        srcs = np.array([dec, ra])
        mask = (enmap.distance_from_healpix(
            self.nside, srcs, rmax=r) >= r)

        print("masking %d/%d pixels (f_sky = %.2f) in fgs"%(
            (~mask).sum(),len(mask),
            float((~mask).sum())/len(mask)))
        return halo_mask

    def get_flux_mask(self, flux_density_map, flux_cut):
        """
        Provide flux_density_map in MJy/sr, and 
        make a mask based on a maximimum flux of flux_cut
        in mJy
        """
        flux_map_mJy = flux_density_map * 1.e9 * hp.nside2pixarea(self.nside)
        mask = flux_map_mJy < flux_cut
        n = len(mask)
        print("flux cut masks %d/%d pixels (=f_sky %.2e)"%(
            n-mask.sum(), n, float(n-mask.sum())/n
        ))
        return mask

    def get_fg_mask(self, halo_mask_fgs=False, m_min=1.e+15, zmax=4.,
                    halo_mask_radius=10.,
                    cib_flux_cut=None, flux_cut_freq=None,
                    radiops_flux_cut=None, nemo_mask_fgs=False,
                    nemo_catalog=None, nemo_snr_min=None,
                    nemo_mask_radius=10.
                 ):
        print("halo_mask_fgs:",halo_mask_fgs)
        print("cib_flux_cut:",cib_flux_cut)
        print("radiops_flux_cut:",radiops_flux_cut)
        fg_mask = np.ones(hp.nside2npix(self.nside),
                          dtype=bool)

        if halo_mask_fgs:
            halo_mask = self.get_halo_mask(
                m_min, zmax,
                halo_mask_radius,
                )
            fg_mask *= halo_mask

        if cib_flux_cut is not None:
            cib_flux_mask = self.get_cib_flux_mask(
                flux_cut_freq, cib_flux_cut)
            fg_mask *= cib_flux_mask
            
        if radiops_flux_cut is not None:
            ps_temp = self.get_radio_ps_temp(flux_cut_freq)
            ps_flux_density = ps_temp / flux_density_to_temp(float(freq))
            print("getting radio point-source mask")
            ps_flux_mask = self.get_flux_mask(
                ps_flux_density, radiops_flux_cut)
            fg_mask *= ps_flux_mask

        if nemo_mask_fgs:
            nemo_mask = self.get_nemo_source_mask(
                nemo_catalog, nemo_mask_radius,
                snr_min=nemo_snr_min)
            fg_mask *= nemo_mask
            
        n = len(fg_mask)
        print("in total masks %d/%d pixels (=f_sky %.2e)"%(
            n-fg_mask.sum(), n, float(n-fg_mask.sum())/n
        ))
        return fg_mask                 

    def get_sky(self, freq, cmb=True, cib=False, tsz=False,
                ksz=False, radiops=False, cmb_unlensed_alms=None,
                cmb_alms=None, lmax=4000, fg_model_alms=None,
                fg_mask=None, mean_fill_fgs=True):

        outputs = {}
        fg_map = np.zeros(hp.nside2npix(self.nside))
        if fg_mask is None:
            fg_mask = np.ones_like(fg_map, dtype=bool)
        if cib:
            cib_temp = self.get_cib_temp(freq)
            fg_map += cib_temp
            has_fgs = True
        if radiops:
            ps_temp = self.get_radio_ps_temp(freq)
            fg_map += ps_temp
            has_fgs = True

        if tsz:
            tsz_temp = self.get_tsz_temp(freq)
            fg_map += tsz_temp
            has_fgs = True

        if ksz:
            ksz_temp = self.get_ksz_temp()
            fg_map += ksz_temp
            has_fgs = True

        #if we've done some fg masking,
        #make the masked map and alms
        if not np.all(fg_mask == True):
            if mean_fill_fgs:
                fg_map_masked = fg_map.copy()
                mask_inds = np.where(~fg_mask)[0]
                fg_map_masked[mask_inds] = (fg_map[fg_mask]).mean()
            else:
                fg_map_masked = fg_map * fg_mask
            fg_masked_alms = hp.map2alm(fg_map_masked, lmax=lmax)
            outputs["fg_masked_alms"] = fg_masked_alms
            outputs["fg_mask"] = fg_mask

        if cmb:
            if cmb_alms is None:
                if (cmb_unlensed_alms is None):
                    cmb_alms = self.get_cmb_lensed_orig_alms(
                        lmax=lmax)
                else:
                    #Lens the cmb alms using the kappa field
                    print("cmb_unlensed_alms provided, lensing these")
                    cmb_lmax = hp.Alm.getlmax(cmb_unlensed_alms[0].size)
                    print("cmb lmax=%d"%cmb_lmax)
                    kappa_alms = self.get_kappa_alms(lmax=cmb_lmax)
                    def kappa_to_phi(kappa_alm):
                        return curvedsky.almxfl(
                            alm=kappa_alm,
                            lfilter=lambda x: 1./(x*(x+1)/2))
                    phi_alms = kappa_to_phi(kappa_alms)
                    res = 0.5*utils.arcmin
                    proj='car'
                    shape, wcs = enmap.fullsky_geometry(res=res, proj=proj)
                    print("doing lensing on %s map with res %f"%(
                        proj, res/utils.arcmin))
                    t_lensed = lensing.lens_map_curved(
                        shape, wcs, phi_alms,
                        cmb_unlensed_alms[0])[0]
                    cmb_alms = curvedsky.map2alm(t_lensed, lmax=lmax)

        total_alms = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
        if cmb:
            print(type(cmb_alms))
            print(cmb_alms)
            total_alms += cmb_alms.astype(np.complex128)
        if has_fgs:
            fg_alms = hp.map2alm(fg_map, lmax=lmax)
            total_alms += fg_alms
            if fg_model_alms is not None:
                fg_model_alms = change_alm_lmax(
                    fg_model_alms, lmax)
                fg_modelsub_alms = fg_alms - fg_model_alms
                outputs["fg_modelsub_alms"] = fg_modelsub_alms
        else:
            fg_alms = np.zeros_like(total_alms)
        outputs["fg_alms"] = fg_alms
        outputs["total_alms"] = total_alms
        outputs["cmb_alms"] = cmb_alms
        outputs["cmb_unlensed_alms"] = cmb_unlensed_alms

        return outputs

    
class SehgalSky(CMBSky):

    
    def __init__(self, data_dir,
                 use_orig_maps=False, rescale_cib=None,
                 rescale_tsz=None):
        super(SehgalSky, self).__init__(data_dir)
        self.use_orig_maps = use_orig_maps
        if self.use_orig_maps:
            self.nside=8192
        else:
            self.nside=4096
        self.rescale_cib = rescale_cib
        if self.rescale_cib is None:
            if self.use_orig_maps:
                self.rescale_cib = True
            else:
                self.rescale_cib = False
        self.rescale_tsz = rescale_tsz
        if self.rescale_tsz is None:
            if self.use_orig_maps:
                self.rescale_tsz = True
            else:
                self.rescale_tsz = False
                
        self.data_dir = data_dir
        if self.data_dir is None:
            if self.use_orig_maps:
                self.data_dir = NERSC_ORIG_DATA_DIR
            else:
                self.data_dir = NERSC_DATA_DIR
        
    def get_cib_temp(self, freq, use_orig_maps=None,
                     rescale_cib=None):
        if use_orig_maps is None:
            use_orig_maps = self.use_orig_maps
        if rescale_cib is None:
            rescale_cib = self.rescale_cib

        if use_orig_maps:
            filename = opj(self.data_dir,
                           "%03d_ir_pts_healpix.fits"%int(freq)
                           )
            m_Jysr = hp.read_map(filename) #this is in Jy/sr
            #convert to MJy/sr
            m_MJysr = m_Jysr / 1.e6
            #convert to deltaT
            m = m_MJysr * get_cib_conversion_factor(freq)
        else:
            filename = opj(self.data_dir,
            "%03d_ir_pts_healpix_nopell_Nside4096_DeltaT_uK_lininterp_CIBrescale0p75.fits"%int(freq)
            )
            m = hp.read_map(filename)
        if rescale_cib:
            m *= 0.75
        return m

    def get_radio_ps_temp(self, freq):
        #This is in cmb units
        filename = opj(
            self.data_dir,
            "%03d_rad_pts_healpix_nopell_Nside4096_DeltaT_uK_fluxcut148_7mJy_lininterp.fits"%int(freq)
        )
        return hp.read_map(filename)
    
    def get_tsz_temp(self, freq, use_orig_maps=None,
                    rescale_tsz=None):
        """frequency in GHz"""
        if use_orig_maps is None:
            use_orig_maps = self.use_orig_maps
        if rescale_tsz is None:
            rescale_cib = self.rescale_tsz
            
        if use_orig_maps:
            filename = opj(self.data_dir,
                           "%03d_tsz_healpix.fits"%int(freq)
                           )
            dT = hp.read_map(filename)
            if rescale_tsz:
                dT *= 0.75
        else:
            y = hp.read_map(
                opj(self.data_dir,
                    "tSZ_skymap_healpix_nopell_Nside4096_y_tSZrescale0p75.fits"
            ))
            #convert y to deltaT
            x = float(freq)/56.8
            T_cmb = 2.726e6
            dT = y * T_cmb * (x * (np.exp(x) + 1)/(np.exp(x) - 1) - 4)
        return dT

    def get_ksz_temp(self):
        return hp.read_map(
            opj(self.data_dir,
                "148_ksz_healpix_nopell_Nside4096_DeltaT_uK.fits"
                ))

    def get_kappa_alms(self, lmax):
        """
        kappa_map = hp.read_map(
            opj(self.data_dir,
                "healpix_4096_KappaeffLSStoCMBfullsky.fits"
                )
            )
        """
        kappa_alms = hp.fitsfunc.read_alm(
            opj(self.data_dir, "healpix_4096_KappaeffLSStoCMBfullsky_almlmax6000.fits")
            )
        kappa_alms = futils.change_alm_lmax(kappa_alms, lmax)
        return kappa_alms

    def load_halo_catalog(self, mmin=0.):
        """
        load the halo catalog
        """
        halodata = np.loadtxt(
            opj(self.data_dir, "halo_nbody.ascii")
        ).T
        print(halodata.shape)
        m_200 = halodata[12]
        use = m_200>mmin
        halodata = halodata[:,use]
        num_halo = len(halodata[0])

        dtype = [("redshift",float),
                 ("ra",float),
                 ("dec",float),
                 ("M_vir",float),
                 ("M_200",float)]
        redshift = halodata[0]
        ra_deg = halodata[1]
        dec_deg = halodata[2]
        M_vir = halodata[10]
        M_200 = halodata[12]

        #Get halo ra/dec
        dec,ra = np.radians(dec_deg),np.radians(ra_deg)
        #For the sehgal sims, the halo catalog only covers
        #one octant. So we need to add the rotated ra/dec
        #also
        theta,phi = np.pi/2 - dec, ra
        all_theta,all_phi = [],[]
        for n in range(4):
            theta_N = theta
            phi_N = phi + n*np.pi/2
            theta_S = np.pi-theta
            phi_S = np.pi/2 - phi + n*np.pi/2
            all_theta += list(theta_N)
            all_theta += list(theta_S)
            all_phi += list(phi_N)
            all_phi += list(phi_S)
        all_theta,all_phi = np.array(all_theta),np.array(all_phi)
        all_dec,all_ra = np.pi/2 - all_theta, all_phi

        halodata_fullsky = np.zeros(num_halo*8, dtype=dtype)
        halodata_fullsky['ra'] = np.degrees(all_ra)
        halodata_fullsky['dec'] = np.degrees(all_dec)
        for i in range(8):
            halodata_fullsky['redshift'][i*num_halo:(i+1)*num_halo] = redshift
            halodata_fullsky['M_vir'][i*num_halo:(i+1)*num_halo] = M_vir
            halodata_fullsky['M_200'][i*num_halo:(i+1)*num_halo] = M_200
        self.halodata = halodata_fullsky

    def get_halo_mask(self, m_min, zmax,
                      mask_radius, num_halo=None,
                      mass_col='M_200'):

        self.load_halo_catalog()
        use = self.halodata[mass_col] > m_min
        halodata = self.halodata[use]
        num_halo = len(halodata)

        print("masking %d halos"%num_halo)
        #Get halo ra/dec
        ra_deg,dec_deg = halodata['ra'], halodata['dec']
        dec,ra = np.radians(dec_deg),np.radians(ra_deg)

        r = mask_radius*utils.arcmin

        srcs = np.array([dec, ra])
        halo_mask = (enmap.distance_from_healpix(
            4096, srcs, rmax=r) >= r)
        print(halo_mask.sum())
        print(len(halo_mask))
        print("halo masking %d/%d pixels (f_sky = %.2f) in fgs"%(
            (~halo_mask).sum(),len(halo_mask),
            float((~halo_mask).sum())/len(halo_mask)))
        return halo_mask

    def get_cmb_lensed_orig_alms(self, lmax=None):
        """
        Get the original (i.e. packaged with the sim)
        lensed cmb alms
        """
        if self.use_orig_maps:
            cmb_alms = hp.read_map(opj(self.data_dir, "030_lensedcmb_healpix.fits"))
        else:
            cmb_alms = hp.read_map(
                opj(self.data_dir,"Sehgalsimparams_healpix_4096_KappaeffLSStoCMBfullsky_phi_SimLens_Tsynfastnopell_fast_lmax8000_nside4096_interp2.5_method1_1_lensed_map.fits")
            )
        if lmax is not None:
            cmb_alms = hp.map2alm(lensed_cmb, lmax=lmax)
        return cmb_alms
    
    def get_sky_old(self, cmb=True, freq="093", cib=False,
		tsz=False, ksz=False, cib_flux_cut=None,
                radiops=False, radiops_flux_cut=None,
                halo_mask_fgs_config=None, cmb_unlensed_alms=None,
	        cmb_alms=None, lmax=4000, fg_model_alms=None):
        
        outputs = {}
        fg_map = np.zeros(hp.nside2npix(self.nside))
        fg_mask = np.ones(len(fg_map), dtype=bool)
        has_fgs=False
        if cib:
            cib_temp = self.get_cib_temp(freq)
            fg_map += cib_temp
            has_fgs = True
            if cib_flux_cut is not None:
                #Convert cib deltaT to flux density in MJy sr^-1
                cib_flux_density = cib_temp / get_cib_conversion_factor(freq)
                print("getting cib flux mask")
                cib_source_mask = self.flux_mask(cib_flux_density,
                  cib_flux_cut)
                fg_mask *= cib_source_mask

        if radiops:
            ps_temp = self.get_radio_ps_temp(freq)
            ps_flux_density = ps_temp / flux_density_to_temp(float(freq))
            fg_map += ps_temp
            has_fgs = True
            if radiops_flux_cut is None:
                if cib_flux_cut is not None:
                    radiops_flux_cut = cib_flux_cut

            if radiops_flux_cut is not None:
                print("getting radio point-source mask")
                ps_flux_mask = self.get_flux_mask(ps_flux_density, radiops_flux_cut)
                fg_mask *= ps_flux_mask
                
        if tsz:
            tsz_temp = self.get_tsz_temp(freq)
            fg_map += tsz_temp
            has_fgs = True

        if ksz:
            ksz_temp = self.get_ksz_temp()
            fg_map += ksz_temp
            has_fgs = True

        if halo_mask_fgs_config is not None:
            halo_mask  = self.get_halo_mask(
                halo_mask_fgs_config['mmin'],
                halo_mask_fgs_config['mask_radius'],
                halo_mask_fgs_config['zmax'])
                                            
            fg_mask *= halo_mask

        #if we've done some fg masking,
        #make the masked map and alms
        fg_masked=False
        if not np.all(fg_mask == True):
            fg_masked=True
            fg_map_masked = fg_map.copy()
            mask_inds = np.where(~fg_mask)[0]
            fg_map_masked[mask_inds] = (fg_map[fg_mask]).mean()
            fg_masked_alms = hp.map2alm(fg_map_masked, lmax=lmax)
            outputs["fg_masked_alms"] = fg_masked_alms
            
        if cmb:
            if cmb_alms is None:
                if (cmb_unlensed_alms is None):
                    if self.use_orig_maps:
                        lensed_cmb = hp.read_map(opj(self.data_dir, "030_lensedcmb_healpix.fits"))
                    else:
                        lensed_cmb = hp.read_map(
                            opj(self.data_dir,"Sehgalsimparams_healpix_4096_KappaeffLSStoCMBfullsky_phi_SimLens_Tsynfastnopell_fast_lmax8000_nside4096_interp2.5_method1_1_lensed_map.fits")
                        )
                    cmb_alms = hp.map2alm(lensed_cmb, lmax=lmax)
                else:
                    #Lens the cmb alms using the kappa field
                    print("cmb_unlensed_alms provided, lensing these")
                    cmb_lmax = hp.Alm.getlmax(cmb_unlensed_alms[0].size)
                    print("cmb lmax=%d"%cmb_lmax)
                    kappa_alms = self.get_kappa_alms(lmax=cmb_lmax)
                    def kappa_to_phi(kappa_alm):
                        return curvedsky.almxfl(
                            alm=kappa_alm,
                            lfilter=lambda x: 1./(x*(x+1)/2))
                    phi_alms = kappa_to_phi(kappa_alms)
                    res = 0.5*utils.arcmin
                    proj='car'
                    shape,wcs = enmap.fullsky_geometry(res=res, proj=proj)
                    print("doing lensing on %s map with res %f"%(
                        proj, res/utils.arcmin))
                    t_lensed = lensing.lens_map_curved(
                        shape, wcs, phi_alms,
                        cmb_unlensed_alms[0])[0]
                    cmb_alms = curvedsky.map2alm(t_lensed, lmax=lmax)

        total_alms = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
        if cmb:
            total_alms += cmb_alms
            
        if has_fgs:
            fg_alms = hp.map2alm(fg_map, lmax=lmax)
            total_alms += fg_alms
            if fg_model_alms is not None:
                fg_model_alms = change_alm_lmax(
                    fg_model_alms, lmax)
                fg_modelsub_alms = fg_alms - fg_model_alms
                outputs["fg_modelsub_alms"] = fg_modelsub_alms
        else:
            fg_alms = np.zeros_like(total_alms)
        outputs["fg_alms"] = fg_alms
        outputs["total_alms"] = total_alms
        outputs["cmb_alms"] = cmb_alms
        outputs['fg_map'] = fg_map
        if fg_masked:
            outputs['fg_masked_map'] = fg_map_masked

        return outputs

class WebSky(CMBSky):
    """class for websky maps and dark matter halo catalogs

    Parameters
    ----------

    directory_path : str
        directory of websky halo catalogs and maps. Default to nersc
    halo_catalog : str
        directory of websky halo catalog. Default to nersc convention
    websky_cosmo : dictionary
        useful cosmological parameters of the websky run
    verbose : bool
        prints information on halo catalogs and maps
    """
    nside = 4096
    def __init__(self,
                 data_dir,
                 halo_catalog_name = 'halos.pksc',
                 kappa_map_name = 'kap.fits',
                 comptony_map_name = 'tsz.fits',
                 ksz_map_name = 'ksz.fits',
                 websky_cosmo = {'Omega_M': 0.31, 'Omega_B': 0.049, 'Omega_L': 0.69, 
                                 'h': 0.68, 'sigma_8': 0.81, 'n_s':0.965},
                 verbose = True
    ):
        super().__init__(data_dir)

        self.websky_cosmo   = websky_cosmo
        self.verbose        = verbose
        self.kappa_map_name = kappa_map_name
        self.comptony_map_name = comptony_map_name
        self.ksz_map_name = ksz_map_name
        self.halo_catalog_name = halo_catalog_name
        self.halodata = None #this will be set
        #on calling self.load_halo_catalog

    def import_astropy(self):
        """load in astropy. Only used if comoving distance to redshift calculations required
        """

        from astropy.cosmology import FlatLambdaCDM, z_at_value
        import astropy.units as u

        self.z_at_value = z_at_value
        self.u = u
        self.astropy_cosmo = FlatLambdaCDM(H0=self.websky_cosmo['h']*100, Om0=self.websky_cosmo['Omega_M'])

    def load_halo_catalog(self, mmin=0., mmax=np.inf, zmin=0., zmax=np.inf, rmin=0., rmax=np.inf, practical=True):
        """load in peak patch dark matter halo catalog

        Requires astropy if using distance to redshift calculations, or redshift cuts

        Returns
        -------

        if practical==True: only generally useful information (including redshifts)
        halodata : np.array((Nhalo, 8))
            numpy array of halo information, 10 floats per halo
            x [Mpc], y [Mpc], z [Mpc], vx [km/s], vy [km/s], vz [km/s], 
            M [M_sun (M_200,m)], redshift (v_pec not included)

        if practical==False: returns everything in halo catalog
        halodata : np.array((Nhalo, 10+))
            numpy array of halo information, 10 floats per halo
            x [Mpc], y [Mpc], z [Mpc], vx [km/s], vy [km/s], vz [km/s], 
            M [M_sun (M_200,m)], x_lag [Mpc], y_lag [Mpc], z_lag [Mpc], ...
        """

        halo_catalog_file = open(opj(self.data_dir, self.halo_catalog_name),"rb")
        
        # load catalog header
        Nhalo            = np.fromfile(halo_catalog_file, dtype=np.int32, count=1)[0]
        RTHMAXin         = np.fromfile(halo_catalog_file, dtype=np.float32, count=1)
        redshiftbox      = np.fromfile(halo_catalog_file, dtype=np.float32, count=1)
        if self.verbose: print("\nNumber of Halos in full catalog %d \n " % Nhalo)

        nfloats_perhalo = 10
        npkdata         = nfloats_perhalo*Nhalo

        # load catalog data
        print('reading from file')
        halodata        = np.fromfile(halo_catalog_file, dtype=np.float32, count=npkdata)
        print('done reading from file')
        halodata        = np.reshape(halodata,(Nhalo,nfloats_perhalo))

        # change from R_th to halo mass (M_200,M)
        rho_mean = 2.775e11 * self.websky_cosmo['Omega_M'] * self.websky_cosmo['h']**2
        halodata[:,6] = 4.0/3*np.pi * halodata[:,6]**3 * rho_mean        
        
        # cut mass range
        print(mmin)
        if mmin > 0 or mmax < np.inf:
            dm = (halodata[:,6] > mmin) & (halodata[:,6] < mmax) 
            halodata = halodata[dm]

        # cut redshift range
        if zmin > 0 or zmax < np.inf:
            self.import_astropy()
            rofzmin = self.astropy_cosmo.comoving_distance(zmin).value
            rofzmax = self.astropy_cosmo.comoving_distance(zmax).value

            rpp =  np.sqrt( np.sum(halodata[:,:3]**2, axis=1))

            dm = (rpp > rofzmin) & (rpp < rofzmax) 
            halodata = halodata[dm]

        # cut distance range
        if rmin > 0 or rmax < np.inf:
            rpp =  np.sqrt(np.sum(halodata[:,:3]**2, axis=1))

            dm = (rpp > rmin) & (rpp < rmax) 
            halodata = halodata[dm]

        Nhalo = len(halodata)

        # get halo redshifts and crop all non practical information
        self.import_astropy()
        dtype = [('x', np.float64), ('y', np.float64), ('z', np.float64),
                 ('vx', np.float64), ('vy', np.float64), ('vz', np.float64),
                 ('M', np.float64), ('redshift', np.float64),
                 ('ra', np.float64), ('dec', np.float64)]
        # set up comoving distance to redshift interpolation table
        rpp =  np.sqrt( np.sum(halodata[:,:3]**2, axis=1))

        zminh = self.z_at_value(self.astropy_cosmo.comoving_distance, rpp.min()*self.u.Mpc)
        zmaxh = self.z_at_value(self.astropy_cosmo.comoving_distance, rpp.max()*self.u.Mpc)
        zgrid = np.linspace(zminh, zmaxh, 10000)
        dgrid = self.astropy_cosmo.comoving_distance(zgrid).value
        redshift = np.interp(rpp, dgrid, zgrid)
        #Get halo ra/dec
        vec = np.zeros((Nhalo, 3))
        vec[:,0] = halodata[:,0]
        vec[:,1] = halodata[:,1]
        vec[:,2] = halodata[:,2]
        ra_deg,dec_deg = hp.vec2ang(vec, lonlat=True)

        halodata_out = np.zeros(halodata.shape[0],
                                    dtype = dtype)
        
        # Add all columns to the more user-friendly output array
        for i,name in enumerate(halodata_out.dtype.names):
            if name == 'redshift':
                halodata_out[name] = redshift
            elif name == 'ra':
                halodata_out[name] = ra_deg
            elif name == 'dec':
                halodata_out[name] = dec_deg
            else:
                halodata_out[name] = halodata[:,i]
                
        Nhalo = len(halodata)
        Nfloat_perhalo = len(halodata.dtype)

        # write out halo catalog information
        if self.verbose: 
            print("Halo catalog after cuts: np.array((Nhalo=%d, floats_per_halo=%d)), containing:\n" % (Nhalo, Nfloat_perhalo))
            if practical:
                print("0:x [Mpc], 1:y [Mpc], 2:z [Mpc], 3:vx [km/s], 4:vy [km/s], 5:vz [km/s],\n"+ 
                      "6:M [M_sun (M_200,m)], 7:redshift(chi_halo) \n")
            else:
                print("0:x [Mpc], 1:y [Mpc], 2:z [Mpc], 3:vx [km/s], 4:vy [km/s], 5:vz [km/s],\n"+ 
                      "6:M [M_sun (M_200,m)], 7:x_lag [Mpc], 8:y_lag [Mpc], 9:z_lag [Mpc]\n")
        self.halodata = halodata_out
        return halodata_out

    def get_halo_mask(self, m_min, zmax,
                      mask_radius, num_halo=None):

        self.load_halo_catalog(
                zmin=0., zmax=zmax,
                mmin=m_min)
        if num_halo is not None:
            halodata = self.halodata[:num_halo]
        else:
            halodata = self.halodata
        num_halo=len(halodata)

        print("masking %d halos"%num_halo)
        #Get halo ra/dec
        vec = np.zeros((num_halo, 3))
        vec[:,0] = halodata['x']
        vec[:,1] = halodata['y']
        vec[:,2] = halodata['z']
        ra_deg,dec_deg = hp.vec2ang(vec, lonlat=True)
        dec,ra = np.radians(halodata['dec']),np.radians(halodata['ra'])

        r = mask_radius*utils.arcmin
        srcs = np.array([dec,ra])
        halo_mask = (enmap.distance_from_healpix(
            4096, srcs, rmax=r) >= r)
        print("halo masking %d/%d pixels (f_sky = %.2f) in fgs"%(
            (~halo_mask).sum(),len(halo_mask),
            float((~halo_mask).sum())/len(halo_mask)))
        return halo_mask

    def flux_mask(self, flux_density_map,
                  flux_cut):
        """flux density in MJy/sr^-1, 
        flux_cut in mJY"""
        flux_map = flux_density_map * hp.nside2pixarea(4096) * 1.e9
        source_mask = (flux_map < flux_cut)
        n=len(source_mask)
        print("source mask masks %d/%d pixels (=f_sky %.2e)"%(
                    n-source_mask.sum(),
                    n, float(n-source_mask.sum())/n
                    ))
        return source_mask
    
    def fmt_freq(self, freq):
        """
        Return frequency in same 
        string format as filenames
        """
        return str(freq).zfill(4)

    def cib_map_file_name(self, freq='545'):
        """get file name of cib map, given a frequency

        Parameters
        ----------

        freq : str or int
            frequency of desired map in GHz

        Returns
        -------

        cib_file_name : str
            name of cib file at given frequency
        """

        cib_file_name = "cib_nu%s.fits"%self.fmt_freq(freq)
        return opj(self.data_dir, cib_file_name)

    def kappa_map_file_name(self):
        """get file name of kappa map

        Returns
        -------

        kappa_file_name : str
            name of kappa map file 
        """

        return opj(self.data_dir, self.kappa_map_name)

    def comptony_map_file_name(self):
        """get file name of compton-y map

        Returns
        -------

        comptony_file_name : str
            name of compton-y map file 
        """

        return opj(self.data_dir, self.comptony_map_name)

    def ksz_map_file_name(self):
        """get file name of ksz map

        Returns
        -------

        ksz_file_name : str
            name of ksz map file 
        """
        
        return opj(self.data_dir, self.ksz_map_name)

    def get_tsz_temp(self, freq):
        fn = self.comptony_map_file_name()
        y_map = hp.read_map(fn)
        return y_map*CONVERSION_FACTORS['Y'][self.fmt_freq(freq)]

    def get_radio_ps_map(self, freq):
        """
        Returns flux density in MJy/sr
        """
        ps_dir = "/global/cfs/cdirs/sobs/www/users/Radio_WebSky/ACT/matched_catalogs"
        filename = opj(ps_dir, "catalog_%.1f.h5"%float(freq))
        f = h5py.File(filename, 'r')
        flux, theta, phi = f['flux'][:], f['theta'][:], f['phi'][:]
        #Flux is in Jansky according to Zack.
        #Let's convert to MJy/sr since that's what the
        #CIB maps are stored as 
        flux_map = np.zeros(hp.nside2npix(4096)) #pixel flux in Jansky
        pix_inds = hp.ang2pix(4096, theta, phi)
        np.add.at(flux_map, pix_inds, flux)
        #Convert to flux density in MJy/sr
        flux_density_map = flux_map / 1.e6 / hp.nside2pixarea(4096)
        return flux_density_map
        
    def get_ksz_temp(self):
        fn = self.ksz_map_file_name()
        return hp.read_map(fn)
    
    def get_kappa_alms(self, lmax=4000):
        f = self.kappa_map_file_name()
        kappa_map = hp.read_map(f)
        kappa_alms = hp.map2alm(kappa_map, lmax=lmax)
        return kappa_alms

    
    def get_cib_flux_mask(self, freq, flux_cut):
        """
        Get a mask which excludes regions with
        flux greater than flux_cut, in mJY.
        """
        cib_map = hp.read_map(
            self.cib_map_file_name(freq=freq)
            )
        return self.get_flux_mask(cib_map, flux_cut)
        """
        #The cib map is flux density in MJy/sr,
        #so the following gets us flux in mJy
        cib_flux = cib_map * hp.nside2pixarea(4096) * 1.e9
        mask = cib_flux < flux_cut
        n=len(mask)
        print("cib flux cut masks %d/%d pixels (=f_sky %.2e)"%(
            n-mask.sum(), n, float(n-mask.sum())/n
        ))
        return mask
        """

    def get_cib_temp(self, freq):
        cib_map = hp.read_map(
            self.cib_map_file_name(freq=freq))
        return cib_map*CONVERSION_FACTORS['CIB'][self.fmt_freq(freq)]

    def get_radio_ps_temp(self, freq):
        ps_map = self.get_radio_ps_map(freq)
        return ps_map*flux_density_to_temp(float(freq))

    def get_cmb_lensed_orig_alms(self, lmax=None):
        """
        Get the original (i.e. packaged with the sim)
        lensed cmb alms
        """
        cmb_alms = hp.fitsfunc.read_alm(
            opj(self.data_dir, "lensed_alm.fits")
        )
        if lmax is not None:
            cmb_alms = change_alm_lmax(
                cmb_alms, lmax)
        return cmb_alms
            
    def get_sky_old(self, cmb=True, freq="0093", cib=False,
                tsz=False, ksz=False, radiops=False,
                cib_flux_cut=None, radiops_flux_cut=None,
                halo_mask_fgs_config=None, cmb_unlensed_alms=None,
                cmb_alms=None, lmax=4000, mean_fill_fgs=True,
                udgrade_fill_factor=None, fg_model_alms=None,
                get_map=False):
        """
        Get sky a_lms 
        """
        outputs = {}
        fg_map = np.zeros(hp.nside2npix(4096))
        fg_mask = np.ones_like(fg_map, dtype=bool)
        has_fgs=False
        if cib:
            cib_temp = self.get_cib_temp(freq)
            fg_map += cib_temp
            has_fgs = True
            if cib_flux_cut is not None:
                print("getting cib flux mask")
                cib_source_mask = self.get_flux_mask(cib_map, cib_flux_cut)
                """
                #the CIB map is flux density in MJy/sr^-1
                #convert to flux in mJY
                cib_flux_map = cib_map * hp.nside2pixarea(4096) * 1.e9
                print("cib_flux_map.max():",cib_flux_map.max())
                cib_source_mask = (cib_flux_map < cib_flux_cut)
                n=len(cib_source_mask)
                print("cib source mask masks %d/%d pixels (=f_sky %.2e)"%(
                    n-cib_source_mask.sum(),
                    n, float(n-cib_source_mask.sum())/n
                    ))
                """
                fg_mask *= cib_source_mask

        if radiops:
            ps_map = self.get_radio_ps_map(freq) #flux density in MJy/sr
            ps_temp = ps_map*CONVERSION_FACTORS['CIB'][self.fmt_freq(freq)]
            fg_map += ps_temp
            has_fgs = True
            if radiops_flux_cut is None:
                if cib_flux_cut is not None:
                    radiops_flux_cut = cib_flux_cut

            if radiops_flux_cut is not None:
                print("getting radio point-source mask")
                ps_flux_mask = self.get_flux_mask(ps_map, radiops_flux_cut)
                fg_mask *= ps_flux_mask

        if tsz:
            tsz_temp = self.get_tsz_temp(freq)
            fg_map += tsz_temp
            has_fgs = True

        if ksz:
            ksz_temp = self.get_ksz_temp()
            fg_map += ksz_temp
            has_fgs = True

        if halo_mask_fgs_config is not None:
            halo_mask = self.get_halo_mask(
                halo_mask_fgs_config['m_min'],
                halo_mask_fgs_config['zmax'],
                halo_mask_fgs_config['mask_radius']
                )
            fg_mask *= halo_mask


            
        #if we've done some fg masking,
        #make the masked map and alms
        if not np.all(fg_mask == True):
            if (mean_fill_fgs or (udgrade_fill_factor is not None)):
                fg_map_mean_masked = fg_map.copy()
                mask_inds = np.where(~fg_mask)[0]
                fg_map_mean_masked[mask_inds] = (fg_map[fg_mask]).mean()
                if udgrade_fill_factor is not None:
                    fg_map_masked = fg_map.copy()
                    fg_map_masked[mask_inds] = (
                        hp.ud_grade(
                            hp.ud_grade(fg_map_mean_masked,
                                        4096//udgrade_fill_factor),
                            4096))[mask_inds]
                else:
                    fg_map_masked = fg_map_mean_masked
            else:
                fg_map_masked = fg_map * fg_mask
            fg_masked_alms = hp.map2alm(fg_map_masked, lmax=lmax)
            outputs["fg_masked_alms"] = fg_masked_alms
            outputs["fg_mask"] = fg_mask

            
        if cmb:
            if cmb_alms is None:
                if (cmb_unlensed_alms is None):
                    cmb_unlensed_alms = hp.fitsfunc.read_alm(
                        opj(self.data_dir, "unlensed_alm.fits")
                        )
                    cmb_alms = hp.fitsfunc.read_alm(
                        opj(self.data_dir, "lensed_alm.fits")
                    )
                    if lmax is not None:
                        cmb_alms = change_alm_lmax(
                            cmb_alms, lmax)

                else:
                    #Lens the cmb alms using the kappa field
                    print("cmb_unlensed_alms provided, lensing these")
                    cmb_lmax = hp.Alm.getlmax(cmb_unlensed_alms[0].size)
                    print("cmb lmax=%d"%cmb_lmax)
                    kappa_alms = self.get_kappa_alms(lmax=cmb_lmax)
                    def kappa_to_phi(kappa_alm):
                        return curvedsky.almxfl(
                            alm=kappa_alm,
                            lfilter=lambda x: 1./(x*(x+1)/2))
                    phi_alms = kappa_to_phi(kappa_alms)
                    res = 0.5*utils.arcmin
                    proj='car'
                    shape,wcs = enmap.fullsky_geometry(res=res, proj=proj)
                    print("doing lensing on %s map with res %f"%(
                        proj, res/utils.arcmin))
                    t_lensed = lensing.lens_map_curved(
                        shape, wcs, phi_alms,
                        cmb_unlensed_alms[0])[0]
                    cmb_alms = curvedsky.map2alm(t_lensed, lmax=lmax)

        total_alms = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
        if cmb:
            total_alms += cmb_alms
        if has_fgs:
            fg_alms = hp.map2alm(fg_map, lmax=lmax)
            total_alms += fg_alms
            if fg_model_alms is not None:
                fg_model_alms = change_alm_lmax(
                    fg_model_alms, lmax)
                fg_modelsub_alms = fg_alms - fg_model_alms
                outputs["fg_modelsub_alms"] = fg_modelsub_alms
        else:
            fg_alms = np.zeros_like(total_alms)
        outputs["fg_alms"] = fg_alms
        outputs["total_alms"] = total_alms
        outputs["cmb_alms"] = cmb_alms
        outputs["cmb_unlensed_alms"] = cmb_unlensed_alms

        return outputs

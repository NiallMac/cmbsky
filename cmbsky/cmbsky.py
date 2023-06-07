from __future__ import print_function
import os,sys
from os.path import join as opj
import numpy as np
import healpy as hp
from pixell import curvedsky
from orphics import maps,io,cosmology,stats,pixcov
from pixell import enmap, curvedsky, utils, enplot, lensing, reproject
from falafel import utils as futils
from falafel.utils import change_alm_lmax
import astropy.io.fits as afits
import h5py
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.units as units

CONVERSION_FACTORS = {"CIB" : 
                      {"0093" : 4.6831e3, "0100" : 4.1877e3, "0145" : 2.6320e3, "0217": 2.06767e3, "0225" : 2.0716e3, "0353" : 3.3710e3, "0545" : 1.7508e4},
                      "Y" : 
                      {"0093" : -4.2840e6, "0100": -4.1103e6, "0145" : -2.8355e6, "0217" : -2.1188e4, "0353" : 6.1071e6, "0545" : 1.5257e7},
}

h=6.62607004e-34
c=2.99792458e8
kb = 1.38064852e-23
T_cmb=2.7255

def y_to_deltaToverT(freq_Ghz):
    x = float(freq_Ghz)/56.8
    return T_cmb * (x * (np.exp(x)+1)/(np.exp(x)-1) - 4) * 1e6
    
def response_fnc_tsz(qid,lmax=None):
    return y_to_deltaToverT(float(qid))/y_to_deltaToverT(145.)


def cib_sed(freq, lmax=None, beta=1.2, T_cib=24.):
    freq = float(freq)*1.e9
    x = h*freq / kb / T_cmb
    #print(x)
    dBdT = 2*h * freq**3 * x * np.exp(x) / c**2 / T_cmb / (np.exp(x)-1)**2
    #print(dBdT)
    x_cib = h*freq / kb / T_cib
    #print(x_cib)
    return freq**(3+beta) / (np.exp(x_cib) - 1) / dBdT

def response_fnc_cib(qid, lmax=None):
    return cib_sed(qid)/cib_sed(145.)

class ClBinner(object):
    def __init__(self, lmin=10, lmax=300, nbin=20):
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

def get_cib_conversion_factor(freq, T_cmb=2.7255):
    #get factor for converting delta flux density in MJy
    #to delta T in CMB units
    freq = float(freq)
    x = freq / 56.8
    return (1.05e3 * (np.exp(x)-1)**2 *
            np.exp(-x) * (freq / 100)**-4)

def flux_density_to_temp(freq):
    #get factor for converting delta flux density in MJy/sr
    #to delta T in CMB units
    freq = float(freq)
    x = freq / 56.8
    return (1.05e3 * (np.exp(x)-1)**2 *
            np.exp(-x) * (freq / 100)**-4)

def get_matched_filter_mask(
        input_map, fl, threshold_flux,
        ):
    """
    Parameters
    ----------
    input_map: np.array
      healpix input map
    fl: np.array
      filter
    """
    lmax = len(filter_of_ell)-1
    nside = hp.npix2nside(len(input_map))
    #filter map
    filtererd_alm = hp.almxfl(hp.map2alm(input_map, lmax=lmax),
                              fl)
    filtered_map = hp.alm2map(filtered_alm, nside)
    
class CMBSky(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.halodata = None

    def get_cmb1_alms(self, cmb_seed, sim_name, lmax):
        f = opj(self.data_dir,
                "T0alm_cmb%d_%s_lmax%d.fits"%(cmb_seed, sim_name, lmax)
                )
        return hp.fitsfunc.read_alm(f)
        
    def get_nemo_source_mask(
            self, nemo_file, mask_radius,
            snr_min=None, rot=None,
            lowz_template_labels=[
                "Arnaud_M1e14_z0p2","Arnaud_M2e14_z0p2","Arnaud_M4e14_z0p2","Arnaud_M8e14_z0p2"],
            lowz_mask_radius=None, true_zmax=None,
            hack_dr6_large_mask_numbers=False):
        """
        rot: list of 3 floats
            If we're producing a rotated sky,
            the nemo catalog provided here is 
            presumnably in that rotated coordinated
            system. Here though, we're still working 
            in the original coordinate system, so we'll
            apply the inverse rotation to the nemo catalog
            to mask the map.
        """
        nemo_data = afits.open(nemo_file)[1].data

        #cut on snr
        if snr_min is not None:
            use = nemo_data['SNR']>=snr_min
            nemo_data = nemo_data[use]
            print("masking %d snr>%.2f nemo sources from catalog %s"%(len(nemo_data), snr_min, nemo_file))
        else:
            print("masking %d nemo sources from catalog %s"%(len(nemo_data), nemo_file))
        
        ra_deg, dec_deg = nemo_data['RADeg'], nemo_data['decDeg']
        dec,ra = np.radians(dec_deg),np.radians(ra_deg)

        #If we are generating a rotated sky, we assume
        #that the nemo catalog provided is in that rotated frame.
        #Here though, we are working in the original frame, so
        #we need apply the inverse rotation to the nemo catalog
        if rot is not None:
            print("applying inverse rotation to nemo catalog coordinates")
            r = hp.rotator.Rotator(rot=rot, deg=False)
            theta,phi = np.pi/2 - dec, ra
            theta_rot, phi_rot = r(theta, phi, inv=True)
            dec, ra = np.pi/2 - theta_rot, phi_rot

        r = mask_radius*utils.arcmin

        srcs = np.array([dec, ra])
        mask = (enmap.distance_from_healpix(
            self.nside, srcs, rmax=r) >= r)
        print("nemo masking %d/%d pixels (f_sky = %.2f) in fgs"%(
            (~mask).sum(),len(mask),
            float((~mask).sum())/len(mask)))

        if lowz_mask_radius is not None:
            #get low z sources
            is_lowz = np.zeros(len(nemo_data), dtype=bool)
            if hack_dr6_large_mask_numbers:
                print("hacking dr6 large mask numbers")
                #mask 40 brightest z<0.1 clusters
                #mask 435-40 random z=0.2 template candidates
                #match to halo catalog
                #self.load_halo_catalog(zmax=true_zmax)
                halodata = self.halodata
                halodata_use = self.halodata[self.halodata['redshift'] < true_zmax]
                cat_halo = SkyCoord(ra=halodata_use["ra"]*u.degree, 
                                        dec=halodata_use["dec"]*u.degree)
                cat_nemo = SkyCoord(ra=nemo_data['RADeg']*u.degree, 
                                     dec=nemo_data['decDeg']*u.degree)
                halo_ind, d2d, _ = cat_nemo.match_to_catalog_sky(cat_halo)
                sepmax=2.5/60 * u.degree # 2.5 arcmin
                has_match = d2d <= sepmax
                print("%d/%d nemo clusters have match to z<0.1 halos"%(
                    has_match.sum(), len(has_match)))
                is_lowz = has_match
                nemo_snr = nemo_data['SNR']
                
                #just keep 40 brightest of these
                snr_thresh = -np.sort(-nemo_snr[is_lowz])[40]
                is_lowz *= nemo_snr>snr_thresh
                print("selected %d brightest lowz clusters"%is_lowz.sum())
                print("snrs",nemo_snr[is_lowz])

                #also keep 395 z=0.2 template candidates
                is_lowz_template = np.zeros(len(nemo_data), dtype=bool)
                for template in lowz_template_labels:
                    is_lowz_template[np.where(nemo_data["template"] == template)[0]] = True
                #remove confirmed 40
                is_lowz_template[is_lowz] = False
                #randomly choose 395 remaining ones
                is_lowz_template_inds, = np.where(is_lowz_template)
                is_lowz_template_inds_choose = np.random.choice(is_lowz_template_inds, 395,
                                                                replace=False)
                is_lowz[is_lowz_template_inds_choose] = True
                print("%d lowz after random templates step"%is_lowz.sum())
                
            elif true_zmax is not None:
                #match to halo catalog
                halodata = self.halodata
                halodata_use = self.halodata[self.halodata['redshift'] < true_zmax]
                cat_halo = SkyCoord(ra=halodata_use["ra"]*u.degree, 
                                        dec=halodata_use["dec"]*u.degree)
                cat_nemo = SkyCoord(ra=nemo_data['RADeg']*u.degree, 
                                     dec=nemo_data['decDeg']*u.degree)
                halo_ind, d2d, _ = cat_nemo.match_to_catalog_sky(cat_halo)
                sepmax=2.5/60 * u.degree # 2.5 arcmin
                has_match = d2d <= sepmax
                print("%d/%d nemo clusters have match to z<0.1 halos"%(
                    has_match.sum(), len(has_match)))
                is_lowz = has_match

            else:
                for template in lowz_template_labels:
                    is_lowz[np.where(nemo_data["template"] == template)[0]] = True
            print("masking %d low z clusters with radius %.2f arcmin"%(is_lowz.sum(), lowz_mask_radius))
            lowz_srcs = np.array([dec[is_lowz], ra[is_lowz]])
            r_lowz = lowz_mask_radius * utils.arcmin
            lowz_mask = (enmap.distance_from_healpix(
                self.nside, lowz_srcs, rmax=r_lowz) >= r_lowz)
            print("low-z nemo masking %d/%d pixels (f_sky = %.2f) in fgs"%(
                (~lowz_mask).sum(),len(lowz_mask),
                float((~lowz_mask).sum())/len(lowz_mask)))
            mask *= lowz_mask
        
        return mask

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

    def get_radio_ps_flux_mask(self, freq, flux_cut):
        """
        Get flux mask for radio point-sources
        """
        print("getting radio ps mask")
        ps_temp = self.get_radio_ps_temp(freq)
        ps_flux_density = ps_temp / flux_density_to_temp(float(freq))
        ps_flux_mask = self.get_flux_mask(
            ps_flux_density, flux_cut)
        return ps_flux_mask
    
    def get_fg_mask(self, halo_mask_fgs=False, mmin=1.e+15, zmax=4.,
                    halo_mask_radius=10.,
                    cib_flux_cut=None, flux_cut_freq=None,
                    radiops_flux_cut=None, nemo_mask_fgs=False,
                    nemo_catalog=None, nemo_snr_min=None,
                    nemo_mask_radius=None, nemo_lowz_mask_radius=None,
                    rot=None, true_zmax=None,
                    hack_dr6_large_mask_numbers=False
                 ):
        """
        Somewhat confusing uberfunction for getting 
        a mask with various different constraints
        """
        print("halo_mask_fgs:",halo_mask_fgs)
        print("cib_flux_cut:",cib_flux_cut)
        print("radiops_flux_cut:",radiops_flux_cut)
        fg_mask = np.ones(hp.nside2npix(self.nside),
                          dtype=bool)

        if halo_mask_fgs:
            halo_mask = self.get_halo_mask(
                mmin, halo_mask_radius,
                zmax=zmax
                )
            fg_mask *= halo_mask

        if cib_flux_cut is not None:
            if isinstance(cib_flux_cut, list):
                assert isinstance(flux_cut_freq, list)
                total_cib_flux_mask = np.ones_like(fg_mask)
                for cut, cut_freq in zip(cib_flux_cut, flux_cut_freq):
                    if cut is None:
                        print("no cib flux cut applied at %s GHz"%cut_freq)
                        continue
                    print("masking cib flux>%f at %s GHz"%(
                        cut, cut_freq))
                    cib_flux_mask = self.get_cib_flux_mask(
                        cut_freq, cut)
                    total_cib_flux_mask *= cib_flux_mask
                n = len(total_cib_flux_mask)
                print("total cib mask masks %d/%d pixels (=f_sky %.2e)"%(
                    n-total_cib_flux_mask.sum(),
                    n, float(n-total_cib_flux_mask.sum())/n
                ))
                fg_mask *= total_cib_flux_mask
            else:
                print("masking cib flux>%f at %s GHz"%(
                        cib_flux_cut, flux_cut_freq))
                cib_flux_mask = self.get_cib_flux_mask(
                    flux_cut_freq, cib_flux_cut)
                fg_mask *= cib_flux_mask
            
        if radiops_flux_cut is not None:
            if isinstance(radiops_flux_cut, list):
                assert isinstance(flux_cut_freq, list)
                total_radiops_flux_mask = np.ones_like(fg_mask)
                for cut, cut_freq in zip(radiops_flux_cut,
                                         flux_cut_freq):
                    if cut is None:
                        print("no radio point-source flux cut applied at %s GHz"%cut_freq)
                        continue
                    print("masking radio ps flux>%f at %s GHz"%(
                        cut, cut_freq))
                    ps_flux_mask = self.get_radio_ps_flux_mask(
                        cut_freq, cut)
                    total_radiops_flux_mask *= ps_flux_mask
                n = len(total_radiops_flux_mask)
                print("total radio ps mask masks %d/%d pixels (=f_sky %.2e)"%(
                    n-total_radiops_flux_mask.sum(),
                    n, float(n-total_radiops_flux_mask.sum())/n
                ))
                fg_mask *= total_radiops_flux_mask
            else:
                print("masking radio ps flux>%f at %s GHz"%(
                        radiops_flux_cut, flux_cut_freq))
                ps_flux_mask = self.get_radio_ps_flux_mask(
                    flux_cut_freq, radiops_flux_cut)
                fg_mask *= ps_flux_mask
                                         
        if nemo_mask_fgs:
            if not isinstance(nemo_catalog, list):
                nemo_catalog = [nemo_catalog]
                assert not isinstance(nemo_snr_min, list)
                assert not isinstance(nemo_mask_radius, list)
                nemo_snr_min = [nemo_snr_min]
                nemo_mask_radius = [nemo_mask_radius]
            else:
                assert isinstance(nemo_snr_min, list)
                assert isinstance(nemo_mask_radius, list)
            for cat, snr, rad in zip(nemo_catalog, nemo_snr_min, nemo_mask_radius):
                nemo_mask = self.get_nemo_source_mask(
                    cat, rad,
                    snr_min=snr,
                    rot=rot, lowz_mask_radius=nemo_lowz_mask_radius,
                    true_zmax=true_zmax,
                    hack_dr6_large_mask_numbers=hack_dr6_large_mask_numbers)
                fg_mask *= nemo_mask
                    
            
        n = len(fg_mask)
        print("in total masks %d/%d pixels (=f_sky %.2e)"%(
            n-fg_mask.sum(), n, float(n-fg_mask.sum())/n
        ))
        return fg_mask                 

    def get_rksz_temp(self):
        return hp.read_map(
            opj(self.data_dir,
                "4e3_2048_50_50_ksz.fits"
                ))
    
    def get_sky(self, freq, cmb=True, cib=False, tsz=False,
                ksz=False, rksz=False, radiops=False,
                cmb_unlensed_alms=None,
                cmb_alms=None, lmax=4000,
                fg_mask=None, mean_fill_fgs=True,
                fg_model_map=None, bl=None,
                survey_mask_hpix=None, mask_beamed_map=False,
                source_flux_cut=None, flux_cut_freq=None):

        if flux_cut_freq is None:
            flux_cut_freq=freq
        outputs = {}
        fg_map = np.zeros(hp.nside2npix(self.nside))
        has_fgs = False
        if fg_mask is None:
            fg_mask = np.ones_like(fg_map, dtype=bool)
            
        if cib:
            cib_temp = self.get_cib_temp(freq)
            if source_flux_cut is not None:
                print("cutting cib flux > %.2f mJY at %s GHz"%(source_flux_cut, flux_cut_freq))
                cib_flux_mask = self.get_cib_flux_mask(flux_cut_freq, source_flux_cut)
                cib_temp[~cib_flux_mask] = (cib_temp[cib_flux_mask]).mean()
            fg_map += cib_temp
            has_fgs = True
        if radiops:
            ps_temp = self.get_radio_ps_temp(freq)
            if source_flux_cut is not None:
                print("cutting radio source flux > %.2f mJY at %s GHz"%(source_flux_cut, flux_cut_freq))
                ps_flux_mask = self.get_radio_ps_flux_mask(flux_cut_freq, source_flux_cut)
                ps_temp *= ps_flux_mask
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

        if rksz:
            rksz_temp = self.get_rksz_temp()
            fg_map += rksz_temp
            has_fgs = True

        if mask_beamed_map:
            if ((fg_model_map is None) and (survey_mask_hpix is None) and (np.all(fg_mask==1))):
                fg_alms = hp.map2alm(fg_map, lmax=lmax)
            else:
                print("masking beamed map")
                print("applying beam")
                mlmax=3*self.nside-1
                fg_alm = hp.map2alm(fg_map, lmax=mlmax)
                fg_alm_beamed = curvedsky.almxfl(fg_alm, bl[:mlmax+1])
                fg_map_beamed = hp.alm2map(fg_alm_beamed, self.nside)

                if fg_model_map is not None:
                    print("subtracting fg_model_map")
                    fg_model_map_hpix = reproject.map2healpix(fg_model_map,
                                                              nside=self.nside)
                    fg_map_beamed -= fg_model_map_hpix

                if fg_mask is not None:
                    if survey_mask_hpix is not None:
                        total_mask=fg_mask*(survey_mask_hpix>0.1)
                    else:
                        total_mask=fg_mask
                    print("applying fg_mask")
                    mean_val = (fg_map_beamed[total_mask]).mean()
                    print("mean filling map with value:", mean_val)
                    print("median =", np.median(fg_map_beamed[total_mask]))
                    fg_map_beamed[~fg_mask] = mean_val

                if survey_mask_hpix is not None:
                    print("applying survey_mask_hpix")
                    fg_map_beamed *= survey_mask_hpix
                    #also calculate w factors
                    outputs["w1"] = (survey_mask_hpix).mean()
                    outputs["w2"] = (survey_mask_hpix**2).mean()
                    outputs["w4"] = (survey_mask_hpix**4).mean()
                    print("applied survey mask with fsky:",outputs["w1"])
                    
                outputs["fg_map_beamed"] = fg_map_beamed
                fg_alms = curvedsky.almxfl(
                    hp.map2alm(fg_map_beamed, lmax=lmax),
                    1./bl[:lmax+1]
                    )
        else:
            if survey_mask_hpix is not None:
                fg_map *= survey_mask_hpix
                #also calculate w factors
                outputs["w1"] = (survey_mask_hpix).mean()
                outputs["w2"] = (survey_mask_hpix**2).mean()
                outputs["w4"] = (survey_mask_hpix**4).mean()
                print("applied survey mask with fsky:",outputs["w1"])

            #if we've done some fg masking,
            #make the masked map and alms
            if not np.all(fg_mask == True):
                if mean_fill_fgs:
                    fg_map_masked = fg_map.copy()
                    mask_inds = np.where(~fg_mask)[0]
                    fg_map_fill_value = (fg_map[fg_mask]).mean()
                    fg_map_masked[mask_inds] = fg_map_fill_value
                    outputs["fg_map_fill_value"] = fg_map_fill_value
                else:
                    fg_map_masked = fg_map * fg_mask.astype(int)
                    outputs["fg_map_fill_value"] = 0.
                    print("fg_map_masked.dtype, survey_mask_hpix.dtype:")
                    print(fg_map_masked.dtype, survey_mask_hpix.dtype)
                    fg_map_masked *= survey_mask_hpix
                fg_alms = hp.map2alm(fg_map_masked, lmax=lmax)
                #outputs["fg_masked_alms"] = fg_masked_alms
                outputs["fg_mask"] = fg_mask
            else:
                fg_alms = hp.map2alm(fg_map, lmax=lmax)

        outputs["fg_alms"] = fg_alms

        if cmb:
            if cmb_alms is None:
                if (cmb_unlensed_alms is None):
                    cmb_alms = self.get_cmb_lensed_orig_alms(
                        lmax=lmax, survey_mask_hpix=survey_mask_hpix)
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
                    phi_alms = kappa_to_phi(kappa_alms).astype(np.complex64)
                    res = 0.5*utils.arcmin
                    proj='car'
                    shape, wcs = enmap.fullsky_geometry(res=res, proj=proj)
                    print("doing lensing on %s map with res %f"%(
                        proj, res/utils.arcmin))
                    print(shape, wcs, phi_alms.dtype, cmb_unlensed_alms[0].dtype)
                    t_lensed = lensing.lens_map_curved(
                        shape, wcs, phi_alms,
                        cmb_unlensed_alms[0])[0]
                    if survey_mask_hpix is not None:
                        cmb_alms = curvedsky.map2alm(t_lensed, lmax=cmb_lmax)
                        cmb_map = hp.alm2map(cmb_alms, self.nside)
                        cmb_alms = hp.map2alm(cmb_map*survey_mask_hpix, lmax=lmax)
                    else:
                        cmb_alms = curvedsky.map2alm(t_lensed, lmax=lmax)


        total_alms = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
        if cmb:
            print(type(cmb_alms))
            print(cmb_alms)
            total_alms += cmb_alms.astype(np.complex128)
        if has_fgs:
            #fg_alms = hp.map2alm(fg_map, lmax=lmax)
            total_alms += fg_alms
            """
            if fg_model_alms is not None:
                fg_model_alms = change_alm_lmax(
                    fg_model_alms, lmax)
                fg_modelsub_alms = fg_alms - fg_model_alms
                outputs["fg_modelsub_alms"] = fg_modelsub_alms
            """
        else:
            fg_alms = np.zeros_like(total_alms)
        #outputs["fg_alms"] = fg_alms
        outputs["total_alms"] = total_alms
        outputs["cmb_alms"] = cmb_alms
        outputs["cmb_unlensed_alms"] = cmb_unlensed_alms

        return outputs
        
    
class SehgalSky(CMBSky):
    """
    Class for generating sims for the 
    rescaled Sehgal sims i.e. those used
    in https://arxiv.org/abs/1808.07445
    """
    
    def __init__(self, data_dir,
                 rescale_cib=None,
                 rescale_tsz=None):
        super(SehgalSky, self).__init__(data_dir)
        self.nside=4096
        #rescale options are None by default
        #since this class reads the already
        #rescaled maps
        self.rescale_cib = rescale_cib
        self.rescale_tsz = rescale_tsz
        self.data_dir = data_dir

    def get_cmb1_alms(self, cmb_seed, lmax):
        return super(SehgalSky, self).get_cmb1_alms(cmb_seed, "sehgal", lmax)
        
    def get_cib_temp(self, freq):
        filename = opj(self.data_dir,
        "%03d_ir_pts_healpix_nopell_Nside4096_DeltaT_uK_lininterp_CIBrescale0p75.fits"%int(freq)
        )
        m = hp.read_map(filename)
        if self.rescale_cib is not None:
            m *= self.rescale_cib
        return m
    
    def get_cib_flux_mask(self, freq, flux_cut):
        """
        Get a mask which excludes regions with
        flux greater than flux_cut, in mJY.
        """
        cib_temp = self.get_cib_temp(freq)
        cib_flux_density_MJy = cib_temp / get_cib_conversion_factor(freq)
        return self.get_flux_mask(
            cib_flux_density_MJy, flux_cut)

    def get_radio_ps_temp(self, freq,
                          ):
        #This is in cmb units
        filename = opj(
                self.data_dir,
                "%03d_rad_pts_healpix_nopell_Nside4096_DeltaT_uK_fluxcut148_7mJy_lininterp.fits"%int(freq)
            )
        m = hp.read_map(filename)
        return m
    
    def get_tsz_temp(self, freq):
        """frequency in GHz"""
        y = hp.read_map(
            opj(self.data_dir,
                "tSZ_skymap_healpix_nopell_Nside4096_y_tSZrescale0p75.fits"
        ))
        #convert y to deltaT
        x = float(freq)/56.8
        T_cmb = 2.726e6
        dT = y * T_cmb * (x * (np.exp(x) + 1)/(np.exp(x) - 1) - 4)
        if self.rescale_tsz is not None:
            dT *= self.rescale_tsz
        return dT

    def get_ksz_temp(self):
        return hp.read_map(
            opj(self.data_dir,
                "148_ksz_healpix_nopell_Nside4096_DeltaT_uK.fits"
                ))

    def get_rksz_temp(self):
        return hp.read_map(
            opj(self.data_dir,
                "4e3_2048_50_50_ksz.fits"
                ))

    def get_kappa_alms(self, lmax):
        kappa_alms = hp.fitsfunc.read_alm(
            opj(self.data_dir, "healpix_4096_KappaeffLSStoCMBfullsky_almlmax6000.fits")
            )
        kappa_alms = futils.change_alm_lmax(kappa_alms, lmax)
        return kappa_alms

    def load_halo_catalog(self, mmin=0., cols=None, zmax=None):
        """
        load the halo catalog
        """
        halodata = np.loadtxt(
            opj(self.data_dir, "halo_nbody.ascii")
        ).T
        print("read %d halos"%halodata.shape[1])
        m_200 = halodata[12]
        use = m_200>mmin
        redshift = halodata[0]
        if zmax is not None:
            use *= (redshift <= zmax)
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

    def get_halo_mask(self, mmin, mask_radius,
                      zmax=None, num_halo=None,
                      mass_col='M_200'):

        self.load_halo_catalog()
        use = self.halodata[mass_col] > mmin
        if zmax is not None:
            use *= self.halodata['redshift']<zmax
            
        halodata = self.halodata[use]
        num_halo = len(halodata)

        print("masking %d halos"%num_halo)
        #Get halo ra/dec
        ra_deg,dec_deg = halodata['ra'], halodata['dec']
        dec,ra = np.radians(dec_deg),np.radians(ra_deg)

        r = mask_radius*utils.arcmin

        srcs = np.array([dec, ra])
        halo_mask = (enmap.distance_from_healpix(
            self.nside, srcs, rmax=r) >= r)
        print("halo masking %d/%d pixels (f_sky = %.2f) in fgs"%(
            (~halo_mask).sum(),len(halo_mask),
            float((~halo_mask).sum())/len(halo_mask)))
        return halo_mask

    def get_cmb_lensed_orig_alms(self, lmax=None, survey_mask_hpix=None):
        """
        Get the original (i.e. packaged with the sim)
        lensed cmb alms. Optionally apply a mask

        """
        lensed_cmb = hp.read_map(
                opj("/global/project/projectdirs/act/data/maccrann/sehgal",
                    "Sehgalsimparams_healpix_4096_KappaeffLSStoCMBfullsky_phi_SimLens_Tsynfastnopell_fast_lmax8000_nside4096_interp2.5_method1_1_lensed_map.fits")
            )
        if survey_mask_hpix is not None:
            lensed_cmb *= hp.ud_grade(survey_mask_hpix, 4096)
        if lmax is not None:
            cmb_alms = hp.map2alm(lensed_cmb, lmax=lmax)
        return cmb_alms
    
class Sehgal10Sky(SehgalSky):
    def __init__(self, data_dir,
                 rescale_cib=True,
                 rescale_tsz=True):
        super(Sehgal10Sky, self).__init__(data_dir)
        self.nside=4096
        self.rescale_cib = rescale_cib
        self.rescale_tsz = rescale_tsz
        self.data_dir = data_dir

    def get_cmb1_alms(self, cmb_seed, lmax):
        return super(Sehgal10Sky, self).get_cmb1_alms(cmb_seed, "sehgal", lmax)
        
    def get_cib_temp(self, freq,
                     rescale_cib=None):

        if int(freq)==545:
            print("hacking by 545GHz cib for Sehgal by rescaling 350GHz")
            t_350 = self.get_cib_temp(350., rescale_cib=rescale_cib)
            t_545 = t_350 * cib_sed(545.) / cib_sed(350.)
            return t_545

        if rescale_cib is None:
            rescale_cib = self.rescale_cib
        filename = opj(self.data_dir,
                       "%03d_ir_pts_healpix.fits"%int(freq)
                       )
        print("reading cib from %s"%filename)
        m_Jysr = hp.read_map(filename) #this is in Jy/sr
        #convert to MJy/sr
        m_MJysr = m_Jysr / 1.e6
        #convert to deltaT
        m = m_MJysr * get_cib_conversion_factor(freq)
        if rescale_cib:
            m *= 0.75
        return hp.ud_grade(m, self.nside)
    
    def get_cib_flux_mask(self, freq, flux_cut,
                          rescale_cib=None):
        """
        Get a mask which excludes regions with
        flux greater than flux_cut, in mJY.
        """
        if rescale_cib is None:
            rescale_cib = self.rescale_cib
        cib_temp = self.get_cib_temp(freq)
        cib_flux_density_MJy = cib_temp / get_cib_conversion_factor(freq)
        return self.get_flux_mask(
            cib_flux_density_MJy, flux_cut)

    def get_radio_ps_temp(self, freq,
                          ):
        if int(freq)==545:
            print("hacking radio-ps at 545 by returning zero map!")
            return np.zeros(hp.nside2npix(self.nside))
        filename = opj(
            self.data_dir,
            "%03d_rad_pts_healpix.fits"%int(freq)
            )
        print("reading radio ps from %s"%filename)
        m_Jysr = hp.read_map(filename)
        m_MJysr = m_Jysr / 1.e6
        m = m_MJysr * get_cib_conversion_factor(freq)
        return hp.ud_grade(m, self.nside)


    
    def get_tsz_temp(self, freq, use_orig_maps=None,
                    rescale_tsz=None):
        """frequency in GHz"""

        if int(freq)==545:
            print("rescaling 350 map for 545 tsz")
            tsz_350 = self.get_tsz_temp(
                350., use_orig_maps=use_orig_maps,
                rescale_tsz=rescale_tsz)
            tsz_545 = tsz_350 * response_fnc_tsz(545.)/response_fnc_tsz(350.)
            return tsz_545
        
        if rescale_tsz is None:
            rescale_tsz = self.rescale_tsz

        filename = opj(self.data_dir,
                       "%03d_tsz_healpix.fits"%int(freq)
                       )
        print("reading tsz from %s"%filename)
        tsz_map = hp.read_map(filename)
        #This is in Jy/sr
        if rescale_tsz:
            tsz_map *= 0.75
        dT = (tsz_map / 1.e6) * flux_density_to_temp(freq)
        return hp.ud_grade(dT, self.nside)
        """
        print("using rescaled tSZ")
        y = hp.ud_grade(hp.read_map(
            opj("/global/project/projectdirs/act/data/maccrann/sehgal",
                "tSZ_skymap_healpix_nopell_Nside4096_y_tSZrescale0p75.fits"
            )), 8192)
        
        #convert y to deltaT
        x = float(freq)/56.8
        T_cmb = 2.726e6
        dT = y * T_cmb * (x * (np.exp(x) + 1)/(np.exp(x) - 1) - 4)
        return dT
        """

    def get_ksz_temp(self):
        ksz_flux_density_Jy = hp.read_map(
            opj(self.data_dir,
                "148_ksz_healpix.fits"
                ))
        return hp.ud_grade(ksz_flux_density_Jy * flux_density_to_temp(148.) / 1.e6, self.nside)

    def get_kappa_alms(self, lmax):
        kappa_alms = hp.fitsfunc.read_alm(
            opj(self.data_dir, "healpix_4096_KappaeffLSStoCMBfullsky_almlmax6000.fits")
            )
        kappa_alms = futils.change_alm_lmax(kappa_alms, lmax)
        return kappa_alms

    def get_cmb_lensed_orig_alms(self, lmax=None, survey_mask_hpix=None):
        """
        Get the original (i.e. packaged with the sim)
        lensed cmb alms
        """
        lensed_cmb = hp.read_map(
                opj("/global/project/projectdirs/act/data/maccrann/sehgal",
                    "Sehgalsimparams_healpix_4096_KappaeffLSStoCMBfullsky_phi_SimLens_Tsynfastnopell_fast_lmax8000_nside4096_interp2.5_method1_1_lensed_map.fits")
            )
        if survey_mask_hpix is not None:
            print("lensed_cmb nside:", hp.pixelfunc.npix2nside(len(lensed_cmb)))
            print("survey_mask_hpix nside:", hp.pixelfunc.npix2nside(len(survey_mask_hpix)))
            print("survey_mask_hpix dgraded nside:", hp.pixelfunc.npix2nside(len(hp.ud_grade(survey_mask_hpix, 4096))))
            lensed_cmb *= hp.ud_grade(survey_mask_hpix, 4096)
        if lmax is not None:
            cmb_alms = hp.map2alm(lensed_cmb, lmax=lmax)
        return cmb_alms


class Seghal10Rot1Sky(Sehgal10Sky):
    def __init__(self, data_dir,
                 rescale_cib=True,
                 rescale_tsz=True):
        super().__init__(data_dir,
                         rescale_cib=rescale_cib,
                         rescale_tsz=rescale_tsz)

    def get_cmb_lensed_orig_alms(self, lmax=None, survey_mask_hpix=None):
        """
        Get the original (i.e. packaged with the sim)
        lensed cmb alms
        """
        lensed_cmb = hp.read_map(
                opj("/global/project/projectdirs/act/data/maccrann/sehgal10_rot1",
                    "Sehgalsimparams_healpix_4096_KappaeffLSStoCMBfullsky_phi_SimLens_Tsynfastnopell_fast_lmax8000_nside4096_interp2.5_method1_1_lensed_map.fits")
            )
        if survey_mask_hpix is not None:
            print("lensed_cmb nside:", hp.pixelfunc.npix2nside(len(lensed_cmb)))
            print("survey_mask_hpix nside:", hp.pixelfunc.npix2nside(len(survey_mask_hpix)))
            print("survey_mask_hpix dgraded nside:", hp.pixelfunc.npix2nside(len(hp.ud_grade(survey_mask_hpix, 4096))))
            lensed_cmb *= hp.ud_grade(survey_mask_hpix, 4096)
        if lmax is not None:
            cmb_alms = hp.map2alm(lensed_cmb, lmax=lmax)
        return cmb_alms

        
class WebSky(CMBSky):
    """class for websky maps and dark matter halo catalogs

    Parameters
    ----------

    directory_path : str
        directory of websky halo catalogs and maps. Default to nersc
    halo_catalog : str
        directory of websky halo catalog. Default to nersc convention
    websky_cosmo : dictionary
        cosmological parameters of the websky run
    """
    nside = 4096
    def __init__(self,
                 data_dir,
                 halo_catalog_name = 'halos.pksc',
                 kappa_map_name = 'kap.fits',
                 comptony_map_name = 'tsz_8192.fits',
                 ksz_map_name = 'ksz.fits',
                 websky_cosmo = {'Omega_M': 0.31, 'Omega_B': 0.049, 'Omega_L': 0.69, 
                                 'h': 0.68, 'sigma_8': 0.81, 'n_s':0.965},
    ):
        super().__init__(data_dir)

        self.websky_cosmo   = websky_cosmo
        self.kappa_map_name = kappa_map_name
        self.comptony_map_name = comptony_map_name
        self.ksz_map_name = ksz_map_name
        self.halo_catalog_name = halo_catalog_name
        self.halodata = None #this will be set
        #on calling self.load_halo_catalog

    def get_cmb1_alms(self, cmb_seed, lmax):
        return super(WebSky, self).get_cmb1_alms(cmb_seed, "websky", lmax)
        
    def load_halo_catalog(self, mmin=0., mmax=np.inf, zmin=0., zmax=np.inf,
                          cols=None):
        """load in peak patch dark matter halo catalog

        Returns
        -------

        halodata : np.array((Nhalo, 8))
            numpy array of halo information, 10 floats per halo
            x [Mpc], y [Mpc], z [Mpc], vx [km/s], vy [km/s], vz [km/s], 
            M [M_sun (M_200,m)], redshift (v_pec not included)
        """

        astropy_cosmo = FlatLambdaCDM(H0=self.websky_cosmo['h']*100,
                                      Om0=self.websky_cosmo['Omega_M'])
        
        halo_catalog_file = open(opj(self.data_dir, self.halo_catalog_name),"rb")
        
        # load catalog header
        Nhalo            = np.fromfile(halo_catalog_file, dtype=np.int32, count=1)[0]
        RTHMAXin         = np.fromfile(halo_catalog_file, dtype=np.float32, count=1)
        redshiftbox      = np.fromfile(halo_catalog_file, dtype=np.float32, count=1)
        #print("\nNumber of Halos in full catalog %d \n " % Nhalo)

        nfloats_perhalo = 10
        npkdata         = nfloats_perhalo*Nhalo

        # load catalog data
        print('reading halo catalog from file')
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
        """
        if zmin > 0 or zmax < np.inf:
            rofzmin = astropy_cosmo.comoving_distance(zmin).value
            rofzmax = astropy_cosmo.comoving_distance(zmax).value

            rpp =  np.sqrt( np.sum(halodata[:,:3]**2, axis=1))

            dm = (rpp > rofzmin) & (rpp < rofzmax) 
            halodata = halodata[dm]
        """
        # set up comoving distance to redshift interpolation table
        rpp =  np.sqrt( np.sum(halodata[:,:3]**2, axis=1))

        zminh = z_at_value(astropy_cosmo.comoving_distance,
                           rpp.min() * units.Mpc)
        zmaxh = z_at_value(astropy_cosmo.comoving_distance,
                           rpp.max() * units.Mpc)
        zgrid = np.linspace(zminh, zmaxh, 10000)
        dgrid = astropy_cosmo.comoving_distance(zgrid).value
        redshift = np.interp(rpp, dgrid, zgrid)
        redshift_use = (redshift>=zmin)*(redshift<=zmax)
        halodata = halodata[redshift_use]
        redshift = redshift[redshift_use]

        Nhalo = len(halodata)
        # get halo redshifts and put everything into a more friendly 
        # recarray
        dtype = [('x', np.float64), ('y', np.float64), ('z', np.float64),
                 ('vx', np.float64), ('vy', np.float64), ('vz', np.float64),
                 ('M', np.float64), ('redshift', np.float64),
                 ('ra', np.float64), ('dec', np.float64)]
        
        #Get halo ra/dec
        vec = np.zeros((Nhalo, 3))
        vec[:,0] = halodata[:,0]
        vec[:,1] = halodata[:,1]
        vec[:,2] = halodata[:,2]
        ra_deg,dec_deg = hp.vec2ang(vec, lonlat=True)

        if cols is not None:
            orig_cols = [d[0] for d in dtype]
            orig_types = [d[1] for d in dtype]
            dtype = [(c, orig_types[orig_cols.index(c)]) for c in cols]
            col_inds = [orig_cols.index(c) for c in cols]
            print("cutting to cols:",cols)
            print("rows in halodata:",col_inds)
        else:
            col_inds = range(len(dtype))
        
        halodata_out = np.zeros(halodata.shape[0],
                                    dtype = dtype)
        
        # Add all columns to output array
        for col_ind, name in zip(col_inds, halodata_out.dtype.names):
            if name == 'redshift':
                halodata_out[name] = redshift
            elif name == 'ra':
                halodata_out[name] = ra_deg
            elif name == 'dec':
                halodata_out[name] = dec_deg
            else:
                halodata_out[name] = halodata[:,col_ind]
                
        Nhalo = len(halodata)
        Nfloat_perhalo = len(halodata.dtype)

        # write out halo catalog information
        print("read halo catalog with %d halos"%Nhalo)
        print("contains the following columns:")
        print(halodata_out.dtype.names)
        self.halodata = halodata_out
        return halodata_out

    def get_halo_mask(
            self, mmin, mask_radius,
            zmax=None):
        """
        Load halo data and generate
        a mask to remove pixels
        containing halos with mass>mmin
        and z<z_max
        """
        self.load_halo_catalog(
            zmin=0., zmax=zmax,
            mmin=mmin, cols=["ra","dec","M"])
        halodata = self.halodata
        num_halo=len(halodata)

        print("masking %d halos"%num_halo)
        #Get halo ra/dec
        """
        vec = np.zeros((num_halo, 3))
        vec[:,0] = halodata['x']
        vec[:,1] = halodata['y']
        vec[:,2] = halodata['z']
        ra_deg,dec_deg = hp.vec2ang(vec, lonlat=True)
        """
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
        y_map = hp.ud_grade(hp.read_map(fn), self.nside)
        return y_map * y_to_deltaToverT(float(freq))
        #return y_map*CONVERSION_FACTORS['Y'][self.fmt_freq(freq)]

    def get_radio_ps_map(self, freq):
        """
        Returns flux density in MJy/sr
        """
        if freq == "0278":
            freq_use = "0275"
        elif freq == "0225":
            freq_use = "0217"
        elif freq=="0039":
            freq_use = "35.9"
        else:
            freq_use = freq
        filename = opj(self.data_dir, "radiops",
                       "catalog_%.1f.h5"%float(freq_use))
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

    def get_cib_temp(self, freq):
        cib_map = hp.read_map(
            self.cib_map_file_name(freq=freq))
        #return cib_map*CONVERSION_FACTORS['CIB'][self.fmt_freq(freq)]
        return cib_map * get_cib_conversion_factor(freq)

    def get_radio_ps_temp(self, freq):
        ps_map = self.get_radio_ps_map(freq)
        return ps_map*flux_density_to_temp(float(freq))

    def get_cmb_lensed_orig_alms(self, lmax=None, survey_mask_hpix=None):
        """
        Get the original (i.e. packaged with the sim)
        lensed cmb alms
        """
        cmb_alms = hp.fitsfunc.read_alm(
            opj(self.data_dir, "lensed_alm.fits")
        )
        if survey_mask_hpix is not None:
            cmb_map = hp.alm2map(cmb_alms, self.nside)
            cmb_alms = hp.map2alm(cmb_map*survey_mask_hpix)
        if lmax is not None:
            cmb_alms = change_alm_lmax(
                cmb_alms, lmax)
        return cmb_alms

class WebSkyOld(WebSky):
    def __init__(self,
                 data_dir,
                 halo_catalog_name = 'halos.pksc',
                 kappa_map_name = 'kap.fits',
                 comptony_map_name = 'tsz_old.fits',
                 ksz_map_name = 'ksz.fits',
                 websky_cosmo = {'Omega_M': 0.31, 'Omega_B': 0.049, 'Omega_L': 0.69, 
                                 'h': 0.68, 'sigma_8': 0.81, 'n_s':0.965},
    ):
        super().__init__(data_dir, halo_catalog_name=halo_catalog_name,
                         kappa_map_name=kappa_map_name, comptony_map_name=comptony_map_name,
                         ksz_map_name=ksz_map_name, websky_cosmo=websky_cosmo)
    """
    def cib_map_file_name(self, freq='545'):
        cib_file_name = "cib_ns4096_nu%s.fits"%self.fmt_freq(freq)
        return opj(self.data_dir, cib_file_name)
    """

    def get_tsz_temp(self, freq):
        fn = self.comptony_map_file_name()
        y_map = hp.read_map(fn)
        return y_map*CONVERSION_FACTORS['Y'][self.fmt_freq(freq)]

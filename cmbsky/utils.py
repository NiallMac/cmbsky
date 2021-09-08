import os
import errno
import yaml
import falafel.utils as futils
import pkg_resources

#defaults_file = pkg_resources.resource_stream(
#    "cmbsky", "defaults.yaml")
defaults_file = os.path.join(os.path.dirname(__file__), "defaults.yaml")
with open(defaults_file,'rb') as f:
    DEFAULTS=yaml.load(f)

def get_disable_mpi():
    try:
        disable_mpi_env = os.environ['DISABLE_MPI']
        disable_mpi = True if disable_mpi_env.lower().strip() == "true" else False
    except:
        disable_mpi = False
    return disable_mpi

def safe_mkdir(d):
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise(e)

config = futils.config
def get_cmb_alm_unlensed(i,iset,path=config['signal_path']):
    sstr = str(iset).zfill(2)
    istr = str(i).zfill(5)
    fname = path + "fullskyUnlensedCMB_alm_set%s_%s.fits" % (sstr,istr)
    return hp.read_alm(fname,hdu=(1,2,3))

def get_cmb_seeds(args):
    #sort out cmb seeds
    if args.cmb_seed_start is not None:
        assert args.cmb_seed_end is not None
        args.cmb_seeds = list(range(
            args.cmb_seed_start, args.cmb_seed_end+1
            ))
    return args.cmb_seeds

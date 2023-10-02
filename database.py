import os
import sys
import numpy as np
import scipy.integrate
import astropy.cosmology
import precession
import pycbc.waveform
import gwdet
import h5py
from tqdm import tqdm
from astropy.cosmology import Planck15


class Database:

    def __init__(
        self, filename='./database/database.h5', store=False, display=False,
        ):

        self.filename = filename
        if not os.path.isfile(self.filename):
            raise ValueError('h5 database not found')
        self.db = h5py.File(self.filename, 'r')

        self.simulations = sorted(list(self.db.keys()))

        self.store = store
        self.stored = {}

        self.display = display

    def checkstore(self, sim, var, func):
        if (sim, var) in self.stored:
            return self.stored[(sim, var)]
        vals = func()
        if self.store:
            self.stored[(sim, var)] = vals
        return vals

    def getattr(self, sim, var):
        func = lambda: self.db[str(sim)].attrs[var]
        return self.checkstore(sim, var, func)

    def getdataset(self, sim, var):
        func = lambda: np.concatenate([np.array(self.db[str(sim)][cluster][var])
                                       for cluster in self.db[str(sim)]])
        return self.checkstore(sim, var, func)

    def getderived(self, sim, var, func, funcvars):
        _func = lambda: func(*[self.getdataset(sim, funcvar) for funcvar in funcvars])
        return self.checkstore(sim, var, _func)

    def __call__(self, sim, var):
        ''' Load data only if necessary '''

        if self.display:
            print(f'Accesing {(sim, var)}')

        # From the attributes
        if var in self.db[str(sim)].attrs:
            return self.getattr(sim, var)

        # From the datasets
        # List of variables is same for each cluster, so get it from the first
        elif var in self.db[str(sim)]['0']:
            return self.getdataset(sim, var)

        else:
            # Derived quantities
            # Some source parameters might be in the datasets or not
            # If they are, it's covered by the early return above
            # If not, compute them below
            # By default, database_consolidate stores Mchirp and q in database.h5
            
            if var == 'm1':
                func = lambda Mchirp, q: Mchirp * (1+q)**(1/5) / q**(3/5)
                funcvars = ['Mchirp', 'q']
                
            elif var == 'm2':
                func = lambda Mchirp, q: Mchirp * (1+q)**(1/5) * q**(2/5)
                funcvars = ['Mchirp', 'q']
                
            elif var == 'M':
                func = lambda Mchirp, q: Mchirp * (1+q)**(6/5) / q**(3/5)
                funcvars = ['Mchirp', 'q']
                
            elif var == 'lumdist':
                func = lambda z: Planck15.luminosity_distance(z).value
                funcvars = ['z']
                
            elif var == 'comdist':
                func = lambda z: Planck15.comoving_distance(z).value
                funcvars = ['z']
                
            else:
                raise ValueError('Variable not available')
                
            return self.getderived(sim, var, func, funcvars)


def place(sim, zmax, epsilon):

    print('Place')

    pars = ['zmax', 'epsilon']
    varis = ['z', 'lumdist']

    filename = f'./database/database_{sim}_place.h5'

    if not os.path.isfile(filename):
        print(f'Saving to {filename}')
        print(f'Simulation {sim}, zmax = {zmax}, epsilon = {epsilon}')

        with h5py.File(filename, 'w') as hplace, \
             h5py.File(f'./database/database_{sim}_eject.h5', 'r') as heject:

            for par in pars:
                hplace.attrs[par] = eval(par)

            # Initialize redshift distribution class
            zpdf = nonuniformredshift(zmax, epsilon)
                
            # Here, cluster is a string from the eject h5 file keys
            for cluster in tqdm(heject):
                    
                n_mergers = heject[cluster].attrs['n_mergers']
                    
                z = zpdf.sample(N=n_mergers)
                lumdist = Planck15.luminosity_distance(z).value

                hplace.create_group(cluster)
                for var in varis:
                    hplace[cluster].create_dataset(
                        var, data=eval(var), compression='gzip', compression_opts=9
                        )

    else:
        print(f'Place file already exists at {filename}')

    return filename


def pycbc_psd_from_txt_hack(filename, length, delta_f, low_freq_cutoff, is_asd_file=True, file_data=None):
    '''
    This is a hack of pycbc.psd.from_txt to handle extrapolation. I changed the interpolation routine from interp1d to InterpolatedUnivariateSpline. Verified that differences in SNR are 1e-14.
    Also adding a file_data optional kwarg to avoid reading in the psd file many times.
    http://pycbc.org/pycbc/latest/html/_modules/pycbc/psd/read.html#from_txt
    '''

    if file_data.all() == None:
        file_data = np.loadtxt(filename)

    if (file_data < 0).any() or np.logical_not(np.isfinite(file_data)).any():
        raise ValueError('Invalid data in ' + filename)

    freq_data = file_data[:, 0]
    noise_data = file_data[:, 1]

    # Only include points above the low frequency cutoff
    if freq_data[0] > low_freq_cutoff:
        raise ValueError('Lowest frequency in input file ' + filename + \
          ' is higher than requested low-frequency cutoff ' + str(low_freq_cutoff))

    kmin = int(low_freq_cutoff / delta_f)
    flow = kmin * delta_f

    data_start = (0 if freq_data[0] == low_freq_cutoff else np.searchsorted(freq_data, flow) - 1)

    # If the cutoff is exactly in the file, start there
    if freq_data[data_start+1] == low_freq_cutoff:
        data_start += 1

    freq_data = freq_data[data_start:]
    noise_data = noise_data[data_start:]

    flog = np.log(freq_data)
    if is_asd_file:
        slog = np.log(noise_data**2)
    else:
        slog = np.log(noise_data)

    psd_interp = scipy.interpolate.InterpolatedUnivariateSpline(flog, slog, k=1)

    kmin = int(low_freq_cutoff / delta_f)
    psd = np.zeros(length, dtype=np.float64)

    vals = np.log(np.arange(kmin, length) * delta_f)
    psd[kmin:] = np.exp(psd_interp(vals))

    return pycbc.types.FrequencySeries(psd, delta_f=delta_f)


def detect(sim, detector, SNRthreshold, approximant):

    print('Detect')

    pars = ['SNRthreshold', 'approximant']
    varis = ['SNR', 'pdet', 'waveformsuccess']

    filename = f'./database/database_{sim}_detect_{detector}.h5'

    if not os.path.isfile(filename):
        print(f'Saving to {filename}')
        print(f'Simulation {sim}, detector = {detector}, SNRthreshold = {SNRthreshold}, approximant = {approximant}')

        if detector == 'O1O2':
            psdtextfile = 'LIGO-P1200087-v18-aLIGO_EARLY_HIGH.txt'
        elif detector == 'O3a':
            psdtextfile = 'aligo_O3actual_L1.txt'
        else:
            raise IOError

        deltaf = 1.
        psd_data = np.loadtxt(f'./database/{psdtextfile}')
        flow = psd_data[:, 0].min()
        fhigh = psd_data[:, 0].max()
        pw = gwdet.averageangles(directory='database')

        with h5py.File(filename, 'w') as hdetect, \
             h5py.File(f'./database/database_{sim}_eject.h5', 'r') as heject, \
             h5py.File(f'./database/database_{sim}_place.h5', 'r') as hplace:

            # eject and place h5 files should have the same clusters
            assert sorted(list(heject.keys())) == sorted(list(hplace.keys()))

            for par in pars:
                hdetect.attrs[par] = eval(par)

            # Here, cluster is a string from the eject h5 file keys
            for cluster in tqdm(heject):
                data = []

                for merger in range(heject[cluster].attrs['n_mergers']):

                    m1 = heject[cluster]['m1'][merger]
                    m2 = heject[cluster]['m2'][merger]
                    chi1 = heject[cluster]['chi1'][merger]
                    chi2 = heject[cluster]['chi2'][merger]
                    theta1 = heject[cluster]['theta1'][merger]
                    theta2 = heject[cluster]['theta2'][merger]
                    phi1 = heject[cluster]['phi1'][merger]
                    phi2 = heject[cluster]['phi2'][merger]
                    z = hplace[cluster]['z'][merger]
                    lumdist = hplace[cluster]['lumdist'][merger]

                    try:
                        hp, hc = pycbc.waveform.get_fd_waveform(
                            approximant=approximant,
                            mass1=m1*(1+z),
                            mass2=m2*(1+z),
                            spin1x=chi1*np.sin(theta1)*np.cos(phi1),
                            spin1y=chi1*np.sin(theta1)*np.sin(phi1),
                            spin1z=chi1*np.cos(theta1),
                            spin2x=chi2*np.sin(theta2)*np.cos(phi2),
                            spin2y=chi2*np.sin(theta2)*np.sin(phi2),
                            spin2z=chi2*np.cos(theta2),
                            distance=lumdist,
                            delta_f=deltaf,
                            f_lower=flow,
                            f_ref=20.,
                            )

                    except:
                        waveformsuccess = False
                        SNR = 0.
                        pdet = 0.

                    else:
                        waveformsuccess = True
                        psd = pycbc_psd_from_txt_hack(
                            None, len(hp), deltaf, flow,
                            is_asd_file=True,
                            file_data=psd_data,
                            )
                        SNR = pycbc.filter.matchedfilter.sigma(
                            hp, psd=psd,
                            low_frequency_cutoff=flow,
                            high_frequency_cutoff=fhigh*(1-1e-3)
                            )
                        if SNR < SNRthreshold:
                            pdet = 0.
                        else:
                            pdet = pw(SNRthreshold/SNR)

                    data.append([float(SNR), float(pdet), waveformsuccess])

                hdetect.create_group(cluster)
                SNR, pdet, waveformsuccess = np.array(data).T
                for var in varis:
                    hdetect[cluster].create_dataset(
                        f'{var}_{detector}', data=eval(var),
                        compression='gzip', compression_opts=9
                        )

    else:
        print(f'Detect file already exists at {filename}')

    return filename

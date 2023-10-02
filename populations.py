import os
import sys
import numpy as np
import h5py
import kalepy
from tqdm import tqdm
import tqdm_pathos

import utils


class Populations:
    
    def __init__(self, pop='chipav', n_points=21):
        
        self.pop = pop
        self.n_points = int(n_points)

        self.limits = eval(f'utils.limits_{pop}').copy()
        self.bounds = list(self.limits['event'].values())

        self.axes = np.linspace(*np.transpose(self.bounds), self.n_points).T
        self._axes = {
            par: self.axes[i] for i, par in enumerate(self.limits['event'])
            }
        self.grid = utils.cartesian_product(self.axes)
        
        widths = np.squeeze(np.abs(np.diff(self.bounds))) / (self.n_points - 1)
        self.bins = np.linspace(
            self.axes[:,0]-widths/2, self.axes[:,-1]+widths/2, self.n_points+1,
            ).T
        
        from database import Database
        self.db = Database()
        self.n_sim = len(self.db.simulations)
        
    def cut(self, theta, return_theta=False):
        
        if 'Mchirp' in self.limits['event']:
        
            lo = min(self.limits['event']['Mchirp'])
            hi = max(self.limits['event']['Mchirp'])
            cut = (lo < theta['Mchirp']) & (theta['Mchirp'] < hi)

            if return_theta:
                return cut, np.array(list(theta.values()))[:, cut]
            return cut

    def get_mergers(self, sim, detection=True):

        theta = {
            par: np.array(self.db(sim, par)) for par in self.limits['event']
            }
        
        if 'chipav' in self.limits:
            notnans = ~np.isnan(theta['chipav'])
            for par in theta:
                theta[par] = theta[par][notnans]

        if not detection:
            return theta
    
        pdet = {
            'O1O2': self.db(sim, 'pdet_O1O2'),
            'O3': self.db(sim, 'pdet_O3a'),
            }

        return theta, pdet
            
    def get_sim(self, sim):
        
        theta, pdet = self.get_mergers(sim, detection=True)
        cut = self.cut(theta)
        
        n_mergers = {
            'horizon': np.shape(list(theta.values()))[-1],
            'posterior': cut.sum(),
            }
        
        sigma = {
            'horizon': {
                run: np.sum(pdet[run]) / n_mergers['horizon']
                for run in pdet
                },
            'posterior': {
                run: np.sum(pdet[run][cut]) / n_mergers['posterior']
                for run in pdet
                },
            }
        
        g = self.db(sim, 'generation')
        gens = {11: '11', 12: '12', 22: '22', 0: '>2'}
        
        f_gen = {'astro': {}, 'det': {}}
        
        f_gen['astro'] = {
            'horizon': {
                gen: (g == key).sum() / n_mergers['horizon']
                for key, gen in gens.items()
                },
            'posterior': {
                gen: (g[cut] == key).sum() / n_mergers['posterior']
                for key, gen in gens.items()
                },
            }
        
        from lvc_data import p_runs
        _pdet = p_runs['O1O2']*pdet['O1O2'] + p_runs['O3']*pdet['O3']
        
        f_gen['det'] = {
            'horizon': {
                gen: _pdet[g == key].sum() / _pdet.sum()
                for key, gen in gens.items()
                },
            'posterior': {
                gen: _pdet[cut][g[cut] == key].sum() / _pdet[cut].sum()
                for key, gen in gens.items()
                },
            }
        
        return n_mergers, sigma, f_gen
    
    def hist(self, _theta):

        hist = np.histogramdd(_theta.T, bins=self.bins, density=True)[0]
        
        edges = [0, -1]
        if len(self.limits['event']) == 1:
            hist[edges] *= 2
        elif len(self.limits['event']) == 2:
            hist[edges, :] *= 2
            hist[:, edges] *= 2
        elif len(self.limits['event']) == 3:
            hist[edges, :, :] *= 2
            hist[:, edges, :] *= 2
            hist[:, :, edges] *= 2
        elif len(self.limits['event']) == 4:
            hist[edges, :, :, :] *= 2
            hist[:, edges, :, :] *= 2
            hist[:, :, edges, :] *= 2
            hist[:, :, :, edges] *= 2
        
        return hist
    
    def kde_fft(self, _theta):
                
        kde = utils.KDEfft(
            _theta, bounds=self.bounds, method='fft', bandwidth='isj',
            )
        
        return kde.pdf(self.axes)
    
    ## TODO: KDEs without cutting chirp mass
    def kde(self, _theta):
        
        ## TODO: move import
        import kalepy
        
        reflect = self.bounds.copy()
        reflect[0] = None # chirp mass unbounded
        
        pdf = kalepy.pdf(_theta, points=self.grid, reflect=reflect)
        pdf = pdf.reshape([len(ax) for ax in self.axes])
        
        return pdf
    
    def get_population(self, sim, directory=None):
        
        filename = \
            f'{directory}/populations_{self.pop}_{self.n_points}_{sim}.npy'
        
        try:
            return np.load(filename, allow_pickle=True)
        
        except:
            n_mergers, sigma, f_gen = self.get_sim(sim)
            theta = self.get_mergers(sim, detection=False)
            cut, _theta = self.cut(theta, return_theta=True)
            hist = self.hist(_theta)
            kde = self.kde_fft(_theta)
#             hist = self.hist(_theta)
#             kde = self.kde(_theta)
            
            if directory is not None:
                np.save(
                    filename,
                    [n_mergers, sigma, f_gen, hist, kde],
                    allow_pickle=True,
                    )
                
            return n_mergers, sigma, f_gen, hist, kde


def save_mergers(sim):
    
    populations = Populations()
    theta, pdet = populations.get_mergers(sim)
    cut = populations.cut(theta)
    
    with h5py.File(f'./populations/populations_{sim}.h5', 'w') as h:
        
        for par in theta:
            h.create_dataset(
                par, data=theta[par], compression='gzip', compression_opts=9,
                )
            
        for par in pdet:
            h.create_dataset(
                'pdet_'+par,
                data=pdet[par],
                compression='gzip',
                compression_opts=9,
                )

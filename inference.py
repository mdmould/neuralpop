import numpy as np
import h5py
import bilby

import utils
limits = utils.limits_chipav


def get_priors(bounds):
    
    priors = bilby.core.prior.PriorDict()
    
    for par in limits['hyper']:
        
        minimum = float(min(bounds[par]))
        maximum = float(max(bounds[par]))
        
        priors[par] = bilby.core.prior.Uniform(
            minimum=minimum, maximum=maximum, name=par,
            )
        
    return priors


class Likelihood(bilby.Likelihood):
    
    def __init__(
        self, posteriors, population_model, norm, n_norm, selection_model=None,
        ):
        
        self.limits = population_model.limits.copy()
        
        if selection_model is None:
            selection_model = lambda _: 0.
        else:
            for par, lim in selection_model.limits.items():
                if par not in self.limits:
                    self.limits[par] = lim
                    
        self.population_model = population_model
        self.norm = norm
        self.n_norm = n_norm
        self.selection_model = selection_model
               
        self.posteriors, self.n_samples = self._trim_posteriors(posteriors)
        self.n_obs = len(self.posteriors)
        self.all_samples = self.n_obs * self.n_samples
        
        try:
            self.priors = np.array(
                [posterior['prior'] for posterior in self.posteriors]
                )
        except:
            self.priors = 1.
        
        ep = {
            par: np.concatenate(
                [posterior[par] for posterior in self.posteriors],
                )
            for par in self.limits['event']
            }
        
        self.ep_unit = np.array([
            utils.rescale_to_unit(ep[par], *lim)
            for par, lim in self.limits['event'].items()
            ]).T
        
        if norm == 'trapezoid' or norm == 'simpson':
            
            self.axes = np.array([
                np.linspace(*lim, n_norm)
                for lim in self.limits['event'].values()
                ])
            
            axes_unit = np.linspace([0.]*len(self.axes), 1., n_norm).T
            grid_unit = utils.cartesian_product(axes_unit).T
            
            self.ep_grid_unit = np.concatenate(
                [self.ep_unit, grid_unit], axis=0,
                )
            
            self.log_likelihood = self._log_likelihood_norm
            
        else:
            self.log_likelihood = self._log_likelihood_nonorm

        super().__init__(parameters={par: None for par in self.limits['hyper']})
        
    def _log_likelihood_norm(self):
        
        hp_unit = np.atleast_2d([
            utils.rescale_to_unit(self.parameters[par], *lim)
            for par, lim in self.limits['hyper'].items()
            ])
        _hp_unit = np.repeat(hp_unit, self.ep_grid_unit.shape[0], axis=0)
        inputs = np.concatenate([_hp_unit, self.ep_grid_unit], axis=-1)
        
        pred = self.population_model(inputs)
        ppop = pred[:self.all_samples].reshape(self.n_obs, self.n_samples)
        
        norm = utils.integrate_nd(
            pred[self.all_samples:].reshape(*[self.n_norm]*len(self.axes)),
            self.axes,
            method=self.norm,
            )
        
        logsigma = self.selection_model(hp_unit)
        
        return (
            np.sum(np.log(np.sum(ppop / self.priors, axis=-1)))
            - self.n_obs * np.log(norm)
            - self.n_obs * np.log(self.n_samples)
            - self.n_obs * logsigma
            )
    
    def _log_likelihood_nonorm(self):

        hp_unit = np.atleast_2d([
            utils.rescale_to_unit(self.parameters[par], *lim)
            for par, lim in self.limits['hyper'].items()
            ])
        _hp_unit = np.repeat(hp_unit, self.all_samples, axis=0)
        inputs = np.concatenate([_hp_unit, self.ep_unit], axis=-1)

        ppop = self.population_model(inputs).reshape(self.n_obs, self.n_samples)
        
        logsigma = self.selection_model(hp_unit)

        return (
            np.sum(np.log(np.sum(ppop / self.priors, axis=-1)))
            - self.n_obs * np.log(self.n_samples)
            - self.n_obs * logsigma
            )
    
    def _trim_posteriors(self, posteriors):
        
        _posteriors = []
        min_samples = np.inf
        for posterior in posteriors:
            n_samples = np.shape(list(posterior.values()))[-1]
            shuffle = np.arange(n_samples)
            np.random.shuffle(shuffle)
            _posteriors.append(
                {par: posterior[par][shuffle] for par in posterior},
                )
            min_samples = min(min_samples, n_samples)
            
        _posteriors = [
            {par: posterior[par][:min_samples] for par in posterior}
            for posterior in _posteriors
            ]
        
        return _posteriors, min_samples
    

def run_sampler(
    priors, posteriors, population_model, norm, n_norm, selection_model, label,
    outdir,
    ):
        
    likelihood = Likelihood(
        posteriors, population_model, norm, n_norm, selection_model,
        )

    kwargs = dict(
        bound='multi', sample='rwalk', nlive=500, walks=5, enlarge=1.5,
        vol_dec=0.5, vol_check=0.8, facc=0.2, slices=5, update_interval=300,
        dlogz=0.1, maxmcmc=5000, nact=2,
        )
    print('kwargs =', kwargs)
   
    result = bilby.run_sampler(
        likelihood=likelihood, priors=priors, sampler='dynesty', label=label,
        outdir=outdir, save='hdf5', check_point_plot=False, **kwargs,
        )
    
    return result

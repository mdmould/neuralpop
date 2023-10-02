import os
import glob
import numpy as np
import h5py
import bilby
import tqdm_pathos
import precession as precession2

import utils
limits = utils.limits_chipav


runs = {
    'O1': {
        2: 48.6 # 51.5 48.6 46.1 48.3
        },
    'O2': {
        2: 118.0,
        3: 15.0,
        },
    'O3a': {
        'h': 130.3,
        'l': 138.5,
        'v': 139.5,
        1: 177.3,
        2: 149.9,
        3: 81.4,
        },
    'O3b': {
        'h': 115.7,
        'l': 115.5,
        'v': 111.3,
        1: 142.0,
        2: 125.5,
        3: 75.0,
        },
    }

runs['O1O2'] = runs['O1'][2] + runs['O2'][2]
runs['O3'] = runs['O3a'][2] + runs['O3b'][2]
runs['O1O2O3'] = runs['O1O2'] + runs['O3']

p_runs = {run: runs[run] / runs['O1O2O3'] for run in ['O1O2', 'O3']}


# p_astro cut = 0.5 for catalog
# FAR cut = 1 for population
# NS cut > 2.5 at >90% confidence
# BBH = 69
# cf. population - NS = 76 - 7 = 69
gwevents = {

    'GWTC-1': [
        # All   = 11
        # BBH   = 10
        # FAR<1 = 11
        # both  = 10
        
        'GW150914',
        'GW151012',
        'GW151226',
        'GW170104',
        'GW170608',
        'GW170729',
        'GW170809',
        'GW170814',
        #'GW170817', # NS
        'GW170818',
        'GW170823',
        ],
    
    'GWTC-2': [
        # All   = 39
        # BBH   = 36
        # FAR<1 = 36
        # both  = 33
        
        'GW190408_181802',
        'GW190412', #_053044
        'GW190413_052954',
        'GW190413_134308',
        'GW190421_213856',
        #'GW190424_180648', # FAR = 0.78 in GWTC-2 but >2 in GWTC-2.1
        #'GW190425', #_081805' # NS
        #'GW190426_152155', # NS, FAR = 1.4 in GWTC-2 but 0.91 in GWTC-2.1
        'GW190503_185404',
        'GW190512_180714',
        'GW190513_205428',
        #'GW190514_065416', FAR = 0.53 in GWTC-2 but 2.8 in GWTC-2.1
        'GW190517_055101',
        'GW190519_153544',
        'GW190521', #_030229
        'GW190521_074359',
        'GW190527_092055',
        'GW190602_175927',
        'GW190620_030421',
        'GW190630_185205',
        'GW190701_203306',
        'GW190706_222641',
        'GW190707_093326',
        'GW190708_232457',
        'GW190719_215514', # FAR = 1.6 in GWTC-2 but 0.63 in GWTC-2.1
        'GW190720_000836',
        'GW190727_060333',
        'GW190728_064510',
        'GW190731_140936',
        'GW190803_022701',
        #'GW190814', #_211039 # NS?
        'GW190828_063405',
        'GW190828_065509',
        'GW190910_112807',
        'GW190915_235702',
        'GW190924_021846',
        'GW190929_012149',
        'GW190930_133541',
        ],
    
    'GWTC-2.1': [
        # All   = 8 (9)
        # BBH   = 7 (7)
        # FAR<1 = 4 (5)
        # both  = 3 (3)
        
        #'GW190403_051519', # FAR
        #'GW190426_190642', # FAR
        #('GW190531_023648', # NS?, p_astro=0.28 but SHOULD BE included as FAR=0.41, however there are no PE samples)
        'GW190725_174728',
        'GW190805_211137',
        #'GW190916_200658', # FAR
        #'GW190917_114630', # NS
        'GW190925_232845',
        #'GW190926_050336, # FAR
        ],
    
    'GWTC-3': [
        # All   = 35 (36)
        # BBH   = 34 (34)
        # FAR<1 = 24 (25)
        # both  = 23 (23)
        
        'GW191103_012549',
        'GW191105_143521',
        'GW191109_010717',
        #'GW191113_071753', # FAR
        #'GW191126_115259', # FAR
        'GW191127_050227',
        'GW191129_134029',
        #'GW191204_110529', # FAR
        'GW191204_171526',
        'GW191215_223052',
        'GW191216_213338',
        #'GW191219_163120', # FAR
        'GW191222_033537',
        'GW191230_180458',
        #('GW200105_162426', # NS, p_astro=0.36 but included as FAR=0.2)
        'GW200112_155838',
        #'GW200115_042309', # NS
        'GW200128_022011',
        'GW200129_065458',
        'GW200202_154313',
        'GW200208_130117',
        #'GW200208_222617', # FAR
        'GW200209_085452',
        #'GW200210_092255', # FAR
        'GW200216_220804',
        'GW200219_094415',
        #'GW200220_061928', # FAR
        #'GW200220_124850', # FAR
        'GW200224_222234',
        'GW200225_060421',
        'GW200302_015811',
        #'GW200306_093714', # FAR
        #'GW200308_173609', # FAR
        'GW200311_115853',
        'GW200316_215756',
        #'GW200322_091133', # FAR
        ],
    
    }
    
    
fars = {
    'GW150914': '$<1\\times10^{-5}$',
    'GW151012': '$7.92\\times10^{-3}$',
    'GW151226': '$<1\\times10^{-5}$',
    'GW170104': '$<1\\times10^{-5}$',
    'GW170608': '$<1\\times10^{-5}$',
    'GW170729': '$1.80\\times10^{-1}$',
    'GW170809': '$<1\\times10^{-5}$',
    'GW170814': '$<1\\times10^{-5}$',
    'GW170818': '$<1\\times10^{-5}$',
    'GW170823': '$<1\\times10^{-5}$',
    'GW190408\\_181802': '$<1\\times10^{-5}$',
    'GW190412\\_053044': '$<1\\times10^{-5}$',
    'GW190413\\_052954': '$8.17\\times10^{-1}$',
    'GW190413\\_134308': '$1.81\\times10^{-1}$',
    'GW190421\\_213856': '$2.83\\times10^{-3}$',
    'GW190503\\_185404': '$<1\\times10^{-5}$',
    'GW190512\\_180714': '$<1\\times10^{-5}$',
    'GW190513\\_205428': '$<1\\times10^{-5}$',
    'GW190517\\_055101': '$3.47\\times10^{-4}$',
    'GW190519\\_153544': '$<1\\times10^{-5}$',
    'GW190521\\_030229': '$<1\\times10^{-5}$',
    'GW190521\\_074359': '$1.00\\times10^{-2}$',
    'GW190527\\_092055': '$2.28\\times10^{-1}$',
    'GW190602\\_175927': '$<1\\times10^{-5}$',
    'GW190620\\_030421': '$1.12\\times10^{-2}$',
    'GW190630\\_185205': '$<1\\times10^{-5}$',
    'GW190701\\_203306': '$5.71\\times10^{-3}$',
    'GW190706\\_222641': '$<1\\times10^{-5}$',
    'GW190707\\_093326': '$<1\\times10^{-5}$',
    'GW190708\\_232457': '$3.09\\times10^{-4}$',
    'GW190719\\_215514': '$6.31\\times10^{-1}$',
    'GW190720\\_000836': '$<1\\times10^{-5}$',
    'GW190725\\_174728': '$4.58\\times10^{-1}$',
    'GW190727\\_060333': '$<1\\times10^{-5}$',
    'GW190728\\_064510': '$<1\\times10^{-5}$',
    'GW190731\\_140936': '$3.35\\times10^{-1}$',
    'GW190803\\_022701': '$7.32\\times10^{-2}$',
    'GW190805\\_211137': '$6.28\\times10^{-1}$',
    'GW190828\\_063405': '$<1\\times10^{-5}$',
    'GW190828\\_065509': '$<1\\times10^{-5}$',
    'GW190910\\_112807': '$2.87\\times10^{-3}$',
    'GW190915\\_235702': '$<1\\times10^{-5}$',
    'GW190924\\_021846': '$<1\\times10^{-5}$',
    'GW190925\\_232845': '$7.20\\times10^{-3}$',
    'GW190929\\_012149': '$1.55\\times10^{-1}$',
    'GW190930\\_133541': '$1.23\\times10^{-2}$',
    'GW191103\\_012549': '$4.58\\times10^{-1}$',
    'GW191105\\_143521': '$1.18\\times10^{-2}$',
    'GW191109\\_010717': '$1.80\\times10^{-4}$',
    'GW191127\\_050227': '$2.49\\times10^{-1}$',
    'GW191129\\_134029': '$<1\\times10^{-5}$',
    'GW191204\\_171526': '$<1\\times10^{-5}$',
    'GW191215\\_223052': '$<1\\times10^{-5}$',
    'GW191216\\_213338': '$<1\\times10^{-5}$',
    'GW191222\\_033537': '$<1\\times10^{-5}$',
    'GW191230\\_180458': '$5.02\\times10^{-2}$',
    'GW200112\\_155838': '$<1\\times10^{-5}$',
    'GW200128\\_022011': '$4.29\\times10^{-3}$',
    'GW200129\\_065458': '$<1\\times10^{-5}$',
    'GW200202\\_154313': '$<1\\times10^{-5}$',
    'GW200208\\_130117': '$3.11\\times10^{-4}$',
    'GW200209\\_085452': '$4.64\\times10^{-2}$',
    'GW200216\\_220804': '$3.50\\times10^{-1}$',
    'GW200219\\_094415': '$9.94\\times10^{-4}$',
    'GW200224\\_222234': '$<1\\times10^{-5}$',
    'GW200225\\_060421': '$<1\\times10^{-5}$',
    'GW200302\\_015811': '$1.12\\times10^{-1}$',
    'GW200311\\_115853': '$<1\\times10^{-5}$',
    'GW200316\\_215756': '$<1\\times10^{-5}$',
    }


def name_to_event(name):
    
    if name in gwevents['GWTC-1']:
        return name
    elif name == 'GW190412\\_053044':
        return 'GW190412'
    elif name == 'GW190521\\_030229':
        return 'GW190521'
    return ''.join(name.split('\\'))
    
    
def event_to_name(event):
    
    if event in gwevents['GWTC-1']:
        return event
    elif event == 'GW190412':
        return 'GW190412\\_053044'
    elif event == 'GW190521':
        return 'GW190521\\_030229'
    return '\\_'.join(event.split('_'))
        

def event_table():

    with h5py.File('./lvc/lvc_data.h5', 'r') as h:
        
        for name in fars:
            
            event = name_to_event(name)
            
            for catalog in gwevents:
                if event in gwevents[catalog]:
            
                    x = f'{name} & {catalog} & {fars[name]} '
                    print(x, end='')

                    p = h[event]['posterior']
                    notnans = ~np.isnan(p['chipav'])

                    for par in limits['event']:

                        l, m, u = np.quantile(
                            p[par][notnans], [0.05, 0.5, 0.95],
                            )

                        x = f'& ${m:.2f}_{{-{m-l:.2f}}}^{{+{u-m:.2f}}}$ '
                        print(x, end='')

                    x = '\\\\'
                    print(x)


def find_bounds():
    
    lo = np.ninf
    hi = -np.inf
    
    for catalog in gwevents:
        print(catalog)
        
        for event in gwevents[catalog]:
            print(event, end=' ')
            
            posterior = get_posterior(event, ['Mchirp'])
            event_lo = posterior['Mchirp'].min()
            event_hi = posterior['Mchirp'].max()
            print(event_lo, event_hi)
            
            lo = min(lo, event_lo)
            hi = max(hi, event_hi)

    print(lo, hi)
    
    return lo, hi
    
    
def find_min_samples():
    
    min_samples = np.inf
    
    for catalog in gwevents:
        print(catalog)
        
        for event in gwevents[catalog]:
            print(event, end=' ')
            
            n_samples = get_posterior(event, ['n_samples'])['n_samples']
            print(n_samples)
            
            min_samples = min(min_samples, n_samples)
            if min_samples == n_samples:
                min_catalog = catalog
                min_event = event
            
    print()
    print(min_catalog, min_event, min_samples)
            
    return min_catalog, min_event, min_samples


def eval_chieff(theta1, theta2, q, chi1, chi2):
    
    return (chi1*np.cos(theta1) + q*chi2*np.cos(theta2)) / (1+q)


def eval_chipav(
    theta1, theta2, deltaphi, f_ref, q, chi1, chi2, Mdet, method='montecarlo',
    n_samples=int(1e4), rng=None,
    ):
    
    r = precession2.gwfrequency_to_pnseparation(
        theta1, theta2, deltaphi, f_ref, q, chi1, chi2, Mdet,
        ).item()
    
    return precession2.eval_chip_averaged(
        theta1=theta1, theta2=theta2, deltaphi=deltaphi, r=r, q=q, chi1=chi1,
        chi2=chi2, method=method, Nsamples=n_samples, rng=rng,
        ).item()


def get_prior(event, n_samples=10000, rng=None):
    
    rng = utils.get_rng(rng)
                
    _f_ref = 11.0 if event == 'GW190521' else 20.0
    
    from inference_injection import UniformSourceFrame
    z_min = 0
    z_max = 2.3
    uniform_source_frame = UniformSourceFrame(z_min=z_min, z_max=z_max)
    
    # For GWTC-1 priors (bilby reanalysis)
    if event in gwevents['GWTC-1']:
        q_min = 0.125
        q_max = 1
        Mchirp_min = min(limits['event']['Mchirp'])
        Mchirp_max = max(limits['event']['Mchirp'])
        Mchirpdet_min = Mchirp_min * (1 + z_max)
        Mchirpdet_max = Mchirp_max * (1 + z_min)
    # For O3 events
    else:
        posterior = get_posterior(event, pars=['mass_1', 'mass_2'])
        mass1det_min = np.min(posterior['mass_1'])
        mass1det_max = np.max(posterior['mass_1'])
        mass2det_min = np.min(posterior['mass_2'])

    #prior = {par: np.zeros(n_samples) for par in limits['event']}
    pars = [
        'Mchirp', 'q', 'chi1', 'chi2', 'theta1', 'theta2', 'deltaphi', 'z',
        'chieff', 'chipav',
        ]
    prior = {par: np.zeros(n_samples) for par in pars}
    nans = np.ones(n_samples, dtype=bool)
    n_nans = nans.sum()
    
    while n_nans > 0:

        z = uniform_source_frame.sample(n=n_nans, rng=rng)
        chi1 = rng.uniform(0, 1, n_nans)
        chi2 = rng.uniform(0, 1, n_nans)
        theta1 = np.arccos(rng.uniform(-1, 1, n_nans))
        theta2 = np.arccos(rng.uniform(-1, 1, n_nans))
        deltaphi = rng.uniform(0, 2*np.pi, n_nans)
        
        if event in gwevents['GWTC-1']:
            Mchirpdet = rng.uniform(Mchirpdet_min, Mchirpdet_max, n_nans)
            q = rng.uniform(q_min, q_max, n_nans)
            Mchirp = Mchirpdet / (1+z)
            Mdet = Mchirpdet * (1+q)**(6/5) / q**(3/5)
        else:
            mass1det = rng.uniform(mass1det_min, mass1det_max, n_nans)
            mass2det = rng.uniform(mass2det_min, mass1det, n_nans)
            Mdet = mass1det + mass2det
            q = mass2det / mass1det
            Mchirp = Mdet * q**(3/5) / (1+q)**(6/5) / (1+z)
            
        chieff = eval_chieff(theta1, theta2, q, chi1, chi2)
        f_ref = np.ones(n_nans) * _f_ref
        chipav = tqdm_pathos.starmap(
            eval_chipav,
            zip(theta1, theta2, deltaphi, f_ref, q, chi1, chi2, Mdet),
            rng=rng,
            )
        
        # prior['Mchirp'][nans] = Mchirp
        # prior['q'][nans] = q
        # prior['chieff'][nans] = chieff
        # prior['chipav'][nans] = chipav
        for par in pars:
            prior[par][nans] = np.asarray(eval(par))
        
        nans = np.isnan(prior['chipav'])
        n_nans = nans.sum()
    
    for par in prior:
        prior[par] = np.asarray(prior[par])
        
    return prior


def get_posterior(event, pars=None, n_samples=None, rng=None):
    
    lvc_dir = '/data/mmould/xwing/lvc'

    gwfiles = {
        'GWTC-1': {
            event: glob.glob(f'{lvc_dir}/GWTC-1_bilby/{event}*')[0]
            for event in gwevents['GWTC-1']
            },
        'GWTC-2': {
            event: f'{lvc_dir}/GWTC-2/{event}_comoving.h5'
            for event in gwevents['GWTC-2']
            },
        'GWTC-2.1': {
            event: glob.glob(f'{lvc_dir}/GWTC-2.1/*{event}*.h5')[0]
            for event in gwevents['GWTC-2.1']
            },
        'GWTC-3': {
            event: glob.glob(f'{lvc_dir}/GWTC-3/*{event}*_cosmo.h5')[0]
            for event in gwevents['GWTC-3']
            },
        }
    
    rng = utils.get_rng(rng)
        
    close = True
    
    if event in gwevents['GWTC-1']:
        close = False
        h = bilby.result.read_in_result(filename=gwfiles['GWTC-1'][event])
        p = h.posterior
    
    elif event in gwevents['GWTC-2']:
        h = h5py.File(gwfiles['GWTC-2'][event], 'r')
        p = h['PrecessingSpinIMRHM']['posterior_samples']
    
    elif event in gwevents['GWTC-2.1']:
        h = h5py.File(gwfiles['GWTC-2.1'][event], 'r')
        p = h['PrecessingSpinIMRHM_comoving']['posterior_samples']
    
    elif event in gwevents['GWTC-3']:
        h = h5py.File(gwfiles['GWTC-3'][event], 'r')
        p = h['C01:Mixed']['posterior_samples']
        
    n_total = p.shape[0]
    
    if n_samples is None:
        idx = np.arange(n_total)
    else:
        # _idx = np.arange(n_total)
        # idx = rng.choice(_idx, n_samples, replace=False)
        idx = rng.choice(n_total, n_samples, replace=False)
    
    posterior = {}
    
    if pars is None:
        pars = [
            'Mchirp', 'q', 'chi1', 'chi2', 'theta1', 'theta2', 'deltaphi', 'z',
            'chieff', 'chipav',
            ]
        
        _f_ref = 11.0 if event == 'GW190521' else 20.0
        
        Mchirp = p['chirp_mass_source'][idx]
        q = p['mass_ratio'][idx]
        chi1 = p['a_1'][idx]
        chi2 = p['a_2'][idx]
        theta1 = p['tilt_1'][idx]
        theta2 = p['tilt_2'][idx]
        deltaphi = p['phi_12'][idx]
        Mdet = p['total_mass'][idx]
        
        chieff = eval_chieff(theta1, theta2, q, chi1, chi2)
        f_ref = np.ones(idx.size) * _f_ref
        chipav = tqdm_pathos.starmap(
            eval_chipav,
            zip(theta1, theta2, deltaphi, f_ref, q, chi1, chi2, Mdet),
            rng=rng,
            )

        z = Mdet * q**(3/5) / (1+q)**(6/5) / Mchirp - 1
        
        # for par in [
        #     'Mchirp', 'q', 'chi1', 'chi2', 'theta1', 'theta2', 'deltaphi',
        #     'chieff', 'chipav',
        #     ]:
        #     posterior[par] = np.asarray(eval(par))
        for par in pars:
            posterior[par] = np.asarray(eval(par))
        
    else:
        if 'n_samples' in pars:
            posterior['n_samples'] = n_total
            pars.remove('n_samples')
        for par in pars:
            posterior[par] = np.array(p[par][idx])
            
    if close:
        h.close()
    del p, h
    
    return posterior


def get_prior_at_posterior(prior, posterior):
    
    prior = np.array([prior[par][:] for par in limits['event']])
    posterior = np.array([posterior[par][:] for par in limits['event']])
    
    bounds = np.array([
        np.min([prior.min(axis=1), posterior.min(axis=1)], axis=0),
        np.max([prior.max(axis=1), posterior.max(axis=1)], axis=0),
        ]).T
    
    return utils.KDE(prior, bounds=bounds).pdf(posterior)
#     import kalepy
#     return kalepy.KDE(prior, reflect=bounds).pdf(posterior)[-1]


def main(seed=None):
    
    rng = utils.get_rng(seed)
    
    #min_catalog, min_event, min_samples = find_min_samples()
    min_catalog, min_event, min_samples = 'GWTC-3', 'GW200129_065458', 1993
    n_samples = min_samples
    
    posteriors = {}
    priors = {}
    
    for catalog in gwevents:
        print(catalog)
        
        for event in gwevents[catalog]:
            print(event)
            
            posteriors[event] = get_posterior(
                event, n_samples=n_samples, rng=rng,
                )
            notnans = ~np.isnan(posteriors[event]['chipav'])
            for par in posteriors[event]:
                posteriors[event][par] = posteriors[event][par][notnans]

            priors[event] = get_prior(event, n_samples=10000, rng=rng)

            min_samples = min(min_samples, notnans.sum())
            
    for event in posteriors:
        for par in posteriors[event]:
            posteriors[event][par] = posteriors[event][par][:min_samples]
            
    priors_at_posteriors = tqdm_pathos.starmap(
        get_prior_at_posterior, zip(priors.values(), posteriors.values()),
        )
    
    for event, pap in zip(posteriors, priors_at_posteriors):
        posteriors[event]['prior'] = pap
    
    os.system('mkdir -p lvc')
    with h5py.File('./lvc/lvc_data.h5', 'w') as h:
        
        for catalog in gwevents:
            for event in gwevents[catalog]:
                h.create_group(event)
                
                for lab, p in zip(['posterior', 'prior'], [posteriors, priors]):
                    h[event].create_group(lab)
                    for par in p[event]:
                        h[event][lab].create_dataset(
                            par, data=p[event][par], compression='gzip',
                            compression_opts=9,
                            )

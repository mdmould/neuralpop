import numpy as np
try:
    from scipy.integrate import trapezoid, simpson
except:
    from scipy.integrate import trapz as trapezoid, simps as simpson
from scipy.stats import gaussian_kde
    
    
limits_all = {
    'hyper' : {
        'alpha'  : [-10.,  10.],
        'beta'   : [-10.,  10.],
        'gamma'  : [-10.,  10.],
        'delta'  : [-10.,  10.],
        'mmax'   : [ 30., 100.],
        'chimax' : [  0.,   1.],
        },
    'event' : {
        'Mchirp' : [ 5., 105.],
        'q'      : [ 0.,   1.],
        'chieff' : [-1.,   1.],
        'z'      : [ 0.,   2.3],
        'chipav' : [ 0.,   2.],
        },
    }

limits_dict = lambda pars: {par: limits_all['event'][par] for par in pars}

limits_1d = {
    'hyper' : limits_all['hyper'],
    'event' : limits_dict(['Mchirp']),
    }

limits_3d = {
    'hyper' : limits_all['hyper'],
    'event' : limits_dict(['Mchirp', 'q', 'chieff']),
    }

limits_z = {
    'hyper' : limits_all['hyper'],
    'event' : limits_dict(['Mchirp', 'q', 'chieff', 'z']),
    }

limits_chipav = {
    'hyper' : limits_all['hyper'],
    'event' : limits_dict(['Mchirp', 'q', 'chieff', 'chipav']),
    }


def get_rng(rng=None):
    
    if (rng is None) or (type(rng) is int):
        rng = np.random.default_rng(rng)
        
    return rng
    
    
def rescale_to_unit(data, minimum, maximum):
    
    return (data - minimum) / (maximum - minimum)


def rescale_to_data(unit, minimum, maximum):
    
    return unit * (maximum - minimum) + minimum


def cartesian_product(axes):

    return np.array(np.meshgrid(*axes, indexing='ij')).reshape(len(axes), -1)


def truncate(data, bounds, exclude_bounds=False):

    data = np.atleast_2d(data)
    bounds = np.atleast_2d(bounds)

    if exclude_bounds:
        return np.logical_and.reduce(np.logical_and(
            data > bounds[:, 0, None], data < bounds[:, 1, None],
            ))
    return np.logical_and.reduce(np.logical_and(
        data >= bounds[:, 0, None], data <= bounds[:, 1, None],
        ))


def integrate_nd(y, x, dims=None, method='trapezoid'):

    assert len(y.shape) == len(x)
    n_dim = len(x)
    for dim in range(n_dim):
        assert y.shape[dim] == len(x[dim])

    if dims is None:
        dims = np.arange(n_dim)
        
    if method == 'simpson':
        integrate = simpson
    elif method == 'trapezoid':
        integrate = trapezoid

    #integral = y.copy()
    for dim in np.flip(np.sort(dims)):
        #integral = trapezoid(integral, x[dim], axis=dim)
        y = integrate(y, x[dim], axis=dim)

    #return integral
    return y


# Split into training and validation sets
def training_split(n_sim=1000, split=.1, seed=42):
     
    n_valid = int(split * n_sim)
    n_train = n_sim - n_valid
    
    np.random.seed(seed)
    valid = np.sort(np.random.choice(n_sim, n_valid, replace=False))
    train = np.setdiff1d(range(n_sim), valid)
    np.random.seed()
    
    return train, valid


class KDE:
    
    def __init__(self, data, bw_method='scott', bounds=None):

        data = np.atleast_2d(data)
        n_dim, n_samples = data.shape
        if bounds is None:
            bounds = np.array([data.min(axis=1), data.max(axis=1)]).T
        
        data_in = data.copy()
        n_reflections = 0
        for dim in range(n_dim):
            for lr in range(2):
                mirror = data_in.copy()
                mirror[dim] = 2*bounds[dim][lr] - mirror[dim]
                data = np.concatenate([data, mirror], axis=1)
                n_reflections += 1
            
        kde = gaussian_kde(data, bw_method=bw_method)
        
        self.bounds = bounds
        self.n_reflections = n_reflections
        self.kde = kde
        
    def __call__(self, data):
        
        return self.pdf(data)
        
    def pdf(self, data):
        
        data = np.atleast_2d(data)
        pdf = self.kde(data) * (self.n_reflections + 1)
        over_bounds = ~truncate(data, self.bounds)
        pdf[over_bounds] = 0
        
        return pdf


class KDEfft:

    def __init__(
        self, data, weights=None, bounds=None, method='fft', bandwidth='isj',
        ):

        data = np.atleast_2d(data)
        n_dim = data.shape[0]
        if weights is not None:
            weights = np.atleast_1d(weights)

        # Construct bounds
        # bounds = list of 2-tuples or Nones
        # _bounds = array (n_dim, 2) where Nones are +/-inf
        if bounds is not None:
            #bounds = np.atleast_2d(bounds)
            _bounds = np.zeros((n_dim, 2))
            for dim in range(n_dim):
                if bounds[dim] is None:
                    _bounds[dim] = [-np.inf, np.inf]
                else:
                    for lr in range(2):
                        if bounds[dim][lr] is None:
                            _bounds[dim, lr] = [-np.inf, np.inf][lr]
                        else:
                            _bounds[dim, lr] = bounds[dim][lr]
            self._bounds = _bounds

            # Filter data and weights by boundaries
            in_bounds = truncate(data, self._bounds, exclude_bounds=True)
            data = data[:, in_bounds]
            if weights is not None:
                weights = weights[in_bounds]

        self.bounds = bounds
        self.n_dim, self.n_obs = data.shape

        # Compute bandwidth of non-reflected data
        if type(bandwidth) is str:
            bandwidth = bandwidth.lower()
            if bandwidth == 'scott':
                _bw_rule = scotts_rule
            elif bandwidth == 'silverman':
                _bw_rule = silvermans_rule
            elif bandwidth == 'isj':
                _bw_rule = improved_sheather_jones
            self.bandwidth = np.apply_along_axis(
                lambda x: _bw_rule(x[:, None]), 1, data,
                )
            self.bandwidth_method = bandwidth
        else:
            self.bandwidth = np.atleast_1d(bandwidth)
            if self.bandwidth.size == 1:
                self.bandwidth = np.repeat(self.bandwidth, self.n_dim)
            self.bandwidth_method = 'given'

        # Reflect data over faces of hypercube
        if bounds is not None:
            # TODO: sort data and only reflect points closest to boundaries
            # e.g. within < x*bandwidth
            data_in = data.copy()
            if weights is not None:
                weights_in = weights.copy()
            n_reflections = 0
            for dim in range(n_dim):
                if bounds[dim] is not None:
                    for lr in range(2):
                        if bounds[dim][lr] is not None:
                            mirror = data_in.copy()
                            mirror[dim] = 2*bounds[dim][lr] - mirror[dim]
                            data = np.concatenate([data, mirror], axis=1)
                            if weights is not None:
                                weights = np.concatenate([weights, weights_in], axis=0)
                            n_reflections += 1
            self.n_reflections = n_reflections

        self.data = data
        self.weights = weights
        
        self.method = method.lower()
        if self.method == 'naive':
            _method = NaiveKDE
        elif self.method == 'tree':
            _method = TreeKDE
        elif self.method == 'fft':
            _method = FFTKDE

        # Wrap KDEpy.FFTKDE
        self._kde = _method(kernel='gaussian', bw=1).fit(
            self.data.T / self.bandwidth, weights=self.weights,
            )

    def pdf(self, axes, grid=True):

        if (self.n_dim == 1) and (len(axes) != 1):
            axes = [axes]

        _axes = axes.copy()
        if self.method == 'fft':
            if self.bounds is not None:
                for dim in range(self.n_dim):
                    if self.bounds[dim] is not None:
                        for lr in range(2):
                            if self.bounds[dim][lr] is not None:
                                mirror = 2*self.bounds[dim][lr] - axes[dim]
                                _axes[dim] = np.concatenate([_axes[dim], mirror])
                                _axes[dim] = np.unique(_axes[dim])

        if grid:
            _axes = cartesian_product(_axes)

        pdf = self._kde.evaluate(_axes.T / self.bandwidth) / np.product(self.bandwidth)

        if self.bounds is not None:
            over_bounds = ~truncate(_axes, self._bounds)
            pdf[over_bounds] = 0
            pdf = pdf[~over_bounds]
            pdf = pdf * (self.n_reflections + 1)

        if grid:
            pdf = pdf.reshape([len(ax) for ax in axes])
            
        # PDF should be normalized by (n_reflections + 1) / product(bandwidth)
        # But the FFTKDE prediction needs to be renormalized anyway
        # So just do it all with one numerical normalization
        if self.method == 'fft':
            norm = integrate_nd(pdf, axes)
            pdf = pdf / norm

        return pdf

    def resample(self, size=1, seed=None):

        np.random.seed(seed)
        # data contains reflections
        # Sample from original (truncated) data which is the first n_obs indices
        idx = np.random.choice(self.n_obs, size, replace=True)
        noise = np.random.randn(self.n_dim, size) * self.bandwidth[:, None]
        sample = self.data[:, idx] + noise

        for _ in range(2):
            for dim in range(self.n_dim):
                if self.bounds[dim] is not None:
                    for lr in range(2):
                        if self.bounds[dim][lr] is not None:
                            if lr == 0:
                                over_bound = sample[dim] < self.bounds[dim][lr]
                            elif lr == 1:
                                over_bound = sample[dim] > self.bounds[dim][lr]
                            sample[dim, over_bound] = 2*self.bounds[dim][lr] - sample[dim, over_bound]

        return sample

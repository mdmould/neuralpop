import os
import sys
import time
import numpy as np
import h5py
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm

import utils


# Population model
#-------------------------------------------------------------------------------

class PopulationModel:
    
    def __init__(
        self,
        pop='chipav',
        n_points=21,
        pdf='kde',
        design_file='./database/database_design.h5',
        model_file=None,
        ):
        
        self.model_file = model_file
        
        if model_file is None:
            self.model = None
            self.pop = pop
            self.n_points = n_points
            self.pdf = pdf  
        else:
            strings = model_file.split('.h5')[0].split('_')
            self.pop = strings[2]
            self.n_points = int(strings[3])
            self.pdf = strings[4]
            self.loss_file = model_file.split('.h5')[0] + '.csv'
            
        self.limits = eval(f'utils.limits_{self.pop}').copy()
        self.n_hp = len(self.limits['hyper'])
        self.n_ep = len(self.limits['event'])
        self.n_dim = self.n_hp + self.n_ep
            
        if model_file is not None:
            try:
                self.model = tf.keras.models.load_model(model_file)
            except:
                n_neurons = int(strings[5])
                n_layers = int(strings[6])
                activation = strings[7]
                self.define_model(
                    n_neurons=n_neurons,
                    n_layers=n_layers,
                    activation=activation,
                    )
                self.model.load_weights(model_file)

        self.design_file = design_file
        self.pop_file = \
            f'./populations/populations_{self.pop}_{self.n_points}.h5'
        
        with h5py.File(self.design_file, 'r') as h:
            self.hp = {par: h[par][:] for par in self.limits['hyper']}
            self._hp = [
                {par: h[par][sim] for par in self.limits['hyper']}
                for sim in range(h.attrs['n_sim'])
                ]
        
        with h5py.File(self.pop_file, 'r') as h:
            self.ep = {par: h['p_pop'][par][:] for par in self.limits['event']}
            self.limits['pdf'] = [0, np.max(h['p_pop'][self.pdf])]
        self.axes = list(self.ep.values())
            
    def __call__(self, inputs):
        
        return utils.rescale_to_data(
            np.squeeze(self.model.predict_on_batch(inputs)),
            *self.limits['pdf'],
            )
            
    def predict(self, hp, ep=None, grid=True, norm=None, n_norm=None):
        
        if ep is None:
            ep = self.ep
            grid = True
        inputs = self._input(hp, ep, grid)
        n_sim, n_points, n_dim = inputs.shape
        inputs = np.concatenate(inputs)
        
        if norm == 'trapezoid' or norm == 'simpson':
            
            if n_norm is None:
                n_norm = 51
            
            _ep = {
                par: np.linspace(*lim, n_norm)
                for par, lim in self.limits['event'].items()
                }
            _axes = list(_ep.values())
            _inputs = np.concatenate(self._input(hp, _ep, True))
            
            if np.array_equal(inputs, _inputs):
                pred = self(inputs).reshape(n_sim, n_points)
                _pred = pred.reshape(n_sim, *[len(ax) for ax in _axes])
                
            else:
                _pred = self(np.concatenate([inputs, _inputs]))
                pred = _pred[:n_sim*n_points].reshape(n_sim, n_points)
                _pred = _pred[n_sim*n_points:].reshape(
                    n_sim, *[len(ax) for ax in _axes],
                    )
 
            norms = np.array([
                utils.integrate_nd(
                    p, list(_ep.values()), dims=None, method=norm,
                    ) 
                for p in _pred
                ])

        elif norm == 'montecarlo':
            
            if n_norm is None:
                n_norm = int(2e5)
            
            _ep = {
                par: np.random.uniform(*lim, n_norm)
                for par, lim in self.limits['event'].items()
                }
            _inputs = np.concatenate(self._input(hp, _ep, False))
            
            _pred = self(np.concatenate([inputs, _inputs]))
            pred = _pred[:n_sim*n_points].reshape(n_sim, n_points)
            _pred = _pred[n_sim*n_points:].reshape(n_sim, n_norm)
            
            maximum = np.max([pred.max(axis=-1), _pred.max(axis=-1)], axis=0)
            n_accepted = np.sum(
                np.random.uniform(0, maximum, [n_norm, maximum.size]).T < _pred,
                axis=-1,
                )
            ep_volume = np.product(np.diff(list(self.limits['event'].values())))
            
            norms = maximum * ep_volume * n_accepted / n_norm
        
        else:
            pred = self(inputs).reshape(n_sim, n_points)
            norms = np.ones(n_sim)
            
        pred /= norms[:, None]
        
        if grid:
            n_points_per_dim = [
                np.atleast_1d(ep[par]).size for par in self.limits['event']
                ]
            pred = pred.reshape(n_sim, *n_points_per_dim)
                    
        return np.squeeze(pred)
    
    def sample(
        self, hp, n_samples, n_montecarlo=int(1e7), seed=None, verbose=True,
        ):
        
        rng = utils.get_rng(seed)
        
        samples = []
        maximum = -np.inf
        
        if verbose:
            pbar = tqdm(total=n_samples)
        
        while len(samples) < n_samples:
            
            ep = {
                par: rng.uniform(*lim, n_montecarlo)
                for par, lim in self.limits['event'].items()
                }
            pred = self.predict(hp, ep=ep, grid=False, norm=None)
            maximum = max(maximum, pred.max()*1.1)
            
            initial = rng.uniform(0, maximum, n_montecarlo) < pred
            n_accept = min(initial.sum(), n_samples-len(samples))
            
            for n in range(n_accept):
                samples.append(
                    {par: ep[par][initial][n] for par in self.limits['event']},
                    )
            
            if verbose:
                pbar.update(n_accept)
                
        if verbose:
            pbar.close()
            
        return samples

    def _input(self, hp, ep, grid):
        
        hp_unit = np.atleast_2d([
            utils.rescale_to_unit(np.atleast_1d(hp[par]), *lim)
            for par, lim in self.limits['hyper'].items()
            ]).T
        n_sim, n_hp = hp_unit.shape
        
        ep_unit = [
            utils.rescale_to_unit(np.atleast_1d(ep[par]), *lim)
            for par, lim in self.limits['event'].items()
            ]
        
        if grid:
            grid = utils.cartesian_product(ep_unit).T
        else:
            grid = np.array(ep_unit).T
        n_points, n_ep = grid.shape
                
        inputs = np.zeros((n_sim, n_points, n_hp+n_ep))
        inputs[:, :, :n_hp] = np.repeat(hp_unit[:, None, :], n_points, axis=1)
        inputs[:, :, n_hp:] = np.repeat(grid[None, :, :], n_sim, axis=0)
    
        return inputs

    def _norm(self, hp, norm='trapezoid', n_norm=None):
        
        if norm == 'trapezoid' or norm == 'simpson':
            
            n_norm = 21 if n_norm is None else n_norm
            ep = {
                par: np.linspace(*lim, n_norm)
                for par, lim in self.limits['event'].items()
                }
            n_ep = len(ep)

            inputs = self._input(hp, ep, True)
            n_sim = inputs.shape[0]

            pred = self(np.concatenate(inputs)).reshape(n_sim, *[n_norm]*n_ep)
        
            norms = np.array([
                utils.integrate_nd(p, list(ep.values()), method=norm) 
                for p in pred
                ])
            
        elif norm == 'montecarlo':
            
            n_norm = int(1e6) if n_norm is None else n_norm
            ep = {
                par: np.random.uniform(*lim, n_norm)
                for par, lim in self.limits['event'].items()
                }
            
            inputs = self._input(hp, ep, False)
            n_sim = inputs.shape[0]
            
            pred = self(np.concatenate(inputs)).reshape(n_sim, n_norm)
            
            maximum = pred.max(axis=-1) * 1.1
            n_accepted = np.sum(
                np.random.uniform(0, maximum, [n_norm, n_sim]).T < pred,
                axis=-1,
                )
            
            ep_volume = np.product(np.diff(list(self.limits['event'].values())))
            
            norms = maximum * ep_volume * n_accepted / n_norm
            
        return norms
            
    def define_model(
        self,
        n_neurons=128,
        n_layers=5,
        activation='rrelu',
        verbose=True,
        ):
        
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.activation = activation
        self.activations = []
        
        # Initialize empty MLP
        self.model = tf.keras.Sequential(
            [tf.keras.layers.InputLayer(input_shape=(self.n_dim,))],
            )
        
        # Hidden layers
        for n in range(n_layers):
            try:
                self.model.add(
                    tf.keras.layers.Dense(n_neurons, activation=activation),
                    )
                self.activations.append(activation)
            except:
                # first-half tanh, second-half relu
                if activation == 'mix':
                    _activation = 'tanh' if n < int(n_layers/2) else 'relu'
                # same activation in each layer
                elif activation == 'prelu':
                    _activation = tf.keras.layes.PReLU()
                else:
                    _activation = eval(f'tfa.activations.{activation}')
                self.activations.append(_activation)
                self.model.add(
                    tf.keras.layers.Dense(n_neurons, activation=_activation),
                    )

        # Output layer, force positive PDF predictions
        self.model.add(tf.keras.layers.Dense(1))
        self.model.add(tf.keras.layers.Lambda(lambda x: x**2))

        if verbose:
            self.model.summary()
    
    def train(
        self,
        optimizer=tf.keras.optimizers.Adam,
        loss='mae',
        learning_rate=0.0001,
        batch_size=None,
        batch_percent=0.01,
        epochs=50000,
        stop_kwargs=None,
        callbacks=[],
        ):
        
        data = self._preprocess()
        
        if batch_percent is not None:
            n_samples = data['train']['inputs'].shape[0]
            batch_size = max(int(n_samples * batch_percent / 100), 1)
        
        self.loss = loss
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.batch_percent = batch_percent
        self.epochs = epochs
        self.stop = False if stop_kwargs is None else True
        
        filename = (
            f'./neural/neural_pop_{self.pop}_{self.n_points}_{self.pdf}'
            f'_{self.n_neurons}_{self.n_layers}_{self.activation}'
            f'_{loss}_{learning_rate}_{batch_percent}'
            )
        if os.path.exists(filename+'.h5') or os.path.exists(filename+'.csv'):
            filename += '_' + str(int(time.time()))
        self.model_file = filename + '.h5'
        self.loss_file = filename + '.csv'
        
        # Callbacks
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.model_file,
                monitor='val_loss',
                mode='min',
                save_weights_only=False,
                save_best_only=True,
                save_freq='epoch',
                verbose=0,
                ),
            tf.keras.callbacks.CSVLogger(self.loss_file),
            ]
        if self.stop:
            self.callbacks.append(
                tf.keras.callbacks.EarlyStopping(**stop_kwargs),
                )
        self.callbacks += callbacks
            
        self.model.compile(
            optimizer=optimizer(learning_rate=learning_rate), loss=loss,
            )
        
        history = self.model.fit(
            x=data['train']['inputs'],
            y=data['train']['outputs'],
            validation_data=(data['valid']['inputs'], data['valid']['outputs']),
            shuffle=True,
            batch_size=self.batch_size,
            epochs=epochs,
            callbacks=self.callbacks,
            )
        
        return history
    
    def _preprocess(self, split=0.1, seed=42, stack=True):
        
        print('| Pre-processing')

        # PDF values
        with h5py.File(self.pop_file, 'r') as h:
            ppop = h['p_pop'][self.pdf][:]
            
        inputs = self._input(self.hp, self.ep)
        n_sim, n_points, n_dim = inputs.shape
        outputs = utils.rescale_to_unit(
            ppop.reshape(n_sim, -1), *self.limits['pdf'],
            )
        
        print('| Populations')
        print('Hyperparameters', self.limits['hyper'].keys())
        print('Source parameters', self.limits['event'].keys())
        print('n_sim, n_points =', n_sim, n_points)
        
        # Split into training and validation sets
        print('| Train / valid')
        train, valid = utils.training_split(n_sim, split, seed)
        
        data = {}
        for train_valid in ['train', 'valid']:
            print(train_valid)
            data[train_valid] = {}
            for in_out in ['inputs', 'outputs']:
                data[train_valid][in_out] = eval(in_out)[eval(train_valid)]
                print(in_out, ':', data[train_valid][in_out].shape)

        # Flatten across first two axes (simulations and points)
        if stack:
            print('| Stack')
            for train_valid in data:
                print(train_valid)
                for in_out in data[train_valid]:
                    data[train_valid][in_out] = np.concatenate(
                        data[train_valid][in_out]
                        )
                    print(in_out, ':', data[train_valid][in_out].shape)

        return data
    
    
def train_population():
    
    pop = 'chipav'
    n_points = 21
    pdf = 'kde'
    
    n_neurons = 128
    n_layers = 5
    activation = 'rrelu'
    
    loss = 'mae'
    learning_rate = 0.0001
    batch_percent = 0.01
    
    population_model = PopulationModel(pop=pop, n_points=n_points, pdf=pdf)
    
    population_model.define_model(
        n_neurons=n_neurons,
        n_layers=n_layers,
        activation=activation,
        output=output,
        )
    
    history = population_model.train(
        loss=loss, learning_rate=learning_rate, batch_percent=batch_percent,
        )
    
    print(population_model.model_file, population_model.loss_file)


# Selection model
#-------------------------------------------------------------------------------
    
class SelectionModel:

    def __init__(
        self,
        pop='chipav',
        n_points=21,
        volume='prime',
        design_file='./database/database_design.h5',
        model_file=None,
        ):

        self.model_file = model_file

        if model_file is None:
            self.model = None
            self.pop = pop
            self.n_points = n_points
            self.volume = volume
        else:
            strings = model_file.split('.h5')[0].split('_')
            self.pop = strings[2]
            self.n_points = int(strings[3])
            self.volume = strings[4]
            self.loss_file = model_file.split('.h5')[0] + '.csv'

        self.limits = eval(f'utils.limits_{self.pop}').copy()
        self.n_dim = len(self.limits['hyper'])

        if model_file is not None:
            try:
                self.model = tf.keras.models.load_model(model_file)
            except:
                n_neurons = int(strings[5])
                n_layers = int(strings[6])
                activation = strings[7]
                self.define_model(
                    n_neurons=n_neurons,
                    n_layers=n_layers,
                    activation=activation,
                    )
                self.model.load_weights(model_file)

        self.design_file = design_file
        self.pop_file = \
            f'./populations/populations_{self.pop}_{self.n_points}.h5'

        with h5py.File(design_file, 'r') as h:

            self.hp = {par: h[par][:] for par in self.limits['hyper']}
            self._hp = [
                {par: h[par][sim] for par in self.limits['hyper']}
                for sim in range(h.attrs['n_sim'])
                ]

        with h5py.File(self.pop_file, 'r') as h:
            
            vol = 'horizon' if self.volume == 'prime' else self.volume

            sigma_O1O2 = h['sigma'][vol]['O1O2'][:]
            sigma_O3 = h['sigma'][vol]['O3'][:]

            from lvc_data import p_runs
            sigma = p_runs['O1O2']*sigma_O1O2 + p_runs['O3']*sigma_O3

            if self.volume == 'prime':

                n_h = h['n_mergers']['horizon'][:]
                n_p = h['n_mergers']['posterior'][:]
                sigma = sigma * n_h / n_p

            self.logsigma = np.log(sigma)

        self.limits['logsigma'] = [self.logsigma.min(), self.logsigma.max()]

    def __call__(self, inputs):

        return utils.rescale_to_data(
            np.squeeze(self.model(inputs).numpy()), *self.limits['logsigma'],
            )

    def _input(self, hp):

        return np.atleast_2d([
            utils.rescale_to_unit(np.atleast_1d(hp[par]), *lim)
            for par, lim in self.limits['hyper'].items()
            ]).T

    def predict(self, hp=None):

        if hp is None:
            hp = self.hp
        inputs = self._input(hp)

        return self(inputs)

    def define_model(
        self,
        n_neurons=128,
        n_layers=3,
        activation='rrelu',
        verbose=True,
        ):

        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.activation = activation

        self.model = tf.keras.Sequential(
            [tf.keras.layers.InputLayer(input_shape=(self.n_dim,)),],
            )

        for n in range(n_layers):
            try:
                self.model.add(
                    tf.keras.layers.Dense(n_neurons, activation=activation),
                    )
            except:
                if activation == 'prelu':
                    _activation = tf.keras.layers.PReLU()
                else:
                    _activation = eval(f'tfa.activations.{activation}')
                self.model.add(
                    tf.keras.layers.Dense(n_neurons, activation=_activation),
                    )

        self.model.add(tf.keras.layers.Dense(1))

        if verbose:
            self.model.summary()

    def train(
        self,
        optimizer=tf.keras.optimizers.Adam,
        loss='mse',
        learning_rate=0.001,
        batch_size=None,
        batch_percent=1,
        epochs=2000,
        stop_kwargs=None,
        callbacks=[],
        ):

        data = self._preprocess()

        if batch_percent is not None:
            n_samples = data['train']['inputs'].shape[0]
            batch_size = max(int(n_samples * batch_percent / 100), 1)

        self.loss = loss
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.batch_percent = batch_percent
        self.epochs = epochs
        self.stop = False if stop_kwargs is None else True

        filename = (
            f'./neural/neural_det_{self.pop}_{self.n_points}_{self.volume}'
            f'_{self.n_neurons}_{self.n_layers}_{self.activation}'
            f'_{loss}_{learning_rate}_{batch_percent}'
            )
        if os.path.exists(filename+'.h5') or os.path.exists(filename+'.csv'):
            filename += '_' + str(int(time.time()))
        self.model_file = filename + '.h5'
        self.loss_file = filename + '.csv'

        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.model_file,
                monitor='val_loss',
                mode='min',
                save_weights_only=False,
                save_best_only=True,
                save_freq='epoch',
                verbose=0,
                ),
            tf.keras.callbacks.CSVLogger(self.loss_file),
            ]
        if self.stop:
            self.callbacks.append(
                tf.keras.callbacks.EarlyStopping(**stop_kwargs),
                )
        self.callbacks += callbacks

        self.model.compile(
            optimizer=optimizer(learning_rate=learning_rate), loss=loss,
            )

        history = self.model.fit(
            x=data['train']['inputs'],
            y=data['train']['outputs'],
            validation_data=(data['valid']['inputs'], data['valid']['outputs']),
            shuffle=True,
            batch_size=self.batch_size,
            epochs=epochs,
            callbacks=self.callbacks,
            )

        return history

    def _preprocess(self, split=0.1, seed=42):

        inputs = self._input(self.hp)
        n_sim, n_dim = inputs.shape
        outputs = utils.rescale_to_unit(self.logsigma, *self.limits['logsigma'])

        train, valid = utils.training_split(n_sim, split, seed)

        data = {}
        for train_valid in ['train', 'valid']:
            data[train_valid] = {}
            for in_out in ['inputs', 'outputs']:
                data[train_valid][in_out] = eval(in_out)[eval(train_valid)]

        return data
    
    
def train_selection():

    pop = 'chipav'
    n_points = 21
    volume = 'nn'

    n_neurons = 128
    n_layers = 3
    activation = 'rrelu'

    loss = 'mse'
    learning_rate = 0.001
    batch_percent = 1

    selection_model = SelectionModel(pop=pop, n_points=n_points, volume=volume)

    selection_model.define_model(
        n_neurons=n_neurons,
        n_layers=n_layers,
        activation=activation,
        )

    t0 = time.time()
    history = selection_model.train(
        loss=loss, learning_rate=learning_rate, batch_percent=batch_percent,
        )
    dt = time.time() - t0
    print(dt/60.)

    print(selection_model.model_file, selection_model.loss_file)


# Generation model
#-------------------------------------------------------------------------------

class GenerationModel:
    
    def __init__(
        self,
        branch='astro',
        pop='chipav',
        n_points=21,
        volume='horizon',
        design_file='./database/database_design.h5',
        model_file=None,
        ):

        self.model_file = model_file
        
        if model_file is None:
            self.model = None
            self.branch = branch
            self.pop = pop
            self.n_points = n_points
            self.volume = volume
        else:
            strings = model_file.split('.h5')[0].split('_')
            self.branch = strings[2]
            self.pop = strings[3]
            self.n_points = int(strings[4])
            self.volume = strings[5]
            self.loss_file = model_file.split('.h5')[0] + '.csv'
                    
        self.limits = eval(f'utils.limits_{self.pop}').copy()
        self.n_dim = len(self.limits['hyper'])
                
        self.design_file = design_file
        self.pop_file = \
            f'./populations/populations_{self.pop}_{self.n_points}.h5'
        
        with h5py.File(design_file, 'r') as h:
            self.hp = {par: h[par][:] for par in self.limits['hyper']}
            self._hp = [
                {par: h[par][sim] for par in self.limits['hyper']}
                for sim in range(h.attrs['n_sim'])
                ]

        with h5py.File(self.pop_file, 'r') as h:
            self.f_gen = {
                gen: vals[:] 
                for gen, vals in h['f_gen'][self.branch][self.volume].items()
                }
            
        #self.limits['f_gen'] = {gen: [0, 1] for gen in self.f_gen}
        
        if model_file is not None:
            try:
                self.model = tf.keras.models.load_model(model_file)
            except:
                if self.det:
                    n_neurons = int(strings[6])
                    n_layers = int(strings[7])
                    activation = strings[8]
                else:
                    n_neurons = int(strings[5])
                    n_layers = int(strings[6])
                    activation = strings[7]                    
                self.define_model(
                    n_neurons=n_neurons,
                    n_layers=n_layers,
                    activation=activation,
                    )
                self.model.load_weights(model_file)
                
    def __call__(self, inputs):

        return np.squeeze(self.model(inputs).numpy())
            
    def _input(self, hp):
        
        hp_unit = np.atleast_2d([
            utils.rescale_to_unit(np.atleast_1d(hp[par]), *lim)
            for par, lim in self.limits['hyper'].items()
            ]).T
        n_sim, n_hp = hp_unit.shape
        
        return hp_unit
    
    def predict(self, hp=None):
        
        if hp is None:
            hp = self.hp
            
        inputs = self._input(hp)
        n_sim, n_dim = inputs.shape
        pred = np.squeeze(self.model(inputs, training=False))
        
        return pred
    
    def define_model(
        self,
        n_neurons=128,
        n_layers=3,
        activation='rrelu',
        verbose=True,
        ):
        
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.activation = activation
        
        self.model = tf.keras.Sequential(
            [tf.keras.layers.InputLayer(input_shape=(self.n_dim,)),],
            )
        
        for n in range(n_layers):
            try:
                self.model.add(
                    tf.keras.layers.Dense(n_neurons, activation=activation)
                    )
            except:
                if activation == 'prelu':
                    _activation = tf.keras.layers.PReLU()
                else:
                    _activation = eval(f'tfa.activations.{activation}')
                self.model.add(
                    tf.keras.layers.Dense(n_neurons, activation=_activation)
                    )
                
        self.model.add(tf.keras.layers.Dense(len(self.f_gen), 'softmax'))
        
        if verbose:
            self.model.summary()
    
    def train(
        self,
        optimizer=tf.keras.optimizers.Adam,
        loss='mse',
        learning_rate=0.001,
        batch_size=None,
        batch_percent=1,
        epochs=2000,
        stop_kwargs=None,
        callbacks=[],
        ):
        
        data = self._preprocess()
        
        if batch_percent is not None:
            n_samples = data['train']['inputs'].shape[0]
            batch_size = max(int(n_samples * batch_percent / 100), 1)
            
        self.loss = loss
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.batch_percent = batch_percent
        self.epochs = epochs
        self.stop = False if stop_kwargs is None else True
        
        filename = f'./neural/neural_gen'
        if self.det:
            filename += '_det'
        filename += (
            f'_{self.pop}_{self.n_points}_{self.volume}'
            f'_{self.n_neurons}_{self.n_layers}_{self.activation}'
            f'_{loss}_{learning_rate}_{batch_percent}'
            )
        if os.path.exists(filename+'.h5') or os.path.exists(filename+'.csv'):
            filename += '_' + str(int(time.time()))
        self.model_file = filename + '.h5'
        self.loss_file = filename + '.csv'
        
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.model_file,
                monitor='val_loss',
                mode='min',
                save_weights_only=False,
                save_best_only=True,
                save_freq='epoch',
                verbose=0,
                ),
            tf.keras.callbacks.CSVLogger(self.loss_file),
            ]
        if self.stop:
            self.callbacks.append(
                tf.keras.callbacks.EarlyStopping(**stop_kwargs),
                )
        self.callbacks += callbacks
        
        self.model.compile(
            optimizer=optimizer(learning_rate=learning_rate), loss=loss,
            )
        
        history = self.model.fit(
            x=data['train']['inputs'],
            y=data['train']['outputs'],
            validation_data=(data['valid']['inputs'], data['valid']['outputs']),
            shuffle=True,
            batch_size=self.batch_size,
            epochs=epochs,
            callbacks=self.callbacks,
            )
        
        return history
    
    def _preprocess(self, split=0.1, seed=42):
        
        inputs = self._input(self.hp)
        n_sim, n_dim = inputs.shape
        outputs = np.array(list(self.f_gen.values())).T
        
        train, valid = utils.training_split(n_sim, split, seed)
        
        data = {}
        for train_valid in ['train', 'valid']:
            data[train_valid] = {}
            for in_out in ['inputs', 'outputs']:
                data[train_valid][in_out] = eval(in_out)[eval(train_valid)]
                
        return data
    
    
def train_generation():
    
    det = True
    pop = 'chipav'
    n_points = 21
    volume = 'horizon'
    
    n_neurons = 128
    n_layers = 3
    activation = 'rrelu'
    
    loss = 'mse'
    learning_rate = 0.001
    batch_percent = 1
    
    generation_model = GenerationModel(
        det=det, pop=pop, n_points=n_points, volume=volume,
        )
    
    generation_model.define_model(
        n_neurons=n_neurons, n_layers=n_layers, activation=activation,
        )
    
    t0 = time.time()
    history = generation_model.train(
        loss=loss, learning_rate=learning_rate, batch_percent=batch_percent,
        )
    dt = time.time() - t0
    print(dt/60.)
    
    print(generation_model.model_file, generation_model.loss_file)

    
# Hellinger distance
#-------------------------------------------------------------------------------    
    
def hellinger(p, q, discrete=False, axes=None):
    
    integrand = (np.sqrt(p) - np.sqrt(q))**2 / 2
    
    if discrete:
        return np.sqrt(np.sum(integrand, axis=-1))
    
#     i = utils.integrate_nd(np.sqrt(p*q), axes)
    
#     if mod:
#         return np.sqrt(1 - min(i, 2-i))
#     if squared:
#         return 1 - i    
#     return np.sqrt(1 - i)

    return np.sqrt(utils.integrate_nd(integrand, axes))


def compute_hellinger():
    
    norm = 'trapezoid'
    n_norm = 51
    
    model_file = \
        './neural/neural_pop_chipav_21_kde_128_5_rrelu_mae_0.0001_0.01.h5'
    population_model = PopulationModel(model_file=model_file)
    
    hds = []
    
    with h5py.File('./populations/populations_chipav_21.h5', 'r') as h:
        
        for sim, hp in enumerate(tqdm(population_model._hp)):
            
            true = h['p_pop']['kde'][sim]
            pred = population_model.predict(hp, norm=norm, n_norm=n_norm)
            
            hds.append(hellinger(
                true, pred, axes=list(population_model.ep.values()),
                ))
            
            np.save(f'./neural/hellinger_{n_norm}.npy', hds)

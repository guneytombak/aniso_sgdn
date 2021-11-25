import numpy as np
import matplotlib.pyplot as plt
import json

import wandb

class LinearRegression():
    def __init__(self, x=None, y=None, name='default'):
        self.x = x
        self.y = y
        self.name = name

        self.m = None
        self.c = None

    def _warn_xy(self):

        if self.x is None:
            print('x is None!')
        if self.y is None:
            print('y is None')

        if self.x is None or self.y is None:
            return False
        else:
            return True

    def del_initial(self, ratio=0.1):
        if ratio < 1.0:
            init_len = int(ratio*len(self.x))
        else:
            init_len = int(ratio)

        self.x = self.x[init_len:]
        self.y = self.y[init_len:]

    def calc(self):
        self._warn_xy()
        A = np.vstack([self.x, np.ones(len(self.x))]).T
        m, c = np.linalg.lstsq(A, self.y, rcond=None)[0]
        self.m, self.c = m, c
        return m, c

    def show(self):
        if self.m is None or self.c is None:
            print('Calculating...')
            self()

        plt.plot(self.x, self.y, 
                 'o', label='Original data', 
                 markersize=2)

        y_hat = self.m*self.x + self.c
        corr = np.corrcoef(self.x, self.y)

        plt.plot(self.x, y_hat, 'r', label='Fitted Line')
        plt.legend()
        title = self.name + \
                f" $\\rho:{corr[0,1]:.4f}$" + \
                f" m:{self.m:.4f}" + \
                f" c:{self.c:.4f}"
        plt.title(title)
        plt.show()
        

    def __call__(self, x=None, y=None):
        self.x = x if x is not None else self.x
        self.y = y if y is not None else self.y
        return self.calc()

class LinearRegression():
    def __init__(self, x=None, y=None, name='default'):
        self.x = x
        self.y = y
        self.name = name

        self.m, self.um, self.lm = None, None, None
        self.c, self.uc, self.lc = None, None, None

    def _warn_xy(self):

        if self.x is None:
            print('x is None!')
        if self.y is None:
            print('y is None')

        if self.x is None or self.y is None:
            return False
        else:
            return True

    def del_initial(self, ratio=0.1):
        if ratio < 1.0:
            init_len = int(ratio*len(self.x))
        else:
            init_len = int(ratio)

        self.x = self.x[init_len:]
        self.y = self.y[init_len:]


    def smooth(self, window_size=0.02, order=2, show=False):
        if self._warn_xy():
            if window_size < 1.0:
                window_size = int(window_size*len(self.x))
        if not (window_size % 2):
            window_size += 1
            
        x_s = savitzky_golay(self.x, window_size=window_size, order=order)
        y_s = savitzky_golay(self.y, window_size=window_size, order=order)

        if show:
            plt.plot(self.x, 'ob', markersize=2)
            plt.plot(x_s, '-b')

            plt.plot(self.y, 'or', markersize=2)
            plt.plot(y_s, '-r')

            plt.title('Smoothing of ' + self.name)
            plt.show()

        self.x, self.y = x_s, y_s


    def calc(self):
        if self._warn_xy():
            A = np.vstack([self.x, np.ones(len(self.x))]).T
            m, c = np.linalg.lstsq(A, self.y, rcond=None)[0]
            self.m, self.c = m, c
            return m, c
        else:
            return None

    def show(self, save=True):
        if self.m is None or self.c is None:
            print('Calculating...')
            self()

        if self.um is None:
            self.find_sup()

        plt.plot(self.x, self.y, 
                 'o', label='Original data', 
                 markersize=2)

        y_hat = self.m*self.x + self.c
        yu_hat = self.um*self.x + self.uc
        yl_hat = self.lm*self.x + self.lc
        
        corr = np.corrcoef(self.x, self.y)

        plt.plot(self.x, y_hat, 
                 'r', label='Fitted Line')
        plt.plot(self.x, yu_hat, 
                 '--k', label='Upper Line')
        plt.plot(self.x, yl_hat, 
                 '-.k', label='Lower Line')
        plt.legend()
        title = self.name + \
                f" $\\rho:{corr[0,1]:.4f}$" + \
                f" m:{self.m:.3f}" + \
                f" c:{self.c:.3f}"
        plt.title(title)

        plt.savefig('../results/figs/' + self.name + '.jpg')
        plt.show()

    def _get_mc(self, pair):

        x1 = self.x[pair[0]]
        y1 = self.y[pair[0]]

        x2 = self.x[pair[1]]
        y2 = self.y[pair[1]]

        m = (y2 - y1) / (x2 - x1)
        c = y1 - m*x1

        return m, c

    def _test_up(self, pair):

        eps = 1e-10

        m, c = self._get_mc(pair)
        func = lambda x, y: y - m*x - c

        if all(func(self.x, self.y) <= eps):
            return -1
        elif all (func(self.x, self.y) >= -eps):
            return +1
        else:
            return 0


    def find_sup(self):
        
        upper_pairs = list()
        lower_pairs = list()

        for k in range(len(self.x)):
            for q in range(len(self.x)):
                if q < k:

                    if self._test_up([q,k]) > 0.5:
                        upper_pairs.append([q,k])
                    elif self._test_up([q,k]) < -0.5:
                        lower_pairs.append([q,k])

        dists_upper = list()
        dists_lower = list()

        for pair in upper_pairs:

            x1 = self.x[pair[0]]
            y1 = self.y[pair[0]]

            x2 = self.x[pair[1]]
            y2 = self.y[pair[1]]

            dists_upper.append((x1-x2)**2 + (y1-y2)**2)

        for pair in lower_pairs:

            x1 = self.x[pair[0]]
            y1 = self.y[pair[0]]

            x2 = self.x[pair[1]]
            y2 = self.y[pair[1]]

            dists_lower.append((x1-x2)**2 + (y1-y2)**2)

        lower_pair = lower_pairs[np.argmax(dists_lower)]
        upper_pair = upper_pairs[np.argmax(dists_upper)]

        self.um, self.uc = self._get_mc(upper_pair)
        self.lm, self.lc = self._get_mc(lower_pair)
        

    def __call__(self, x=None, y=None):
        self.x = x if x is not None else self.x
        self.y = y if y is not None else self.y
        return self.calc()


def lu_anly(name, del_init=0.1, 
            smooth_size=0.05, smooth_order=1,
            show_smoothing=True):

    api = wandb.Api()
    run = api.run("guneytombak/aniso_sgdn/" + name)

    cfg = get_config(run)
    data = run.history()

    t = get_title(cfg)

    u = data['U'].to_numpy()
    l = data['D'].to_numpy()
    g = data['G'].to_numpy()

    loss = data['L'].to_numpy()
    #dh2 = data['Dh'].to_numpy()

    reg_l = LinearRegression(l, g, t+'lower')
    if del_init:
        reg_l.del_initial(del_init)
    if smooth_size:
        reg_l.smooth(smooth_size, smooth_order, show=show_smoothing)
    reg_l.show()
    
    reg_u = LinearRegression(loss, g, t+'loss')
    if del_init:
        reg_u.del_initial(del_init)
    if smooth_size:
        reg_u.smooth(smooth_size, smooth_order, show=show_smoothing)
    reg_u.show()

    return reg_l, reg_u

def get_config(run):

    dict_cfg = json.loads(run.json_config)
    cfg = dict()

    for key, value in dict_cfg.items():
        cfg[key] = value['value']

    return cfg


def get_title(cfg):

    t = ''

    if cfg['dataset_name'] == 'DataName.GRID':
        t += 'grid'
    elif cfg['dataset_name'] == 'DataName.ENERGY':
        t += 'energy'
    elif cfg['dataset_name'] == 'DataName.DIGITS':
        t += 'digits'    
    else:
        t += cfg['dataset_name']

    t += '_'

    if cfg['activ_type'] == 'ActivType.RELU':
        t += 'relu'
    elif cfg['activ_type'] == 'ActivType.SIGMOID':
        t += 'sigm'
    else:
        t += cfg['activ_type']

    t += '_'

    t += str(cfg['hidden_size'])

    t += '_'

    return t

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

import numpy as np
import matplotlib.pyplot as plt
from funcs import savitzky_golay

class LinearRegression():
    def __init__(self, x=None, y=None, name='default', 
                 save_name='../results/figs/'):
        self.x = x
        self.y = y
        self.name = name

        self.m = None
        self.c = None
        self.save_folder = save_name

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

        plt.savefig(self.save_folder + self.name + '.jpg')
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

import numpy as np
import math

class CSA:
    def __init__(self, init_mean, init_variance=1.0, popsize=None):
        # variables of the normal distribution
        self.mean = init_mean
        self.variance = init_variance

        # integer variables for population and dimensionality
        self.n = init_mean.shape[0]
        self.n_off = popsize or int(4 + math.floor(3 * math.log(self.n)))

        # selection weights
        self.weights = np.zeros(self.n_off)
        mu = self.n_off // 2
        for i in range(mu):
            self.weights[i] = math.log(mu + 0.5) - math.log(1. + i)
        self.weights /= np.sum(self.weights)
        self._mu_eff = 1.0 / np.sum(self.weights**2)

        # variables for CSA
        self._path = np.zeros(self.n)
        self._gamma_path = 0.0

        self.avg_loss = 0.0

    def ask(self):
        # generate offspring
        sigma = math.sqrt(self.variance)
        z = np.random.normal(np.zeros((self.n_off, self.n)))
        x = self.mean[np.newaxis, :] + sigma * z

        # evaluate offspring
        return x

    def tell(self, x, fvals):
        x, fvals = np.asarray(x), np.asarray(fvals)
        sigma = math.sqrt(self.variance)
        z = (x - self.mean) / sigma

        # store estimate for current loss
        self.avg_loss = np.mean(fvals)

        self._update(x, z, fvals)

    def _update(self, x, z, fvals):
        order = np.argsort(fvals)
        weights = np.zeros(self.n_off)
        for i in range(self.n_off):
            weights[order[i]] = self.weights[i]

        # compute individual learning-rates
        cPath = 2*(self._mu_eff + 2.)/(self.n + self._mu_eff + 5.)
        damping_path = cPath * 4.0 / (1 + cPath)

        # compute gradient of mean and normalized step-length
        new_mean = np.sum(weights[:, np.newaxis] * x, axis=0)
        step_z = np.sum(weights[:, np.newaxis] * z, axis=0)

        # update evolution-path
        self._path = (1-cPath) * self._path + math.sqrt(cPath * (2-cPath) * self._mu_eff) * step_z
        self._gamma_path = (1-cPath)**2 * self._gamma_path + cPath * (2-cPath)
        deviation_step_len = math.sqrt(np.mean(self._path**2)) - math.sqrt(self._gamma_path)

        # update evariables
        self.mean = new_mean
        self.variance *= np.exp(deviation_step_len*damping_path)

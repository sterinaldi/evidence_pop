import numpy as np
import ray
from figaro.mixture import DPGMM, HDPGMM
from figaro.utils import get_priors

# Remote actor for probability density reconstruction
@ray.remote
class worker_post:
    def __init__(self, bounds,
                       sigma   = None,
                       samples = None,
                       probit  = True,
                       scale   = None,
                       ):
        self.dim     = bounds.shape[-1]
        self.mixture = DPGMM(bounds, prior_pars = get_priors(bounds, samples = samples, std = sigma, scale = scale, probit = probit, hierarchical = False), probit = probit)
        self.samples = np.copy(samples)
        self.samples.setflags(write = True)

    def draw_sample(self):
        return self.mixture.density_from_samples(self.samples, make_comp = False)

# Remote actor for evidence inference
@ray.remote
class worker_evidence:
    def __init__(self, bounds,
                       out_folder  = '.',
                       hier_sigma  = None,
                       events      = None,
                       ):
        self.out_folder           = out_folder
        self.dim                  = 1
        self.bounds               = np.atleast_2d(bounds)
        self.mixture              = DPGMM(self.bounds, probit = False)
        self.hier_sigma           = hier_sigma,
        self.hierarchical_mixture = HDPGMM(self.bounds,
                                           probit     = False,
                                           prior_pars = get_priors(self.bounds,
                                                                   samples      = events,
                                                                   std          = hier_sigma,
                                                                   probit       = False,
                                                                   hierarchical = True,
                                                                   )
                                            )

    def run_Zi(self, pars):
        # Unpack data
        samples, name, n_draws = pars
        # Copying (issues with shuffling)
        ev = np.copy(samples)
        ev.setflags(write = True)
        # Actual inference
        prior_pars = get_priors(self.bounds, samples = ev, probit = False, hierarchical = False)
        self.mixture.initialise(prior_pars = prior_pars)
        draws      = [self.mixture.density_from_samples(ev, make_comp = False) for _ in range(n_draws)]
        return draws

    def draw_hierarchical(self):
        self.hierarchical_mixture.exp_sigma = self.hier_sigma/np.exp(np.random.uniform(np.log(1), np.log(20)))
        return self.hierarchical_mixture.density_from_samples(self.posteriors, make_comp = False)
    
    def load_posteriors(self, posteriors):
        self.posteriors = np.copy(posteriors)
        self.posteriors.setflags(write = True)
        for i in range(len(self.posteriors)):
            self.posteriors[i].setflags(write = True)

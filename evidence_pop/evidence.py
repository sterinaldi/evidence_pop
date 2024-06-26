import numpy as np
import optparse
import warnings
import ray

from pathlib import Path
from tqdm import tqdm
from ray.util import ActorPool

from figaro.utils import save_options, load_options
from figaro.plot import plot_median_cr, plot_multidim
from figaro.load import save_density, load_density

from evidence_pop.utils import load_data, plot_evidence, save_evidence
from evidence_pop.workers import worker_post, worker_evidence

def main():
    parser = optparse.OptionParser(prog = 'evidence', description = 'Inference of Bayesian evidence')
    # Input/output
    parser.add_option("-s", "--samples", type = "string", dest = "samples_file", help = "File with samples", default = None)
    parser.add_option("-l", "--logP", type = "string", dest = "logp_file", help = "File with log probability", default = None)
    parser.add_option("-b", "--bounds", type = "string", dest = "bounds", help = "Density bounds. Must be a string formatted as '[[xmin, xmax], [ymin, ymax],...]'. For 1D distributions use '[xmin, xmax]'. Quotation marks are required and scientific notation is accepted", default = None)
    parser.add_option("-o", "--output", type = "string", dest = "output", help = "Output folder. Default: same directory as samples", default = None)
    parser.add_option("--logz", dest = "logz", help = "log evidence value (if known)", default = None)
    # Plot
    parser.add_option("-p", "--postprocess", dest = "postprocess", action = 'store_true', help = "Postprocessing", default = False)
    parser.add_option("--symbol", type = "string", dest = "symbol", help = "LaTeX-style quantity symbol, for plotting purposes", default = None)
    parser.add_option("--unit", type = "string", dest = "unit", help = "LaTeX-style quantity unit, for plotting purposes", default = None)
    # Settings
    parser.add_option("-r", "--skip_reconstruction", dest = "skip_reconstruction", action = 'store_true', help = "Skip posterior reconstruction", default = False)
    parser.add_option("-z", "--skip_samples_z", dest = "skip_samples_z", action = 'store_true', help = "Skip Zi reconstruction", default = False)
    parser.add_option("--save_posterior", dest = "save_posterior", action = 'store_true', help = "Save plot for posterior probability density", default = False)
    parser.add_option("--draws_p", type = "int", dest = "draws_p", help = "Number of draws for posterior reconstruction", default = 1000)
    parser.add_option("--draws_zi", type = "int", dest = "draws_zi", help = "Number of draws for individual Z_i", default = 1)
    parser.add_option("--draws", type = "int", dest = "draws", help = "Number of draws for Z", default = 1000)
    parser.add_option("-n", "--n_post_samples", type = "int", dest = "n_post_samples", help = "Number of samples to use for the inference", default = 200)
    parser.add_option("--n_samples_dsp", type = "int", dest = "n_samples_dsp", help = "Number of samples to analyse (downsampling). Default: all", default = -1)
    parser.add_option("--sigma_prior", dest = "sigma_prior", type = "string", help = "Expected standard deviation (prior) - single value or n-dim values", default = None)
    parser.add_option("--fraction", dest = "fraction", type = "float", help = "Fraction of samples standard deviation for sigma prior. Overrided by sigma_prior.", default = 3.)
    parser.add_option("--no_probit", dest = "probit", action = 'store_false', help = "Disable probit transformation for posterior reconstruction", default = True)
    parser.add_option("--config", dest = "config", type = "string", help = "Config file. Warning: command line options override config options", default = None)
    parser.add_option("--n_parallel", dest = "n_parallel", type = "int", help = "Number of parallel threads", default = 1)

    (options, args) = parser.parse_args()

    if options.config is not None:
        options = load_options(options, parser)
    # Paths
    if options.samples_file is not None:
        options.samples_file = Path(options.samples_file).resolve()
    else:
        raise Exception("Please provide path to samples.")
    if options.logp_file is not None:
        options.logp_file = Path(options.logp_file).resolve()
    else:
        raise Exception("Please provide path to log probability.")
    if options.output is not None:
        options.output = Path(options.output).resolve()
        if not options.output.exists():
            options.output.mkdir(parents=True)
    else:
        options.output = options.samples_file.parent
    if options.config is not None:
        options.config = Path(options.config).resolve()
    if options.config is None:
        save_options(options, options.output)

    # Read bounds
    if options.bounds is not None:
        options.bounds = np.array(np.atleast_2d(eval(options.bounds)), dtype = np.float64)
    elif options.bounds is None and not (options.skip_reconstruction or options.skip_samples_z or options.postprocess):
        raise Exception("Please provide sampling bounds (use -b '[[xmin,xmax],[ymin,ymax],...]')")
    if options.sigma_prior is not None:
        options.sigma_prior = np.array([float(s) for s in options.sigma_prior.split(',')])
    
    # Load samples
    samples, logP, name = load_data(options.samples_file, options.logp_file, n_samples = options.n_samples_dsp)
    try:
        dim = np.shape(samples)[-1]
    except IndexError:
        dim = 1
    
    # Log Z (if provided)
    if options.logz is not None:
        options.logz = eval(options.logz)
    
    if not options.postprocess:
        ray.init(num_cpus = options.n_parallel)
    
    # Normalised posterior probability density reconstruction
    if not (options.skip_reconstruction or options.skip_samples_z or options.postprocess):
        pool = ActorPool([worker_post.remote(bounds  = options.bounds,
                                             sigma   = options.sigma_prior,
                                             scale   = options.fraction,
                                             samples = samples,
                                             probit  = options.probit,
                                             )
                          for _ in range(options.n_parallel)])
        draws = []
        for s in tqdm(pool.map_unordered(lambda a, v: a.draw_sample.remote(), [_ for _ in range(options.draws_p)]), total = options.draws_p, desc = 'Posterior'):
            draws.append(s)
        draws = np.array(draws)
        save_density(draws, folder = options.output, name = 'draws_'+name, ext = 'json')
    else:
        draws = load_density(Path(options.output, 'draws_'+name+'.json'), make_comp = False)

    # Plot posterior distribution P(x)
    if options.save_posterior:
        # Plot
        if dim == 1:
            plot_median_cr(draws, samples = samples, out_folder = options.output, name = name, label = options.symbol, unit = options.unit)
        else:
            if options.symbol is not None:
                symbols = options.symbol.split(',')
            else:
                symbols = options.symbol
            if options.unit is not None:
                units = options.unit.split(',')
            else:
                units = options.unit
            plot_multidim(draws, samples = samples, out_folder = options.output, name = name, labels = symbols, units = units)
    
    # Evidence inference
    if not (options.skip_samples_z or options.postprocess):
        # Random sampling from samples within 68th percentile
        samples   = samples[np.argsort(logP)][::-1]
        logP      = np.sort(logP)[::-1]
        idx       = np.random.choice(int(len(samples)*0.5), np.min([int(len(samples)*0.5), options.n_post_samples]), replace = False)
        samples_Z = np.array([logP[idx] - d.logpdf(samples[idx]) for d in tqdm(draws, desc = 'Evaluating Zi')]).T
        np.savetxt(Path(options.output, 'samples_Z.txt'), samples_Z)
    else:
        samples_Z = np.genfromtxt(Path(options.output, 'samples_Z.txt'))
    if not options.postprocess:
        sigma_Z   = np.std(np.median(samples_Z, axis = 1))
        bounds_Z  = np.atleast_2d([np.median(samples_Z) - 5*sigma_Z, np.median(samples_Z) + 5*sigma_Z])
        pool = ActorPool([worker_evidence.remote(bounds     = bounds_Z,
                                                 out_folder = options.output,
                                                 hier_sigma = sigma_Z,
                                                 events     = samples_Z,
                                                 )
                      for _ in range(options.n_parallel)])
    
        if not options.skip_samples_z:
            # Run each single-event analysis
            posteriors = []
            for s in tqdm(pool.map_unordered(lambda a, v: a.run_Zi.remote(v), [[Zi, str(i), options.draws_zi] for i, Zi in enumerate(samples_Z)]), total = len(samples_Z), desc = 'Reconstructing Zi'):
                posteriors.append(s)
            # Save all single-event draws together
            posteriors = np.array(posteriors)
            save_density(posteriors, folder = options.output, name = 'realisations_Zi', ext = 'json')
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                posteriors = load_density(Path(options.output, 'realisations_Zi.json'), make_comp = False)
        # Load posteriors
        for s in pool.map(lambda a, v: a.load_posteriors.remote(v), [posteriors for _ in range(options.n_parallel)]):
            pass
        # Run hierarchical analysis
        draws = []
        for s in tqdm(pool.map_unordered(lambda a, v: a.draw_hierarchical.remote(), [_ for _ in range(options.draws)]), total = options.draws, desc = 'Inferring Z'):
            draws.append(s)
        draws = np.array(draws)
        # Save draws
        save_density(draws, folder = options.output, name = 'draws_evidence', ext = 'json')
    else:
        draws = load_density(Path(options.output, 'draws_evidence.json'))
    # Plot
    plot_evidence(draws, out_folder = options.output, logZ = options.logz)
    # Save evidence value
    save_evidence(draws, out_folder = options.output)

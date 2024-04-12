import numpy as np
from pathlib import Path
from figaro.load import load_single_event
from figaro.plot import plot_median_cr, plot_1d_dist
from figaro.utils import rvs_median

def load_data(samples_file, logP_file, n_samples = -1):
    '''
    Load posterior samples from .txt/.dat files.
    Samples are sorted in descending order according to logP.
    
    Arguments:
        str or Path samples_file: file with samples
        str or Path logP_file:    file with evaluated logL + logPrior
        int n_samples:    number of samples for (random) downsampling. Default -1: all samples
        
    Returns:
        np.ndarray: samples
        np.ndarray: logP
        np.ndarray: name
    '''
    samples, name = load_single_event(samples_file, n_samples = -1)
    logP          = np.genfromtxt(logP_file)
    samples       = samples[np.argsort(logP)][::-1]
    logP          = np.sort(logP)[::-1]
    if n_samples == -1:
        return samples, logP, name
    else:
        idx = np.random.choice(len(samples), size = np.min([len(samples), n_samples]), replace = False)
        return samples[idx], logP[idx], name

def significative_digits(vals):
    """
    Compute the larges numer of significative digits among an array of values
    
    Arguments:
        np.ndarray vals: values
    
    Returns:
        float: number of significative digits
    """
    vals      = np.atleast_1d(vals)
    sig_digit = np.array([int(-np.floor(np.log10(np.abs(v)))) for v in vals])
    prec      = int(np.max(sig_digit))+1
    return prec

def plot_evidence(draws, bounds = None, out_folder = '.', logZ = None):
    """
    Mahe the plots for evidence and log evidence
    
    Arguments:
        iterable draws:         container for realisations
        np.ndarray bounds:      Z plot boundaries
        str or Path out_folder: output folder
        float logZ:             true logZ value (if known)
    """
    if logZ is not None:
        true_Z = np.exp(logZ)
    else:
        true_Z = None
    if bounds is None:
        bounds = np.exp(draws[0].bounds[0])
    else:
        bounds = np.atleast_1d(bounds)
    # Z
    try:
        Z       = np.linspace(bounds[0], bounds[1], 1000)
        draws_Z = np.array([d.pdf(np.log(Z))/Z for d in draws])
        plot_1d_dist(Z,
                     draws_Z,
                     out_folder       = out_folder,
                     name             = 'evidence',
                     label            = 'Z',
                     median_label     = '\\mathrm{(H)DPGMM}',
                     true_value       = true_Z,
                     true_value_label = 'Z_\\mathrm{true}',
                     )
        bounds = np.log(bounds)
    except:
        bounds = draws[0].bounds[0]
    # logZ
    plot_median_cr(draws,
                   bounds           = bounds,
                   out_folder       = out_folder,
                   name             = 'log_evidence',
                   label            = '\\log{\\mathcal{Z}}',
                   hierarchical     = True,
                   true_value       = logZ,
                   true_value_label = '\\log{\\mathcal{Z}}_\\mathrm{true}'
                   )
    Path(out_folder, 'observed_log_evidence.pdf').rename(Path(out_folder, 'log_evidence.pdf'))
    Path(out_folder, 'log_observed_log_evidence.pdf').unlink()

def save_evidence(draws, out_folder = '.'):
    """
    Save evidence value to .txt file
    
    Arguments:
        iterable draws:         container for realisations
        str or Path out_folder: output folder
    """
    with open(Path(out_folder, 'evidence.txt'), 'w') as f:
        # Draw samples
        logZ_samples = rvs_median(draws, size = 10000)
        # logZ
        logZ_med, logZ_up, logZ_down = np.percentile(logZ_samples, [50, 84, 16])
        prec = significative_digits([logZ_up-logZ_med, logZ_med-logZ_down])
        f.write('logZ = {:.{prec}f} + {:.{prec}f} - {:.{prec}f}\n'.format(logZ_med, logZ_up-logZ_med, logZ_med-logZ_down, prec = prec))
        try:
            # Z
            Z_med, Z_up, Z_down = np.percentile(np.exp(logZ_samples), [50, 84, 16])
            prec = significative_digits([Z_up-Z_med, Z_med-Z_down])
            f.write('Z = {:.{prec}f} + {:.{prec}f} - {:.{prec}f}\n'.format(Z_med, Z_up-Z_med, Z_med-Z_down, prec = prec))
        # Prevent underflows
        except:
            pass

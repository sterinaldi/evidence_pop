import numpy as np
from figaro.load import load_single_event

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

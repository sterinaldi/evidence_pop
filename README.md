# evidence_pop

Parallelised non-parametric inference of evidence using posterior samples. This code implements the method described in [Rinaldi et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024arXiv240507504R/abstract). In a nutshell, the inference scheme follows these steps:

1. Approximate $P(x)$ with a DPGMM;
2. Evaluate $p(Z(x_i))$ for different samples $x_i$ and approximate each of these with a DPGMM;
3. Reconstruct $p(Z)$ using (H)DPGMM.

The (H)/DPGMM reconstructions are made using [FIGARO](https://github.com/sterinaldi/FIGARO) ([Rinaldi & Del Pozzo 2024](https://joss.theoj.org/papers/10.21105/joss.06589)).

## Getting started
You can install `evidence_pop` from this repository:
```
git clone git@github.com:sterinaldi/evidence_pop.git
cd evidence_pop
pip install .
```
`evidence_pop` is meant to be used via its CLI, `evidence`. The complete list of options can be displayed with `evidence -h`: here we summarise the most important ones:

* `-s path/to/samples.txt`: file storing the posterior samples;
* `-l path/to/log_p.txt`: file storing the evaluated $\log\Pi(x_i) + \log\mathcal{L}(x_i)$ values;
* `-b "[[xmin, xmax],[ymin, ymax], ...]"`: domain bounds for $P(x)$ (remember the quotation marks);
* `--n_parallel N`: use N parallel threads for the analysis
* `-r`: skip the $P(x)$ reconstruction (if already done);
* `-z`: skip the individual $p(Z(x_i))$ reconstructons (if already done);
* `-p`: postprocessing.

The basic instruction for running the evidence inference from start to finish looks somewhat like this

## Acknowledgments
If you use this code in your research, please cite [Rinaldi et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024arXiv240507504R/abstract):
```
@ARTICLE{2024arXiv240507504R,
       author = {{Rinaldi}, Stefano and {Demasi}, Gabriele and {Del Pozzo}, Walter and {Hannuksela}, Otto A.},
        title = "{Hierarchical inference of evidence using posterior samples}",
      journal = {arXiv e-prints},
     keywords = {Statistics - Methodology, 62F15, 62C10},
         year = 2024,
        month = may,
          eid = {arXiv:2405.07504},
        pages = {arXiv:2405.07504},
          doi = {10.48550/arXiv.2405.07504},
archivePrefix = {arXiv},
       eprint = {2405.07504},
 primaryClass = {stat.ME},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv240507504R},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

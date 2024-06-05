# evidence_pop

Parallelised non-parametric inference of evidence $Z$ using posterior samples. This code implements the method described in [Rinaldi et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024arXiv240507504R/abstract). In a nutshell, the inference scheme follows these steps:

1. Approximate the posterior distribution $P(x)$ with a DPGMM;
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

* `-s path/to/samples.txt`: file storing the posterior samples $x_i$;
* `-l path/to/log_p.txt`: file storing the evaluated $\log\Pi(x_i) + \log\mathcal{L}(x_i)$;
* `-b "[[xmin, xmax],[ymin, ymax], ...]"`: domain bounds for $P(x)$ (remember the quotation marks);
* `--n_parallel N`: use N parallel threads for the analysis
* `-r`: skip the $P(x)$ reconstruction (if already done);
* `-z`: skip the individual $p(Z(x_i))$ reconstructons (if already done);
* `-p`: postprocessing;
* `--save_posterior`: produce a plot with the inferred $P(x)$ (it may take a long time for more than 3 dimensions). 

Assuming that the folder you are working in is 
```
folder
├─ samples.txt
└─ log_p.txt
```
the basic instruction for running the evidence inference from start to finish looks somewhat like this:
```
evidence -s samples.txt -l log_p.txt -b "[[0,1],[0,1]]"
```
Once the run is finished, the folder will look like this:
```
folder
├─ draws_evidence.json
├─ draws_samples.json
├─ evidence.pdf (optional)
├─ evidence.txt
├─ log_evidence.txt
├─ options.ini
├─ prob_evidence.txt (optional)
├─ prob_log_evidence.txt
├─ realisations_Zi.json
├─ samples_Z.txt
├─ samples.pdf (optional)
├─ samples.txt
└─ log_p.txt
```

Among these files, the most important ones are:

* `log_evidence.pdf`: plot showing the posterior distribution for $Z$;
* `evidence.txt`: stores the expected value and credible interval for $Z$;
* `draws_evidence.json`: stores the FIGARO reconstructions for $p(Z)$ (see [here](https://figaro.readthedocs.io/en/latest/use_mixture.html) for a tutorial on how to use them).

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

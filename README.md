# Functionally-informed Z-score Imputation (FIZI)
FIZI leverages functional information together with reference linkage-disequilibrium (LD) to
impute GWAS summary statistics (Z-score).

This README is a working draft and will be expanded soon.

[//]: # (This repository serves as the home for the python implementation of the algorithm described in XX.)

Installation
----
1. First grab the latest version of FIZI using git as

    `git clone https://github.com/bogdanlab/fizi`
    
2. FIZI can be installed using setuptools as 

    `python setup.py install --user` or optionally as
    
    `sudo python setup.py install` if you have root access and wish to install for all users
    
3. Check that FIZI was installed by typing

    `fizi.py --help`

4. If that did not work, and `--user` was specified, please check that your local user path is included in
`$PATH` environment variable. `--user` is typically defined as `.local/bin` and can be appended to `$PATH`
by executing

    `export PATH=~/.local/bin:$PATH`
    
    which can be saved in `.bashrc` or `.bash_profile`

Incorporating functional data to improve summary statistics imputation
-----
Usage consists of several steps. We outline the general workflow here when the intention to perform imputation on
chromosome 1 of our data:

1. Munge/clean _all_ GWAS summary data before imputation
2. Partitioning cleaned GWAS summary data into chr1 and everything else (loco-chr1).
3. Run LDSC on locoChr to obtain tau estimates
4. Perform functionally-informed imputation on chr1 data using tau estimates from loco-chr

Imputing summary statistics using only reference LD
------
When functional annotations and LDSC estimates are not provided to FIZI, it will fallback to the classic ImpG
algorithm described in ref[1]. To impute missing summary statistics using the ImpG algorithm simply enter the
command 

    fimpy.py cleaned.gwas.sumstat.gz --chr 1 --out imputed.cleaned.gwas.sumstat


Software and support
-----
If you have any questions or comments please contact nmancuso@mednet.ucla.edu and/or meganroytman@gmail.com

For performing various inferences using summary data from large-scale GWASs please find the following useful software:

1. Association between predicted expression and complex trait/disease [FUSION](https://github.com/gusevlab/fusion_twas)
2. Estimating local heritability or genetic correlation [HESS](https://github.com/huwenboshi/hess)
3. Estimating genome-wide heritability or genetic correlation [UNITY](https://github.com/bogdanlab/UNITY)
4. Fine-mapping using summary-data [PAINTOR](https://github.com/gkichaev/PAINTOR_V3.0)

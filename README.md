# Functionally-informed Z-score Imputation (FIZI)
FIZI leverages functional information together with reference linkage-disequilibrium (LD) to
impute GWAS summary statistics (Z-score).

This README is a working draft and will be expanded soon.

[//]: # (This repository serves as the home for the python implementation of the algorithm described in XX.)

Installation
----
The easiest way to install `fizi` and `pyfizi` is through conda and conda-forge:

    conda config --add channels conda-forge
    conda install pyfizi
    
Alternatively you can use pip for installation:

    pip install pyfizi
    
Check that FIZI was installed by typing

    fizi --help

If that did not work, and `pip install pyfizi --user` was specified, please check that your local user path is included in
`$PATH` environment variable. `--user` location and can be appended to `$PATH`
by executing

    export PATH=`python -m site --user-base`/bin/:$PATH
    
which can be saved in `~/.bashrc` or `~/.bash_profile`. To reload the environment type `source ~/.bashrc` or `source ~/.bash_profile` depending where you entered it.

*We currently only support Python3.6+. [Python2.7 and below is not supported](https://pythonclock.org/)*

Overview
--------
`fizi` has two main functions: `munge` and `impute`. The `munge` subcommand is a pruned down version of the LDSC munge_sumstats software with a few bells and whistles needed for our imputation algorithm. The `impute` subcommand performs summary statistic imputation using either the functionally informed algorithm (i.e. `fizi`) or using only reference-LD-only algorithm (i.e. ImpG). For a full list of features please refer to the help command: `fizi munge -h` or `fizi impute -h`. 

Imputing summary statistics using only reference LD
------
When functional annotations and LDSC estimates are not provided to `fizi`, it will fallback to the classic ImpG
algorithm described in ref [1]. To impute missing summary statistics only for chromosome 1 using the ImpG algorithm 
simply enter the commands

    1. fizi munge gwas.sumstat.gz --out cleaned.gwas
    2. fizi impute cleaned.gwas.sumstat.gz plink_data_path --chr 1 --out imputed.cleaned.gwas.chr1.sumstat

By default `fizi` requires that at least 50% of SNPs to be observed for imputation at a region. This can be changed with the `--min-prop PROP` flag in step 2.

Incorporating functional data to improve summary statistics imputation
-----
Usage consists of several steps. We outline the general workflow here when the intention to perform imputation on
chromosome 1 of our data:

1. Munge/clean _all_ GWAS summary data before imputation

    `fizi munge gwas.sumstat.gz --out cleaned.gwas`

2. Partitioning cleaned GWAS summary data into chr1 and everything else (loco-chr1).
3. Run LDSC on locoChr to obtain tau estimates
4. Perform functionally-informed imputation on chr1 data using tau estimates from loco-chr

Software and support
-----
If you have any questions or comments please contact nicholas.mancuso@med.usc.edu and/or meganroytman@gmail.com

For performing various inferences using summary data from large-scale GWASs please find the following useful software:

1. Association between predicted expression and complex trait/disease [FUSION](https://github.com/gusevlab/fusion_twas)
2. Estimating local heritability or genetic correlation [HESS](https://github.com/huwenboshi/hess)
3. Estimating genome-wide heritability or genetic correlation [UNITY](https://github.com/bogdanlab/UNITY)
4. Fine-mapping using summary-data [PAINTOR](https://github.com/gkichaev/PAINTOR_V3.0)

[1]: https://academic.oup.com/bioinformatics/article/30/20/2906/2422225

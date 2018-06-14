import pandas as pd

from pysnptools.snpreader import Bed


__all__ = ["parse_plink", "write_output"]

def parse_plink(ref):
    pass

def write_output(imputed_gwas, output, append=False):
    imputed_gwas.to_csv(output, sep="\t", mode="a" if append else "w", header=not append)
    return

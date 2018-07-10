import pandas as pd

__all__ = ["get_compression", "write_output"]


def get_compression(fh):
    # This function from LDSC regression
    # (c) 2014 Brendan Bulik-Sullivan and Hilary Finucane
    '''Which sort of compression should we use with read_csv?'''
    if fh.endswith('gz'):
        compression = 'gzip'
    elif fh.endswith('bz2'):
        compression = 'bz2'
    else:
        compression = None

    return compression


def write_output(imputed_gwas, output, append=False):
    imputed_gwas.to_csv(output, sep="\t", mode="a" if append else "w", header=not append, index=False)
    return

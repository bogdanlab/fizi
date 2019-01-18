import os
import pandas as pd

__all__ = ["get_compression", "write_output"]


def get_compression(fh):
    """
    Get a string object that encodes what compression algorithm to use.
    gz = gzip
    bz2 = bz2
    otherwise None

    :param fh: File object or string containing the file path

    :return: string object encoding the compression algorithm.
    """
    # This function from LDSC regression
    # (c) 2014 Brendan Bulik-Sullivan and Hilary Finucane
    """Which sort of compression should we use with read_csv?"""
    if hasattr(fh, "name"):
        _, ext = os.path.splitext(fh.name)
    elif isinstance(fh, str):
        _, ext = os.path.splitext(fh)
    else:
        raise ValueError("get_compression: argument must be file handle or path")

    if ext.endswith('gz'):
        compression = 'gzip'
    elif ext.endswith('bz2'):
        compression = 'bz2'
    else:
        compression = None

    return compression


def write_output(imputed_gwas, output, append=False):
    """
    Write GWAS data to file in tab-delimited format

    :param imputed_gwas: pandas.DataFrame object containing GWAS data
    :param output: file-object or path to write file to
    :param append: bool indicated whether or not to append to the file

    :return: None
    """
    with pd.option_context('display.float_format', '{:.3}'.format):
        imputed_gwas.to_csv(output, sep="\t", mode="a" if append else "w", header=not append, index=False)

    return

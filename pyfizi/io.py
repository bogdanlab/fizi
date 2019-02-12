import os
import pandas as pd

__all__ = ["write_output"]


def write_output(imputed_gwas, output, append=False):
    """
    Write GWAS data to file in tab-delimited format

    :param imputed_gwas: pandas.DataFrame object containing GWAS data
    :param output: file-object or path to write file to
    :param append: bool indicated whether or not to append to the file

    :return: None
    """
    imputed_gwas.to_csv(output, sep="\t", mode="a" if append else "w", header=not append, index=False,
                        float_format="%.3g")

    return

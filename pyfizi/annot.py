import numpy as np
import pandas as pd

import pyfizi


class AnnotSeries(pd.Series):
    @property
    def _constructor(self):
        return pyfizi.AnnotSeries

    @property
    def _constructor_expanddim(self):
        return pyfizi.Annot


class Annot(pd.DataFrame):
    """
    Thin wrapper for a pandas DataFrame object containing annotation data.
    Assumes the annotation data to be in LDSC format.
    """

    CHRCOL = "CHR"
    SNPCOL = "SNP"
    BPCOL = "BP"
    CMCOL = "CM"

    BASE_NAME = "base"

    REQ_COLS = [CHRCOL, SNPCOL, BPCOL]

    def __init__(self, *args, **kwargs):
        super(Annot, self).__init__(*args, **kwargs)
        return

    @property
    def _constructor(self):
        return pyfizi.Annot

    @property
    def _constructor_sliced(self):
        return pyfizi.AnnotSeries

    @property
    def names(self):
        flag = self.columns.map(lambda x: x not in [Annot.CHRCOL, Annot.SNPCOL, Annot.BPCOL, Annot.CMCOL]).values
        return self.columns[flag.astype(bool)]

    def get_matrix(self, snps, names=None):
        """
        Get the binary annotation matrix for the SNPs found in snps data.

        :param snps: pandas.DataFrame containing SNP info (must have `SNP` column)
        :param names: list or numpy.ndarray containing the names of the annotations to look up
        :return: numpy.ndarray 0/1 matrix indicating which annotations the SNPs fall in
        """
        snps = pd.merge(snps, self, how="left", left_on=pyfizi.RefPanel.SNPCOL, right_on=Annot.SNPCOL)
        if names is None:
            names = self.names

        A = snps[names].values
        A[np.isnan(A)] = 0

        return A

    def subset_by_pos(self, chrom, start, stop):
        if pd.api.types.is_string_dtype(self[Annot.CHRCOL]) or pd.api.types.is_categorical_dtype(self[Annot.CHRCOL]):
            chrom = str(chrom)
        else:
            chrom = int(chrom)

        if start is not None and stop is not None:
            snps = self.loc[(self[Annot.CHRCOL] == chrom) & (self[Annot.BPCOL] >= start) & (self[Annot.BPCOL] <= stop)]
        elif start is not None and stop is None:
            snps = self.loc[(self[Annot.CHRCOL] == chrom) & (self[Annot.BPCOL] >= start)]
        elif start is None and stop is not None:
            snps = self.loc[(self[Annot.CHRCOL] == chrom) & (self[Annot.BPCOL] <= stop)]
        else:
            snps = self.loc[(self[Annot.CHRCOL] == chrom)]

        return Annot(snps)

    @classmethod
    def parse_annot(cls, stream, names=None):
        dtype_dict = {'SNP': str, 'BP': int}
        df = pd.read_csv(stream, dtype=dtype_dict, delim_whitespace=True, compression='infer')
        for column in Annot.REQ_COLS:
            if column not in df:
                raise ValueError("{}-column not found in annotation file".format(column))

        # if user-specified only a subset of annotations to be included
        if names is not None:
            if 'base' not in names:
                names = ['base'] + names
                for name in names:
                    if name not in df:
                        raise ValueError("{}-column not found in annotation file".format(name))
                columns = Annot.REQ_COLS + names
                df = df[columns]

        return cls(df)

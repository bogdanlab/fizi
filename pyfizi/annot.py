import pandas as pd

import fizi


class AnnotSeries(pd.Series):
    @property
    def _constructor(self):
        return fizi.AnnotSeries

    @property
    def _constructor_expanddim(self):
        return fizi.Annot


class Annot(pd.DataFrame):
    """
    Thin wrapper for a pandas DataFrame object containing annotation data.
    Assumes the annotation data to be in LDSC format.
    """

    CHRCOL = "CHR"
    SNPCOL = "SNP"
    BPCOL = "BP"
    CMCOL = "CM"

    REQ_COLS = [CHRCOL, SNPCOL, BPCOL]

    def __init__(self, *args, **kwargs):
        super(Annot, self).__init__(*args, **kwargs)
        return

    @property
    def _constructor(self):
        return fizi.Annot

    @property
    def _constructor_sliced(self):
        return fizi.AnnotSeries

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
    def parse_annot(cls, stream):
        dtype_dict = {'SNP': str, 'BP': int}
        cmpr = fizi.get_compression(stream)
        df = pd.read_csv(stream, dtype=dtype_dict, delim_whitespace=True, compression=cmpr)
        for column in Annot.REQ_COLS:
            if column not in df:
                raise ValueError("{}-column not found in annotation file".format(column))

        return cls(df)

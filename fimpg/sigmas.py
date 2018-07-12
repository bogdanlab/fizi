import pandas as pd
import scipy.stats as stats

import fimpg


class SigmasSeries(pd.Series):
    @property
    def _constructor(self):
        return fimpg.SigmasSeries

    @property
    def _constructor_expanddim(self):
        return fimpg.Sigmas


class Sigmas(pd.DataFrame):
    """
    Thin wrapper for a pandas DataFrame object containing LDSC tau-estimates.
    """

    NAMECOL = "Category"
    SIGMACOL = "Coefficient"
    SIGMASECOL = "Coefficient_std_error"
    SIGMAZCOL = "Coefficient_z-score"

    REQ_COLS = [NAMECOL, SIGMACOL, SIGMASECOL, SIGMAZCOL]

    def __init__(self, *args, **kwargs):
        super(Sigmas, self).__init__(*args, **kwargs)
        return

    @property
    def _constructor(self):
        return fimpg.Sigmas

    @property
    def _constructor_sliced(self):
        return fimpg.SigmasSeries

    def subset_by_pvalue(self, pvalue, keep_baseline=True):
        zscores = self[Sigmas.SIGMAZCOL].values
        pval_flag = 2 * stats.norm.sf(zscores) < pvalue
        if keep_baseline:
            sigmas = self.loc[(self[Sigmas.NAMECOL]=="base") | (pval_flag)]
        else:
            sigmas = self.loc[pval_flag]

        return Sigmas(sigmas)

    @classmethod
    def parse_sigmas(cls, stream):
        dtype_dict = {'Category': str}
        cmp = fimpg.get_compression(stream)
        df = pd.read_csv(stream, dtype=dtype_dict, delim_whitespace=True, compression=cmp)
        for column in Sigmas.REQ_COLS:
            if column not in df:
                raise ValueError("{}-column not found in LDSC tau-estimates file".format(column))

        # remove L2_0 from variable names
        df[Sigmas.NAMECOL] = df[Sigmas.NAMECOL].str.replace("L2_0", "")

        return cls(df)

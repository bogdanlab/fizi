import pandas as pd
import numpy as np
import scipy.stats as stats

import pyfizi


class TausSeries(pd.Series):
    @property
    def _constructor(self):
        return pyfizi.TausSeries

    @property
    def _constructor_expanddim(self):
        return pyfizi.Taus


class Taus(pd.DataFrame):
    """
    Thin wrapper for a pandas DataFrame object containing LDSC tau-estimates.
    """

    NAMECOL = "Category"
    TAUCOL = "Coefficient"
    TAUSECOL = "Coefficient_std_error"
    TAUZCOL = "Coefficient_z-score"
    ENRICHP = "Enrichment_p"

    REQ_COLS = [NAMECOL, TAUCOL, TAUSECOL, TAUZCOL, ENRICHP]

    def __init__(self, *args, **kwargs):
        super(Taus, self).__init__(*args, **kwargs)
        return

    @property
    def _constructor(self):
        return pyfizi.Taus

    @property
    def _constructor_sliced(self):
        return pyfizi.TausSeries

    @property
    def estimates(self):
        return self[Taus.TAUCOL].values

    @property
    def std_errors(self):
        return self[Taus.TAUSECOL].values

    @property
    def names(self):
        return self[Taus.NAMECOL].values.flatten()

    def subset_by_tau_pvalue(self, pvalue, keep_baseline=True):
        zscores = self[Taus.TAUZCOL].values
        pval_flag = 2 * stats.norm.sf(zscores) < pvalue
        if keep_baseline:
            taus = self.loc[(self[Taus.NAMECOL] == "base") | pval_flag]
        else:
            taus = self.loc[pval_flag]

        return Taus(taus)

    def subset_by_enrich_pvalue(self, pvalue, keep_baseline=True):
        pvalues = self[Taus.ENRICHP].values
        pvalues[np.isnan(pvalues)] = 1.0
        pval_flag = pvalues < pvalue
        if keep_baseline:
            taus = self.loc[(self[Taus.NAMECOL] == "base") | pval_flag]
        else:
            taus = self.loc[pval_flag]

        return Taus(taus)

    def set_nonnegative(self):
        taus = self[Taus.TAUCOL].values
        taus[taus < 0] = 0
        self[Taus.TAUCOL] = taus

        return

    @classmethod
    def parse_taus(cls, stream, names=None):
        dtype_dict = {'Category': str}
        df = pd.read_csv(stream, dtype=dtype_dict, delim_whitespace=True, compression='infer')
        for column in Taus.REQ_COLS:
            if column not in df:
                raise ValueError("{}-column not found in LDSC tau-estimates file".format(column))

        # remove L2_0 from variable names
        df[Taus.NAMECOL] = df[Taus.NAMECOL].str.replace("L2_0", "")
        if names is not None:
            df = df[df[Taus.NAMECOL].isin(names)]

        return cls(df)

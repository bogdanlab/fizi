import pandas as pd
import scipy.stats as stats

import fimpg

class Sigmas(pd.DataFrame):
    """
    Thin wrapper for a pandas DataFrame object containing LDSC partitions.
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

    @classmethod
    def parse_sigmas(cls, stream):
        dtype_dict = {'Category': str}
        df = pd.read_csv(stream, dtype=dtype_dict, delim_whitespace=True)
        for column in Sigmas.REQ_COLS:
            if column not in df:
                raise ValueError("{}-column not found in partition file".format(column))

        return cls(df)

import itertools
import numpy as np
import pandas as pd


def asinhmag_dm(flux, flux_err=None, flux_ivar=None, dm=0):
    """
    Calculate asinh mognitude with dm shift.
    ::flux      - flux in [nanomaggies]
    ::flux_ivar - inverse variance of flux in [1/nanomaggies**2]
    ::flux_err  - flux error in [nanomaggies]
    ::dm        - magnitude shift
    """
    assert (flux_err is not None) ^ (
                flux_ivar is not None), 'specify only flux_err or flux_ivar'
    f = flux / 1e9 * np.power(10, 0.4 * dm)
    if flux_ivar is not None:
        b = np.power(flux_ivar, -0.5) / 1e9 * np.power(10, 0.4 * dm)
    else:
        b = flux_err / 1e9 * np.power(10, 0.4 * dm)

    f, b = f.astype(np.float64), b.astype(
        np.float64)  # otherwise type error like
    # TypeError: loop of ufunc does not support argument 0 of
    # type numpy.float64 which has no callable arcsinh method

    return (np.arcsinh(f / (2 * b)) + np.log(b)) * (-2.5 / np.log(10))


def calculate_features_on_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate features for dataset
    :param df: dataset
    :return: dataset with features
    """
    # SDSS features
    for pb in 'ugriz':
        df[f'sdss_cModelAsinhmag_{pb}'] = asinhmag_dm(
            df[f'sdss_cModelFlux_{pb}'],
            flux_ivar=df[f'sdss_cModelFluxIvar_{pb}']
        )
        df[f'sdss_psfAsinhmag_{pb}'] = asinhmag_dm(
            df[f'sdss_psfFlux_{pb}'],
            flux_ivar=df[f'sdss_psfFluxIvar_{pb}']
        )
        df[f'sdss_psf-cModelAsinhmag_{pb}'] = \
            df[f'sdss_psfAsinhmag_{pb}'] - df[f'sdss_cModelAsinhmag_{pb}']

    for pb1, pb2 in itertools.combinations('ugriz', 2):
        df[f'sdss_psfAsinhmag_{pb1}-{pb2}'] = \
            df[f'sdss_psfAsinhmag_{pb1}'] - df[f'sdss_psfAsinhmag_{pb2}']

    # Pan-STARRS features
    for pb in 'grizy':
        df[f'ps_{pb}KronAsinhmag'] = asinhmag_dm(
            df[f'ps_{pb}KronFlux'], flux_err=df[f'ps_{pb}KronFluxErr']
        )
        df[f'ps_{pb}PSFAsinhmag'] = asinhmag_dm(
            df[f'ps_{pb}PSFFlux'], flux_err=df[f'ps_{pb}PSFFluxErr']
        )

        mask = df[f'ps_{pb}KronAsinhmag'].isna()
        df.loc[mask, f'ps_{pb}KronAsinhmag'] = df.loc[mask, f'ps_{pb}PSFAsinhmag']

        df[f'ps_{pb}PSF-KronAsinhmag'] = \
            df[f'ps_{pb}PSFAsinhmag'] - df[f'ps_{pb}KronAsinhmag']

    for pb1, pb2 in itertools.combinations('grizy', 2):
        df[f'ps_{pb1}-{pb2}PSFAsinhmag'] = \
            df[f'ps_{pb1}PSFAsinhmag'] - df[f'ps_{pb2}PSFAsinhmag']

    # DESI LIS features
    for pb in 'grz':
        df[f'ls_asinhmag_{pb}'] = asinhmag_dm(
            df[f'ls_flux_{pb}'], flux_ivar=df[f'ls_flux_ivar_{pb}']
        )

    for pb1, pb2 in itertools.combinations('grz', 2):
        df[f'ls_asinhmag_{pb1}-{pb2}'] = \
            df[f'ls_asinhmag_{pb1}'] - df[f'ls_asinhmag_{pb2}']

    # WISE features
    for pb in ['w1', 'w2']:
        df[f'wise_asinhmag_{pb}'] = asinhmag_dm(
            df[f'ls_flux_{pb}'], flux_ivar=df[f'ls_flux_ivar_{pb}']
        )

    df[f'wise_asinhmag_w1-w2'] = \
        df[f'wise_asinhmag_w1'] - df[f'wise_asinhmag_w2']

    passbands = {
        'sdss_cModelAsinhmag_{}': 'ugriz',
        'ps_{}KronAsinhmag': 'grizy',
        'ls_asinhmag_{}': 'grz',
    }
    for col_fmt, pbs in passbands.items():
        for pb in pbs:
            for pb_wise in ['w1', 'w2']:
                df[col_fmt.format(pb) + '-' + pb_wise] = \
                    df[col_fmt.format(pb)] - df[f'wise_asinhmag_{pb_wise}']

    return df

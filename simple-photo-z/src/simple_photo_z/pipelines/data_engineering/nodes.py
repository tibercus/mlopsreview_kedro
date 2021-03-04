from astropy.table import Table
import numpy as np
import pandas as pd


def get_GQ(df: pd.DataFrame) -> pd.DataFrame:
    mask = df['LabelS'] == 0
    return df.loc[mask]


def rename_input(x: str):
    if x[-6:] == '_input':
        return x[:-6]
    else:
        return x


def preprocess_xray_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    catalog['__workxid__'] = np.arange(len(catalog))
    catalog = get_GQ(catalog)
    return catalog


def preprocess_panstarrs(catalog: Table) -> pd.DataFrame:
    mask = catalog['primaryDetection'] == 1
    catalog = catalog[mask]

    dst_columns = [
        '__workcid___input', 'objID', 'raBest', 'decBest',

        'gKronFluxErr', 'gKronFlux', 'rKronFluxErr', 'rKronFlux',
        'iKronFluxErr', 'iKronFlux', 'zKronFluxErr', 'zKronFlux',
        'yKronFluxErr', 'yKronFlux',

        'gPSFFluxErr', 'gPSFFlux', 'rPSFFluxErr', 'rPSFFlux',
        'iPSFFluxErr', 'iPSFFlux', 'zPSFFluxErr', 'zPSFFlux',
        'yPSFFluxErr', 'yPSFFlux',

        'w1flux', 'dw1flux', 'w2flux', 'dw2flux'
    ]
    catalog = catalog[dst_columns]
    catalog = catalog.to_pandas()
    catalog = catalog.rename(columns=rename_input)
    catalog = catalog.drop_duplicates(subset='__workcid__', keep='last')
    catalog = catalog.astype({'objID': str})
    catalog = catalog.rename(
        columns=lambda x: 'ps_'+x if x != '__workcid__' else x
    )
    return catalog


def preprocess_desilis(catalog: Table) -> pd.DataFrame:
    dst_columns = [
        '__workxid___input', 'ra', 'dec', 'brickid', 'objid', 'ebv',

        'flux_g', 'flux_r', 'flux_z', 'flux_w1', 'flux_w2',
        'flux_ivar_g', 'flux_ivar_r', 'flux_ivar_z',
        'flux_ivar_w1', 'flux_ivar_w2'
    ]
    catalog = catalog[dst_columns]
    catalog = catalog.to_pandas()
    catalog = catalog.rename(columns=rename_input)
    catalog['__workcid__'] = np.arange(len(catalog))
    catalog = catalog.drop_duplicates(subset='__workxid__', keep='last')
    catalog = catalog.astype({'brickid': str, 'objid': str})
    catalog = catalog.rename(
        columns=lambda x: 'ls_'+x if x[:6] != '__work' else x
    )
    return catalog


def preprocess_sdss(catalog: Table) -> pd.DataFrame:
    mask = catalog['MODE'] == 1
    catalog = catalog[mask]
    dst_columns = [
        '__workcid___input', 'ra', 'dec', 'objID',

        'psfFlux_u', 'psfFluxIvar_u', 'psfFlux_g',
        'psfFluxIvar_g', 'psfFlux_r', 'psfFluxIvar_r', 'psfFlux_i',
        'psfFluxIvar_i', 'psfFlux_z', 'psfFluxIvar_z',

        'cModelFlux_u',
        'cModelFluxIvar_u', 'cModelFlux_g', 'cModelFluxIvar_g', 'cModelFlux_r',
        'cModelFluxIvar_r', 'cModelFlux_i', 'cModelFluxIvar_i', 'cModelFlux_z',
        'cModelFluxIvar_z'
    ]
    catalog = catalog[dst_columns]
    catalog = catalog.to_pandas()
    catalog = catalog.rename(columns=rename_input)
    catalog = catalog.drop_duplicates(subset='__workcid__', keep='last')
    catalog = catalog.astype({'objID': str})
    catalog = catalog.rename(
        columns=lambda x: 'sdss_'+x if x != '__workcid__' else x
    )
    return catalog


def create_master_dataset(
        sample: pd.DataFrame, ls: pd.DataFrame,
        ps: pd.DataFrame, sdss:pd.DataFrame
) -> pd.DataFrame:
    """Combines data from different sources into single dataframe
    :param sample:
    :param ls:
    :param sdss:
    :param ps:
    :return:
    """
    df = pd.merge(left=sample, right=ls, on='__workxid__', how='left')
    df = pd.merge(left=df, right=sdss, on='__workcid__', how='left')
    df = pd.merge(left=df, right=ps, on='__workcid__', how='left')
    return df

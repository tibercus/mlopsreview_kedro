from pathlib import PurePosixPath
from typing import Any, Dict

from kedro.io import AbstractVersionedDataSet, Version
from kedro.io.core import get_protocol_and_path, get_filepath_str

import fsspec
import numpy as np
from astropy.table import Table


class AstropyDataSet(AbstractVersionedDataSet):
    """Loads/saves Astropy fits tables
    """

    def __init__(self, filepath: str, version: Version = None):
        """Creates a new instance of AstropyDataSet to load / save fits tables.

        Args:
            filepath: The location of the fits file to load / save data.
            version: The version of the fits being saved and loaded.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._fs = fsspec.filesystem(self._protocol)

        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )

    def _load(self) -> Table:
        """Loads data from the fits file.

        Returns:
            Data from the fits file as astropy.table.Table
        """
        load_path = get_filepath_str(self._get_load_path(), self._protocol)
        print(load_path, type(load_path))
        return Table.read(load_path, format='fits')

    def _save(self, data: np.ndarray) -> None:
        """Saves astropy.table.Table at specified path
        """
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        Table.write(save_path, overwrite=True, format='fits')

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset.
        """
        return dict(
            filepath=self._filepath,
            version=self._version,
            protocol=self._protocol
        )

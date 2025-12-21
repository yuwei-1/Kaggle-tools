import pytest
from ktools.base.preprocessor import BasePreprocessor
from ktools.config.dataset import DatasetConfig
from ktools.preprocessing.categorical import CategoricalEncoder
from ktools.preprocessing.core import ReduceMemory
from ktools.preprocessing.numerical import StandardScale


DUMMY_CONFIG = DatasetConfig(
    training_col_names=[],
    categorical_col_names=[],
    numerical_col_names=[],
    target_col_name="",
)


@pytest.mark.parametrize(
    "preprocessor_class", [StandardScale, CategoricalEncoder, ReduceMemory]
)
def test_preprocessor_save_load(tmpdir, preprocessor_class):
    preprocessor: BasePreprocessor = preprocessor_class(DUMMY_CONFIG)
    preprocessor.save(tmpdir)

    loaded_preprocessor = preprocessor_class.load(tmpdir)
    assert isinstance(loaded_preprocessor, preprocessor_class)

from ktools.preprocessing.categorical_string_label_error_imputator import CategoricalLabelErrorImputator
from ktools.preprocessing.categorical_features_embedder import SortMainCategories
from ktools.preprocessing.interfaces.i_preprocessing_utility import IPreprocessingUtility
from ktools.preprocessing.kaggle_dataset_manager import KaggleDatasetManager
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings


class DataSciencePipeline:

    def __init__(self,
                 pipeline_settings : DataSciencePipelineSettings,
                 data_preprocessor : IPreprocessingUtility,
                 ) -> None:
        self._ps = pipeline_settings
        self.data_preprocessor = data_preprocessor

    def run(self):

        

        # preprocess data
        training_dataframe, testing_dataframe, updated_settings = self.data_preprocessor.process()

        # TODO: EDA / data visualisation

        # Scale data

        # Data splitting
        train_split = updated_settings.training_data_percentage
        data_manager = KaggleDatasetManager(training_dataframe,
                                            updated_settings.training_col_names,
                                            updated_settings.target_col_name,
                                            train_split,
                                            1 - train_split,
                                            0)
        (X_train, 
        X_valid, 
        X_test, 
        y_train, 
        y_valid, 
        y_test) = data_manager.dataset_partition()


        pass
from copy import deepcopy
from functools import reduce
import unittest
from sklearn.model_selection import StratifiedKFold
from ktools.fitting.cross_validation_executor import CrossValidationExecutor
from ktools.modelling.ktools_models.pytorch_embedding_model import PytorchEmbeddingModel
from ktools.modelling.model_transform_wrappers.survival_model_wrapper import SupportedSurvivalTransformation
from ktools.preprocessing.basic_feature_transformers import *
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings
from post_HCT_survival_notebooks.hct_utils import score


class TestPytorchODST(unittest.TestCase):

    def setUp(self) -> None:

        train_csv_path = "data/post_hct_survival/train.csv"
        test_csv_path = "data/post_hct_survival/test.csv"
        target_col_name = ['efs', 'efs_time']

        def scci_metric(y_test, y_pred, id_col_name : str = "ID",
               survived_col_name : str = "efs",
               survival_time_col_name : str = "efs_time",
               stratify_col_name : str = "race_group"):
            idcs = y_test.index
            og_train = pd.read_csv(train_csv_path)
            
            y_true = og_train.loc[idcs, [id_col_name, survived_col_name, survival_time_col_name, stratify_col_name]].copy()
            y_pred_df = og_train.loc[idcs, [id_col_name]].copy()
            y_pred_df["prediction"] = y_pred
            scci = score(y_true.copy(), y_pred_df.copy(), id_col_name)
            return scci
        
        self.scci = scci_metric

        class GenerateSurvivalTarget():
            
            def __init__(self, transform_string : str, times_col : str = "efs_time", event_col : str = "efs", **kwargs):
                self.transform_func = SupportedSurvivalTransformation[transform_string.upper()].value
                self._times_col = times_col
                self._event_col = event_col
                self.kwargs = kwargs

            def transform(self, original_settings : DataSciencePipelineSettings):
                settings = deepcopy(original_settings)

                times = settings.combined_df.loc['train', self._times_col]
                events = settings.combined_df.loc['train', self._event_col]
                
                settings.combined_df.loc['train', 'target'] = self.transform_func(times, events, **self.kwargs)
                settings.combined_df = settings.combined_df.drop(columns=[self._times_col, self._event_col])
                settings.target_col = "target"
                return settings
            
        self.settings = settings = DataSciencePipelineSettings(train_csv_path,
                                                                test_csv_path,
                                                                target_col_name,
                                                                )
        transforms = [
                    MinMaxScalerNumerical.transform,
                    FillNullValues.transform,
                    OrdinalEncode.transform,
                    ConvertObjectToCategorical.transform,
                    GenerateSurvivalTarget('kaplanmeier').transform
                    ]

        settings = reduce(lambda acc, func: func(acc), transforms, settings)
        settings.update()

        train, test_df = settings.update()
        test_df.drop(columns=settings.target_col, inplace=True)
        self.X, self.y = train.drop(columns=settings.target_col), train[[settings.target_col]]

    
    def test_pytorch_odst(self):
        
        cat_names = self.settings.categorical_col_names
        cat_sizes = [int(x) for x in self.X[cat_names].nunique().values]
        cat_emb = [int(np.sqrt(x)) for x in cat_sizes]
        categorical_idcs = [self.X.columns.get_loc(col) for col in cat_names]

        pynn = PytorchEmbeddingModel(
                                    "ODST",
                                    len(self.settings.training_col_names),
                                    output_dim=1,
                                    categorical_idcs=categorical_idcs,
                                    categorical_sizes=cat_sizes,
                                    categorical_embedding=cat_emb,
                                    last_activation='none',
                                    batch_size=1000,
                                    epochs=50,
                                    learning_rate=1e-2,
                                    patience=10,
                                    decay_period=3,
                                    decay_rate=0.9
                                    )
        
        kf = StratifiedKFold(5, shuffle=True, random_state=42)
        score_tuple, oofs, model_list, test_preds = CrossValidationExecutor(pynn,
                                                                          self.scci,
                                                                          kf,
                                                                          verbose=2).run(self.X, self.y, groups=self.X['race_group'])
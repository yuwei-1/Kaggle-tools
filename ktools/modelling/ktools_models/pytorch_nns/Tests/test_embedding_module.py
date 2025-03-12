import unittest
import torch
from ktools.modelling.ktools_models.pytorch_nns.embedding_module import EmbeddingCategoricalModule



class TestEmbeddingModule(unittest.TestCase):

    def test_embedding_forward(self):

        # arrange
        X = torch.cat([torch.randint(0, 6, (10, 1)), torch.randint(0, 4, (10, 1))], dim=-1)


        # act
        embedding_mod = EmbeddingCategoricalModule(category_cardinalities=[6, 4],
                                                    embedding_sizes=[2, 2],
                                                    projection_dim=5)
        y = embedding_mod(X)

        embedding_mod = EmbeddingCategoricalModule(category_cardinalities=[6, 4],
                                                    embedding_sizes=[2, 2],
                                                    projection_dim=None)
        y_no_projection = embedding_mod(X)


        # assert
        self.assertEqual(y.shape, (10, 5))
        self.assertEqual(y_no_projection.shape, (10, 4))
        self.assertTrue(y.requires_grad)
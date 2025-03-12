from typing import List, Union
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.api.callbacks import EarlyStopping
import keras
from keras import layers, models, Sequential
from keras import backend as K
from ktools.modelling.Interfaces.i_ktools_model import IKtoolsModel



class KerasEmbeddingModel(IKtoolsModel):

    def __init__(self,
                 continuous_feature_idcs : List[int],
                 categorical_feature_idcs : List[int],
                 categorical_feature_sizes : List[int],
                 feature_names : List[str],
                 categorical_embedding_sizes : Union[List[int], int] = None,
                 dense_hidden_sizes : List[int] = [256, 128],
                 dense_activation : str = 'mish',
                 dense_dropout : float = 0.3,
                 output_activation : str = 'sigmoid',
                 batch_size : int = 256,
                 patience : int = 2,
                 epochs : int = 10,
                 loss : str = 'binary_crossentropy',
                 metrics : List[str] = ['auc'],
                 callbacks : List[keras.callbacks.Callback] = [],
                 random_state=42,
                 verbose : int = 0,
                 plot_model : bool = False
                 ) -> None:
        keras.utils.set_random_seed(random_state)
        self._continuous_feature_idcs = continuous_feature_idcs
        self._categorical_feature_idcs = categorical_feature_idcs
        self._categorical_feature_sizes = categorical_feature_sizes
        self._feature_names = feature_names
        if isinstance(categorical_embedding_sizes, int):
            self._categorical_embedding_sizes = [categorical_embedding_sizes] * len(categorical_feature_sizes)
        self._dense_hidden_sizes = dense_hidden_sizes
        self._dense_activation = dense_activation
        self._dense_dropout = dense_dropout
        self._output_activation = output_activation
        self._batch_size = batch_size
        self._epochs = epochs
        self._loss = loss
        self._metrics = metrics
        self._callbacks = [EarlyStopping(
                                        monitor='val_' + metrics[0],
                                        patience=patience,
                                        verbose=verbose,
                                        mode='max',
                                        restore_best_weights=True
                                    )] + callbacks
        
        self._random_state = random_state
        self._verbose = verbose
        self._categorical_embedding_sizes = categorical_embedding_sizes
        self._all_features = continuous_feature_idcs + categorical_feature_idcs
        self._model_input_idcs = categorical_feature_idcs + [continuous_feature_idcs]
        self.model = self._build_model()
        if plot_model:
            tf.keras.utils.plot_model(
                model=self.model, 
                show_shapes=True, 
                rankdir='TB')

    def _build_model(self):
        cat_inputs = [layers.Input(shape=(1,), name=f'cat{i}') for i in self._categorical_feature_idcs]
        cont_inputs = layers.Input(shape=(len(self._continuous_feature_idcs),))
                                    
        flat_embeddings = []
        for i in range(len(self._categorical_feature_idcs)):
            input_dim = self._categorical_feature_sizes[i]

            if self._categorical_embedding_sizes is not None:
                output_dim = self._categorical_embedding_sizes[i]
            else:
                output_dim = int(min(128, round(1.6 * input_dim ** .56)))

            embedding = layers.Embedding(
                input_dim=input_dim, output_dim=output_dim)(cat_inputs[i])
            if output_dim > 32:
                embedding = layers.SpatialDropout1D(min(0.5, self._dense_dropout + 0.2))(embedding)
            else:
                embedding = layers.SpatialDropout1D(self._dense_dropout)(embedding)
            flat_embeddings.append(layers.Flatten()(embedding))
                                    
        concatenated_inputs = layers.Concatenate()(flat_embeddings + [cont_inputs, ])
        concatenated_inputs_bn = layers.BatchNormalization()(concatenated_inputs)
        x = layers.Dense(self._dense_hidden_sizes[0], 
                         activation=self._dense_activation)(concatenated_inputs_bn)

        for units in self._dense_hidden_sizes[1:]:
            inp = layers.Concatenate()([x, concatenated_inputs_bn])
            x = layers.Dense(units=units, activation=self._dense_activation)(inp)
            x = layers.Dropout(self._dense_dropout)(x)

        outputs = layers.Dense(1, activation=self._output_activation)(x)
        return keras.Model(cat_inputs + [cont_inputs], outputs)
    
    def _convert_to_nn_input(self, df):
        X = df[self._feature_names].values
        result = []
        for f_idx in self._model_input_idcs:
            result.append(X[:, f_idx])
        return result
    
    def fit(self, X, y, validation_set = None, val_size=0.05):

        if validation_set is None:
            X_train, X_valid, y_train, y_valid = train_test_split(X, 
                                                                  y, 
                                                                  test_size=val_size, 
                                                                  random_state=self._random_state)
        else:
            X_train, y_train = X, y
            X_valid, y_valid = validation_set

        self.model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=3E-4),
            loss=self._loss,
            metrics=self._metrics)

        _ = self.model.fit(
            self._convert_to_nn_input(X_train), y_train,
            validation_data=(self._convert_to_nn_input(X_valid), y_valid),
            batch_size=self._batch_size,
            epochs=self._epochs,
            callbacks=self._callbacks,
            verbose=self._verbose
        )
        return self
    
    def predict(self, X):
        y_pred = self.model.predict(self._convert_to_nn_input(X), verbose=0, batch_size=self._batch_size).flatten()
        return y_pred
import numpy as np
from sklearn.base import BaseEstimator
import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.api.callbacks import EarlyStopping
from keras.api.layers import Input, Embedding, Dense, Flatten, Activation, Add, Dot
from keras.api.models import Model
from keras.api.regularizers import l2 as l2_reg
import itertools


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]


def batch_generator(X,y,batch_size=128,shuffle=True):
    sample_size = X[0].shape[0]
    index_array = np.arange(sample_size)
    while 1:
        if shuffle:
            np.random.shuffle(index_array)
        batches = make_batches(sample_size, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            X_batch = [X[i][batch_ids] for i in range(len(X))]
            y_batch = y[batch_ids]
            yield X_batch,y_batch


def test_batch_generator(X,y,batch_size=128):
    sample_size = X[0].shape[0]
    index_array = np.arange(sample_size)
    batches = make_batches(sample_size, batch_size)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = index_array[batch_start:batch_end]
        X_batch = [X[i][batch_ids] for i in range(len(X))]
        y_batch = y[batch_ids]
        yield X_batch,y_batch


def predict_batch(model,X_t,batch_size=128):
    outcome = []
    for X_batch,y_batch in test_batch_generator(X_t,np.zeros(X_t[0].shape[0]),batch_size=batch_size):
        outcome.append(model.predict(X_batch,batch_size=batch_size))
    outcome = np.concatenate(outcome).ravel()
    return outcome



def build_model(max_features,K=8,solver='adam',l2=0.0,l2_fm = 0.0):

    inputs = []
    flatten_layers=[]
    columns = range(len(max_features))
    for c in columns:
        inputs_c = Input(shape=(1,), dtype='int32',name = 'input_%s'%c)
        num_c = max_features[c]

        embed_c = Embedding(
                        num_c,
                        K,
                        input_length=1,
                        name = 'embed_%s'%c,
                        embeddings_regularizer=l2_reg(l2_fm)
                        )(inputs_c)

        flatten_c = Flatten()(embed_c)

        inputs.append(inputs_c)
        flatten_layers.append(flatten_c)

    fm_layers = []
    for emb1,emb2 in itertools.combinations(flatten_layers, 2):
        dot_layer = Dot(axes=(1, 1))([emb1,emb2])#merge([emb1,emb2],mode='dot',dot_axes=1)
        fm_layers.append(dot_layer)

    for c in columns:
        num_c = max_features[c]
        embed_c = Embedding(
                        num_c,
                        1,
                        input_length=1,
                        name = 'linear_%s'%c,
                        embeddings_regularizer=l2_reg(l2)
                        )(inputs[c])

        flatten_c = Flatten()(embed_c)

        fm_layers.append(flatten_c)
        
        
    flatten = Add()(fm_layers)#merge(fm_layers,mode='sum')
    outputs = Activation('sigmoid',name='outputs')(flatten)
    
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
                optimizer=solver,
                loss= 'binary_crossentropy',
                metrics=['auc']
              )

    return model


class KerasFM():
    def __init__(self,
                 max_features=[],
                 feature_names=[],
                 K=8,
                 solver='adam',
                 l2=0.0,
                 l2_fm = 0.0,
                 batch_size=128,
                 epochs=10,
                 patience=2,
                 shuffle=True,
                 verbose=1,
                 random_state=42,
                 callbacks = []
                 ):
        self._batch_size = batch_size
        self._epochs = epochs
        self._shuffle = shuffle
        self._verbose = verbose
        self._random_state = random_state
        self._feature_names = feature_names
        self._callbacks = [EarlyStopping(
                                        monitor='val_auc',
                                        patience=patience,
                                        verbose=verbose,
                                        mode='max',
                                        restore_best_weights=True
                                    )] + callbacks
        
        self.model = build_model(max_features,
                                 K,
                                 solver,
                                 l2=l2,
                                 l2_fm = l2_fm)
        
    def _to_nn(self, X):
        X = X[self._feature_names].values
        result = []
        for i in range(X.shape[-1]):
            result.append(X[:, i])
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

        self.model.fit(self._to_nn(X_train),
                       y_train.to_numpy(),
                       batch_size=self._batch_size,
                       epochs=self._epochs,
                       shuffle=self._shuffle,
                       verbose=self._verbose,
                       callbacks=self._callbacks,
                       validation_data=[self._to_nn(X_valid), y_valid.to_numpy()])

    def predict(self, X):
        y_preds = predict_batch(self.model, self._to_nn(X),batch_size=self._batch_size)
        return y_preds
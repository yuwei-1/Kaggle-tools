
from copy import deepcopy
import numpy as np
import math
from typing import List
from ktools.fitting.cross_validation_executor import CrossValidationExecutor


class RecursiveFeatureEliminationExecutor:

    def __init__(self,
                 cross_validation_executor : CrossValidationExecutor,
                 initial_features : List[str],
                 minimum_num_features : int = 1,
                 direction = "minimize",
                 verbose : bool = False
                 ) -> None:
        self._cross_validation_executor = cross_validation_executor
        self._initial_features = deepcopy(initial_features)
        self._minimum_num_features = minimum_num_features
        self._direction = direction
        self._verbose = verbose

    def run(self, X, y):
        
        benchmark, oof, _ = self._cross_validation_executor.run(X, y)
        running_scores = [benchmark[0]]
        best_score = benchmark[0]
        best_feature_set = deepcopy(self._initial_features)
        best_oof = oof

        if self._verbose:
            print("#"*15 + f" Running CV with all features " + "#"*15)
            print(f"Score achieved: {best_score}")


        is_minimize = (self._direction == "minimize")
        while (len(self._initial_features) > self._minimum_num_features) and (running_scores[-1] <= best_score) * is_minimize:

            scores = np.zeros(len(self._initial_features))

            for i, feat in enumerate(self._initial_features):
                used_features = [f for f in self._initial_features if f != feat]
                if self._verbose:
                    print()
                    print("#"*15 + f" Running CV without feature: {self._initial_features[i]} " + "#"*15)
                
                score_tuple, oof, _ = self._cross_validation_executor.run(X[used_features], y)

                if is_minimize:
                    if score_tuple[0] < best_score:
                        best_oof = oof
                        best_feature_set = used_features
                        best_score = score_tuple[0]

                    scores[i] = score_tuple[0]
                else:
                    if -score_tuple[0] < -best_score:
                        best_oof = oof
                        best_feature_set = used_features
                        best_score = score_tuple[0]
                    
                    scores[i] = -score_tuple[0]
                
            worst_idx = scores.argmin()
            worst_feature = self._initial_features.pop(worst_idx)
            curr_score = scores[worst_idx]
            running_scores += [curr_score]

            if self._verbose:
                print("#"*100)
                print(f"Removed feature: {worst_feature}")
                print(f"Score achieved: {curr_score}")
                print("#"*100)

        return best_feature_set, best_oof, best_score
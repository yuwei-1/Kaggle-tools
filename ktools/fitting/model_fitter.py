from ktools.fitting.supported_metrics import SupportedMetrics



class BasicModelEvaluation:

    def evaluate(self, X_train, y_train, X_test, y_test, model, metric='accuracy'):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        evaluation_metric = SupportedMetrics[metric].value
        return evaluation_metric(y_test, y_pred)
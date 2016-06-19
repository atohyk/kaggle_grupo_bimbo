class base_estimator:
    def __init__(self, est):
        self.est = est
    def predict(self, X):
        return self.est.predict_proba(X)[:,1][:,numpy.newaxis]
    def fit(self, X, y, sample_weight = None, monitor = None):
        self.est.fit(X, y, sample_weight, monitor)

acpGB = ensemble.GradientBoostingRegressor(loss= 'lad',n_estimators = 6,
	learning_rate = 0.9, max_depth = 8, verbose = 1)

acpGB_BE = base_estimator(acpGB)

acpGB2 = ensemble.GradientBoostingRegressor(loss= 'ls',n_estimators = 6,
	learning_rate = 0.9, max_depth = 8, init = acpGB_BE, verbose = 1)
acpGB2.fit(train[['acpMed','routeMed','chanMed','cliMed','ageMed']],
	train['Demanda_uni_equil'])
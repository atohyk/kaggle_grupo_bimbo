import time
t0 = time.clock()
startTime = t0
import pandas as pd
import numpy as np
from sklearn import ensemble


t1 = time.clock()
print 'Import Done in ', t1-t0
t0 = t1

def loss(pred,act):
    return np.sqrt(sum((np.log(pred+1)-np.log(act+1))**2)/len(pred))

def mseLoss(pred,act):
	return np.sqrt(sum((pred-act)**2)/len(pred))

df = pd.read_csv('train.csv',usecols = ['Agencia_ID','Canal_ID','Ruta_SAK',
	'Cliente_ID', 'Producto_ID', 'Demanda_uni_equil'])

t1 = time.clock()
print 'Loading Done in', t1-t0
t0 = t1

msk = np.random.randn(df.shape[0])<0.8

train = df[msk]
test = df[~msk]

t1 = time.clock()
print 'Splitting Done in', t1-t0
t0 = t1

del df

acpMed = train.groupby(['Agencia_ID','Cliente_ID',
	'Producto_ID'])['Demanda_uni_equil'].median()
acpMed = acpMed.to_frame()
acpMed.columns = ['acpMed']
acpMed = acpMed.reset_index()

routeMed = train.groupby('Ruta_SAK')['Demanda_uni_equil'].median()
routeMed = routeMed.reset_index()
routeMed.columns = ['Ruta_SAK','routeMed']

pcMed = train.groupby(['Cliente_ID','Producto_ID'])['Demanda_uni_equil'].median()
pcMed = pcMed.to_frame()
pcMed.columns = ['pcMed']
pcMed = pcMed.reset_index()

prodMed = train.groupby('Producto_ID')['Demanda_uni_equil'].median()
prodMed = prodMed.reset_index()
prodMed.columns = ['Producto_ID','prodMed']

chanMed = train.groupby('Canal_ID')['Demanda_uni_equil'].median()
chanMed = chanMed.reset_index()
chanMed.columns = ['Canal_ID','chanMed']

cliMed = train.groupby('Cliente_ID')['Demanda_uni_equil'].median()
cliMed = cliMed.reset_index()
cliMed.columns = ['Cliente_ID','cliMed']

ageMed = train.groupby('Agencia_ID')['Demanda_uni_equil'].median()
ageMed = ageMed.reset_index()
ageMed.columns = ['Agencia_ID','ageMed']

allMed = np.median(train['Demanda_uni_equil'])

t1 = time.clock()
print 'Extracting Medians done in ', t1-t0
t0 = t1

train = pd.merge(train, acpMed, how='left')
train = pd.merge(train, pcMed, how='left')
train = pd.merge(train, routeMed, how='left')
train = pd.merge(train, prodMed, how='left')
train = pd.merge(train, chanMed, how='left')
train = pd.merge(train, cliMed, how='left')
train = pd.merge(train, ageMed, how='left')

t1 = time.clock()
print 'Merging done in ', t1-t0
t0 = t1

#don't need any replacing of nulls because merges which
#come from train merge perfectly to train
acpLoss = loss(train['acpMed'],train['Demanda_uni_equil'])
pcLoss = loss(train['pcMed'],train['Demanda_uni_equil'])
routeLoss = loss(train['routeMed'],train['Demanda_uni_equil'])
prodLoss = loss(train['prodMed'],train['Demanda_uni_equil'])
chanLoss = loss(train['chanMed'],train['Demanda_uni_equil'])
cliLoss = loss(train['cliMed'],train['Demanda_uni_equil'])
ageLoss = loss(train['ageMed'],train['Demanda_uni_equil'])

del train['Agencia_ID']
del train['Canal_ID']
del train['Ruta_SAK']
del train['Cliente_ID']
del train['Producto_ID']

#regr = linear_model.LogisticRegression()
#regr.fit(train[['acpMed','prodMed','cliMed']],
#	train['Demanda_uni_equil'])

t1 = time.clock()
print 'Calculating Loss and Clearing up in ', t1-t0
t0 = t1


test = pd.merge(test, acpMed, how='left')
# test.loc[test.acpMed.isnull(),'acpMed'] = allMed
test = pd.merge(test, pcMed, how='left')
# test.loc[test.pcMed.isnull(),'pcMed'] = allMed
test = pd.merge(test, routeMed, how='left')
test.loc[test.routeMed.isnull(),'routeMed'] = allMed
test = pd.merge(test, prodMed, how='left')
# test.loc[test.prodMed.isnull(),'prodMed'] = allMed
test = pd.merge(test, chanMed, how='left')
test.loc[test.chanMed.isnull(),'chanMed'] = allMed
test = pd.merge(test, cliMed, how='left')
test.loc[test.cliMed.isnull(),'cliMed'] = allMed
test = pd.merge(test, ageMed, how='left')
test.loc[test.ageMed.isnull(),'ageMed'] = allMed
del test['Agencia_ID']
del test['Canal_ID']
del test['Cliente_ID']
del test['Producto_ID']
del test['Ruta_SAK']

t1 = time.clock()
print 'Merging and Cleaning on Test done in ', t1-t0
t0 = t1


#GBREG
# gbreg = ensemble.GradientBoostingRegressor(n_estimators = 10,
# 	learning_rate = 0.7, max_depth = 4, verbose = 1)
# gbreg.fit(train[['acpMed','pcMed','routeMed','prodMed','chanMed',
# 	'cliMed','ageMed']],train['Demanda_uni_equil'])

# t1 = time.clock()
# print 'GB Training done in ', t1-t0
# t0 = t1

# gbPredictions = gbreg.predict(test[['acpMed','pcMed', 'routeMed',
# 	'prodMed','chanMed','cliMed','ageMed']])
# print sum(gbPredictions<0)
# gbPredictions[gbPredictions<0]=0.0
# gbLoss = loss(gbPredictions,test['Demanda_uni_equil'])
# print 'gb loss: ',gbLoss
# t1 = time.clock()
# print 'Scoring on Test done in ', t1-t0
# t0 = t1

#SUBMISSION
'''
subname = 'Submission12_RouteMed.csv'
submit = pd.read_csv('test.csv')
print 'Submission Test Set loaded in ', t1-t0
t0 = t1

submit = pd.merge(submit, acpMed, how='left')
submit.loc[submit.acpMed.isnull(),'acpMed'] = allMed
submit = pd.merge(submit, pcMed, how='left')
submit.loc[submit.pcMed.isnull(),'pcMed'] = allMed
submit = pd.merge(submit, prodMed, how='left')
submit.loc[submit.prodMed.isnull(),'prodMed'] = allMed
submit = pd.merge(submit, chanMed, how='left')
submit.loc[submit.chanMed.isnull(),'chanMed'] = allMed
submit = pd.merge(submit, cliMed, how='left')
submit.loc[submit.cliMed.isnull(),'cliMed'] = allMed
submit = pd.merge(submit, ageMed, how='left')
print 'Submission Set Merged in ', t1-t0
t0 = t1

submit['Demanda_uni_equil'] = sgdr.predict(submit[['acpMed',
	'pcMed','prodMed','chanMed','cliMed','ageMed']])
submit.loc[submit['Demanda_uni_equil']<0, 'Demanda_uni_equil'] = 0.0

print 'Submission Set Merged in ', t1-t0
t0 = t1

submit[['id','Demanda_uni_equil']].to_csv(subname,index=False)
'''



# rccaGB = ensemble.GradientBoostingRegressor(n_estimators = 10,
# 	learning_rate = 0.7, max_depth = 4, verbose = 1)
# rccaGB.fit(train[['routeMed','chanMed','cliMed','ageMed']],
# 	train['Demanda_uni_equil'])

# prodGB = ensemble.GradientBoostingRegressor(n_estimators = 12,
# 	learning_rate = 0.7, max_depth = 4, verbose = 1)
# prodGB.fit(train[['prodMed','routeMed','chanMed','cliMed','ageMed']],
# 	train['Demanda_uni_equil'])

# pcGB = ensemble.GradientBoostingRegressor(n_estimators = 10,
# 	learning_rate = 0.7, max_depth = 4, verbose = 1)
# pcGB.fit(train[['pcMed','routeMed','chanMed','cliMed','ageMed']],
# 	train['Demanda_uni_equil'])

# acpGB = ensemble.GradientBoostingRegressor(n_estimators = 10,
# 	learning_rate = 0.7, max_depth = 4, verbose = 1)
# acpGB.fit(train[['acpMed','routeMed','chanMed','cliMed','ageMed']],
# 	train['Demanda_uni_equil'])



'''
test['Preds'] = 0
for i in xrange(test.shape[0]):
	if i%100000 == 0:
		print 'i',i
		t1 = time.clock()
		print 'Time taken:', t1-t0
		t0 = t1
	test.loc[i,'Preds'] = rowPred(test[i:i+1])

t1 = time.clock()
print 'Results computed in ', t1-t0
t0 = t1

enLoss = loss(test['Preds'],test['Demanda_uni_equil'])
'''
test['acpMedNull'] = test['acpMed'].isnull()
test['pcMedNull'] = test['pcMed'].isnull()
test['prodMedNull'] = test['prodMed'].isnull()

test.loc[test.acpMed.isnull(),'acpMed'] = allMed
test.loc[test.pcMed.isnull(),'pcMed'] = allMed
test.loc[test.prodMed.isnull(),'prodMed'] = allMed

# test['rccaGB'] = rccaGB.predict(test[['routeMed','chanMed','cliMed',
# 	'ageMed']])
# test['prodGB'] = prodGB.predict(test[['prodMed','routeMed',
# 	'chanMed','cliMed','ageMed']])
# test['pcGB'] = pcGb.predict(test[['pcMed','routeMed','chanMed',
# 	'cliMed','ageMed']])
# test['acpGB'] = acpGB.predict(test[['acpMed','routeMed','chanMed',
# 	'cliMed','ageMed']])

test['Preds'] = 0

rowSel = ~test.acpMedNull
test.loc[rowSel,'Preds'] = test.loc[rowSel,'acpMed']
rowSel = (test.acpMedNull) & (~test.pcMedNull)
test.loc[rowSel,'Preds'] = test.loc[rowSel,'pcMed']
rowSel = (test.acpMedNull) & (test.pcMedNull) & (~test.prodMedNull) 
test.loc[rowSel,'Preds'] = test.loc[rowSel,'prodMed']
rowSel = (test.acpMedNull) & (test.pcMedNull) & (test.prodMedNull) 
test.loc[rowSel,'Preds'] = allMed

test.loc[test['Preds']<0,'Preds'] = 0.0

hierMedLoss = loss(test['Preds'],test['Demanda_uni_equil'])

t1 = time.clock()
print 'Hierarchical Medians done in ', t1-t0
t0 = t1

#preparing the base estimator 
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

t1 = time.clock()
print 'GB Fitted in ', t1-t0
t0 = t1

test['acpGB2'] = acpGB2.predict(test[['acpMed','routeMed','chanMed','cliMed','ageMed']])
test.loc[test['acpGB2']<0,'acpGB2'] = 0.0
print 'acpGB2', loss(test['acpGB2'], test['Demanda_uni_equil'])



print 'Total time taken', time.clock() -startTime
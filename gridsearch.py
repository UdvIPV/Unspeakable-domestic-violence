import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import GridSearchCV
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import make_scorer,recall_score,precision_score

def pre_dealing(filename,y_colum_index):
    data_dealed = np.genfromtxt(filename, delimiter=',',encoding='utf-8', skip_header=1)
    print(data_dealed.shape)
    data_have = data_dealed[(data_dealed[:,y_colum_index]==1)|(data_dealed[:,y_colum_index]==0)] 
    data_none = data_dealed[(data_dealed[:,y_colum_index]!=1)&(data_dealed[:,y_colum_index]!=0)]  
    print(data_have.shape, data_none.shape)
    x_have_dealed = data_have[:, 0:-3]
    x_none_dealed = data_none[:, 0:-3]
    print(x_have_dealed.shape,x_none_dealed.shape)
    y_have = data_have[:, y_colum_index].astype(int)
    return data_dealed,data_have,x_have_dealed, x_none_dealed, y_have, data_none

def canshu(name,x_have_dealed, y_have):
	max_depth = [5, 10, 15, 20, 25, 30]
	max_features = [20, 25, 30, 35, 40, 45]
	param_grid = {'max_depth': max_depth, 'max_features': max_features}
	scoring = {'AUC': 'roc_auc', 'Accuracy': 'accuracy', 'f1': 'f1', 'precision': make_scorer(precision_score),'recall':make_scorer(recall_score)}
	clf = GridSearchCV(estimator=BalancedRandomForestClassifier(n_estimators=500,replacement=True,random_state=0), n_jobs=-1 ,cv=10, param_grid=param_grid, scoring=scoring, refit='AUC', return_train_score=True)
	clf.fit(x_have_dealed, y_have)
	results = pd.DataFrame.from_dict(clf.cv_results_)


	plt.rcParams["figure.dpi"] = 300
	fig, ax = plt.subplots()
	
	results['params_str'] = results.params.apply(str)
	values='mean_test_AUC'
	scores_matrix = results.pivot(index='param_max_depth', columns='param_max_features',values=values)
	im = ax.imshow(scores_matrix)

	ax.set_xticks(np.arange(len(max_features)))
	ax.set_xticklabels(max_features)
	ax.set_xlabel('max_features', fontsize=15)

	ax.set_yticks(np.arange(len(max_depth)))
	ax.set_yticklabels(max_depth)
	ax.set_ylabel('max_depth', fontsize=15)

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.82, 0.2, 0.04, 0.6])
	fig.colorbar(im, cax=cbar_ax)
	cbar_ax.set_ylabel(values, rotation=-90, va="bottom",fontsize=10)
	

	ax.set_title(name, fontsize=15)
	
	best_clf=clf.best_estimator_ 
	best_score=clf.best_score_
	best_params=clf.best_params_ 
	print(name,'best_score',best_score,'best_params',best_params)
	return best_clf


if __name__ == '__main__':
    filename='data/data-dealed.csv'
    y_colum_index=-1    #-1 physical violence -2 verbal violence -3cold violence
    name='Physical Violence'
    data_dealed,data_have,x_have_dealed, x_none_dealed, y_have, data_none=pre_dealing(filename,y_colum_index) 
    best_clf=canshu(name,x_have_dealed, y_have)
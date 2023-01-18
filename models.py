import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier,BalancedBaggingClassifier 
from sklearn.linear_model import LogisticRegression,Lasso
from sklearn.metrics import accuracy_score,recall_score,confusion_matrix,roc_curve,auc,f1_score,plot_det_curve,plot_roc_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
import collections

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
    return data_dealed,data_have,x_have_dealed, x_none_dealed, y_have


if __name__ == '__main__':
    filename='data/data-dealed.csv'
    y_colum_index=-1     #-1 physical violence -2 verbal violence -3cold violence
    data_dealed,data_have,x_have_dealed, x_none_dealed, y_have=pre_dealing(filename,y_colum_index)  #数据预处理
    X_train, X_test, y_train, y_test = train_test_split(x_have_dealed,y_have,test_size=.3,stratify=y_have,random_state=22)

    classifiers = {
        "Random Forest":make_pipeline(StandardScaler(),SelectFromModel(estimator=Lasso(alpha=0.001,max_iter=8000)),BalancedRandomForestClassifier(n_estimators=100,replacement=True)), 
        "AdaBoost":make_pipeline(StandardScaler(),SelectFromModel(estimator=Lasso(alpha=0.001,max_iter=8000)),BalancedBaggingClassifier(n_estimators=100,replacement=True,base_estimator=AdaBoostClassifier())),
        "GaussianNB": make_pipeline(StandardScaler(),SelectFromModel(estimator=Lasso(alpha=0.001,max_iter=5000)),BalancedBaggingClassifier(n_estimators=100,base_estimator=GaussianNB(),replacement=True)), 
        "SVM": make_pipeline(StandardScaler(),SelectFromModel(estimator=Lasso(alpha=0.001,max_iter=5000)),BalancedBaggingClassifier(n_estimators=100,base_estimator=SVC(probability=True),replacement=True)),
        "Logistic": make_pipeline(StandardScaler(),SelectFromModel(estimator=Lasso(alpha=0.001,max_iter=5000)),BalancedBaggingClassifier(n_estimators=100,base_estimator=LogisticRegression(max_iter=5000),replacement=True)), 
        "Neural network": make_pipeline(StandardScaler(),SelectFromModel(estimator=Lasso(alpha=0.001,max_iter=8000)),BalancedBaggingClassifier(n_estimators=100,base_estimator=MLPClassifier(),replacement=True))
        } 
    
    columns=['Train Acc','Val Acc','Train Rec','Val Rec','Train F1','Value F1','00','10','01','11','0','1']
    big_list=[]
    index_list=[]
    plt.rcParams["figure.dpi"] = 300
    fig, [ax_roc, ax_det] = plt.subplots(1, 2, figsize=(11, 5))
    for name, clf in classifiers.items():
        index_list.append(name)
        clf.fit(X_train, y_train)
       
        y_train_pred=clf.predict(X_train)
        y_test_pred=clf.predict(X_test)

        accuracy_score_train = accuracy_score(y_train, y_train_pred)
        accuracy_score_test = accuracy_score(y_test, y_test_pred)
        recall_score_train = recall_score(y_train, y_train_pred)
        recall_score_test = recall_score(y_test, y_test_pred)
        f1_score_train = f1_score(y_train, y_train_pred)
        f1_score_test = f1_score(y_test, y_test_pred)

        cm = confusion_matrix(y_test, y_test_pred, labels=None, sample_weight=None)

        probas_ = clf.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        print(name,roc_auc)
        
        y_none_pred=clf.predict(x_none_dealed)
        print(name,collections.Counter(y_none_pred),collections.Counter(y_none_pred)[0],collections.Counter(y_none_pred)[1])
        evaulate_list=[accuracy_score_train, accuracy_score_test, recall_score_train, recall_score_test, f1_score_train, f1_score_test, cm[0,0],cm[1,0],cm[0,1],cm[1,1], collections.Counter(y_none_pred)[0], collections.Counter(y_none_pred)[1]]
        print(evaulate_list)
        big_list.append(evaulate_list)
        plot_roc_curve(clf, X_test, y_test, ax=ax_roc, name=name)
        plot_det_curve(clf, X_test, y_test, ax=ax_det, name=name)
        

    table=pd.DataFrame(data=big_list,columns=columns,index =index_list)

    ax_roc.set_title('Receiver operating characteristic (ROC) curves')
    ax_det.set_title('Detection error tradeoff (DET) curves')

    ax_roc.grid(linestyle='--')
    ax_det.grid(linestyle='--')

    plt.legend()
    plt.show()
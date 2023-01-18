import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import ClusterCentroids
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score,recall_score,confusion_matrix,roc_curve,auc,f1_score,plot_det_curve,plot_roc_curve
from sklearn.feature_selection import SelectFromModel
import collections


def pre_dealing(filename,y_colum_index):
    data_dealed = np.genfromtxt(filename, delimiter=',',encoding='utf-8', skip_header=1)
    print(data_dealed.shape)
    data_have = data_dealed[(data_dealed[:,y_colum_index]==1)|(data_dealed[:,y_colum_index]==0)]  #目标变量为非缺失值的行，用于训练和评估
    data_none = data_dealed[(data_dealed[:,y_colum_index]!=1)&(data_dealed[:,y_colum_index]!=0)]   #目标变量为缺失值的行，用于预测
    print(data_have.shape, data_none.shape)
    x_have_dealed = data_have[:, 0:-3]
    x_none_dealed = data_none[:, 0:-3]
    print(x_have_dealed.shape,x_none_dealed.shape)
    y_have = data_have[:, y_colum_index].astype(int)
    return data_dealed,data_have,x_have_dealed, x_none_dealed, y_have


def deal_unbalance(x_have_dealed,y_have):  
    X_train, X_test, y_train, y_test = train_test_split(x_have_dealed,y_have,test_size=.3,stratify=y_have,random_state=0)
    clfs={}
    for i in range(0,5):
        print(i)
        if i == 0:  
            x_balanced,y_balanced = RandomOverSampler(random_state=0).fit_resample(X_train,y_train)
            clf = make_pipeline(StandardScaler(),SelectFromModel(estimator=Lasso(alpha=0.001,max_iter=8000)),RandomForestClassifier()).fit(x_balanced,y_balanced)
            clfs['Random Over Sampler']=clf
        elif i == 1:  
            x_balanced,y_balanced = SMOTE().fit_resample(X_train,y_train)  
            clf = make_pipeline(StandardScaler(),SelectFromModel(estimator=Lasso(alpha=0.001,max_iter=8000)),RandomForestClassifier()).fit(x_balanced,y_balanced)
            clfs['SMOTE']=clf
        elif i == 2: 
            clf = make_pipeline(StandardScaler(),SelectFromModel(estimator=Lasso(alpha=0.001,max_iter=8000)),BalancedRandomForestClassifier(replacement=True)).fit(X_train,y_train)
            clfs['Random Under Sampler-ensemble']=clf
        elif i == 3:  
            x_balanced,y_balanced = ClusterCentroids(random_state=0).fit_resample(X_train,y_train) 
            clf = make_pipeline(StandardScaler(),SelectFromModel(estimator=Lasso(alpha=0.001,max_iter=8000)),RandomForestClassifier()).fit(x_balanced,y_balanced)
            clfs['K-means']=clf
        elif i == 4: 
            x_balanced,y_balanced = SMOTEENN(random_state=0).fit_resample(X_train,y_train)
            clf = make_pipeline(StandardScaler(),SelectFromModel(estimator=Lasso(alpha=0.001,max_iter=8000)),RandomForestClassifier()).fit(x_balanced,y_balanced)
            clfs['SMOTEENN']=clf    
    return X_train, X_test, y_train, y_test,clfs


if __name__ == '__main__':
    filename='data/data-dealed.csv'
    y_colum_index=-1    #-1 physical violence -2 verbal violence -3cold violence
    data_dealed,data_have,x_have_dealed, x_none_dealed, y_have=pre_dealing(filename,y_colum_index)  #数据预处理
    X_train, X_test, y_train, y_test,clfs = deal_unbalance(x_have_dealed,y_have)

    columns=['Train Acc','Val Acc','Train Rec','Val Rec','Train F1','Value F1','roc_auc','00','10','01','11','0','1']
    big_list=[]
    index_list=[]
    plt.rcParams["figure.dpi"] = 300
    fig, [ax_roc, ax_det] = plt.subplots(1, 2, figsize=(11, 5))
    for name, clf in clfs.items():
        index_list.append(name)
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
        y_none_pred=clf.predict(x_none_dealed)
        print(name,collections.Counter(y_none_pred),collections.Counter(y_none_pred)[0],collections.Counter(y_none_pred)[1])
        evaulate_list=[accuracy_score_train, accuracy_score_test, recall_score_train, recall_score_test, f1_score_train, f1_score_test, roc_auc,cm[0,0],cm[0,1],cm[1,0],cm[1,1], collections.Counter(y_none_pred)[0], collections.Counter(y_none_pred)[1]]
        big_list.append(evaulate_list)
        plot_roc_curve(clf, X_test, y_test, ax=ax_roc, name=name)
        plot_det_curve(clf, X_test, y_test, ax=ax_det, name=name)
    table=pd.DataFrame(data=big_list,columns=columns,index =index_list)
    ax_roc.set_title('Receiver Operating Characteristic (ROC) curves')
    ax_det.set_title('Detection Error Tradeoff (DET) curves')
    ax_roc.grid(linestyle='--')
    ax_det.grid(linestyle='--')
    plt.legend()
    plt.show()
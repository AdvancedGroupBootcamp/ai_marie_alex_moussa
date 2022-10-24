from sklearn import metrics
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.model_selection import cross_val_score
import numpy as np

import time

def attribute_impact(models,X,y,n_attributes_in_range):

        """
        Parameter:
        X: Données d'entrainnement contenant des variables independantes
        y: les labels des variables indépendantes pour les données d'entrainnement
        n_attributes_in_range: Le range du nombre de variable à extraire de l'ensemble des variables indépendantes
        threshold: La proportion de l'information à selectionner afin de supprimer des données abérantes

        Return:
        
        """
    
        scores_means={}
        scores_stds={}
        times={}
        
        
        for name,model in zip(models.keys(),models.values()):
            
            #train_scores,test_scores=initiate_dict(models)

            #train_scores[name]=[]
            
            check_mean={"accuracy_mean":[],"precision_mean":[],"recall_mean":[],"f1_score_mean":[]}
            check_std={"accuracy_std":[],"precision_std":[],"recall_std":[],"f1_score_std":[]}
            time_dict={}

            for num_attributes in n_attributes_in_range:

                #Intialisation de la méthode de selection

                selector=SelectKBest(score_func=f_classif, k=num_attributes)

                # Entrainnement de la méthode
                selector_fitted=selector.fit(X,y)

                # Récuperation des variables sélectionnées
                columns=X.columns[selector.get_support()]

                #test_scores[name]=check
                
                start_time=time.time()

                score_accuracy=cross_val_score(model,X[columns],y,cv=5,scoring='accuracy',n_jobs=-1)
                score_precision=cross_val_score(model,X[columns],y,cv=5,scoring='precision',n_jobs=-1)
                score_recall=cross_val_score(model,X[columns],y,cv=5,scoring='recall',n_jobs=-1)
                score_f1=cross_val_score(model,X[columns],y,cv=5,scoring='f1',n_jobs=-1)
                
                end_time=time.time()

                time_dict[num_attributes]=end_time-start_time

                #Testing score
                #Accuracy
                check_mean["accuracy_mean"].append(np.mean(score_accuracy))
                check_std["accuracy_std"].append(np.std(score_accuracy))

                #Presicion
                check_mean["precision_mean"].append(np.mean(score_precision))
                check_std["precision_std"].append(np.std(score_precision))


                #Recall
                check_mean["recall_mean"].append(np.mean(score_recall))
                check_std["recall_std"].append(np.std(score_recall))

                #F1-Score
                check_mean["f1_score_mean"].append(np.mean(score_f1))
                check_std["f1_score_std"].append(np.std(score_f1))

            scores_means[name]=check_mean
            scores_stds[name]=check_std
            times[name]=time_dict
                
        return scores_means,scores_stds,times
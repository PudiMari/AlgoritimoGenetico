import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import *
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import *
from sklearn.decomposition import *
from sklearn.impute import *
from sklearn.pipeline import Pipeline
from sklearn.datasets import *
from sklearn.feature_selection import *
from deap import base, creator, tools, algorithms
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import random
import time
import csv

RANDOM_STATE = 42

df = pd.read_csv("arrhythmia.csv", header=None).replace("?", np.nan)
data = df.to_numpy()
x, y = data[:, :-1], data[:, -1]
y = pd.Series(y).apply(str)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,test_size=0.2,stratify=y, random_state=RANDOM_STATE)

#Corrigir aviso de UserWarning
selector = VarianceThreshold(threshold=0.01)
x_train = selector.fit_transform(x_train)
x_test = selector.transform(x_test)

def criar_individuo_randomForest(params_a):
    
    column_names_modelos = [' Modelos', 'F1 Score']
    
    start_time = time.time()
    
    with_mean, with_std, strategy, k, n_estimators, max_depth, min_samples_split, random_state = params_a
    
    imputer = SimpleImputer(strategy=strategy)
    selector = SelectKBest(k=k)

    x_train_imputed = imputer.fit_transform(x_train)
    x_test_imputed = imputer.transform(x_test)
        
    x_train_selected = selector.fit_transform(x_train_imputed, y_train)
    x_test_selected = selector.fit_transform(x_test_imputed, y_test)
    
    def evaluate(individual):

        x_train_imputed = imputer.fit_transform(x_train)
        x_test_imputed = imputer.transform(x_test)
        
        x_train_selected = selector.fit_transform(x_train_imputed, y_train)
        x_test_selected = selector.fit_transform(x_test_imputed, y_test)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
        model.fit(x_train_selected, y_train)
        
        scores = cross_val_score(model, x_train_selected, y_train, cv=5, scoring='f1_weighted')
        f1 = scores.mean()
        
        with open('modelos_randomForest.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
        
            if file.tell() == 0:
                writer.writerow(column_names_modelos)
            writer.writerow([model, f1])
        
        return f1,

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", np.random.randint, 1, 100)
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_int, ), n=4)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=100, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=10)

    #modificar gerações
    algorithms.eaMuPlusLambda(population, toolbox, mu=10, lambda_=50, cxpb=0.7, mutpb=0.3, ngen=3, stats=None, halloffame=None)

    best_individual = tools.selBest(population, k=36)[0]
    print("Melhores hiperparametros encontrados:", best_individual)
                                               
    n_estimators, max_depth, min_samples_split, random_state = best_individual
    best_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    
    best_model.fit(x_train_selected, y_train)
    scores = cross_val_score(best_model, x_train_selected, y_train, cv=5, scoring='f1_weighted')
    f1 = scores.mean()
    
    print("F1-score:", f1)
    return best_individual, start_time, f1

def main():
    column_names = [' Melhor Individuo', 'Tempo de execucao', 'F1 Score']
    params_a = [True, True, 'mean', 36, 18, 17, 4, RANDOM_STATE]

    for _ in range(1):
        best_individual, start_time, f1 = criar_individuo_randomForest(params_a)
    
        end_time = time.time()
        tempo = end_time - start_time

        with open('resultados_randomForest.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
        
            if file.tell() == 0:
                writer.writerow(column_names)
            writer.writerow([best_individual, tempo, f1])
        
if __name__ == "__main__":
    main()
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import *
from sklearn.impute import *
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import *
from deap import base, creator, tools, algorithms
from datetime import datetime
from sklearn.svm import SVC
import warnings
import numpy as np
import random

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

IND_SIZE = 10
RANDOM_STATE = 42
random.seed(RANDOM_STATE)

column_names_modelos = [' Modelos', 'F1 Score']

df = pd.read_csv("arrhythmia.csv", header=None).replace("?", np.nan)
data = df.to_numpy()
x, y = data[:, :-1], data[:, -1]
y = pd.Series(y).apply(str)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, stratify=y,
                                                    random_state=RANDOM_STATE)
records = list()

def evaluate_RF(individual):

    pipe = Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler()),
        ('feature-selection', SelectKBest()),
        ('randomForest', RandomForestClassifier())])

    start_time = datetime.now()
    pipe.fit(x_train, y_train)
    scores = cross_val_score(pipe, x_test, y_test, cv=5, scoring='f1_weighted')
    f1 = scores.mean()
    end_time = datetime.now()

    records.append({
        "individual": individual,
        "f1": f1,
        "elapsed_time": (end_time - start_time).total_seconds()
    })

    return f1,


def evaluate_SVC(individual):

    pipe = Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler()),
        ('feature-selection', SelectKBest()),
        ('svc', SVC())])

    start_time = datetime.now()
    pipe.fit(x_train, y_train)
    scores = cross_val_score(pipe, x_test, y_test, cv=5, scoring='f1_weighted')
    f1 = scores.mean()
    end_time = datetime.now()

    records.append({
        "individual": individual,
        "f1": f1,
        "elapsed_time": (end_time - start_time).total_seconds()
    })

    return f1,


def evaluate_LogReg(individual):

    pipe = Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler()),
        ('feature-selection', SelectKBest()),
        ('logistic regression', LogisticRegression())])

    start_time = datetime.now()
    pipe.fit(x_train, y_train)
    scores = cross_val_score(pipe, x_test, y_test, cv=5, scoring='f1_weighted')
    f1 = scores.mean()
    end_time = datetime.now()

    records.append({
        "individual": individual,
        "f1": f1,
        "elapsed_time": (end_time - start_time).total_seconds()
    })

    return f1,


def evaluate_KNN(individual):
    
    pipe = Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler()),
        ('feature-selection', SelectKBest()),
        ('knn', KNeighborsClassifier())])

    start_time = datetime.now()
    pipe.fit(x_train, y_train)
    scores = cross_val_score(pipe, x_test, y_test, cv=5, scoring='f1_weighted')
    f1 = scores.mean()
    end_time = datetime.now()

    records.append({
        "individual": individual,
        "f1": f1,
        "elapsed_time": (end_time - start_time).total_seconds()
    })

    return f1,


def evaluate_AB(individual):

    pipe = Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler()),
        ('feature-selection', SelectKBest()),
        ('adaBoost', AdaBoostClassifier())])

    start_time = datetime.now()
    pipe.fit(x_train, y_train)
    scores = cross_val_score(pipe, x_test, y_test, cv=5, scoring='f1_weighted')
    f1 = scores.mean()
    end_time = datetime.now()

    records.append({
        "individual": individual,
        "f1": f1,
        "elapsed_time": (end_time - start_time).total_seconds()
    })

    return f1,

def criar_individuo_randomForest():
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_RF)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=10, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=10)

    # modificar gerações
    start_time = datetime.now()
    algorithms.eaMuPlusLambda(population, toolbox, mu=10, lambda_=50, cxpb=0.7, mutpb=0, ngen=3, stats=None, halloffame=None)
    end_time = datetime.now()

    best_individual = tools.selBest(population, k=36)[0]
    print(
        f"Melhores hiperparametros encontrados com Random Forest: {best_individual} duration: {end_time - start_time}")

def criar_individuo_SVC():
        
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_SVC)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=10, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=10)

    # modificar gerações
    start_time = datetime.now()
    algorithms.eaMuPlusLambda(population, toolbox, mu=10, lambda_=50, cxpb=0.7, mutpb=0, ngen=3, stats=None, halloffame=None)
    end_time = datetime.now()

    best_individual = tools.selBest(population, k=36)[0]
    print(f"Melhores hiperparametros encontrados com SVC: {best_individual} duration: {end_time - start_time}")


def criar_individuo_LogReg():
        
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_LogReg)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=10, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=10)

    # modificar gerações
    start_time = datetime.now()
    algorithms.eaMuPlusLambda(population, toolbox, mu=10, lambda_=50, cxpb=0.7, mutpb=0, ngen=3, stats=None, halloffame=None)
    end_time = datetime.now()

    best_individual = tools.selBest(population, k=36)[0]
    print(f"Melhores hiperparametros encontrados com LogReg: {best_individual} duration: {end_time - start_time}")


def criar_individuo_KNN():    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_KNN)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=10, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=10)

    # modificar gerações
    start_time = datetime.now()
    algorithms.eaMuPlusLambda(population, toolbox, mu=10, lambda_=50, cxpb=0.7, mutpb=0, ngen=3, stats=None, halloffame=None)
    end_time = datetime.now()

    best_individual = tools.selBest(population, k=36)[0]
    print(f"Melhores hiperparametros encontrados com KNN: {best_individual} duration: {end_time - start_time}")


def criar_individuo_AB():    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_AB)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=10, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=10)

    # modificar gerações
    start_time = datetime.now()
    algorithms.eaMuPlusLambda(population, toolbox, mu=10, lambda_=50, cxpb=0.7, mutpb=0, ngen=3, stats=None, halloffame=None)
    end_time = datetime.now()

    best_individual = tools.selBest(population, k=36)[0]
    print(f"Melhores hiperparametros encontrados com AB: {best_individual} duration: {end_time - start_time}")


def main():
    global records
    # RandomForest
    criar_individuo_randomForest()
    data_frame = pd.DataFrame.from_records(records)
    data_frame.to_csv(f'resultados_randomForest_Padrao.csv', index=False, header=True)
    records = list()

    # SVC
    criar_individuo_SVC()
    data_frame = pd.DataFrame.from_records(records)
    data_frame.to_csv(f'resultados_SVC_Padrao.csv', index=False, header=True)
    records = list()

    # Logistic Regression
    criar_individuo_LogReg()
    data_frame = pd.DataFrame.from_records(records)
    data_frame.to_csv(f'resultados_LogReg_Padrao.csv', index=False, header=True)
    records = list()

    # KNN
    criar_individuo_KNN()
    data_frame = pd.DataFrame.from_records(records)
    data_frame.to_csv(f'resultados_KNN_Padrao.csv', index=False, header=True)
    records = list()

    # AdaBoost
    criar_individuo_AB()
    data_frame = pd.DataFrame.from_records(records)
    data_frame.to_csv(f'resultados_AdaBoost_Padrao.csv', index=False, header=True)
    records = list()


if __name__ == "__main__":
    main()

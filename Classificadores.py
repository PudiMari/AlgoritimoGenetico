import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import *
from sklearn.decomposition import *
from sklearn.impute import *
from sklearn.pipeline import Pipeline
from sklearn.datasets import *
from sklearn.feature_selection import *
from deap import base, creator, tools, algorithms
import warnings

from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import random
from datetime import datetime
import csv

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

PARAMS_STRATEGY = ['mean', 'median', 'most_frequent']
PARAMS_K = list(range(3, x_train.shape[1], 5))
PARAMS_N_ESTIMATORS = list(range(1, 100, 5))
PARAMS_MAX_DEPTH = list(range(1, 20))
PARAMS_MIN_SAMPLES_SPLIT = list(range(2, 20))
PARAMS_MIN_SAMPLES_LEAF = list(range(2, 20))

param_grid_RF = {
    'strategy': range(len(PARAMS_STRATEGY)),
    'k': range(len(PARAMS_K)),
    'n_estimators': range(len(PARAMS_N_ESTIMATORS)),
    'max_depth': range(len(PARAMS_MAX_DEPTH)),
    'min_samples_split': range(len(PARAMS_MIN_SAMPLES_SPLIT)),
    'min_samples_leaf': range(len(PARAMS_MIN_SAMPLES_LEAF)),
}

PARAMS_STRATEGY = ['mean', 'median', 'most_frequent']
PARAMS_K = list(range(3, x_train.shape[1], 5))
PARAMS_KERNEL = ['linear', 'poly', 'rbf', 'sigmoid']
PARAMS_C = list(range(1, 20))
PARAMS_DEGREE = list(range(3, 20))

param_grid_SVC = {
    'strategy': range(len(PARAMS_STRATEGY)),
    'k': range(len(PARAMS_K)),
    'kernel': range(len(PARAMS_KERNEL)),
    'c': range(len(PARAMS_C)),
    'degree': range(len(PARAMS_DEGREE)),
}

PARAMS_STRATEGY = ['mean', 'median', 'most_frequent']
PARAMS_K = list(range(3, x_train.shape[1], 5))
PARAMS_PENALTY = ['l2', None]
PARAMS_C = list(range(1, 20))
PARAMS_SOLVER = ['newton-cg', 'lbfgs', 'sag', 'saga', 'newton-cholesky']

param_grid_LogReg = {
    'strategy': range(len(PARAMS_STRATEGY)),
    'k': range(len(PARAMS_K)),
    'penalty': range(len(PARAMS_PENALTY)),
    'c': range(len(PARAMS_C)),
    'solver': range(len(PARAMS_SOLVER)),
}

PARAMS_STRATEGY = ['mean', 'median', 'most_frequent']
PARAMS_K = list(range(3, x_train.shape[1], 5))
PARAMS_N_NEIGHBORS = list(range(2, 20))
PARAMS_WEIGHTS = ['uniform', 'distance', None]
PARAMS_ALGORITHM = ['auto', 'ball_tree', 'kd_tree', 'brute']
PARAMS_LEAF_SIZE = list(range(10, 50))

param_grid_KNN = {
    'strategy': range(len(PARAMS_STRATEGY)),
    'k': range(len(PARAMS_K)),
    'n_neighbors': range(len(PARAMS_N_NEIGHBORS)),
    'weights': range(len(PARAMS_WEIGHTS)),
    'algorithm': range(len(PARAMS_ALGORITHM)),
    'leaf_size': range(len(PARAMS_LEAF_SIZE)),
}
def evaluate_RF(individual):
    strategy, k, n_estimators, max_depth, min_samples_split, min_samples_leaf = individual

    # print(individual)

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy=PARAMS_STRATEGY[strategy], copy=True)),
        ('scaler', StandardScaler()),
        ('feature-selection', SelectKBest(k=PARAMS_K[k])),
        ('randomForest', RandomForestClassifier(
            n_estimators=PARAMS_N_ESTIMATORS[n_estimators],
            max_depth=PARAMS_MAX_DEPTH[max_depth],
            min_samples_split=PARAMS_MIN_SAMPLES_SPLIT[min_samples_split],
            min_samples_leaf=PARAMS_MIN_SAMPLES_LEAF[min_samples_leaf],
            random_state=RANDOM_STATE))])

    start_time = datetime.now()
    pipe.fit(x_train, y_train)
    scores = cross_val_score(pipe, x_test, y_test, cv=5, scoring='f1_weighted')
    f1 = scores.mean()
    end_time = datetime.now()

    records.append({
        "strategy": PARAMS_STRATEGY[strategy],
        "k": PARAMS_K[k],
        "n_estimators": PARAMS_N_ESTIMATORS[n_estimators],
        "max_depth": PARAMS_MAX_DEPTH[max_depth],
        "min_samples_split": PARAMS_MIN_SAMPLES_SPLIT[min_samples_split],
        "min_samples_leaf": PARAMS_MIN_SAMPLES_LEAF[min_samples_leaf],
        "f1": f1,
        "elapsed_time": (end_time - start_time).total_seconds()
    })

    return f1,

def evaluate_SVC(individual):
    strategy, k, kernel, c, degree = individual

    # print(individual)

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy=PARAMS_STRATEGY[strategy], copy=True)),
        ('scaler', StandardScaler()),
        ('feature-selection', SelectKBest(k=PARAMS_K[k])),
        ('svc', SVC(
            kernel=PARAMS_KERNEL[kernel],
            C=PARAMS_C[c],
            degree=PARAMS_DEGREE[degree],
            random_state=RANDOM_STATE))])

    start_time = datetime.now()
    pipe.fit(x_train, y_train)
    scores = cross_val_score(pipe, x_test, y_test, cv=5, scoring='f1_weighted')
    f1 = scores.mean()
    end_time = datetime.now()

    records.append({
        "strategy": PARAMS_STRATEGY[strategy],
        "k": PARAMS_K[k],
        "kernel": PARAMS_KERNEL[kernel],
        "c": PARAMS_C[c],
        "degree": PARAMS_DEGREE[degree],
        "f1": f1,
        "elapsed_time": (end_time - start_time).total_seconds()
    })

    return f1,

def evaluate_LogReg(individual):
    strategy, k, penalty, c, solver = individual

    # print(individual)

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy=PARAMS_STRATEGY[strategy], copy=True)),
        ('scaler', StandardScaler()),
        ('feature-selection', SelectKBest(k=PARAMS_K[k])),
        ('logistic regression', LogisticRegression(
            penalty=PARAMS_PENALTY[penalty],
            C=PARAMS_C[c],
            solver=PARAMS_SOLVER[solver],
            random_state=RANDOM_STATE))])

    start_time = datetime.now()
    pipe.fit(x_train, y_train)
    scores = cross_val_score(pipe, x_test, y_test, cv=5, scoring='f1_weighted')
    f1 = scores.mean()
    end_time = datetime.now()

    records.append({
        "strategy": PARAMS_STRATEGY[strategy],
        "k": PARAMS_K[k],
        "penalty": PARAMS_PENALTY[penalty],
        "c": PARAMS_C[c],
        "solver": PARAMS_SOLVER[solver],
        "f1": f1,
        "elapsed_time": (end_time - start_time).total_seconds()
    })

    return f1,

def evaluate_KNN(individual):
    strategy, k, n_neighbors, weights, algorithm, leaf_size = individual

    # print(individual)

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy=PARAMS_STRATEGY[strategy], copy=True)),
        ('scaler', StandardScaler()),
        ('feature-selection', SelectKBest(k=PARAMS_K[k])),
        ('knn', KNeighborsClassifier(
            n_neighbors=PARAMS_N_NEIGHBORS[n_neighbors],
            weights=PARAMS_WEIGHTS[weights],
            algorithm=PARAMS_ALGORITHM[algorithm],
            leaf_size=PARAMS_LEAF_SIZE[leaf_size]))])

    start_time = datetime.now()
    pipe.fit(x_train, y_train)
    scores = cross_val_score(pipe, x_test, y_test, cv=5, scoring='f1_weighted')
    f1 = scores.mean()
    end_time = datetime.now()

    records.append({
        "strategy": PARAMS_STRATEGY[strategy],
        "k": PARAMS_K[k],
        "n_neighbors": PARAMS_N_NEIGHBORS[n_neighbors],
        "weights": PARAMS_WEIGHTS[weights],
        "algorithm": PARAMS_ALGORITHM[algorithm],
        "leaf_size": PARAMS_LEAF_SIZE[leaf_size],
        "f1": f1,
        "elapsed_time": (end_time - start_time).total_seconds()
    })

    return f1,

def criar_individuo(ind_class, param_grid):
    individuo = []
    for key, values in param_grid.items():
        numero = np.random.randint(0, len(values))
        individuo.append(numero)
    # print(individuo)
    return ind_class(individuo)


def criar_individuo_randomForest():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # toolbox.register("attr_int", np.random.randint, 1, 100)
    toolbox.register("individual", criar_individuo, creator.Individual, param_grid=param_grid_RF)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_RF)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt,
                     low=np.zeros(6),
                     up=[
                         len(PARAMS_STRATEGY) - 1,
                         len(PARAMS_K) - 1,
                         len(PARAMS_N_ESTIMATORS) - 1,
                         len(PARAMS_MAX_DEPTH) - 1,
                         len(PARAMS_MIN_SAMPLES_SPLIT) - 1,
                         len(PARAMS_MIN_SAMPLES_LEAF) - 1,
                     ], indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=10)

    # modificar gerações
    start_time = datetime.now()
    algorithms.eaMuPlusLambda(population, toolbox, mu=10, lambda_=50, cxpb=0.7, mutpb=0, ngen=3, stats=None,
                              halloffame=None)
    end_time = datetime.now()

    best_individual = tools.selBest(population, k=36)[0]
    print(f"Melhores hiperparametros encontrados com Random Forest: {best_individual} duration: {end_time - start_time}")

    strategy, k, n_estimators, max_depth, min_samples_split, min_samples_leaf = best_individual

    best_model = Pipeline([
        ('imputer', SimpleImputer(strategy=PARAMS_STRATEGY[strategy], copy=True)),
        ('scaler', StandardScaler()),
        ('feature-selection', SelectKBest(k=PARAMS_K[k])),
        ('randomForest', RandomForestClassifier(
            n_estimators=PARAMS_N_ESTIMATORS[n_estimators],
            max_depth=PARAMS_MAX_DEPTH[max_depth],
            min_samples_split=PARAMS_MIN_SAMPLES_SPLIT[min_samples_split],
            min_samples_leaf=PARAMS_MIN_SAMPLES_LEAF[min_samples_leaf],
            random_state=RANDOM_STATE))])

    best_model.fit(x_train, y_train)
    scores = cross_val_score(best_model, x_test, y_test, cv=5, scoring='f1_weighted')
    f1 = scores.mean()

    print("F1-score:", f1)
    return best_individual, start_time, f1

def criar_individuo_SVC():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # toolbox.register("attr_int", np.random.randint, 1, 100)
    toolbox.register("individual", criar_individuo, creator.Individual, param_grid=param_grid_SVC)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_SVC)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt,
                     low=np.zeros(5),
                     up=[
                         len(PARAMS_STRATEGY) - 1,
                         len(PARAMS_K) - 1,
                         len(PARAMS_KERNEL) - 1,
                         len(PARAMS_C) - 1,
                         len(PARAMS_DEGREE) - 1,
                     ], indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=10)

    # modificar gerações
    start_time = datetime.now()
    algorithms.eaMuPlusLambda(population, toolbox, mu=10, lambda_=50, cxpb=0.7, mutpb=0, ngen=3, stats=None,
                              halloffame=None)
    end_time = datetime.now()

    best_individual = tools.selBest(population, k=36)[0]
    print(f"Melhores hiperparametros encontrados com SVC: {best_individual} duration: {end_time - start_time}")

    strategy, k, kernel, c, degree = best_individual

    best_model = Pipeline([
        ('imputer', SimpleImputer(strategy=PARAMS_STRATEGY[strategy], copy=True)),
        ('scaler', StandardScaler()),
        ('feature-selection', SelectKBest(k=PARAMS_K[k])),
        ('svc', SVC(
            kernel=PARAMS_KERNEL[kernel],
            C=PARAMS_C[c],
            degree=PARAMS_DEGREE[degree],
            random_state=RANDOM_STATE))])

    best_model.fit(x_train, y_train)
    scores = cross_val_score(best_model, x_test, y_test, cv=5, scoring='f1_weighted')
    f1 = scores.mean()

    print("F1-score:", f1)
    return best_individual, start_time, f1

def criar_individuo_LogReg():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # toolbox.register("attr_int", np.random.randint, 1, 100)
    toolbox.register("individual", criar_individuo, creator.Individual, param_grid=param_grid_LogReg)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_LogReg)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt,
                     low=np.zeros(5),
                     up=[
                         len(PARAMS_STRATEGY) - 1,
                         len(PARAMS_K) - 1,
                         len(PARAMS_PENALTY) - 1,
                         len(PARAMS_C) - 1,
                         len(PARAMS_SOLVER) - 1,
                     ], indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=10)

    # modificar gerações
    start_time = datetime.now()
    algorithms.eaMuPlusLambda(population, toolbox, mu=10, lambda_=50, cxpb=0.7, mutpb=0, ngen=3, stats=None,
                              halloffame=None)
    end_time = datetime.now()

    best_individual = tools.selBest(population, k=36)[0]
    print(f"Melhores hiperparametros encontrados com Logistic Regression: {best_individual} duration: {end_time - start_time}")

    strategy, k, penalty, c, solver = best_individual

    best_model = Pipeline([
        ('imputer', SimpleImputer(strategy=PARAMS_STRATEGY[strategy], copy=True)),
        ('scaler', StandardScaler()),
        ('feature-selection', SelectKBest(k=PARAMS_K[k])),
        ('logistic regression', LogisticRegression(
            penalty=PARAMS_PENALTY[penalty],
            C=PARAMS_C[c],
            solver=PARAMS_SOLVER[solver],
            random_state=RANDOM_STATE))])

    best_model.fit(x_train, y_train)
    scores = cross_val_score(best_model, x_test, y_test, cv=5, scoring='f1_weighted')
    f1 = scores.mean()

    print("F1-score:", f1)
    return best_individual, start_time, f1

def criar_individuo_KNN():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # toolbox.register("attr_int", np.random.randint, 1, 100)
    toolbox.register("individual", criar_individuo, creator.Individual, param_grid=param_grid_KNN)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_KNN)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt,
                     low=np.zeros(6),
                     up=[
                         len(PARAMS_STRATEGY) - 1,
                         len(PARAMS_K) - 1,
                         len(PARAMS_N_NEIGHBORS) - 1,
                         len(PARAMS_WEIGHTS) - 1,
                         len(PARAMS_ALGORITHM) - 1,
                         len(PARAMS_LEAF_SIZE) - 1,
                     ], indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=10)

    # modificar gerações
    start_time = datetime.now()
    algorithms.eaMuPlusLambda(population, toolbox, mu=10, lambda_=50, cxpb=0.7, mutpb=0, ngen=3, stats=None,
                              halloffame=None)
    end_time = datetime.now()

    best_individual = tools.selBest(population, k=36)[0]
    print(f"Melhores hiperparametros encontrados com KNN: {best_individual} duration: {end_time - start_time}")

    strategy, k, n_neighbors, weights, algorithm, leaf_size = best_individual

    best_model = Pipeline([
        ('imputer', SimpleImputer(strategy=PARAMS_STRATEGY[strategy], copy=True)),
        ('scaler', StandardScaler()),
        ('feature-selection', SelectKBest(k=PARAMS_K[k])),
        ('knn', KNeighborsClassifier(
            n_neighbors=PARAMS_N_NEIGHBORS[n_neighbors],
            weights=PARAMS_WEIGHTS[weights],
            algorithm=PARAMS_ALGORITHM[algorithm],
            leaf_size=PARAMS_LEAF_SIZE[leaf_size]))])

    best_model.fit(x_train, y_train)
    scores = cross_val_score(best_model, x_test, y_test, cv=5, scoring='f1_weighted')
    f1 = scores.mean()

    print("F1-score:", f1)
    return best_individual, start_time, f1


def main():
    global records
    for i in range(5):

        # RandomForest
        best_individual, start_time, f1 = criar_individuo_randomForest()
        df = pd.DataFrame.from_records(records)
        df.to_csv(f'resultados_randomForest_{i}.csv', index=False, header=True)
        records = list()

        # SVC
        best_individual, start_time, f1 = criar_individuo_SVC()
        df = pd.DataFrame.from_records(records)
        df.to_csv(f'resultados_SVC_{i}.csv', index=False, header=True)
        records = list()

        # Logistic Regression
        best_individual, start_time, f1 = criar_individuo_LogReg()
        df = pd.DataFrame.from_records(records)
        df.to_csv(f'resultados_LogReg_{i}.csv', index=False, header=True)
        records = list()

        # KNN
        best_individual, start_time, f1 = criar_individuo_KNN()
        df = pd.DataFrame.from_records(records)
        df.to_csv(f'resultados_KNN_{i}.csv', index=False, header=True)
        records = list()


if __name__ == "__main__":
    main()

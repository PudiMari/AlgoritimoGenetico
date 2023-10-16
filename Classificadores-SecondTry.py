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
from sklearn.model_selection import GridSearchCV
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

column_names_modelos = [' Modelos', 'F1 Score']

#Corrigir aviso de UserWarning
selector = VarianceThreshold(threshold=0.01)
x_train = selector.fit_transform(x_train)
x_test = selector.transform(x_test)

imputer = SimpleImputer(strategy='mean')
selector = SelectKBest(k=36)

x_train_imputed = imputer.fit_transform(x_train)
x_test_imputed = imputer.transform(x_test)
        
x_train_selected = selector.fit_transform(x_train_imputed, y_train)
x_test_selected = selector.fit_transform(x_test_imputed, y_test)

param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

model = RandomForestClassifier(random_state=RANDOM_STATE)

grid_search = GridSearchCV(model, param_grid, scoring='f1_weighted', cv=5)

grid_search.fit(x_train_selected, y_train)
best_model = grid_search.best_estimator_

print("Best Model: ", best_model)
    

#model = RandomForestClassifier(
#            n_estimators=18,
#            max_depth=17,
#            min_samples_split=4,
#            random_state=RANDOM_STATE
#        )

#model.fit(x_train_selected, y_train)

#y_pred = model.predict(x_test_selected)

#print("Classification Report:\n", classification_report(y_test, y_pred))
#print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#scores = cross_val_score(model, x_train_selected, y_train, cv=5, scoring='accuracy')
#print("Cross-Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
    
ind = creator.Individual([1, 0, 1, 1, 0])

#print(ind)
#print(type(ind))
#print(type(ind.fitness))

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
bit = toolbox.attr_bool()
ind = toolbox.individual()
pop = toolbox.population(n=10)

#print("bit is of type %s and has value\n%s" % (type(bit), bit))
#print("ind is of type %s and contains %d bits\n%s" % (type(ind), len(ind), ind))
#print("pop is of type %s and contains %d individuals\n%s" % (type(pop), len(pop), pop))
    
def evalOneMax(individual):
    selected_features = [i for i, val in enumerate(individual) if val == 1]

    if len(selected_features) == 0:
        return 0.0, 
    
    #print("selected: ", selected_features)
    
    #model = RandomForestClassifier(n_estimators=18,
            #max_depth=17,
            #min_samples_split=4,
            #random_state=RANDOM_STATE)
    x_train_selected = x_train[:, selected_features]
    x_test_selected = x_test[:, selected_features]
    best_model.fit(x_train_selected, y_train)

    y_pred = best_model.predict(x_test_selected)

    f1 = f1_score(y_test, y_pred, average='weighted')
    
    with open('modelos_randomForest.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
        
            if file.tell() == 0:
                writer.writerow(column_names_modelos)
            writer.writerow([best_model, f1])

    return f1,

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.10)
toolbox.register("select", tools.selTournament, tournsize=7)

#print("Classification Report:\n", classification_report(y_test, y_pred))
#print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

ind = toolbox.individual()
mutant = toolbox.clone(ind)
        
population = toolbox.population(n=10)

#modificar gerações
algorithms.eaMuPlusLambda(population, toolbox, mu=10, lambda_=50, cxpb=0.7, mutpb=0.3, ngen=3, stats=None, halloffame=None)

def main():
    import numpy
    
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, stats=stats, halloffame=hof, verbose=True)
    
    return pop, logbook, hof
        
if __name__ == "__main__":
    pop, log, hof = main()
    print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))
    
    import matplotlib.pyplot as plt
    gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")
    plt.plot(gen, avg, label="average")
    plt.plot(gen, min_, label="minimum")
    plt.plot(gen, max_, label="maximum")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="lower right")
    plt.show()
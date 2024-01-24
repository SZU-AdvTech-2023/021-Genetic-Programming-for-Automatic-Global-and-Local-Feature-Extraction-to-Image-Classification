# python packages
import random
import time
import operator
import warnings

from sklearn.decomposition import PCA

import pca_evalGP_main as evalGP
# only for strongly typed GP
import gp_restrict
import numpy as np
# deap package
from deap import base, creator, tools, gp
from strongGPDataType import Int1, Int2, Int3, Img, Region, Vector, Vector1
import feature_function as fe_fs
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from functools import partial
from sklearn.decomposition import KernelPCA

# 将 ConvergenceWarning 添加到过滤器中
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

'FLGP'

dataSetName='KTH2'
randomSeeds=2

x_train = np.load(dataSetName + '_train_data.npy')/ 255.0
y_train = np.load(dataSetName + '_train_label.npy')
x_test = np.load(dataSetName + '_test_data.npy')/ 255.0
y_test = np.load(dataSetName + '_test_label.npy')

print(x_train.shape)
print(x_test.shape)

# parameters:
population = 100
generation = 50
cxProb = 0.8
mutProb = 0.19
elitismProb = 0.01
totalRuns = 1
initialMinDepth = 2
initialMaxDepth = 6
maxDepth = 8

bound1, bound2 = x_train[1, :, :].shape

##GP
pset = gp.PrimitiveSetTyped('MAIN', [Img], Vector1, prefix='Image')
# PCA
# pset.addPrimitive(fe_fs.PCA_func, [Vector1], Vector1, name='PCA_func')
#Feature concatenation
pset.addPrimitive(fe_fs.root_con, [Vector1, Vector1], Vector1, name='FeaCon')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector], Vector1, name='FeaCon2')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector, Vector], Vector1, name='FeaCon3')
# Global feature extraction
pset.addPrimitive(fe_fs.all_dif, [Img], Vector, name='Global_DIF')
pset.addPrimitive(fe_fs.all_histogram, [Img], Vector, name='Global_Histogram')
pset.addPrimitive(fe_fs.global_hog, [Img], Vector, name='Global_HOG')
pset.addPrimitive(fe_fs.all_lbp, [Img], Vector, name='Global_uLBP')
pset.addPrimitive(fe_fs.all_sift, [Img], Vector, name='Global_SIFT')
# Local feature extraction
pset.addPrimitive(fe_fs.all_dif, [Region], Vector, name='Local_DIF')
pset.addPrimitive(fe_fs.all_histogram, [Region], Vector, name='Local_Histogram')
pset.addPrimitive(fe_fs.local_hog, [Region], Vector, name='Local_HOG')
pset.addPrimitive(fe_fs.all_lbp, [Region], Vector, name='Local_uLBP')
pset.addPrimitive(fe_fs.all_sift, [Region], Vector, name='Local_SIFT')
# Region detection operators
pset.addPrimitive(fe_fs.regionS, [Img, Int1, Int2, Int3], Region, name='Region_S')
pset.addPrimitive(fe_fs.regionR, [Img, Int1, Int2, Int3, Int3], Region, name='Region_R')

# 定义生成函数，并使用functools.partial代替lambda函数
# 定义生成函数
def generate_X(bound):
    return random.randint(0, bound - 20)
def generate_Y(bound):
    return random.randint(0, bound - 20)
def generate_Size():
    return random.randint(20, 51)
X_func = partial(generate_X, bound1)
Y_func = partial(generate_Y, bound2)
Size_func = generate_Size

# Terminals
pset.renameArguments(ARG0='Grey')
pset.addEphemeralConstant('X', X_func, Int1)
pset.addEphemeralConstant('Y', Y_func, Int2)
pset.addEphemeralConstant('Size', Size_func, Int3)
'''
pset.addEphemeralConstant('X', lambda: random.randint(0, bound1 - 20), Int1)
pset.addEphemeralConstant('Y', lambda: random.randint(0, bound2 - 20), Int2)
pset.addEphemeralConstant('Size', lambda: random.randint(20, 51), Int3)
'''
#fitnesse evaluaiton
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("mapp", map)

def evalTrain(individual):
    # print('individual:',individual)
    func = toolbox.compile(expr=individual)
    # print('编译完成')
    train_tf = []
    for i in range(0, len(y_train)):
        # print('i = ',i) 针对每一行数据都要进行一次，为什么这个可以feacon2，但是另一个不可以？
        # 可以看到这里是一条一条的数据进入，通过个体转化到特征空间的，所以PCA应该在这个阶段进行
        # print('转换成功的特征：',np.asarray(func(x_train[i, :, :])))
        train_tf.append(np.asarray(func(x_train[i, :, :])))
    # print('是否跳出来')
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
    # print(train_norm.shape)
    '''pca过程'''
    # pca = PCA(n_components=0.95) # 经验主义，能量取95%
    # train_pca = pca.fit_transform(train_norm)
    '''核pca过程'''
    # kpca = KernelPCA(n_components = int((train_norm.shape[1])/3),kernel='rbf', gamma=0.1)  # 根据数据特性选择合适的核函数和参数
    # train_kpca = kpca.fit_transform(train_norm)
    # print('（PCA之前的特征数量，PCA后的特征数量）：',(train_norm.shape[1],train_pca.shape[1]))
    '''
    ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
    '''
    lsvm = LinearSVC(dual='auto', max_iter=1000)
    # norm_begin_time = time.time()
    accuracy = round(100 * cross_val_score(lsvm, train_norm, y_train, cv=3).mean(), 2)
    # norm_end_time = time.time()
    # pca_begin_time = time.time()
    # accuracy_pca = round(100 * cross_val_score(lsvm, train_pca, y_train, cv=3).mean(), 2)
    # pca_end_time = time.time()
    # kpca_begin_time = time.time()
    # accuracy_kpca = round(100 * cross_val_score(lsvm, train_kpca, y_train, cv=3).mean(), 2)
    # kpca_end_time = time.time()
    # print('evalTrain accuracy:',accuracy_pca)
    # print('evalTrain accuracy-pca:', accuracy_pca,pca_end_time-pca_begin_time)
    # print('evalTrain accuracy-kpca:', accuracy_kpca,kpca_end_time-kpca_begin_time)
    # print('KTH acc:',accuracy)
    return accuracy,

# genetic operator
toolbox.register("evaluate", evalTrain)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))

def GPMain(randomSeeds):
    random.seed(randomSeeds)

    pop = toolbox.population(population)
    hof = tools.HallOfFame(10)
    log = tools.Logbook()
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size_tree = tools.Statistics(key=len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size_tree=stats_size_tree)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    log.header = ["gen", "evals"] + mstats.fields

    pop, log = evalGP.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generation,
                               stats=mstats, halloffame=hof, verbose=True)

    return pop, log, hof

def evalTest(individual):
    func = toolbox.compile(expr=individual)
    train_tf = []
    test_tf = []
    for i in range(0, len(y_train)):
        train_tf.append(np.asarray(func(x_train[i, :, :])))
    for j in range(0, len(y_test)):
        test_tf.append(np.asarray(func(x_test[j, :, :])))
    train_tf = np.asarray(train_tf)
    test_tf = np.asarray(test_tf)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
    test_norm = min_max_scaler.transform(np.asarray(test_tf))
    lsvm= LinearSVC()
    lsvm.fit(train_norm, y_train)
    accuracy = round(100*lsvm.score(test_norm, y_test),2)
    return train_tf.shape[1], accuracy

if __name__ == "__main__":
    beginTime = time.process_time()
    pop, log, hof = GPMain(randomSeeds)
    endTime = time.process_time()
    trainTime = endTime - beginTime

    num_features, testResults = evalTest(hof[0])
    endTime1 = time.process_time()
    testTime = endTime1 - endTime

    print('Best individual ', hof[0])
    print('Test results  ', testResults)
    print('Train time  ', trainTime)
    print('Test time  ', testTime)
    print('End')

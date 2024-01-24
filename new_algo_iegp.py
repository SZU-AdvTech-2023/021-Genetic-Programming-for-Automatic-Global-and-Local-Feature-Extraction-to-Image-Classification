import random
from deap import tools
from collections import defaultdict


def pop_compare(ind1, ind2):
    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    for idx, node in enumerate(ind1[1:],1):
        types1[node.ret].append(idx)
    for idx, node in enumerate(ind2[1:],1):
        types2[node.ret].append(idx)
    return types1==types2

def varAnd(population, toolbox, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param elitpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.  A
    first loop over :math:`P_\mathrm{o}` is executed to mate pairs of
    consecutive individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting :math:`P_\mathrm{o}`
    is returned.

    This variation is named *And* beceause of its propention to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematicaly, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
    offspring = [toolbox.clone(ind) for ind in population]
    new_cxpb=cxpb/(cxpb+mutpb)

    #num_cx=int(new_cxpb*len(offspring))
    #num_mu=len(offspring)-num_cx
    #print(new_cxpb, new_mutpb)
    # Apply crossover and mutation on the offspring
    i = 1
    while i < len(population):
        if random.random() < new_cxpb:
            if (offspring[i - 1] == offspring[i]) or pop_compare(offspring[i - 1], offspring[i]):
                offspring[i - 1], = toolbox.mutate(offspring[i - 1])
                offspring[i], = toolbox.mutate(offspring[i])
            else:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            i = i + 2
        else:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            i = i + 1
        if i == len(population)-1:
            # 检查是否有重复
            offspring_value = []
            for inv in offspring:
                inv_str = str(inv)  # 获取个体的字符串表示形式
                # print(inv_str)
                offspring_value.append(inv_str)
            # print('offspring_value:', offspring_value)
            # 确保子代中每个个体是唯一的
            unique_population = list(set(offspring_value))
            # print('去重成功')
            # print('此时子代的长度：', len(unique_population))
            i = len(unique_population) - 1
            # print('去重后开始的个数：',i)
    '''
    print('offspring:',offspring)
    offspring_value = []
    for inv in offspring:
        inv_str = str(inv)  # 获取个体的字符串表示形式
        print(inv_str)
        offspring_value.append(inv_str)
    print('offspring_value:', offspring_value)
    # 确保子代中每个个体是唯一的
    unique_population = list(set(offspring_value))
    print('去重成功')
    print('此时子代的长度：',len(unique_population))
    ####
    while len(offspring) < len(population):
        ind = toolbox.individual()
        while ind in seen:
            ind = toolbox.individual()
            del ind.fitness.values
        seen.add(ind)
        offspring.append(ind)
        '''
    # print('最后输出的offspring的个数是否够数？',len(offspring))
    return offspring


def eaSimple(population, toolbox, cxpb, mutpb, elitpb, ngen, randomseed, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param etilpb: The probability of elitism
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evalutions for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            elitismNum
            offspringE=selectElitism(population,elitismNum)
            population = select(population, len(population)-elitismNum)
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            offspring=offspring+offspringE
            evaluate(offspring)
            population = offspring.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` and :meth::`toolbox.selectElitism`,
     aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    # Evaluate the individuals with an invalid fitness
    # invalid_ind = [ind for ind in population if not ind.fitness.valid]
    # print(len(invalid_ind))
    # k = 0 # 计数作用
    for i in population:
        # print('评估fitness i:',k,i)
        # 一开始传的就是一个空的hof
        # k = k+1
        i.fitness.values = toolbox.evaluate(individual=i, hof=[])

    if halloffame is not None:
        halloffame.update(population)
    hof_store = tools.HallOfFame(5 * len(population))
    hof_store.update(population)
    cop_po = population
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    for gen in range(1, ngen + 1):
        # 演化开始的时候传入hof是一个种群，所以evalTrain中的hof不为空
        # Select the next generation individuals by elitism
        elitismNum = int(elitpb * len(population))
        population_for_eli = [toolbox.clone(ind) for ind in population]
        offspringE = toolbox.selectElitism(population_for_eli, k=elitismNum)

        # Select the next generation individuals for crossover and mutation
        offspring = toolbox.select(population, len(population) - elitismNum)
        # Vary the pool of individuals 产生子代
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        # add offspring from elitism into current offspring
        # generate the next generation individuals

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        print('invalid_ind size',len(invalid_ind)) # 这个统计的是无效的个体
        k = 0
        for i in invalid_ind:
            i.fitness.values = toolbox.evaluate(individual=i, hof=cop_po)
            # print('评估fitness i:',k,i)

        offspring[0:0] = offspringE

        # Update the hall of fame with the generated
        if halloffame is not None:
            halloffame.update(offspring)
        cop_po = offspring.copy()
        hof_store.update(offspring)
        for i in hof_store:
            cop_po.append(i)
        population[:] = offspring
        # Append the current generation statistics to the logbook 将当前生成的统计数据附加到日志中
        record = stats.compile(population) if stats else {}
        # print(record)
        logbook.record(gen=gen, nevals=len(offspring), **record)
        # print(record)
        if verbose:
            print(logbook.stream)
    return population, logbook


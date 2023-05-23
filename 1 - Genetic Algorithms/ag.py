print('Author: Gabriel Padilha Alves') # Author: GABRIEL PADILHA ALVES

import numpy as np
from piece_code_ag import get_bits, get_float
import matplotlib.pyplot as plt


def mutation(child, Pm, L, lower_bound, upper_bound):
    """Mutation of a new individual (child)

    Args:
        child (string): strings of bits representing
            the genes of the child
        Pm (float): probability of mutation
        L (int): number of bits
        lower_bound (float): lower bound of the domain
        upper_bound (float): upper bound of the domain
        
    Returns:
        child (float): equivalent float of the mutated child
    """
    c = child
    point = np.random.randint(L)
    if np.random.rand(1) <= Pm:
        c = child[:point] + str((int(child[point]) + 1) % 2) + child[point+1:]

    child = get_float(c)
    
    # NaN is a float (in bits is like: 111111111... or 01111111...)
    # so we have to put this huge value inside the bounds
    if c[0] == '0': # if is a positive infinity/nan, adjust to the upper bound
        return upper_bound if np.isnan(child) else child
    else: # if is a negative infinity/nan, adjust to the lower bound
        return lower_bound if np.isnan(child) else child


def crossover(parents, Pc, L, lower_bound, upper_bound):
    """Crossover between two parents

    Args:
        parents (list of size 2): contains one float in each dimension
        Pc (float): probability of crossover
        L (int): number of bits

    Returns:
        child0, child1: two strings of bits, generated from the crossover between the parents
    """
    parent0 = get_bits(parents[0])
    parent1 = get_bits(parents[1])

    point = np.random.randint(L)  # locus point

    child0 = parent0[0:point] + parent1[point:]
    child1 = parent1[0:point] + parent0[point:]
    if np.random.rand(1) > Pc:
        child0 = parent0
        child1 = parent1
    
    c0 = get_float(child0)
    if child0[0] == '0': # if is a positive infinity/nan, adjust to the upper bound
        child0 = get_bits(upper_bound) if np.isnan(c0) else child0
    else: # if is a negative infinity/nan, adjust to the lower bound
        child0 = get_bits(lower_bound) if np.isnan(c0) else child0
    
    c1 = get_float(child1)
    if child1[0] == '0': # if is a positive infinity/nan, adjust to the upper bound
        child1 = get_bits(upper_bound) if np.isnan(c1) else child1
    else: # if is a negative infinity/nan, adjust to the lower bound
        child1 = get_bits(lower_bound) if np.isnan(c1) else child1

    return child0, child1


def fitness(pop):
    """Objective function evaluation (fitness of every individual)

    Args:
        pop (vector with n dimensions): individuals to be evaluated

    Returns:
        evaluation of the population
    """
    return pop + np.abs(np.sin(32*pop))


def roulette_selection(fobj):
    """Roulette Selection

    Args:
        fobj (vector with n dimensions): fitness of every individual

    Returns:
        P (vector with n dimensions): probability of selection of individuals
    """
    P = fobj/np.sum(fobj)
    return np.ones(n)/n if all(np.isnan(P)) else P


def boundary_control(pop, lower_bound, upper_bound):
    """ Adjusts the population to be inside the bounds
    50% chance to adjust to the bounds of the domain
    50% chance to change to some random value inside the domain

    Args:
        pop (vector with n dimensions): individuals to be evaluated
        lower_bound (float): lower bound of the domain
        upper_bound (float): upper bound of the domain

    Returns:
        pop: population inside the bounds
    """
    if np.random.rand() > np.random.rand():
        pop[pop > upper_bound] = upper_bound
        pop[pop < lower_bound] = lower_bound
    else:
        pop[pop > upper_bound] = np.random.uniform(low=lower_bound, high=upper_bound, size=np.sum(pop > upper_bound))
        pop[pop < lower_bound] = np.random.uniform(low=lower_bound, high=upper_bound, size=np.sum(pop < lower_bound))
    return pop


def ag(n=100, ngen=50, nruns=5, Pc=0.5, Pm=0.2, lower_bound=0, upper_bound=np.pi, L=32, selectionII=False):
    """
    Genetic algorithm that runs for 'nruns'
    with 'n' population and 'ngen' generations
    
    Args:
        n (int): population size
        ngen (int): number of iterations (generations)
        nruns (int): number of runs
        Pc (float): probability of crossover
        Pm (float): probability of mutation
        lower_bound (float): lower bound of the domain
        upper_bound (float): upper bound of the domain
        L (int): number of bits
        selectionII (bool): if True, then the GA has a second selection
            where some members of the old population (parents) are maintaned
            instead of the children
    
    Returns:
        pop_run (numpy array n x ngen x nruns): population of each generation in every run
        fit_run (numpy array n x ngen x nruns): fitness of all the population for each generation in every run
    """

    # Preallocate variables
    pop_run = np.zeros((n, ngen, nruns))
    fit_run = np.zeros((n, ngen, nruns))

    for run in range(nruns):
        # Generate random population
        pop = np.float32(np.random.uniform(low=0.0, high=np.pi, size=n))
        
        for gen in range(ngen):
            # Fitness evaluation
            fobj = fitness(pop)
            
            # Save variables
            pop_run[:, gen, run] = pop
            fit_run[:, gen, run] = fobj

            # Probability of selection
            P = roulette_selection(fobj)

            # Preallocate new population
            new_pop = np.zeros(shape=(round(n/2)*2), dtype='float32')

            for i in range(int(round(n/2))):
                # Select a pair
                pair = np.random.choice(n, 2, p=P)
                parents = pop[pair]

                # Crossover
                child0, child1 = crossover(parents, Pc, L, lower_bound, upper_bound)

                # Mutation
                child0 = mutation(child0, Pm, L, lower_bound, upper_bound)
                child1 = mutation(child1, Pm, L, lower_bound, upper_bound)

                new_pop[i*2:i*2+2] = np.array([child0, child1], dtype='float32')

            # if n is odd, delete one random individual
            if n % 2 != 0:
                new_pop = np.delete(new_pop, np.random.randint(n), 0)

            # Boundary control
            new_pop = boundary_control(new_pop, lower_bound, upper_bound)
            
            if selectionII:
                # maintain the best between new_pop and pop
                f = fitness(new_pop)
                idx = f < fobj
                new_pop[idx] = pop[idx]
            
            pop = np.copy(new_pop)

    return pop_run, fit_run


def plot_average_fitness():
    """
    Plot average fitness
    """
    # Calculate average fitness of each iteration for each run
    fit_avg = np.mean(fit_run, axis=0, dtype=np.float64)

    plt.rc('font', size=24)      # controls default text sizes
    plt.rc('axes', titlesize=24)  # fontsize of the axes title
    linestyle = [
        'solid',   # '-'
        'dotted',  # ':'
        'dashed',  # '--'
        'dashdot',  # '-.'
    ]
    leg = []

    for run in range(nruns):
        leg.append('Run ' + str(run))
        plt.plot(np.arange(ngen), fit_avg[:, run], linestyle=linestyle[run % len(
            linestyle)], linewidth=2.5, alpha=1-(run/nruns)*0.8)

    # Labels and legend
    plt.xlabel('Generation')
    plt.ylabel('Average Cost')
    plt.legend(leg)
    plt.title(
        f'Average fitness, P_m = {Pm}, P_c = {Pc}, Population: {n}, Generations: {ngen}')

    # Set figure size and save
    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    fig.savefig(f'{nruns}runs_{n}pop_{ngen}gen_Pc{int(Pc*100)}_Pm{int(Pm*100)}.png', dpi=300)

    plt.show()
    pass


def plot_generation_fitness():
    """
    Plot chromosomes fitness in the function (for run 0)
    """
    generations = [0, round(ngen*0.25)-1, round(ngen*0.5)-1, round(ngen*0.75)-1, ngen-1]
    pop = pop_run[:, generations, 0]
    fit = fit_run[:, generations, 0]

    x = np.linspace(0, np.pi, num=1000)
    y = x + np.abs(np.sin(32*x))
    color = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4']
    for idx, gen in enumerate(generations):
        plt.figure(idx+2)
        plt.plot(x, y, linewidth=2.5)
        plt.scatter(pop[:, idx], fit[:, idx], c=color[idx])
        plt.legend(['Function', f'Gen {gen+1}'])
        plt.title(f'Generation {gen+1}, P_m = {Pm}, P_c = {Pc}, Population: {n}')
        fig = plt.gcf()
        fig.set_size_inches(16, 9)
        fig.savefig(f'gen{gen+1}fitness_Pc{int(Pc*100)}_Pm{int(Pm*100)}.png', dpi=300)
        
    plt.show()
    pass


if __name__ == "__main__":
    """
    Run Genetic Algorithm
    """
    nruns = 5
    ngen = 500
    n = 100
    Pc = 0.9
    Pm = 0.9
    selectionII = True
    # High Pc and/or High Pm are usually bad, but selectionII improves the GA substantially

    pop_run, fit_run = ag(n=n, ngen=ngen, nruns=nruns, Pc=Pc, Pm=Pm, selectionII=selectionII)
    plot_average_fitness()
    plot_generation_fitness()
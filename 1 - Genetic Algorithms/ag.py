import numpy as np
from piece_code_ag import get_bits, get_float
import matplotlib.pyplot as plt

def ag(n=100, ngen=50, nruns=5, Pc=0.5, Pm=0.2):
    """
    n: population size
    ngen: number of iterations (generations)
    nruns: number of runs
    Pc: probability of crossover
    Pm: probability of mutation
    """
    
    # Preallocate variables
    pop_run = np.zeros((n, ngen, nruns))
    fit_run = np.zeros((n, ngen, nruns))

    for run in range(nruns):
        # Generate random population
        pop = np.float32(np.random.uniform(low=0.0, high=np.pi, size=n))

        globalmax = -np.inf
        for gen in range(ngen):
            # Fitness evaluation of the population
            fobj = pop + np.abs(np.sin(32*pop))

            fobj_max = np.max(fobj)
            if fobj_max > globalmax:
                globalmax = fobj_max

            pop_run[:, gen, run] = pop
            fit_run[:, gen, run] = fobj

            # Probability of selection
            P = fobj/np.sum(fobj)

            if all(np.isnan(P)):
                P = np.ones(n)/n

            new_pop = np.zeros(shape=(n), dtype='float32')
            for i in range(int(n/2)):
                # Select a pair
                pair = np.random.choice(n, 2, p=P)
                parents = pop[pair]

                # Crossover
                parent0 = get_bits(parents[0])
                parent1 = get_bits(parents[1])

                point = np.random.randint(n) # locus point

                child0 = parent0[0:point] + parent1[point:]
                child1 = parent1[0:point] + parent0[point:]
                if np.random.rand(1) < Pc:
                    child0 = parent0
                    child1 = parent1
                
                # Mutation
                # c0, c1 = '', ''
                # for l in range(len(child0)):
                #     c0 = c0 + str( (int(child0[l]) + int(1*(np.random.rand(1) >= Pm))) % 2 )
                #     c1 = c1 + str( (int(child1[l]) + int(1*(np.random.rand(1) >= Pm))) % 2 )

                c0, c1 = child0, child1
                point = np.random.randint(32)
                if np.random.rand(1) >= Pm:
                    c0 = child0[:point] + str((int(child0[point]) + 1) % 2) + child0[point+1:]

                point = np.random.randint(32)
                if np.random.rand(1) >= Pm:
                    c1 = child1[:point] + str((int(child1[point]) + 1) % 2) + child1[point+1:]

                child0 = get_float(c0)
                child1 = get_float(c1)

                if np.isnan(child0):
                    child0 = 0
                    
                if np.isnan(child1):
                    child1 = 0
                
                new_pop[i*2:i*2+2] = np.array([child0, child1], dtype='float32')

            # if n is odd, delete one random individual
            if n % 2 != 0:
                new_pop = np.delete(new_pop, np.random.randint(n), 0)
            
            # Boundary control
            new_pop[new_pop > np.pi] = np.pi
            new_pop[new_pop < 0] = 0

            pop = new_pop

    return pop_run, fit_run, globalmax


"""
Run Genetic Algorithm
"""
nruns = 5
ngen = 500
n = 100
Pc = 0.5
Pm = 0.2

pop_run, fit_run, globalmax = ag(n=n, ngen=ngen, nruns=nruns, Pc=Pc, Pm=Pm)


"""
Plot average fitness
"""
# Calculate average fitness of each iteration for each run
fit_avg = np.mean(fit_run, axis=0, dtype=np.float64)

plt.rc('font', size=24)      # controls default text sizes
plt.rc('axes', titlesize=24) # fontsize of the axes title
linestyle = [
    'solid',   # '-'
    'dotted',  # ':'
    'dashed',  # '--'
    'dashdot', # '-.'
    ]
leg = []

for run in range(nruns):
    leg.append('Run ' + str(run))
    plt.plot(np.arange(ngen), fit_avg[:, run], linestyle=linestyle[run%len(linestyle)], linewidth=2.5, alpha=1-(run/nruns)*0.6)

# Labels and legend
plt.xlabel('Generation')
plt.ylabel('Average Cost')
plt.legend(leg)
plt.title(f'Average fitness, P_m = {Pm}, P_c = {Pc}, Population: {n}, Generations: {ngen}')

# Set figure size and save
fig = plt.gcf()
fig.set_size_inches(16, 9)
fig.savefig(f'{n}runs_{n}pop_{ngen}gen.png', dpi=300)

plt.show()



"""
Plot chromossomes fitness in the function (for run 0)
"""
generations = [0, round(ngen*0.25), round(ngen*0.5), round(ngen*0.75), ngen-1]
pop = pop_run[:, generations, 0]
fit = fit_run[:, generations, 0]

x = np.linspace(0, np.pi, num=1000)
y = x + np.abs(np.sin(32*x))
color = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf' ,'#1f77b4']
for idx, gen in enumerate(generations):
    plt.figure(idx+2)
    plt.plot(x, y, linewidth=2.5)
    plt.scatter(pop[:, idx], fit[:, idx], c=color[idx])
    plt.legend(['Function', f'Gen {gen+1}'])
    plt.title(f'Generation {gen+1}, P_m = {Pm}, P_c = {Pc}, Population: {n}')
    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    fig.savefig(f'gen{gen+1}_fitness.png', dpi=300)


plt.show()

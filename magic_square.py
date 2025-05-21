import numpy as np
import random
import matplotlib.pyplot as plt 
# Parameters
n = 3  # 3x3 magic square
population_size = 100
mutation_rate = 0.01
valid_set = set(np.random.permutation(n * n) + 1)

# Generate initial population (random permutations of 1 to n²)
def generate_individual(n):
    return np.random.permutation(n * n) + 1  # e.g., [8,1,6,3,5,7,4,9,2]

# fixed sum that all rows, columns, and both main diagonals of a magic square must equal.
def magic_constant(n):
    return n * (n**2 + 1) // 2 

# Fitness function: calculate the fitness score for a regular magic square (not perfect)
def fitness_score_regular(individual, n):
    # Reshape into n×n matrix
    square = np.array(individual).reshape(n, n)
    M = magic_constant(n)
    total_deviation = 0

    # Check rows and columns
    for i in range(n):
        row_sum = np.sum(square[i, :])
        col_sum = np.sum(square[:, i])
        total_deviation += abs(row_sum - M) + abs(col_sum - M)

    # Check diagonals
    diag1_sum = np.sum(np.diag(square))  # Main diagonal
    diag2_sum = np.sum(np.diag(np.fliplr(square)))  # Anti-diagonal
    total_deviation += abs(diag1_sum - M) + abs(diag2_sum - M)

    return 1 / (1 + total_deviation)  # Higher is better

def crossover(first, second, n):
    #get 2 matrixes and crossover them but the child must be valid
    #all unique numbers
    child = np.zeros_like(first)
    child_set = set()

    # crossover
    i = random.randint(0, n*n - 1)
    for j in range(len(first)):
        if j < i:
            child[j] = first[i]
        else:
            if second[j] not in child_set:
                child[j] = second[j]
            else:
                # we need to put a diffrent number to make it a valid permuttion
                child[j] = [e for e in valid_set if e not in child_set][0]
        child_set.add(child[j])


    if random.random() < mutation_rate:
        # do mutation on child, swap 2 random places
        i = random.randint(0, len(child) - 1)
        j = random.randint(0, len(child) - 1)
        child[i] , child[j] = child[j], child[i] 
    return child

population = [generate_individual(n) for _ in range(population_size)]
avg_fitness = []
best_fitness = []
age = 1
while(age < 500):
    avg_fitness.append(0)
    best_fitness.append(-1)
    population_fitness = [fitness_score_regular(individual, n) for individual in population]
    best_fitness[-1] = max(population_fitness)
    avg_fitness[-1]  = sum(population_fitness) / population_size

    if(best_fitness == 1):
        break;
    
    new_pop = []
    for i in range(population_size):
        # choose 2 for cross over
        first, second = random.choices(population, weights=population_fitness, k=2)      
        new_pop.append(crossover(first, second, n))
    population = new_pop
    age+=1

plt.plot(range(1, age), avg_fitness[:age], c='r')
plt.plot(range(1, age), best_fitness[:age], c='g')
plt.show()

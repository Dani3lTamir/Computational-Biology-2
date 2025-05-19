import numpy as np

# Parameters
n = 3  # 3x3 magic square
population_size = 100

# Generate initial population (random permutations of 1 to n²)
def generate_individual(n):
    return np.random.permutation(n * n) + 1  # e.g., [8,1,6,3,5,7,4,9,2]

population = [generate_individual(n) for _ in range(population_size)]
print("Population (first 3 individuals):")
for ind in population[:3]:
    print(ind.reshape(n, n))  # Reshape to see as matrix
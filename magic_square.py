import numpy as np

# Parameters
n = 3  # 3x3 magic square
population_size = 100

# Generate initial population (random permutations of 1 to n²)
def generate_individual(n):
    return np.random.permutation(n * n) + 1  # e.g., [8,1,6,3,5,7,4,9,2]

population = [generate_individual(n) for _ in range(population_size)]

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

    # Penalize duplicate numbers in the individual
    unique_numbers = len(set(individual))
    if unique_numbers != n**2:
        total_deviation += 1000  # Large penalty for duplicates

    return 1 / (1 + total_deviation)  # Higher is better

import numpy as np
import random
import matplotlib.pyplot as plt

# Parameters
n = 3  # 3x3 magic square
population_size = 100
mutation_rate = 0.2
generations = 500


# Generate initial population (random permutations of 1 to n²)
def generate_individual(n):
    return np.random.permutation(n * n) + 1  # e.g., [8,1,6,3,5,7,4,9,2]


# fixed sum that all rows, columns, and both main diagonals of a magic square must equal.
def magic_constant(n):
    return n * (n**2 + 1) // 2  # 15 for n=3


# Fitness function (only valid permutations reach here)
def fitness_score(individual, n):
    square = np.array(individual).reshape(n, n)
    M = magic_constant(n)
    total_deviation = 0

    # Check rows and columns
    for i in range(n):
        row_sum = np.sum(square[i, :])
        col_sum = np.sum(square[:, i])
        total_deviation += abs(row_sum - M) + abs(col_sum - M)

    # Check diagonals
    diag1_sum = np.sum(np.diag(square))
    diag2_sum = np.sum(np.diag(np.fliplr(square)))
    total_deviation = total_deviation + abs(diag1_sum - M) + abs(diag2_sum - M)

    return 1 / (1 + total_deviation)  # Higher is better


# Crossover that always produces valid permutations, discarding invalid children
def crossover(parent1, parent2, n):
    size = n * n
    while True:  # Keep trying until we get a valid child
        # Create a template child
        child = np.zeros(size, dtype=int)

        # Select a random segment from parent1
        start = random.randint(0, size - 1)
        end = random.randint(start + 1, size)
        child[start:end] = parent1[start:end]

        # Fill remaining positions from parent2
        remaining_positions = [i for i in range(size) if i not in range(start, end)]
        remaining_values = [x for x in parent2 if x not in child]

        # Assign remaining values
        for i, pos in enumerate(remaining_positions):
            child[pos] = remaining_values[i]

        # Apply mutation if needed
        if random.random() < mutation_rate:
            i, j = random.sample(range(size), 2)
            child[i], child[j] = child[j], child[i]

        # Verify this is a valid permutation
        if len(set(child)) == size and all(1 <= x <= size for x in child):
            return child
        # If not valid, the loop continues


# Initialize population with valid permutations
population = [generate_individual(n) for _ in range(population_size)]
avg_fitness = []
best_fitness = []

for age in range(generations):
    # Evaluate fitness
    population_fitness = [fitness_score(ind, n) for ind in population]
    current_best = max(population_fitness)
    best_fitness.append(current_best)
    avg_fitness.append(np.mean(population_fitness))

    # Early exit if magic square found
    if current_best == 1:
        print(f"Magic square found at generation {age}!")
        break

    # Selection and reproduction
    new_population = []
    for _ in range(population_size):
        # Select parents based on fitness
        parent1, parent2 = random.choices(population, weights=population_fitness, k=2)
        child = crossover(parent1, parent2, n)  # This always returns valid children
        new_population.append(child)
    population = new_population

# Plot results
plt.plot(range(len(avg_fitness)), avg_fitness, "r-", label="Avg Fitness")
plt.plot(range(len(best_fitness)), best_fitness, "g-", label="Best Fitness")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.show()

# Print the best individual found
best_idx = np.argmax([fitness_score(ind, n) for ind in population])
best_square = np.array(population[best_idx]).reshape(n, n)
print("Magic constant:", magic_constant(n))
print("Best solution found:")
print(best_square)
print(f"Fitness: {fitness_score(population[best_idx], n)}")
print(f"Row sums: {[sum(best_square[i,:]) for i in range(n)]}")
print(f"Col sums: {[sum(best_square[:,i]) for i in range(n)]}")
print(f"Diagonals: {sum(np.diag(best_square))}, {sum(np.diag(np.fliplr(best_square)))}")

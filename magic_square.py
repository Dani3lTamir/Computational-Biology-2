import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter

# Parameters
n = 4  # nxn magic square
population_size = 100  
base_mutation_rate = 0.7
adaptive_mutation = True
generations = 500 
diversity_threshold = 0.3  # Trigger diversity measures if similarity > 80%
tournament_size = 7  # For tournament selection


# Generate initial population (random permutations of 1 to n²)
def generate_individual(n):
    return np.random.permutation(n * n) + 1


# Calculate population diversity
def calculate_diversity(population):
    """Calculate diversity as average pairwise differences"""
    if len(population) < 2:
        return 1.0
    total_differences = 0
    comparisons = 0
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            # Count different positions
            differences = np.sum(population[i] != population[j])
            total_differences += differences
            comparisons += 1

    # Normalize by maximum possible differences
    max_differences = len(population[0])
    return (total_differences / comparisons) / max_differences


# Fixed sum that all rows, columns, and both main diagonals of a magic square must equal
def magic_constant(n):
    return n * (n**2 + 1) // 2



def fitness_score(individual, n, population=None, diversity_bonus=False):
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
    total_deviation += abs(diag1_sum - M) + abs(diag2_sum - M)

    # Penalty for non-unique elements
    counts = Counter(individual)
    duplicate_penalty = sum((count - 1) * 100 for count in counts.values() if count > 1)

    # Exponential fitness
    k = 0.1
    base_fitness = 1 / (1 + total_deviation + duplicate_penalty)
    
    # Diversity bonus
    if diversity_bonus and population is not None:
        uniqueness = calculate_individual_uniqueness(individual, population)
        base_fitness += 0.1 * uniqueness

    return base_fitness


def calculate_individual_uniqueness(individual, population):
    """Calculate how unique this individual is compared to population"""
    if len(population) <= 1:
        return 1.0
    differences = []
    for other in population:
        if not np.array_equal(individual, other):
            diff = np.sum(individual != other) / len(individual)
            differences.append(diff)
    return np.mean(differences) if differences else 0.0


# Tournament selection
def tournament_selection(population, fitness_scores, tournament_size):
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitness)]
    return population[winner_idx]


# Partially Matched Crossover (PMX)
def pmx_crossover(parent1, parent2, n):
    size = n * n
    child = np.zeros(size, dtype=int)
    start, end = sorted(random.sample(range(size), 2))

    # Copy segment from parent1
    child[start:end] = parent1[start:end]

    # Map values from parent2
    mapping = {parent1[i]: parent2[i] for i in range(start, end)}
    for i in range(size):
        if i < start or i >= end:
            value = parent2[i]
            while value in child[start:end]:
                value = mapping.get(value, value)
            child[i] = value

    return child


def mutate(individual, mutation_rate, n, method="swap"):
    if random.random() > mutation_rate:
        return individual
    individual = individual.copy()
    size = len(individual)

    if method == "swap":
        i, j = random.sample(range(size), 2)
        individual[i], individual[j] = individual[j], individual[i]

    elif method == "scramble":
        start, end = sorted(random.sample(range(size), 2))
        segment = individual[start:end]
        np.random.shuffle(segment)
        individual[start:end] = segment

    elif method == "inversion":
        start, end = sorted(random.sample(range(size), 2))
        individual[start:end] = individual[start:end][::-1]


    return individual


# Validate magic square
def is_magic_square(individual, n):
    square = np.array(individual).reshape(n, n)
    M = magic_constant(n)
    row_sums = [sum(square[i, :]) for i in range(n)]
    col_sums = [sum(square[:, i]) for i in range(n)]
    diag1 = sum(np.diag(square))
    diag2 = sum(np.diag(np.fliplr(square)))
    return (
        all(s == M for s in row_sums)
        and all(s == M for s in col_sums)
        and diag1 == M
        and diag2 == M
        and len(set(individual)) == n * n
    )


# Main genetic algorithm
population = [generate_individual(n) for _ in range(population_size)]
avg_fitness = []
best_fitness = []
diversity_history = []
mutation_rates = []

print(f"Starting evolution for {n}x{n} magic square...")
print(f"Magic constant should be: {magic_constant(n)}")

for age in range(generations):
    diversity = calculate_diversity(population)
    diversity_history.append(diversity)

    # Adaptive mutation rate
    current_mutation_rate = base_mutation_rate
    if adaptive_mutation:
        if diversity < 0.3:
            current_mutation_rate = min(0.9, base_mutation_rate * 3)
        elif diversity < 0.5:
            current_mutation_rate = base_mutation_rate * 1.5
    mutation_rates.append(current_mutation_rate)

    # Diversity injection
    if diversity < diversity_threshold:
        new_individuals = [generate_individual(n) for _ in range(population_size // 10)]
        population = population[: -len(new_individuals)] + new_individuals

    # Evaluate fitness
    use_diversity_bonus = diversity < 0.4
    population_fitness = [
        fitness_score(
            ind, n, population if use_diversity_bonus else None, use_diversity_bonus
        )
        for ind in population
    ]

    current_best = max(population_fitness)
    best_fitness.append(current_best)
    avg_fitness.append(np.mean(population_fitness))

    # Check for valid magic square
    best_idx = np.argmax(population_fitness)
    if is_magic_square(population[best_idx], n):
        print(f"Magic square found at generation {age}!")
        break

    if age % 25 == 0:
        print(
            f"Gen {age}: Best={current_best:.4f}, Avg={avg_fitness[-1]:.4f}, "
            f"Diversity={diversity:.3f}, MutRate={current_mutation_rate:.3f}"
        )

    # Selection and reproduction
    new_population = []

    # Elitism: keep best individuals
    elite_count = max(1, population_size // 10)  # Keep top 10%
    elite_indices = np.argsort(population_fitness)[-elite_count:]
    for idx in elite_indices:
        new_population.append(population[idx].copy())

    mutation_methods = ["swap", "scramble", "inversion"]

    while len(new_population) < population_size:
        parent1 = tournament_selection(population, population_fitness, tournament_size)
        parent2 = tournament_selection(population, population_fitness, tournament_size)
        child = pmx_crossover(parent1, parent2, n)
        mut_method = random.choice(mutation_methods)
        child = mutate(child, current_mutation_rate, n, mut_method)
        new_population.append(child)

    population = new_population

# Plot results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Fitness over time
ax1.plot(range(len(avg_fitness)), avg_fitness, "r-", label="Avg Fitness", alpha=0.7)
ax1.plot(range(len(best_fitness)), best_fitness, "g-", label="Best Fitness")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness")
ax1.set_title("Fitness Evolution")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Diversity over time
ax2.plot(
    range(len(diversity_history)), diversity_history, "b-", label="Population Diversity"
)
ax2.axhline(
    y=0.3, color="r", linestyle="--", alpha=0.5, label="Low Diversity Threshold"
)
ax2.set_xlabel("Generation")
ax2.set_ylabel("Diversity")
ax2.set_title("Population Diversity")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Adaptive mutation rate
ax3.plot(range(len(mutation_rates)), mutation_rates, "m-", label="Mutation Rate")
ax3.set_xlabel("Generation")
ax3.set_ylabel("Mutation Rate")
ax3.set_title("Adaptive Mutation Rate")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Final solution
best_idx = np.argmax([fitness_score(ind, n) for ind in population])
best_square = np.array(population[best_idx]).reshape(n, n)
im = ax4.imshow(best_square, cmap="viridis")
ax4.set_title("Best Magic Square Found")
for i in range(n):
    for j in range(n):
        ax4.text(
            j,
            i,
            str(best_square[i, j]),
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
ax4.set_xticks([])
ax4.set_yticks([])

plt.tight_layout()
plt.show()

# Print results
print("\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)
print("Magic constant:", magic_constant(n))
print("Best solution found:")
print(best_square)
print(f"Final fitness: {fitness_score(population[best_idx], n):.6f}")
print(f"Final diversity: {diversity_history[-1]:.3f}")

# Verify magic square properties
M = magic_constant(n)
row_sums = [sum(best_square[i, :]) for i in range(n)]
col_sums = [sum(best_square[:, i]) for i in range(n)]
diag1 = sum(np.diag(best_square))
diag2 = sum(np.diag(np.fliplr(best_square)))

print(f"Row sums: {row_sums} (target: {M})")
print(f"Col sums: {col_sums} (target: {M})")
print(f"Diagonals: {diag1}, {diag2} (target: {M})")

is_magic = is_magic_square(population[best_idx], n)
print(f"Is valid magic square: {is_magic}")
if is_magic:
    print("🎉 SUCCESS! Perfect magic square found!")
else:
    total_error = (
        sum(abs(s - M) for s in row_sums + col_sums) + abs(diag1 - M) + abs(diag2 - M)
    )
    print(f"Total deviation from magic constant: {total_error}")

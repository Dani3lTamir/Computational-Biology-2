import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter

# Parameters
n = 4  # nxn magic square
population_size = 100
base_mutation_rate = 0.7  # Base mutation rate
adaptive_mutation = True  # Enable adaptive mutation
generations = 500
diversity_threshold = 0.8  # Trigger diversity measures if similarity > 80%
tournament_size = 4  # For tournament selection


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


# Fitness function with diversity bonus
max_fit = 1


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
    total_deviation = total_deviation + abs(diag1_sum - M) + abs(diag2_sum - M)

    base_fitness = (1 / (1 + total_deviation)) ** 2

    # Add diversity bonus if requested
    if diversity_bonus and population is not None:
        uniqueness = calculate_individual_uniqueness(individual, population)
        return base_fitness + 0.1 * uniqueness  # Small diversity bonus

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


# Tournament selection (better than fitness-proportionate for maintaining diversity)
def tournament_selection(population, fitness_scores, tournament_size=3):
    """Select parent using tournament selection"""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitness)]
    return population[winner_idx]


# Enhanced crossover with multiple methods
def crossover(parent1, parent2, n):
    size = n * n
    attempts = 0
    max_attempts = 50

    while attempts < max_attempts:
        child = original_crossover(parent1, parent2, size)

        # Verify this is a valid permutation
        if len(set(child)) == size and all(1 <= x <= size for x in child):
            return child
        attempts += 1

    # Fallback: return a random permutation
    return generate_individual(n)


def original_crossover(parent1, parent2, size):
    """Original crossover method"""
    child = np.zeros(size, dtype=int)
    start = random.randint(0, size - 1)
    end = random.randint(start + 1, size)
    child[start:end] = parent1[start:end]

    remaining_positions = [i for i in range(size) if i not in range(start, end)]
    remaining_values = [x for x in parent2 if x not in child]

    for i, pos in enumerate(remaining_positions):
        child[pos] = remaining_values[i]

    return child


# Enhanced mutation with multiple strategies
def mutate(individual, mutation_rate, n, method="swap"):
    """Apply mutation with different strategies"""
    if random.random() > mutation_rate:
        return individual

    individual = individual.copy()
    size = len(individual)

    if method == "swap":
        # Simple swap mutation
        i, j = random.sample(range(size), 2)
        individual[i], individual[j] = individual[j], individual[i]

    elif method == "scramble":
        # Scramble a random segment
        start, end = sorted(random.sample(range(size), 2))
        segment = individual[start:end]
        np.random.shuffle(segment)
        individual[start:end] = segment

    elif method == "inversion":
        # Invert a random segment
        start, end = sorted(random.sample(range(size), 2))
        individual[start:end] = individual[start:end][::-1]

    return individual



# Initialize population with valid permutations
population = [generate_individual(n) for _ in range(population_size)]
avg_fitness = []
best_fitness = []
diversity_history = []
mutation_rates = []

print(f"Starting evolution for {n}x{n} magic square...")
print(f"Magic constant should be: {magic_constant(n)}")

for age in range(generations):
    # Calculate diversity
    diversity = calculate_diversity(population)
    diversity_history.append(diversity)

    # Adaptive mutation rate
    current_mutation_rate = base_mutation_rate
    if adaptive_mutation:
        if diversity < 0.3:  # Low diversity
            current_mutation_rate = min(0.5, base_mutation_rate * 3)
        elif diversity < 0.5:  # Medium diversity
            current_mutation_rate = base_mutation_rate * 1.5

    mutation_rates.append(current_mutation_rate)

    # Evaluate fitness (with diversity bonus if diversity is low)
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

    # Progress reporting
    if age % 50 == 0:
        print(
            f"Gen {age}: Best={current_best:.4f}, Avg={avg_fitness[-1]:.4f}, "
            f"Diversity={diversity:.3f}, MutRate={current_mutation_rate:.3f}"
        )

    # Early exit if magic square found
    if current_best >= max_fit * 0.999:  # Allow for floating point precision
        print(f"Magic square found at generation {age}!")
        break

    # Selection and reproduction
    new_population = []

    # Elitism: keep best individuals
    elite_count = max(1, population_size // 10)  # Keep top 10%
    elite_indices = np.argsort(population_fitness)[-elite_count:]
    for idx in elite_indices:
        new_population.append(population[idx].copy())

    # Generate rest of population
    mutation_methods = ["swap", "scramble", "inversion"]

    while len(new_population) < population_size:
        # Parent selection using tournament
        parent1 = tournament_selection(population, population_fitness, tournament_size)
        parent2 = tournament_selection(population, population_fitness, tournament_size)

        child = crossover(parent1, parent2, n)

        # Mutation with random method selection
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

# Display the magic square
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

# Print detailed results
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

# Check if it's a valid magic square
is_magic = (
    all(s == M for s in row_sums)
    and all(s == M for s in col_sums)
    and diag1 == M
    and diag2 == M
)

print(f"Is valid magic square: {is_magic}")
if is_magic:
    print("🎉 SUCCESS! Perfect magic square found!")
else:
    total_error = (
        sum(abs(s - M) for s in row_sums + col_sums) + abs(diag1 - M) + abs(diag2 - M)
    )
    print(f"Total deviation from magic constant: {total_error}")

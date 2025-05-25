import numpy as np
import random
import math
import matplotlib.pyplot as plt
from collections import Counter
from enum import Enum


class EvolutionType(Enum):
    ORIGINAL = 0
    LAMARCKIAN = 1
    DARWINIAN = 2


# Parameters
n = 3 # nxn magic square
population_size = 100
base_mutation_rate = 0.7
adaptive_mutation = True
generations = 500
diversity_threshold = 0.3  # Trigger diversity measures if similarity > 80%
tournament_size = 7  # For tournament selection
evolution_type = EvolutionType.LAMARCKIAN # Change this to switch between versions
optimization_steps = n  # Number of local optimization steps to perform
prioritize_most_perfect = False


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


def is_magic_square(individual, n):
    """Check if individual forms a valid magic square"""
    square = np.array(individual).reshape(n, n)
    M = magic_constant(n)

    # Check if all numbers 1 to n² are present exactly once
    if sorted(individual) != list(range(1, n * n + 1)):
        return False

    # Check rows
    for i in range(n):
        if sum(square[i, :]) != M:
            return False

    # Check columns
    for i in range(n):
        if sum(square[:, i]) != M:
            return False

    # Check diagonals
    if sum(np.diag(square)) != M:
        return False
    if sum(np.diag(np.fliplr(square))) != M:
        return False

    return True


def is_most_perfect_magic_square(individual, n):
    """Check if individual forms a most-perfect magic square (only for n%4=0)"""
    if n % 4 != 0:
        return False

    if not is_magic_square(individual, n):
        return False

    square = np.array(individual).reshape(n, n)
    M = magic_constant(n)

    # For most-perfect magic squares, every 2x2 subsquare must sum to M
    expected_sum = M

    # Check all possible 2x2 subsquares
    for i in range(n - 1):
        for j in range(n - 1):
            subsquare_sum = (
                square[i, j]
                + square[i, j + 1]
                + square[i + 1, j]
                + square[i + 1, j + 1]
            )
            if subsquare_sum != expected_sum:
                return False

    return True


def most_perfect_fitness_component(individual, n):
    """Calculate fitness component for most-perfect property"""
    if n % 4 != 0:
        return 0

    square = np.array(individual).reshape(n, n)
    M = magic_constant(n)
    expected_sum = M

    total_deviation = 0
    subsquare_count = 0

    # Check all 2x2 subsquares
    for i in range(n - 1):
        for j in range(n - 1):
            subsquare_sum = (
                square[i, j]
                + square[i, j + 1]
                + square[i + 1, j]
                + square[i + 1, j + 1]
            )
            total_deviation += abs(subsquare_sum - expected_sum)
            subsquare_count += 1

    # Return normalized score (higher is better)
    if total_deviation == 0:
        return 1.0
    return 1.0 / (1.0 + total_deviation / subsquare_count)


def fitness_score(
    individual, n, population=None, diversity_bonus=False
):
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

    # Base fitness (magic square fitness)
    base_fitness = 1 / (1 + total_deviation + duplicate_penalty)

    # Most-perfect fitness component (only for n%4=0)
    most_perfect_bonus = 0
    if n % 4 == 0 and prioritize_most_perfect:
        most_perfect_score = most_perfect_fitness_component(individual, n)
        most_perfect_bonus = (
            most_perfect_score * 0.5
        )  # Weight the most-perfect component

    # Diversity bonus
    diversity_bonus_score = 0
    if diversity_bonus and population is not None:
        uniqueness = calculate_individual_uniqueness(individual, population)
        diversity_bonus_score = 0.1 * uniqueness

    return base_fitness + most_perfect_bonus + diversity_bonus_score


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
    child1 = [-1] * size  # Use -1 to indicate unfilled positions
    child2 = [-1] * size
    
    # Choose crossover points
    start, end = sorted(random.sample(range(size), 2))
    
    # Copy the middle section from each parent
    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]
    
    # Create mapping dictionaries
    mapping1 = {}  # parent1 -> parent2
    mapping2 = {}  # parent2 -> parent1
    
    for i in range(start, end):
        mapping1[parent1[i]] = parent2[i]
        mapping2[parent2[i]] = parent1[i]
    
    # Fill the remaining positions
    for i in range(size):
        if i < start or i >= end:
            # For child1, try to place parent2[i]
            value = parent2[i]
            while value in child1[start:end]:  # If value is already in the copied section
                value = mapping1.get(value, value)  # Map it
            child1[i] = value
            
            # For child2, try to place parent1[i]
            value = parent1[i]
            while value in child2[start:end]:  # If value is already in the copied section
                value = mapping2.get(value, value)  # Map it
            child2[i] = value
    
    # Return one of the children randomly
    return np.array(child1 if random.random() < 0.5 else child2)


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


def optimize_individual(individual, n, max_steps=None, temp=1.0, cooling_rate=0.95):
    """Optimize magic square using targeted hill climbing with intelligent swap selection."""
    if max_steps is None:
        max_steps = n * 5  # More steps for thorough optimization

    current = individual.copy()
    current_fitness = fitness_score(current, n)
    M = magic_constant(n)
    best = current.copy()
    best_fitness = current_fitness

    for step in range(max_steps):
        # Find the best swap among all possible swaps
        best_swap = None
        best_swap_fitness = current_fitness
        
        # Try a subset of swaps to balance exploration and efficiency
        swap_candidates = []
        square = current.reshape(n, n)
        
        # Identify problematic positions (in rows/cols with wrong sums)
        row_sums = np.sum(square, axis=1)
        col_sums = np.sum(square, axis=0)
        
        problematic_positions = []
        for i in range(n):
            if abs(row_sums[i] - M) > 0:  # Row i has wrong sum
                problematic_positions.extend([(i, j) for j in range(n)])
            if abs(col_sums[i] - M) > 0:  # Column i has wrong sum
                problematic_positions.extend([(j, i) for j in range(n)])
        
        # Remove duplicates
        problematic_positions = list(set(problematic_positions))
        
        # If we have problematic positions, focus swaps on them
        if problematic_positions and len(problematic_positions) > 1:
            # Try swaps involving problematic positions
            for i in range(min(20, len(problematic_positions))):  # Limit to 20 swaps
                pos1 = random.choice(problematic_positions)
                pos2 = random.choice(problematic_positions)
                if pos1 != pos2:
                    idx1 = pos1[0] * n + pos1[1]
                    idx2 = pos2[0] * n + pos2[1]
                    swap_candidates.append((idx1, idx2))
        else:
            # If no clear problematic positions, try random swaps
            for _ in range(20):  # Try 20 random swaps
                idx1, idx2 = random.sample(range(len(current)), 2)
                swap_candidates.append((idx1, idx2))
        
        # Evaluate all swap candidates
        for idx1, idx2 in swap_candidates:
            candidate = current.copy()
            candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
            candidate_fitness = fitness_score(candidate, n)
            
            if candidate_fitness > best_swap_fitness:
                best_swap = (idx1, idx2)
                best_swap_fitness = candidate_fitness
        
        # Apply the best swap if it improves fitness
        if best_swap and best_swap_fitness > current_fitness:
            current[best_swap[0]], current[best_swap[1]] = current[best_swap[1]], current[best_swap[0]]
            current_fitness = best_swap_fitness
            
            if current_fitness > best_fitness:
                best = current.copy()
                best_fitness = current_fitness
        else:
            # No improvement found, stop early
            break
        
        # If we found a perfect solution, return immediately
        if current_fitness >= 0.999:  # Close to 1.0 (perfect)
            return current

    return best

# Main genetic algorithm
population = [generate_individual(n) for _ in range(population_size)]
avg_fitness = []
best_fitness = []
diversity_history = []
mutation_rates = []

# Track if we found solutions
magic_square_found = False
most_perfect_found = False
magic_square_generation = -1
most_perfect_generation = -1

print(f"Starting evolution for {n}x{n} magic square...")
print(f"Evolution type: {evolution_type.name}")
print(f"Magic constant should be: {magic_constant(n)}")
if n % 4 == 0:
    print(
        f"Also searching for most-perfect magic square (2x2 subsquares sum to {magic_constant(n)})"
    )

for age in range(generations):
    diversity = calculate_diversity(population)
    diversity_history.append(diversity)

    # Adaptive mutation rate
    current_mutation_rate = base_mutation_rate
    if adaptive_mutation:
        if diversity < 0.3:
            current_mutation_rate = min(0.9, base_mutation_rate * 3)
        elif diversity < 0.5:
             current_mutation_rate = min(0.9, base_mutation_rate * 1.5)
    mutation_rates.append(current_mutation_rate)

    # Diversity injection
    if diversity < diversity_threshold:
        new_individuals = [generate_individual(n) for _ in range(population_size // 10)]
        population += new_individuals
        population = sorted(population, key=lambda ind: fitness_score(ind, n), reverse=True)[:population_size]

    # Evaluate fitness with optional optimization
    use_diversity_bonus = diversity < 0.4
    optimized_population = []

    if evolution_type != EvolutionType.ORIGINAL:
        # Create optimized versions for Lamarckian or Darwinian
        optimized_population = [
            optimize_individual(ind, n, optimization_steps) for ind in population
        ]

    if evolution_type == EvolutionType.LAMARCKIAN:
        # Use optimized individuals for fitness calculation AND pass them to next generation
        population_fitness = [
            fitness_score(
                opt_ind,
                n,
                optimized_population if use_diversity_bonus else None,
                use_diversity_bonus,
            )
            for opt_ind in optimized_population
        ]
        population = optimized_population
    elif evolution_type == EvolutionType.DARWINIAN:
        # Use optimized individuals for fitness calculation but original for next generation
        population_fitness = [
            fitness_score(
                opt_ind,
                n,
                population if use_diversity_bonus else None,
                use_diversity_bonus,
            )
            for opt_ind in optimized_population
        ]
    else:  # ORIGINAL
        population_fitness = [
            fitness_score(
                ind, n, population if use_diversity_bonus else None, use_diversity_bonus
            )
            for ind in population
        ]

    current_best = max(population_fitness)
    best_fitness.append(current_best)
    avg_fitness.append(np.mean(population_fitness))

    # Check for valid magic square and most-perfect magic square
    best_idx = np.argmax(population_fitness)
    best_individual = population[best_idx]

    if not magic_square_found and is_magic_square(best_individual, n):
        magic_square_found = True
        magic_square_generation = age
        print(f"Regular magic square found at generation {age}!")
        if n % 4 != 0:  # if n is not a multiple of 4, we can stop here
            break

    if (
        n % 4 == 0
        and not most_perfect_found
        and is_most_perfect_magic_square(best_individual, n)
    ):
        most_perfect_found = True
        most_perfect_generation = age
        print(f"Most-perfect magic square found at generation {age}!")
        break  # Stop if we found the most-perfect square

    if age % 25 == 0:
        print(
            f"Gen {age}: Best={current_best:.4f}, Avg={avg_fitness[-1]:.4f}, "
            f"Diversity={diversity:.3f}, MutRate={current_mutation_rate:.3f}"
        )

    # Selection and reproduction
    new_population = []

    # Elitism: keep best individuals
    elite_count = max(1, population_size // 20)  # Keep top candidates
    elite_indices = np.argsort(population_fitness)[-elite_count:]
    for idx in elite_indices:
        if evolution_type == EvolutionType.LAMARCKIAN:
            # In Lamarckian, we pass the optimized versions
            new_population.append(optimized_population[idx].copy())
        else:
            # In Darwinian and Original, we pass the original versions
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
ax1.set_title(f"Fitness Evolution ({evolution_type.name})")
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
print(f"Evolution type: {evolution_type.name}")
print("Magic constant:", magic_constant(n))
print("Best solution found:")
print(best_square)
print(f"Final fitness: {fitness_score(population[best_idx], n):.6f}")
print(f"Final diversity: {diversity_history[-1]:.3f}")

M = magic_constant(n)
row_sums = [sum(best_square[i, :]) for i in range(n)]
col_sums = [sum(best_square[:, i]) for i in range(n)]
diag1 = sum(np.diag(best_square))
diag2 = sum(np.diag(np.fliplr(best_square)))

print(f"Row sums: {row_sums} (target: {M})")
print(f"Col sums: {col_sums} (target: {M})")
print(f"Diagonals: {diag1}, {diag2} (target: {M})")

is_magic = is_magic_square(population[best_idx], n)
is_most_perfect = is_most_perfect_magic_square(population[best_idx], n)

print(f"\nIs valid magic square: {is_magic}")
if magic_square_found:
    print(f"🎉 Regular magic square found at generation {magic_square_generation}!")

if n % 4 == 0:
    print(f"Is most-perfect magic square: {is_most_perfect}")
    if most_perfect_found:
        print(
            f"🌟 Most-perfect magic square found at generation {most_perfect_generation}!"
        )
    elif is_magic:
        print("Found regular magic square but not most-perfect.")

        # Show 2x2 subsquare analysis
        print("\n2x2 Subsquare Analysis:")
        expected_sum = M
        for i in range(n - 1):
            for j in range(n - 1):
                subsquare_sum = (
                    best_square[i, j]
                    + best_square[i, j + 1]
                    + best_square[i + 1, j]
                    + best_square[i + 1, j + 1]
                )
                print(
                    f"Subsquare at ({i},{j}): {subsquare_sum} (target: {expected_sum})"
                )

if not is_magic:
    total_error = (
        sum(abs(s - M) for s in row_sums + col_sums) + abs(diag1 - M) + abs(diag2 - M)
    )
    print(f"Total deviation from magic constant: {total_error}")

import numpy as np
import random
import math
import matplotlib.pyplot as plt
from collections import Counter
from enum import Enum
import sys


class EvolutionType(Enum):
    ORIGINAL = 0
    LAMARCKIAN = 1
    DARWINIAN = 2


# Parameters
population_size = 100
base_mutation_rate = 0.7
adaptive_mutation = True
generations = 500
diversity_threshold = 0.4  # Trigger diversity measures if too similar
tournament_size = 7  # For tournament selection
prioritize_most_perfect = True


# Generate initial population (random permutations of 1 to n²)
def generate_individual(n):
    return np.random.permutation(n * n) + 1


# Calculate population diversity
def calculate_diversity(population):
    """Calculate diversity as average pairwise differences"""
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
    total_deviation += abs(diag1_sum - M) + abs(diag2_sum - M)

    # Base fitness (magic square fitness)
    base_fitness = 1 / (1 + total_deviation)

    # Most-perfect fitness component (only for n%4=0)
    most_perfect_bonus = 0
    if n % 4 == 0 and prioritize_most_perfect:
        most_perfect_score = most_perfect_fitness_component(individual, n)
        most_perfect_bonus = (
            most_perfect_score * 0.5
        )  # Weight the most-perfect component

    return base_fitness + most_perfect_bonus


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


def crossover(parent1, parent2, n):
    size = n * n

    # Choose crossover points
    start, end = sorted(random.sample(range(size), 2))

    # Create child with -1 placeholders
    child = [-1] * size

    # Copy segment from parent1
    child[start:end] = parent1[start:end]

    # Fill remaining positions with parent2's order
    child_idx = end % size
    parent2_idx = end % size

    while -1 in child:
        if parent2[parent2_idx] not in child:
            child[child_idx] = parent2[parent2_idx]
            child_idx = (child_idx + 1) % size
        parent2_idx = (parent2_idx + 1) % size

    return np.array(child)


def mutate(individual, mutation_rate, n, method="scramble"):
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


def get_problematic_positions(square, n):
    """Find positions that are in problematic rows/columns/diagonals"""
    M = magic_constant(n)
    problematic_positions = set()

    # Check rows
    for i in range(n):
        if abs(np.sum(square[i, :]) - M) > 0:
            for j in range(n):
                problematic_positions.add((i, j))

    # Check columns
    for j in range(n):
        if abs(np.sum(square[:, j]) - M) > 0:
            for i in range(n):
                problematic_positions.add((i, j))

    # Check main diagonal
    if abs(np.sum(np.diag(square)) - M) > 0:
        for i in range(n):
            problematic_positions.add((i, i))

    # Check anti-diagonal
    if abs(np.sum(np.diag(np.fliplr(square))) - M) > 0:
        for i in range(n):
            problematic_positions.add((i, n - 1 - i))

    return list(set(problematic_positions))


def optimize_individual(individual, n, max_steps=None, temp=1.0, cooling_rate=0.95):
    if max_steps is None:
        max_steps = n * 2  # Increase for larger squares

    current = individual.copy()
    current_fitness = fitness_score(current, n)
    best = current.copy()
    best_fitness = current_fitness

    for step in range(max_steps):
        # Try multiple swaps and keep the best
        best_swap = None
        best_swap_fitness = current_fitness

        # Try more swaps for larger squares
        num_swaps_to_try = min(50, n * n)  # Try more swaps for larger squares

        for _ in range(num_swaps_to_try):
            idx1, idx2 = random.sample(range(len(current)), 2)
            candidate = current.copy()
            candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
            candidate_fitness = fitness_score(candidate, n)

            if candidate_fitness > best_swap_fitness:
                best_swap = (idx1, idx2)
                best_swap_fitness = candidate_fitness

        if best_swap:
            current[best_swap[0]], current[best_swap[1]] = (
                current[best_swap[1]],
                current[best_swap[0]],
            )
            current_fitness = best_swap_fitness

            if current_fitness > best_fitness:
                best = current.copy()
                best_fitness = current_fitness

        # Simulated annealing acceptance
        elif random.random() < math.exp(-1 / temp):
            # Accept worse solution to escape local optima
            idx1, idx2 = random.sample(range(len(current)), 2)
            current[idx1], current[idx2] = current[idx2], current[idx1]
            current_fitness = fitness_score(current, n)

        temp *= cooling_rate

        if current_fitness >= 1:  # Perfect solution found
            return current

    return best


def get_user_input():
    """Get user input for n and evolution type"""
    print("=" * 50)
    print("MAGIC SQUARE EVOLUTION")
    print("=" * 50)

    # Get n value
    while True:
        try:
            n = int(input("Enter the size of the magic square (n): "))
            if n < 3:
                print("Please enter a value of 3 or greater.")
                continue
            break
        except ValueError:
            print("Please enter a valid integer.")

    # Get evolution type
    print("\nEvolution types:")
    print("0 - ORIGINAL (basic genetic algorithm)")
    print(
        "1 - LAMARCKIAN (with local optimization, optimized individuals passed to next generation)"
    )
    print(
        "2 - DARWINIAN (with local optimization, but original individuals passed to next generation)"
    )

    while True:
        try:
            choice = int(input("Enter evolution type (0, 1, or 2): "))
            if choice in [0, 1, 2]:
                evolution_type = EvolutionType(choice)
                break
            else:
                print("Please enter 0, 1, or 2.")
        except ValueError:
            print("Please enter a valid integer.")

    return n, evolution_type


def run_evolution(n, evolution_type):
    """Run the evolution algorithm with given parameters"""
    optimization_steps = n  # Number of local optimization steps to perform

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

    print(f"\nStarting evolution for {n}x{n} magic square...")
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

        # Diversity injection - replace worst individuals with new random ones
        if diversity < diversity_threshold:
            # Calculate fitness for current population to identify worst individuals
            current_fitness = [fitness_score(ind, n) for ind in population]

            # Number of new individuals to inject
            num_new = population_size // 10

            # Find indices of worst individuals
            worst_indices = np.argsort(current_fitness)[:num_new]

            # Generate new random individuals
            new_individuals = [generate_individual(n) for _ in range(num_new)]

            # Replace worst individuals with new ones
            for i, worst_idx in enumerate(worst_indices):
                population[worst_idx] = new_individuals[i]

        # Evaluate fitness with optional optimization

        optimized_population = []

        if evolution_type != EvolutionType.ORIGINAL:
            # Create optimized versions for Lamarckian or Darwinian
            optimized_population = [
                optimize_individual(ind, n, optimization_steps) for ind in population
            ]

        if evolution_type == EvolutionType.LAMARCKIAN:
            # Use optimized individuals for fitness calculation AND pass them to next generation
            population_fitness = [
                fitness_score(opt_ind, n) for opt_ind in optimized_population
            ]
            population = optimized_population
        elif evolution_type == EvolutionType.DARWINIAN:
            # Use optimized individuals for fitness calculation but original for next generation
            population_fitness = [
                fitness_score(opt_ind, n) for opt_ind in optimized_population
            ]
        else:  # ORIGINAL
            population_fitness = [fitness_score(ind, n) for ind in population]

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

        # Elitism: keep best individuals, bigger elite count for larger lamarckian populations
        if evolution_type == EvolutionType.LAMARCKIAN:
            elite_count = int(population_size * 0.9)
        else:
            elite_count = int(population_size * 0.1)
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
            parent1 = tournament_selection(
                population, population_fitness, tournament_size
            )
            parent2 = tournament_selection(
                population, population_fitness, tournament_size
            )
            child = crossover(parent1, parent2, n)
            mut_method = random.choice(mutation_methods)
            child = mutate(child, current_mutation_rate, n, mut_method)
            new_population.append(child)

        population = new_population

    return (
        population,
        avg_fitness,
        best_fitness,
        diversity_history,
        mutation_rates,
        magic_square_found,
        most_perfect_found,
        magic_square_generation,
        most_perfect_generation,
    )


def display_results(
    n,
    evolution_type,
    population,
    avg_fitness,
    best_fitness,
    diversity_history,
    mutation_rates,
    magic_square_found,
    most_perfect_found,
    magic_square_generation,
    most_perfect_generation,
):
    """Display results in plots and console output"""

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
        range(len(diversity_history)),
        diversity_history,
        "b-",
        label="Population Diversity",
    )
    ax2.axhline(
        y=diversity_threshold,
        color="r",
        linestyle="--",
        alpha=0.5,
        label="Low Diversity Threshold",
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

    # Use non-blocking show
    plt.show(block=False)

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
            sum(abs(s - M) for s in row_sums + col_sums)
            + abs(diag1 - M)
            + abs(diag2 - M)
        )
        print(f"Total deviation from magic constant: {total_error}")


def main():
    """Main program loop"""
    while True:
        # Get user input
        n, evolution_type = get_user_input()

        # Run evolution
        results = run_evolution(n, evolution_type)

        # Display results
        display_results(n, evolution_type, *results)

        # Ask if user wants to run again
        print("\n" + "=" * 50)
        while True:
            choice = (
                input("Do you want to run another evolution? (y/n): ").lower().strip()
            )
            if choice in ["y", "yes"]:
                break
            elif choice in ["n", "no"]:
                print("Thank you for using Magic Square Evolution!")
                return
            else:
                print("Please enter 'y' for yes or 'n' for no.")


if __name__ == "__main__":
    main()

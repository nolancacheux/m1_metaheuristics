import itertools
import time

def read_distance_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        distance_matrix = []
        for line in lines:
            row = list(map(int, line.split()))
            distance_matrix.append(row)
    return distance_matrix

def calculate_total_distance(route, distance_matrix):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i]][route[i + 1]]
    total_distance += distance_matrix[route[-1]][route[0]]  # Return to the starting city
    return total_distance

def tsp_brute_force(distance_matrix, n):
    cities = list(range(n))
    min_distance = float('inf')
    best_route = None
    for route in itertools.permutations(cities):
        current_distance = calculate_total_distance(route, distance_matrix)
        if current_distance < min_distance:
            min_distance = current_distance
            best_route = route
    return best_route, min_distance

# Read the distance matrix from the file
file_path = 'p01.15.291.tsp'
distance_matrix = read_distance_matrix(file_path)

# Optimized version using dynamic programming (Held-Karp algorithm)
def tsp_dynamic_programming(distance_matrix, n):
    memo = {}
    def dp(mask, pos):
        if (mask, pos) in memo:
            return memo[(mask, pos)]
        if mask == (1 << n) - 1:
            return distance_matrix[pos][0]
        min_cost = float('inf')
        for city in range(n):
            if mask & (1 << city) == 0:
                new_cost = distance_matrix[pos][city] + dp(mask | (1 << city), city)
                min_cost = min(min_cost, new_cost)
        memo[(mask, pos)] = min_cost
        return min_cost

    return dp(1, 0)
# Analyze the running time and obtained solutions for n = 5 to 15
print(f"{'n':<3} {'Method':<20} {'Distance':<10} {'Time (s)':<10}")
print("="*45)

# Dynamic Programming results
for n in range(5, 16):
    start_time = time.time()
    min_distance = tsp_dynamic_programming(distance_matrix, n)
    end_time = time.time()
    print(f"{n:<3} {'Dynamic Programming':<20} {min_distance:<10} {end_time - start_time:.4f}")

print("\n" + "="*45)

# Analyze the running time and obtained solutions for n = 5 to 15
print(f"{'n':<3} {'Method':<20} {'Distance':<10} {'Time (s)':<10}")
print("="*45)
# Brute Force results
for n in range(5, 16):
    start_time = time.time()
    best_route, min_distance = tsp_brute_force(distance_matrix, n)
    end_time = time.time()
    print(f"{n:<3} {'Brute Force':<20} {min_distance:<10} {end_time - start_time:.4f}")

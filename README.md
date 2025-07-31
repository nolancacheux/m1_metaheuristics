# Metaheuristics Algorithms Implementation

A comprehensive collection of metaheuristic algorithms implemented in Python for solving optimization problems, with a focus on the Traveling Salesman Problem (TSP). This repository contains both theoretical materials and practical implementations of various optimization techniques.

## Algorithms Implemented

### Single-Solution Metaheuristics

Single-solution metaheuristics work with one solution at a time, iteratively improving it through local search operations.

#### 1. Local Search Algorithm (LS)
- **Implementation**: `Single-Solution-Metaheuristics/Local Search Algorithm (LS)/Localsearch_steepestdescent.ipynb`
- **Description**: Basic local search using steepest descent approach
- **Key Features**:
  - Neighborhood exploration through 2-opt moves
  - Steepest descent selection strategy
  - Termination when no improving neighbor is found

#### 2. Simulated Annealing (SA)
- **Implementation**: `Single-Solution-Metaheuristics/Simulated Annealing (SA)/SimulatedAnnealing.ipynb`
- **Description**: Probabilistic optimization algorithm inspired by metallurgical annealing
- **Key Features**:
  - Temperature-based acceptance probability
  - Geometric cooling schedule (T = T * alpha)
  - Solution degradation acceptance for escaping local optima
  - Visualization of cost evolution and solution paths

#### 3. Tabu Search (TS)
- **Standard Implementation**: `Single-Solution-Metaheuristics/Tabu Search (TS)/Standard/`
- **Optimized Implementation**: `Single-Solution-Metaheuristics/Tabu Search (TS)/Optimized/`
- **Description**: Memory-based metaheuristic using tabu list to avoid cycling
- **Key Features**:
  - Tabu list management
  - Aspiration criteria
  - Intensification and diversification strategies (optimized version)
  - Long-term memory mechanisms

### Population-Solution Metaheuristics

Population-based metaheuristics maintain and evolve a set of solutions simultaneously.

#### 1. Genetic Algorithm (GA)
- **Implementation**: `Population-Solution-Metaheuristics/Genetic Algorithms (GA)/GeneticAlgorithm-Solution.ipynb`
- **Description**: Evolution-inspired algorithm using selection, crossover, and mutation
- **Key Features**:
  - Tournament and roulette wheel selection methods
  - Partially Matched Crossover (PMX) for TSP
  - Swap mutation operator
  - Elitism strategy
  - Steady-state and generational replacement options

#### 2. Ant Colony Optimization (ACO)
- **Implementation**: `Population-Solution-Metaheuristics/Ant Colony Optimization (ACO)/CACHEUXNolan_AntColony-TSP_optimized.ipynb`
- **Description**: Swarm intelligence algorithm inspired by ant foraging behavior
- **Key Features**:
  - Pheromone trail management
  - Probabilistic solution construction
  - Local and global pheromone updates
  - Optimization enhancements

## Problem Domain: Traveling Salesman Problem (TSP)

The primary optimization problem addressed in this repository is the TSP, where the goal is to find the shortest possible route visiting each city exactly once and returning to the starting city.

### TSP Implementation Details

#### Exact Methods
- **Brute Force**: Complete enumeration of all possible permutations
- **Dynamic Programming**: Held-Karp algorithm with memoization
- **Implementation**: `Introduction/TSPProblem.py`

#### Distance Matrix Format
The TSP instances use symmetric distance matrices where:
- `distance_matrix[i][j]` represents the distance between cities i and j
- The diagonal elements are typically 0 (distance from a city to itself)
- The matrix is symmetric: `distance_matrix[i][j] = distance_matrix[j][i]`

#### Objective Function
```python
def objectivefunction(solution):
    cost = 0
    for i in range(len(solution) - 1):
        cost += distance_matrix[solution[i]][solution[i + 1]]
    cost += distance_matrix[solution[-1]][solution[0]]  # Return to start
    return cost
```

## Dataset

The `TSPDataset/` folder contains various TSP instances:

| Instance | Cities | Optimal Distance | Description |
|----------|--------|------------------|-------------|
| five.19.tsp | 5 | 19 | Small test instance |
| gr17.2085.tsp | 17 | 2085 | Medium-sized instance |
| fri26_d.937.tsp | 26 | 937 | Medium-sized instance |
| dantzig42_.d.699.tsp | 42 | 699 | Larger instance |
| att48_d.33523.tsp | 48 | 33523 | Large instance |

## Requirements

### Python Dependencies
```
numpy
matplotlib
scikit-learn
jupyter
```

### Installation
```bash
pip install numpy matplotlib scikit-learn jupyter
```

## Usage

### Running Jupyter Notebooks
1. Navigate to the project directory
2. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open the desired algorithm implementation
4. Modify file paths in the notebooks to point to your TSP dataset location
5. Execute cells to run the algorithms

### Running Python Scripts
```bash
python Introduction/TSPProblem.py
```

### Modifying TSP Data Paths
Update the file paths in notebook cells to match your dataset location:
```python
# Example modification needed in notebooks
tsp_data = np.loadtxt('TSPDataset/gr17.2085.tsp')
```

## Algorithm Parameters

### Simulated Annealing
- **Initial Temperature**: 500
- **Cooling Rate**: 0.99 (geometric cooling)
- **Internal Iterations**: 5 per temperature level
- **Termination**: Temperature < 1

### Genetic Algorithm
- **Population Size**: 200
- **Generations**: 100
- **Mutation Probability**: 0.3
- **Tournament Size**: 8
- **Elitism Size**: 40

### Tabu Search
- **Tabu List Size**: Variable (implementation dependent)
- **Aspiration Criteria**: Best solution improvement
- **Diversification**: Long-term memory mechanisms

## Performance Analysis

The implementations include performance analysis features:

1. **Convergence Tracking**: Cost evolution over iterations/generations
2. **Solution Visualization**: 2D plotting of TSP tours using multidimensional scaling
3. **Comparative Analysis**: Runtime and solution quality comparisons
4. **Statistical Reporting**: Best, average, and worst case performance metrics

## Theoretical Background

The repository includes comprehensive theoretical materials:

- **Session 1**: Introduction to metaheuristics
- **Session 2**: Single-solution metaheuristics overview
- **Session 3**: Local search algorithms
- **Session 4**: Simulated annealing theory
- **Session 5**: Tabu search methodology
- **Session 6**: Genetic algorithms principles
- **Session 7**: Ant colony optimization concepts

## Educational Content

This repository serves as educational material covering:

1. **Algorithm Design**: Implementation patterns and best practices
2. **Parameter Tuning**: Guidelines for algorithm configuration
3. **Performance Evaluation**: Metrics and comparison methodologies
4. **Problem Modeling**: TSP representation and solution encoding
5. **Optimization Strategies**: Intensification vs. diversification trade-offs

## Contributing

When extending this repository:

1. Follow the existing directory structure
2. Include both implementation and theoretical documentation
3. Provide parameter tuning guidelines
4. Add performance benchmarks
5. Update this README with new algorithm descriptions

## References

The implementations are based on established metaheuristic literature and optimization principles. Refer to the PDF materials in each algorithm directory for theoretical foundations and mathematical formulations.

## License

This project is developed for educational purposes as part of metaheuristics coursework.
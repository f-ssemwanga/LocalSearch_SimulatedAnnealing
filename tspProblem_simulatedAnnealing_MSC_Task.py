import numpy as np
import matplotlib.pyplot as plt

def read_csv(filename):
  """Reads a csv file and returns a list of points."""
  points = []
  with open(filename, 'r') as f:
    for line in f:
      points.append([float(x) for x in line.split(',')])
  return points

def distance(p1, p2):
  """Returns the distance between two points."""
  return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def distance(p1, p2):
  """Returns the distance between two points."""
  return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def tsp(points, alpha, max_temp, min_temp, num_iterations):
  """Solve the TSP problem using simulated annealing."""
  # Initialize the solution.
  solution = np.arange(len(points))

  # Initialize the temperature.
  temp = max_temp
  
  # Initialize the cost to a value as high as possible - in this case it is infinity
  cost = np.inf 

  # Iterate for the specified number of iterations.
  for i in range(num_iterations):
    # Choose two random cities.
    c1 = np.random.randint(len(points))
    c2 = np.random.randint(len(points))

    # Swap the cities.
    solution[c1], solution[c2] = solution[c2], solution[c1]

    # Calculate the new cost.
    new_cost = 0
    for i in range(len(solution) - 1):
      new_cost += distance(points[solution[i]], points[solution[i + 1]])
    new_cost += distance(points[solution[-1]], points[solution[0]])

    # Accept the new solution if it is better than the old solution.
    if new_cost < cost:
      cost = new_cost
    else:
      # Otherwise, accept the new solution with a probability of e^(-(new_cost - cost) / temp).
      if np.random.random() < np.exp(-(new_cost - cost) / temp):
        cost = new_cost

    # Decrease the temperature.
    temp *= alpha
     
  # Return the solution.
  return solution
def plot_cost(costs):
    """Plots the cost/distance versus iteration."""
    iterations = range(len(costs))
    plt.plot(iterations, costs)
    plt.title('Cost/Distance vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Cost/Distance')
    plt.show()

def plot_best_solution(points, best_solutions):
    """Plots the best solution versus iteration."""
    plt.figure(figsize=(10, 6))
    for i, solution in enumerate(best_solutions):
        plt.plot([points[i][0] for i in solution], [points[i][1] for i in solution], label=f'Iteration {i+1}')
    plt.title('Best Solution vs Iteration')
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.legend()
    plt.show()

def main():
    # Read the data from the csv file.
    points = read_csv('TSPMatrix.csv')

    # Set the parameters.
    alpha = 0.99
    max_temp = 10
    min_temp = 0.01
    num_iterations = 1000

    # Solve the TSP problem and obtain the best solution.
    best_solution = tsp(points, alpha, max_temp, min_temp, num_iterations)

    # Calculate the cost of the best solution.
    best_cost = 0
    for i in range(len(best_solution) - 1):
        best_cost += distance(points[best_solution[i]], points[best_solution[i + 1]])
    best_cost += distance(points[best_solution[-1]], points[best_solution[0]])

    # Print the best cost.
    print("Best Cost:", best_cost)

    # Calculate costs for all iterations.
    costs = []
    best_solutions = []
    for iteration in range(num_iterations):
        # Solve the TSP problem for each iteration.
        solution = tsp(points, alpha, max_temp, min_temp, iteration+1)

        # Calculate the cost for the current solution.
        cost = 0
        for i in range(len(solution) - 1):
            cost += distance(points[solution[i]], points[solution[i + 1]])
        cost += distance(points[solution[-1]], points[solution[0]])

        # Append the cost and solution to the respective lists.
        costs.append(cost)
        best_solutions.append(solution)

    # Plot the cost/distance versus iteration.
    plot_cost(costs)

    # Plot the best solution versus iteration.
    plot_best_solution(points, best_solutions)


if __name__ == '__main__':
  main()
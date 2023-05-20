# LocalSearch_SimulatedAnnealing
Solving the TSP using simulated Annealing Algorithm
## Pseudo Algorithm
# Read and Prepare Data
procedure read_data(filename)
  open file filename for reading
  create a csv reader object for the file
  create an empty list to store the data
  iterate over the rows in the csv file
    add the row to the list as a tuple of floats
  return the list
# calculate straight line distance between points
procedure euclidean_distance(point1, point2)
  x1, y1 = point1
  x2, y2 = point2
  distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
  return distance
# calculatetotal distance of a given order of points
procedure total_distance(points, order)
  total = 0
  for i in range(len(order) - 1):
    point1 = points[order[i]]
    point2 = points[order[i + 1]]
    distance = euclidean_distance(point1, point2)
    total += distance
  return total
# perform simulated annealing for TSP 
procedure simulated_annealing_tsp(points, initial_order, temperature, cooling_rate):
  //Initialize the current order and the best order.'''
  current_order = initial_order[:]
  best_order = current_order[:]
  best_distance = total_distance(points, current_order)
  
  while temperature > 0:
	  //Generate a random neighbor of the current order.'''
    neighbor = generate_random_neighbor(current_order)
    //Calculate the cost of the neighbor.'''
	neighbor_distance = total_distance(points, neighbor)

	// If the cost of the neighbor is less than the cost of the current order, then accept the neighbor.
	//Otherwise, accept the neighbor with a probability that is given by the Boltzmann distribution.
	if neighbor_distance < best_distance:
	  best_order = neighbor
	  best_distance = neighbor_distance
	else:
	  //Calculate the probability of accepting the neighbor.
	  probability = exp(-(neighbor_distance - best_distance) / temperature)

	  // If the random number is less than the probability, then accept the neighbor.
	  if random() < probability:
		current_order = neighbor

	//Decrease the temperature.
	temperature *= cooling_rate
# calculate the acceptance probability based on the temperature and the differences in distances
procedure acceptance_probability(current_distance, new_distance, temperature):
  //If the new distance is less than the current distance, then return 1.0.
  if new_distance < current_distance:
    return 1.0

  //Otherwise, calculate the probability of accepting the new solution.
  probability = exp((current_distance - new_distance) / temperature)

  //Return the probability.
  return probability
# Iterate through the algorithm multiple times and return best orders
procedure run_algorithm(points, num_runs, initial_temperature, cooling_rate):
  best_orders = []
  for _ in range(num_runs):
    initial_order = list(range(len(points)))
    random.shuffle(initial_order)
    best_order = simulated_annealing_tsp(points, initial_order, initial_temperature, cooling_rate)
    best_orders.append(best_order)
  return best_orders
# visualise results
procedure plot_tsp(points, order):
	plot(points, 'bo-')
	for i in range(len(order) - 1):
		plot([points[order[i]][0], points[order[i + 1]][0]], [points[order[i]][1], points[order[i + 1]][1]], 'r-')
		plot([points[order[-1]][0], points[order[0]][0]], [points[order[-1]][1], points[order[0]][1]], 'r-')
		xlabel('X')
		ylabel('Y')
		title('TSP Solution')
		show()

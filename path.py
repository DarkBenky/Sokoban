import math
import json
import os
import random

def convert_coordinates_to_direction(x1, y1, x2, y2):
    if x1 < x2:
        return 'right'
    elif x1 > x2:
        return 'left'
    elif y1 < y2:
        return 'down'
    elif y1 > y2:
        return 'up'

def convert_direction_to_coordinates(x, y, direction):
    if direction == 'right':
        return x + 1, y
    elif direction == 'left':
        return x - 1, y
    elif direction == 'down':
        return x, y + 1
    elif direction == 'up':
        return x, y - 1
    else:
        return 'up' ,x, y

def load_function_from_json(map_name=None):
    if map_name is None:
        random_file = random.choice(os.listdir('maps'))
        with open('maps/' + random_file) as f:
            data = json.load(f)
        # Extract information from the data
        barriers = data.get('barriers', [])
        player_x, player_y = data.get('player', [0, 0])
        box = data.get('box', [])
        goals = data.get('goals', [])
        print(f"Loaded map from {random_file}")
        return barriers, player_x, player_y, box, goals
        
    else:
        with open('maps/' + map_name) as f:
            data = json.load(f)
        # Extract information from the data
        barriers = data.get('barriers', [])
        player_x, player_y = data.get('player', [0, 0])
        box = data.get('box', [])
        goals = data.get('goals', [])
        
        print(f"Loaded map from {map_name}")

        return barriers, player_x, player_y, box, goals

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def is_valid_move(matrix, new_x, new_y):
    # 1 for player and 2 for box and 3 for wall and 4 for goal and 0 for empty
    return 0 <= new_x < len(matrix) and 0 <= new_y < len(matrix[0]) and matrix[new_x][new_y] != 3

def prioritize_moves(current_x, current_y, finish_x, finish_y):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # Calculate Euclidean distances for each direction
    distances = [euclidean_distance(current_x + dr, current_y + dc, finish_x, finish_y) for dr, dc in directions]

    # Prioritize moves based on distances
    sorted_moves = [move for _, move in sorted(zip(distances, directions))]
    return sorted_moves

def pathfinder(current_x, current_y, visited, finish_x, finish_y, width, height, matrix):
    if not (0 <= current_x < width and 0 <= current_y < height) or not is_valid_move(matrix, current_x, current_y):
        return None
    
    if current_x == finish_x and current_y == finish_y:
        return visited
    else:
        # Prioritize moves based on distance to finish
        moves = prioritize_moves(current_x, current_y, finish_x, finish_y)

        for dr, dc in moves:
            new_x, new_y = current_x + dr, current_y + dc
            if is_valid_move(matrix, new_x, new_y) and (new_x, new_y) not in visited:
                visited.append((new_x, new_y))
                result = pathfinder(new_x, new_y, visited, finish_x, finish_y, width, height, matrix)
                if result:
                    return result

        # If no valid path found, backtrack
        visited.pop()
        return None


barriers, player_x, player_y, box, goals = load_function_from_json()

def createEnv(barriers, box, goals):
    matrix = [[0 for i in range(10)] for j in range(10)]
    for barrier in barriers:
        matrix[barrier[0]][barrier[1]] = 3
    return matrix , box[0] , box[1], goals[0] , goals[1]



# matrix , start_x , start_y , finish_x , finish_y = createEnv(barriers, box, goals)
# visited = [(start_x, start_y)]
# result = pathfinder(start_x, start_y, visited, finish_x, finish_y, 10, 10, matrix)

    

# print(result)

# for i in range(10):
#     for j in range(10):
#         if matrix[i][j] == 0:
#             matrix[i][j] = '.'
#         if matrix[i][j] == 4:
#             matrix[i][j] = '#'

# matrix[start_x][start_y] = '*'
# matrix[finish_x][finish_y] = '+'
# print('\n'.join([' '.join([str(cell) for cell in row]) for row in matrix]))
# print('\n\n\n')
# for value in result:
#     matrix[value[0]][value[1]] = '>'
# print('\n'.join([' '.join([str(cell) for cell in row]) for row in matrix]))



# Example usage
# start_x, start_y = 0, 0
# finish_x, finish_y = 3, 3
# width, height = 4, 4
# matrix = [
#     [1, 0, 1, 1],
#     [1, 1, 1, 0],
#     [0, 0, 1, 1],
#     [0, 0, 1, 1]
# ]
# visited = [(start_x, start_y)]
# result = pathfinder(start_x, start_y, visited, finish_x, finish_y, width, height, matrix)
# print(result)
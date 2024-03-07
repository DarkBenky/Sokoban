from main import Game , load_function_from_json
from path import pathfinder , convert_coordinates_to_direction , convert_direction_to_coordinates
import time
    

data = load_function_from_json(map_name='map-EWwUZ.json')
walls , player_x , player_y , box , goals = data
game = Game(player_x=player_x, player_y=player_y, player_char=1, wall_char=2, empty_char=0, box_char=3, final_char=4, wall_cords=walls, final_cords=[goals], box_cords=[box] , map_size=(10, 10))

print(game.show_map('human'))

game_map = game.show_map('array')

path = pathfinder(current_x=box[0], current_y=box[1], visited=[(box[0], box[1])], finish_x=goals[0], finish_y=goals[0], width=10, height=10, matrix=game_map)

print(path)

new_path = []
for i in range(0,len(path)-1):
    direction = convert_coordinates_to_direction(*(path[i] + path[i+1]))
    new_path.append((direction,path[i]))

player_x , player_y = game.x , game.y
for path in new_path:
    # move player close to box base on direction
    direction , (x,y) = path
    path = pathfinder(current_x=player_x, current_y=player_y, visited=[(player_x, player_y)], finish_x=x, finish_y=y, width=10, height=10, matrix=game_map)
    
    # temp = game_map.copy()
    # for cord in path:
    #     temp[cord[0]][cord[1]] = '*'
    
    # print('\n'.join([' '.join([str(cell) for cell in row]) for row in temp]))
    
    for p in path:
        game.move(direction=None , new_x_y=p)
        print(game.show_map('human'))
        print('\n')
        time.sleep(0.5)
    print('done')
        


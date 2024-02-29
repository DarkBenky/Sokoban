import pygame
import sys
import json
import random
import string

pygame.init()

WIDTH, HEIGHT = 800, 800
GRID_SIZE = 80
GRID_ROWS, GRID_COLS = 10, 10
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PLAYER_COLOR = (0, 128, 255)
BOX_COLOR = (255, 165, 0)
TARGET_COLOR = (255, 0, 0)
BARRIER_COLOR = (128, 128, 128)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Map Editor")

def draw_grid():
    for i in range(0, WIDTH, GRID_SIZE):
        pygame.draw.line(screen, WHITE, (i, 0), (i, HEIGHT))
    for j in range(0, HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, WHITE, (0, j), (WIDTH, j))

def draw_cell(row, col, color):
    pygame.draw.rect(screen, color, (col * GRID_SIZE, row * GRID_SIZE, GRID_SIZE, GRID_SIZE))

def main():
    grid = [['' for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                
                mouse_x, mouse_y = pygame.mouse.get_pos()
                col = mouse_x // GRID_SIZE
                row = mouse_y // GRID_SIZE

                keys = pygame.key.get_pressed()
                if keys[pygame.K_b]:  # B for barrier
                    grid[row][col] = 'barrier' if grid[row][col] != 'barrier' else ''
                elif keys[pygame.K_p]:  # P for player
                    grid[row][col] = 'player' if grid[row][col] != 'player' else ''
                elif keys[pygame.K_x]:  # X for box
                    grid[row][col] = 'box' if grid[row][col] != 'box' else ''
                elif keys[pygame.K_g]:  # G for target goal
                    grid[row][col] = 'goal' if grid[row][col] != 'goal' else ''
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    barriers = []
                    player = None
                    box = None
                    goals = None
                    for row in range(GRID_ROWS):
                        for col in range(GRID_COLS):
                            if grid[row][col] == 'barrier':
                                barriers.append((row, col))
                            elif grid[row][col] == 'player':
                                player = (row, col)
                            elif grid[row][col] == 'box':
                                box = (row, col)
                            elif grid[row][col] == 'goal':
                                goals = (row, col)
                    if player and box and goals:
                        file_name = f'maps/map-{"".join(random.choice(string.ascii_letters) for _ in range(5))}.json'
                        with open(file_name, 'w') as f:
                            json.dump({'barriers': barriers, 'player': player, 'box': box, 'goals': goals}, f)
                            print('Map saved to map.json')
                # reset level
                elif event.key == pygame.K_r:
                    grid = [['' for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
                    print('Level reset')
                    

        screen.fill(BLACK)
        draw_grid()
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                if grid[row][col] == 'barrier':
                    draw_cell(row, col, BARRIER_COLOR)
                elif grid[row][col] == 'player':
                    draw_cell(row, col, PLAYER_COLOR)
                elif grid[row][col] == 'box':
                    draw_cell(row, col, BOX_COLOR)
                elif grid[row][col] == 'goal':
                    draw_cell(row, col, TARGET_COLOR)

        pygame.display.flip()

if __name__ == "__main__":
    main()

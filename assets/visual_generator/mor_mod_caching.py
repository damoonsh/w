import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)

# Grid settings
GRID_SIZE = 40
PADDING = 20
TOP_PADDING = 60  # Add extra padding at the top
FONT_SIZE = 20

class GridDisplay:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Grid Display")
        self.font = pygame.font.Font(None, FONT_SIZE)
        
        # Generate random values for grids
        self.left_grids = self.generate_left_grids()
        self.right_grids = self.generate_right_grids()
        
    def generate_left_grids(self):
        """Generate three 5x5 grids with values from -1 to 1"""
        grids = []
        
        # Choose 2 random columns for the second and third grids
        zero_cols_base = random.sample(range(5), 2)
        
        for grid_idx in range(3):
            grid = []
            for row in range(5):
                grid_row = []
                for col in range(5):
                    value = round(random.uniform(-1, 1), 2)
                    grid_row.append(value)
                grid.append(grid_row)
            grids.append(grid)
        
        # First grid stays as is (no zeros)
        
        # For the second grid, zero out the base columns and corresponding rows
        for col in zero_cols_base:
            for row in range(5):
                grids[1][row][col] = 0
        
        # Also zero out the corresponding rows
        for row in zero_cols_base:
            for col in range(5):
                grids[1][row][col] = 0
        
        # For the third grid, use the same base columns plus one additional random column
        remaining_cols = [i for i in range(5) if i not in zero_cols_base]
        additional_col = random.choice(remaining_cols)
        zero_cols_third = zero_cols_base + [additional_col]
        
        # Zero out the columns
        for col in zero_cols_third:
            for row in range(5):
                grids[2][row][col] = 0
        
        # Also zero out the corresponding rows
        for row in zero_cols_third:
            for col in range(5):
                grids[2][row][col] = 0
        
        return grids
    
    def generate_right_grids(self):
        """Generate 5x5, 3x3, and 2x2 grids with values from -1 to 1"""
        grids = []
        sizes = [5, 3, 2]
        
        for size in sizes:
            grid = []
            for row in range(size):
                grid_row = []
                for col in range(size):
                    value = round(random.uniform(-1, 1), 2)
                    grid_row.append(value)
                grid.append(grid_row)
            grids.append(grid)
        return grids
    
    def value_to_color(self, value):
        """Convert value to color based on range"""
        if value == 0:
            return BLACK  # Zero values are black
        elif value >= 0.5:
            return RED
        elif value >= 0:
            return GREEN
        elif value >= -0.5:
            return BLUE
        else:
            return PURPLE
    
    def draw_grid(self, grid, start_x, start_y, title=""):
        """Draw a single grid with values and colors"""
        rows = len(grid)
        cols = len(grid[0])
        
        # Draw title
        if title:
            title_surface = self.font.render(title, True, BLACK)
            self.screen.blit(title_surface, (start_x, start_y - 25))
        
        for row in range(rows):
            for col in range(cols):
                x = start_x + col * (GRID_SIZE + 2)
                y = start_y + row * (GRID_SIZE + 2)
                
                value = grid[row][col]
                color = self.value_to_color(value)
                
                # Draw cell background
                pygame.draw.rect(self.screen, color, (x, y, GRID_SIZE, GRID_SIZE))
                pygame.draw.rect(self.screen, BLACK, (x, y, GRID_SIZE, GRID_SIZE), 2)
                
                # Draw value text
                text = str(value)
                # Use white text for better contrast on dark backgrounds, black for light backgrounds
                text_color = WHITE if color in [BLACK, BLUE, PURPLE] else BLACK
                text_surface = self.font.render(text, True, text_color)
                text_rect = text_surface.get_rect(center=(x + GRID_SIZE//2, y + GRID_SIZE//2))
                self.screen.blit(text_surface, text_rect)
    
    def draw_legend(self):
        """Draw color legend"""
        legend_x = WINDOW_WIDTH - 200
        legend_y = 50
        
        legend_title = self.font.render("Color Legend:", True, BLACK)
        self.screen.blit(legend_title, (legend_x, legend_y))
        
        legend_items = [
            ("= 0", BLACK),
            ("≥ 0.5", RED),
            ("0 to 0.5", GREEN),
            ("-0.5 to 0", BLUE),
            ("< -0.5", PURPLE)
        ]
        
        for i, (text, color) in enumerate(legend_items):
            y = legend_y + 30 + i * 25
            pygame.draw.rect(self.screen, color, (legend_x, y, 20, 20))
            pygame.draw.rect(self.screen, BLACK, (legend_x, y, 20, 20), 1)
            text_surface = self.font.render(text, True, BLACK)
            self.screen.blit(text_surface, (legend_x + 25, y + 2))
    
    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        # Regenerate grids on 'R' key press
                        self.left_grids = self.generate_left_grids()
                        self.right_grids = self.generate_right_grids()
            
            # Clear screen
            self.screen.fill(WHITE)
            
            # Draw left grids (three 5x5 grids)
            left_start_x = PADDING
            for i, grid in enumerate(self.left_grids):
                y_pos = TOP_PADDING + i * (5 * (GRID_SIZE + 2) + PADDING * 2)  # Added bottom padding
                self.draw_grid(grid, left_start_x, y_pos, f"Grid {i+1} (5x5)")
            
            # Draw right grids
            right_start_x = left_start_x + 5 * (GRID_SIZE + 2) + PADDING * 3
            grid_sizes = [(5, "5x5"), (3, "3x3"), (2, "2x2")]
            
            current_y = TOP_PADDING
            for i, (grid, (size, label)) in enumerate(zip(self.right_grids, grid_sizes)):
                self.draw_grid(grid, right_start_x, current_y, f"Grid {label}")
                current_y += size * (GRID_SIZE + 2) + PADDING * 3  # Added more bottom padding
            
            # Draw legend
            self.draw_legend()
            
            # Draw instructions
            # instructions = [
            #     "Press 'R' to regenerate grids",
            #     "Colors represent value ranges"
            # ]
            # for i, instruction in enumerate(instructions):
            #     text_surface = self.font.render(instruction, True, BLACK)
            #     self.screen.blit(text_surface, (PADDING, WINDOW_HEIGHT - 60 + i * 25))
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = GridDisplay()
    app.run()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Any

def plot_2d_trace_with_proper_front_cells(x_0, x_f, path, all_front_cells, intersection_coords, y_coords_history, obstacles, new_front_cells=None, new_coords=None, l=None, isGoalReached=None, title="2D Ray Trace - Proper Front Cells"):
    """
    Visualization showing the corrected front cells - cells that the ray will enter next.
    Also plots the history of y corner coordinates.
    Displays new_front_cells, new_coords, l, isGoalReached on the plot.
    """
    plt.figure(figsize=(16, 16))

    # Determine plot limits dynamically
    all_x_coords = []
    all_y_coords = []

    if path:
        path_np_temp = np.array(path)
        if path_np_temp.ndim == 2 and path_np_temp.shape[1] >= 1:
             all_x_coords.extend(path_np_temp[:, 0])
        if path_np_temp.ndim == 2 and path_np_temp.shape[1] >= 2:
             all_y_coords.extend(path_np_temp[:, 1])

    for fcs in all_front_cells:
        for fc_val in fcs: # fc_val is a coordinate array like [x,y]
            if hasattr(fc_val, '__len__') and len(fc_val) >= 1:
                all_x_coords.append(fc_val[0])
            if hasattr(fc_val, '__len__') and len(fc_val) >= 2:
                all_y_coords.append(fc_val[1])
    
    all_x_coords.extend([x_0[0], x_f[0]])
    all_y_coords.extend([x_0[1], x_f[1]])

    if y_coords_history:
        valid_y_coords = [y for y in y_coords_history if hasattr(y, '__len__') and len(y) >= 2]
        if valid_y_coords:
            all_x_coords.extend([y[0] for y in valid_y_coords])
            all_y_coords.extend([y[1] for y in valid_y_coords])

    min_coord_x = min(all_x_coords) - 1 if all_x_coords else -2
    max_coord_x = max(all_x_coords) + 1 if all_x_coords else 7
    min_coord_y = min(all_y_coords) - 1 if all_y_coords else -2
    max_coord_y = max(all_y_coords) + 1 if all_y_coords else 7
    
    min_coord_line = min(min_coord_x, min_coord_y)
    max_coord_line = max(max_coord_x, max_coord_y)

    for i in range(int(np.floor(min_coord_line)), int(np.ceil(max_coord_line)) + 1):
        plt.axhline(i, color='lightgray', linestyle='--', lw=0.5)
        plt.axvline(i, color='lightgray', linestyle='--', lw=0.5)

    # Plot ray path
    if path:
        path_np = np.array(path)
        if path_np.ndim == 2 and path_np.shape[1] >=2:
             plt.plot(path_np[:, 0], path_np[:, 1], 'b-', label="Ray Path", linewidth=4, alpha=0.9, zorder=3)
    
    # Plot intersection points
    if intersection_coords:
        intersection_np = np.array(intersection_coords)
        if intersection_np.ndim == 2 and intersection_np.shape[1] >=2:
            plt.scatter(intersection_np[:, 0], intersection_np[:, 1], c='red', s=120,
                        label="Hyperplane Intersections", zorder=5, marker='x', linewidths=3)
    
    # Plot y corner coordinates
    if y_coords_history:
        y_coords_np = np.array([y for y in y_coords_history if hasattr(y, '__len__') and len(y) >= 2]) # Filter for valid coords
        if y_coords_np.ndim == 2 and y_coords_np.shape[0] > 0 and y_coords_np.shape[1] >= 2:
            plt.scatter(y_coords_np[:, 0], y_coords_np[:, 1], c='darkviolet', s=80,
                        label="y corner coordinates (self.y)", zorder=4, marker='s', alpha=0.7)

    # Plot start and goal
    plt.plot(x_0[0], x_0[1], 'go', label="Start", markersize=10, zorder=6)
    plt.plot(x_f[0], x_f[1], 'ro', label="Goal", markersize=10, zorder=6)
    plt.text(x_0[0], x_0[1] + 0.2, f'Start ({x_0[0]:.2f}, {x_0[1]:.2f})', ha='center', va='bottom', fontsize=9)
    plt.text(x_f[0], x_f[1] + 0.2, f'Goal ({x_f[0]:.2f}, {x_f[1]:.2f})', ha='center', va='bottom', fontsize=9)
    
    # Plot obstacles
    if obstacles:
        for obs_idx, obs in enumerate(obstacles):
             if hasattr(obs, '__len__') and len(obs) >= 2:
                rect = plt.Rectangle((obs[0], obs[1]), 1, 1, color='black', alpha=0.8, zorder=2,
                                   label='Obstacle' if obs_idx == 0 and 'Obstacle' not in plt.gca().get_legend_handles_labels()[1] else "")
                plt.gca().add_patch(rect)

    # Plot front cells
    colors = ['cyan', 'magenta', 'yellow', 'orange', 'lightgreen', 'pink', 'purple', 'brown']
    num_steps_to_plot_front_cells = len(all_front_cells)

    for step_idx in range(num_steps_to_plot_front_cells):
        front_cells_at_step = all_front_cells[step_idx]
        
        current_intersection_point_for_arrow = x_0 # Default to start
        if intersection_coords: # Ensure intersection_coords is not empty
            if step_idx < len(intersection_coords):
                current_intersection_point_for_arrow = intersection_coords[step_idx]
            else:
                current_intersection_point_for_arrow = intersection_coords[-1] # Use last known if step_idx is out of bounds

        color = colors[step_idx % len(colors)]
        alpha_val = 0.4 + 0.15 * (step_idx % 2)
        
        label_prefix = f'Front Cells Step {step_idx}'
        label_text = f'{label_prefix} (Initial)' if step_idx == 0 else label_prefix
        
        for cell_idx, fc in enumerate(front_cells_at_step):
            if hasattr(fc, '__len__') and len(fc) >= 2: # Check if fc is a valid coordinate
                legend_label = label_text if cell_idx == 0 else ""
                rect = plt.Rectangle((fc[0], fc[1]), 1, 1, color=color, alpha=alpha_val,
                                   edgecolor='darkblue', linewidth=2.5, label=legend_label, zorder=2)
                plt.gca().add_patch(rect)
                
                plt.text(fc[0] + 0.5, fc[1] + 0.5, f'S{step_idx}',
                        ha='center', va='center', fontsize=10, fontweight='bold', zorder=3,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9, edgecolor='black'))
                
                fc_center = fc + 0.5
                if hasattr(current_intersection_point_for_arrow, '__len__') and len(current_intersection_point_for_arrow) >=2:
                    plt.annotate('', xy=fc_center, xytext=current_intersection_point_for_arrow, zorder=3,
                               arrowprops=dict(arrowstyle='->', color='darkred', alpha=0.8, lw=2))
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("X-coordinate", fontsize=14)
    plt.ylabel("Y-coordinate", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.3)
    
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xticks(np.arange(0, 11, 1))
    plt.yticks(np.arange(0, 11, 1))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    info_text_lines = []
    if new_front_cells is not None:
        info_text_lines.append(f"new_front_cells: {np.array(new_front_cells)}")
    if new_coords is not None:
        info_text_lines.append(f"new_coords: {np.array(new_coords)}")
    if l is not None:
        info_text_lines.append(f"l (length): {round(l, 4)}")
    if isGoalReached is not None:
        info_text_lines.append(f"isGoalReached: {isGoalReached}")
    
    if info_text_lines:
        info_text = "\n".join(info_text_lines)
        plt.gcf().text(0.02, 0.98, info_text, fontsize=13, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    plt.show()

def plot_3d_trace_with_proper_front_cells(x_0, x_f, path, all_front_cells, intersection_coords, y_coords_history, obstacles, isGoalReached=None, title="3D Ray Trace - Proper Front Cells"):
    """
    3D Visualization showing the corrected front cells - cells that the ray will enter next.
    Also plots the history of y corner coordinates as 3D points.
    """
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    min_coord_line = -1
    max_coord_line = 6

    for i in range(min_coord_line, max_coord_line + 1):
        ax.plot([min_coord_line, max_coord_line], [i, i], [min_coord_line, min_coord_line], 
                color='lightgray', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.plot([i, i], [min_coord_line, max_coord_line], [min_coord_line, min_coord_line], 
                color='lightgray', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.plot([i, i], [min_coord_line, min_coord_line], [min_coord_line, max_coord_line], 
                color='lightgray', linestyle='--', alpha=0.3, linewidth=0.5)

    path_np = np.array(path)
    ax.plot(path_np[:, 0], path_np[:, 1], path_np[:, 2], 'b-', label="Ray Path", linewidth=4, alpha=0.9)
    
    intersection_np = np.array(intersection_coords)
    ax.scatter(intersection_np[:, 0], intersection_np[:, 1], intersection_np[:, 2], 
               c='red', s=120, label="Hyperplane Intersections", marker='x')
    
    ax.scatter(x_0[0], x_0[1], x_0[2], c='green', s=300, label="Start", marker='o')
    ax.scatter(x_f[0], x_f[1], x_f[2], c='red', s=300, label="Goal", marker='o')
    ax.text(x_0[0], x_0[1], x_0[2] + 0.2, f'Start ({x_0[0]:.2f}, {x_0[1]:.2f}, {x_0[2]:.2f})', ha='center', va='bottom', fontsize=9, color='black')
    ax.text(x_f[0], x_f[1], x_f[2] + 0.2, f'Goal ({x_f[0]:.2f}, {x_f[1]:.2f}, {x_f[2]:.2f})', ha='center', va='bottom', fontsize=9, color='black')
    
    obstacle_edges_to_plot = []
    if obstacles:
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        for obs in obstacles:
            x, y, z = obs[0], obs[1], obs[2]
            
            vertices = np.array([
                [x, y, z], [x+1, y, z], [x+1, y+1, z], [x, y+1, z],
                [x, y, z+1], [x+1, y, z+1], [x+1, y+1, z+1], [x, y+1, z+1]
            ])
            
            for edge in edges:
                obstacle_edges_to_plot.append(vertices[edge])
            
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]],
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[4], vertices[7], vertices[3], vertices[0]]
            ]
            ax.add_collection3d(Poly3DCollection(faces, alpha=0.5, facecolor='black', edgecolor=None, hatch='..', zorder=2))

    colors = ['cyan', 'magenta', 'yellow', 'orange', 'lightgreen', 'pink', 'purple', 'brown']
    
    for step_idx, (front_cells_at_step, intersection_point) in enumerate(zip(all_front_cells, intersection_coords)):
        color = colors[step_idx % len(colors)]
        alpha = 0.4
        
        for cell_idx, fc in enumerate(front_cells_at_step):
            x, y, z = fc[0], fc[1], fc[2]
            
            vertices = np.array([
                [x, y, z], [x+1, y, z], [x+1, y+1, z], [x, y+1, z],
                [x, y, z+1], [x+1, y, z+1], [x+1, y+1, z+1], [x, y+1, z+1]
            ])
            
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]],
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[4], vertices[7], vertices[3], vertices[0]]
            ]
            
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            ax.add_collection3d(Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='darkblue', linewidth=1))
            
            ax.text(x + 0.5, y + 0.5, z + 0.5, f'S{step_idx}',
                   ha='center', va='center', fontsize=8, fontweight='bold')
            
            if step_idx < len(intersection_coords):
                fc_center = fc + 0.5
                ax.quiver(intersection_point[0], intersection_point[1], intersection_point[2],
                         fc_center[0] - intersection_point[0],
                         fc_center[1] - intersection_point[1],
                         fc_center[2] - intersection_point[2],
                         color='darkred', alpha=0.6, arrow_length_ratio=0.1, linewidth=1.5)
    
    for points in obstacle_edges_to_plot:
        ax.plot3D(*points.T, color='black', linewidth=2, alpha=1.0)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("X-coordinate", fontsize=14)
    ax.set_ylabel("Y-coordinate", fontsize=14)
    ax.set_zlabel("Z-coordinate", fontsize=14)
    ax.legend(loc='upper left', fontsize=12)
    
    ax.set_xlim(min_coord_line - 0.5, max_coord_line + 0.5)
    ax.set_ylim(min_coord_line - 0.5, max_coord_line + 0.5)
    ax.set_zlim(min_coord_line - 0.5, max_coord_line + 0.5)
    
    ax.set_box_aspect([1,1,1])
    
    if y_coords_history is not None and len(y_coords_history) > 0:
        y_coords_np = np.array([y for y in y_coords_history if hasattr(y, '__len__') and len(y) >= 3])
        if y_coords_np.ndim == 2 and y_coords_np.shape[0] > 0 and y_coords_np.shape[1] >= 3:
            ax.scatter(y_coords_np[:, 0], y_coords_np[:, 1], y_coords_np[:, 2], c='purple', s=80,
                       label="y corners (self.y)", zorder=4, marker='s', alpha=0.7)

    plt.tight_layout()
    plt.show()

    info_text_lines = []
    if isGoalReached is not None:
        info_text_lines.append(f"isGoalReached: {isGoalReached}")

    if info_text_lines:
        info_text = "\n".join(info_text_lines)
        fig.text(0.02, 0.98, info_text, fontsize=13, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

def plot_trace(dim, label, x_0, x_f, path, front_cells, intersections, y_history, obstacles, goal_reached, loose_dimension):
    if dim == 2:
        title = f"2D Test: {label}\nStart: {np.round(x_0, 2)}, Goal: {np.round(x_f, 2)}, Loose: {loose_dimension}"
        plot_2d_trace_with_proper_front_cells(
            x_0, x_f,
            path, front_cells, intersections, y_history,
            obstacles,
            title=title
        )
    elif dim == 3:
        title = f"3D Test: {label}\nStart: {np.round(x_0, 2)}, Goal: {np.round(x_f, 2)}, Loose: {loose_dimension}"
        plot_3d_trace_with_proper_front_cells(
            x_0, x_f,
            path, front_cells, intersections, y_history,
            obstacles,
            isGoalReached=goal_reached,
            title=title
        )
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def add_box(ax, text, xy, width, height, facecolor='#f4f7fb', edgecolor='#184589', textcolor='black', textsize=12):
    # Create a rounded rectangle
    box = patches.FancyBboxPatch(
        (xy[0] - width/2, xy[1] - height/2), 
        width, height, 
        boxstyle="round,pad=0.05,rounding_size=0.1",
        facecolor=facecolor, 
        edgecolor=edgecolor, 
        linewidth=2.5
    )
    ax.add_patch(box)
    
    # Add text
    ax.text(xy[0], xy[1], text, ha='center', va='center', 
            fontsize=textsize, fontweight='bold', color=textcolor, wrap=True)
    return box

def add_arrow(ax, start_xy, end_xy):
    ax.annotate("", xy=end_xy, xycoords='data',
                xytext=start_xy, textcoords='data',
                arrowprops=dict(arrowstyle="-|>,head_length=0.6,head_width=0.4", color="#184589", lw=2.5, shrinkA=0, shrinkB=0))

def add_branch_arrow(ax, start_xy, end_xy):
    mid_y = start_xy[1] - 0.45
    # Vertical down from start
    ax.plot([start_xy[0], start_xy[0]], [start_xy[1], mid_y], color="#184589", lw=2.5)
    # Horizontal across
    ax.plot([start_xy[0], end_xy[0]], [mid_y, mid_y], color="#184589", lw=2.5)
    # Vertical down with arrow
    ax.annotate("", xy=end_xy, xycoords='data',
                xytext=(end_xy[0], mid_y), textcoords='data',
                arrowprops=dict(arrowstyle="-|>,head_length=0.6,head_width=0.4", color="#184589", lw=2.5, shrinkA=0, shrinkB=0))

def add_merge_arrow(ax, start_xy, end_xy):
    mid_y = end_xy[1] + 0.45
    # Vertical down from start
    ax.plot([start_xy[0], start_xy[0]], [start_xy[1], mid_y], color="#184589", lw=2.5)
    # Horizontal across
    ax.plot([start_xy[0], end_xy[0]], [mid_y, mid_y], color="#184589", lw=2.5)
    
    # Only draw the downward arrow exactly at the center merge point once to avoid overlaps
    if start_xy[0] == 0: # We can draw it when processing the center route
        ax.annotate("", xy=end_xy, xycoords='data',
                    xytext=(end_xy[0], mid_y), textcoords='data',
                    arrowprops=dict(arrowstyle="-|>,head_length=0.6,head_width=0.4", color="#184589", lw=2.5, shrinkA=0, shrinkB=0))

def generate_flowchart(output_path):
    fig, ax = plt.subplots(figsize=(10, 13))
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.5, 10.5)
    ax.axis('off')

    # Coordinates
    y_starts = [10, 9, 8, 7, 6, 5, 4, 3]
    steps = [
        "1. Input images",
        "2. Remove black borders",
        "3. Extract Green, Lab-a, Lab-b channels",
        "4. Normalize each channel",
        "5. Extract DWT-17 features per channel",
        "6. Extract PyRadiomics GLCM features per channel",
        "7. Concatenate features",
        "8. Z-score standardization"
    ]
    
    width = 6.0
    height = 0.55
    
    # Draw main sequential steps
    for i in range(len(steps)):
        add_box(ax, steps[i], (0, y_starts[i]), width, height, facecolor='#f4f7fb', textsize=13)
        if i < len(steps) - 1:
            add_arrow(ax, (0, y_starts[i]-height/2), (0, y_starts[i+1]+height/2))
            
    # Branching
    route_y = 1.3
    route_height = 1.0
    route_width = 2.4
    
    # Route 1
    add_box(ax, "Clustering route 1:\nRaw standardized\nfeatures \u2192 K-means", 
            (-2.7, route_y), route_width, route_height, facecolor='#effcf4', edgecolor='#339966', textsize=12)
            
    # Route 2
    add_box(ax, "Clustering route 2:\nPCA \u2192 K-means", 
            (0, route_y), route_width, route_height, facecolor='#f4f7fb', edgecolor='#184589', textsize=12)
            
    # Route 3
    add_box(ax, "Clustering route 3:\nUMAP \u2192 K-means", 
            (2.7, route_y), route_width, route_height, facecolor='#f8f4fc', edgecolor='#7030a0', textsize=12)
            
    # Arrows from step 8 to routes
    add_branch_arrow(ax, (0, y_starts[7]-height/2), (-2.7, route_y+route_height/2))
    add_branch_arrow(ax, (0, y_starts[7]-height/2), (0, route_y+route_height/2))
    add_branch_arrow(ax, (0, y_starts[7]-height/2), (2.7, route_y+route_height/2))
    
    # Merge step
    final_y = -0.2
    add_box(ax, "Reveal MES labels for post-hoc validation", (0, final_y), width, height, facecolor='#f4f7fb', textsize=13)
    
    # Arrows from routes to merge
    add_merge_arrow(ax, (-2.7, route_y-route_height/2), (0, final_y+height/2))
    add_merge_arrow(ax, (2.7, route_y-route_height/2), (0, final_y+height/2))
    add_merge_arrow(ax, (0, route_y-route_height/2), (0, final_y+height/2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Flowchart saved to {output_path}")

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'Texture_Analysis_Presentation_Figures')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'architecture_flowchart.png')
    generate_flowchart(output_file)

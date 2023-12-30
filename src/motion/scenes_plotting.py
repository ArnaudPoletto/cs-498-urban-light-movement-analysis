import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
import matplotlib.pyplot as plt

from src.motion.scenes_analysis import TYPE_COLORS

DATA_PATH = str(GLOBAL_DIR / "data") + "/"
GENERATED_PATH = str(GLOBAL_DIR / "generated") + "/"
MOTION_PATH = f"{DATA_PATH}motion/"
GENERATED_MOTION_PATH = f"{GENERATED_PATH}motion/"

def plot_class_comparison(statistics, scene_list):
    # Define object classes to plot
    object_classes = ['vegetation', 'person', 'vehicle']
    
    # Data preparation
    data = {obj_class: [] for obj_class in object_classes}
    mean_counts = {obj_class: [] for obj_class in object_classes}
    for scene in scene_list:
        for obj_class in object_classes:
            data[obj_class].append(statistics[scene]['object_class_dict'].get(obj_class, 0))
            mean_counts[obj_class].append(statistics[scene]['mean_class_counts'].get(obj_class, 0))

    # Create subplots for each object class
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))

    for i, obj_class in enumerate(object_classes):
        bars = axes[i].bar(scene_list, data[obj_class], color=np.array(TYPE_COLORS[obj_class]) / 255.0)
        axes[i].set_title(obj_class.capitalize())
        axes[i].set_ylabel('Magnitude')
        axes[i].set_xticks(np.arange(len(scene_list)))
        axes[i].set_xticklabels([scene[2:-4] for scene in scene_list], rotation=45)
        axes[i].set_ylim(0, 1.1 * max(data[obj_class]))
        
        # Add mean object numbers as text on the bars
        if obj_class != 'vegetation':
            for bar, mean_count in zip(bars, mean_counts[obj_class]):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width() / 2., height,
                            f'{mean_count:.2f}', ha='center', va='bottom')
                
    fig.suptitle('Object Class Comparison')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"{GENERATED_MOTION_PATH}part3_class_comparison.png")
    print(f"💾 Plot saved at {statistics_path}")

    # Show figure
    plt.show()




if __name__ == "__main__":
    # Scenes ordered by group
    scene_list = ['P3Scene11.mp4', 'P3Scene12.mp4', 'P3Scene13.mp4', 'P3Scene14.mp4', 'P3Scene15.mp4', 
                  'P3Scene02.mp4', 'P3Scene04.mp4', 'P3Scene05.mp4', 'P3Scene06.mp4', 'P3Scene10.mp4', 
                  'P3Scene01.mp4', 'P3Scene03.mp4', 'P3Scene07.mp4', 'P3Scene08.mp4', 'P3Scene09.mp4']
    
    # Get statistics
    statistics_path = f"{MOTION_PATH}statistics_part3.npy"
    statistics = np.load(statistics_path, allow_pickle=True).item()

    plot_class_comparison(statistics, scene_list)
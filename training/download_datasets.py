"""
Download datasets from Roboflow - Free emergency exit detection datasets
"""

import os
import requests
from pathlib import Path

def download_roboflow_dataset(project_url: str, dataset_name: str, output_dir: str):
    """
    Download dataset from Roboflow using direct URL
    
    Example URLs:
    - Emergency Exit Signs v2: https://universe.roboflow.com/emergency-exit-signs/emergency-exit-signs-v2
    - Stairs Detection: https://universe.roboflow.com/stairs-detection/stairs-fo4v5
    - Escalator-Stairs: https://universe.roboflow.com/escalatorstairsdetection/escalator-stairs
    """
    
    print(f"\n{'='*60}")
    print(f"Downloading {dataset_name}...")
    print(f"{'='*60}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Instructions for manual download
    print(f"""
    To download {dataset_name}:
    
    1. Visit: {project_url}
    2. Click 'Export' button
    3. Select 'YOLOv8' format
    4. Choose 'Download zip'
    5. Extract to: {output_dir}
    
    Or use Roboflow Python API:
    
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR_API_KEY")
    project = rf.workspace("...").project("...")
    dataset = project.versions(1).download("yolov8")
    """)
    
    return output_dir

def setup_datasets():
    """Setup all required datasets"""
    
    base_dir = Path(__file__).parent.parent / "datasets"
    
    datasets = [
        {
            "name": "Emergency Exit Signs v2",
            "url": "https://universe.roboflow.com/emergency-exit-signs/emergency-exit-signs-v2",
            "dir": "exit_signs_v2",
            "images": 1070,
        },
        {
            "name": "Stairs Detection",
            "url": "https://universe.roboflow.com/stairs-detection/stairs-fo4v5",
            "dir": "stairs_detection",
            "images": 7890,
        },
        {
            "name": "Escalator-Stairs",
            "url": "https://universe.roboflow.com/escalatorstairsdetection/escalator-stairs",
            "dir": "escalator_stairs",
            "images": 8690,
        },
        {
            "name": "Exit-Detection (Doors, Obstacles, Exits)",
            "url": "https://universe.roboflow.com/project1exits/exit-detection-w00yi",
            "dir": "exit_detection",
            "images": 36,
        },
    ]
    
    print("\n" + "="*80)
    print("EMERGENCY PATH FINDER - DATASET SETUP")
    print("="*80)
    print("\nYou have two options to download datasets:\n")
    print("OPTION 1: Automatic download (requires Roboflow account)")
    print("  - Sign up free at: https://roboflow.com")
    print("  - Get API key from account settings")
    print("  - Run: python download_datasets.py --api-key YOUR_KEY\n")
    print("OPTION 2: Manual download")
    print("  - Download each dataset from Roboflow")
    print("  - Extract to respective folders\n")
    print("="*80)
    
    for dataset in datasets:
        output_path = base_dir / dataset["dir"]
        print(f"\n{dataset['name']}")
        print(f"  Location: {output_path}")
        print(f"  Images: ~{dataset['images']}")
        print(f"  Link: {dataset['url']}")
        
        download_roboflow_dataset(
            dataset["url"],
            dataset["name"],
            str(output_path)
        )
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Download datasets from Roboflow")
    print("2. Run: python train_exit_detector.py")
    print("3. Run: python train_stairs_detector.py")
    print("4. Models will be saved in ml_models/ folder")
    print("="*80)

if __name__ == "__main__":
    setup_datasets()

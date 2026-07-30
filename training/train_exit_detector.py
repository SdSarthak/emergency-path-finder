"""
Train YOLOv8 model for exit sign detection
Quick training on laptop - supports both CPU and GPU
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch

def check_environment():
    """Check available hardware"""
    print("\n" + "="*60)
    print("Environment Check")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("⚠ GPU Not Found - Using CPU (slower but still works)")
        device = "cpu"
    
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    print("="*60 + "\n")
    
    return device

def train_exit_detector():
    """Train YOLOv8 model for exit detection"""
    
    device = check_environment()
    
    # Dataset path
    dataset_dir = Path(__file__).parent.parent / "datasets" / "exit_signs_v2"
    
    if not dataset_dir.exists():
        print(f"ERROR: Dataset not found at {dataset_dir}")
        print("Please run: python download_datasets.py")
        return False
    
    # Create YAML config if needed
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        print("Creating data.yaml configuration...")
        yaml_content = """
path: {path}
train: images/train
val: images/val
test: images/test

nc: 3
names: ['exit', 'stairs', 'door']
""".format(path=str(dataset_dir))
        yaml_path.write_text(yaml_content)
    
    print("\n" + "="*60)
    print("Training YOLOv8-nano for Exit Detection")
    print("="*60)
    print(f"Dataset: {dataset_dir}")
    print(f"Device: {device}")
    print("="*60 + "\n")
    
    try:
        # Load model
        model = YOLO('yolov8n.pt')
        
        # Train
        results = model.train(
            data=str(yaml_path),
            epochs=50,  # Increase for better accuracy, 50 is good for demo
            imgsz=416,
            batch=8,  # Reduce if OOM, increase if GPU memory available
            patience=20,  # Early stopping
            device=device,
            project=str(Path(__file__).parent.parent / "ml_models"),
            name="exit_detector",
            pretrained=True,
            optimizer='SGD',
            lr0=0.01,
            amp=True,  # Mixed precision for faster training
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10,
            translate=0.1,
            scale=0.5,
            mosaic=1.0,
            flipud=0.5,
            fliplr=0.5,
            verbose=True,
        )
        
        print("\n" + "="*60)
        print("✓ Training Complete!")
        print("="*60)
        print(f"Best model: {results.save_dir / 'weights' / 'best.pt'}")
        
        # Convert to TFLite
        convert_to_tflite(results.save_dir / 'weights' / 'best.pt')
        
        return True
        
    except Exception as e:
        print(f"✗ Training Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def convert_to_tflite(model_path):
    """Convert PyTorch model to TensorFlow Lite"""
    
    print("\n" + "="*60)
    print("Converting to TensorFlow Lite...")
    print("="*60)
    
    try:
        model = YOLO(str(model_path))
        
        # Export to TFLite
        export_path = model.export(
            format='tflite',
            imgsz=416,
            half=True,  # Use FP16 for smaller model
        )
        
        print(f"✓ TFLite Model: {export_path}")
        
        # Copy to flutter assets
        flutter_models = Path(__file__).parent.parent / "flutter_app" / "assets" / "models"
        flutter_models.mkdir(parents=True, exist_ok=True)
        
        import shutil
        tflite_file = list(Path(export_path).parent.glob("*.tflite"))[0]
        dest = flutter_models / "exit_detector.tflite"
        shutil.copy(tflite_file, dest)
        
        print(f"✓ Copied to Flutter: {dest}")
        
    except Exception as e:
        print(f"⚠ TFLite Conversion Note: {e}")
        print("This is optional - model can still be used in PyTorch format")

def main():
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  Emergency Path Finder - ML Training Pipeline  ".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    # Check if dataset exists
    dataset_dir = Path(__file__).parent.parent / "datasets" / "exit_signs_v2"
    if not dataset_dir.exists():
        print(f"\n⚠ Dataset not found: {dataset_dir}")
        print("\nSetup Instructions:")
        print("=" * 60)
        print("1. Download dataset from Roboflow:")
        print("   - Visit: https://universe.roboflow.com/emergency-exit-signs/emergency-exit-signs-v2")
        print("   - Click 'Export'")
        print("   - Select 'YOLOv8' format")
        print("   - Click 'Download zip'")
        print("   - Extract to: " + str(dataset_dir))
        print("=" * 60)
        print("\n2. Then run this script again")
        return
    
    # Start training
    success = train_exit_detector()
    
    if success:
        print("\n✓ All done! Model is ready for mobile app")
    else:
        print("\n✗ Training failed. Check errors above")
        sys.exit(1)

if __name__ == "__main__":
    main()

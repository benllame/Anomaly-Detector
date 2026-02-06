from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass
class ProjectConfig:
    """Configuración global del proyecto"""
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw" / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"
    
    # Dataset
    DATASET_NAME: str = "screw"  # MVTec category
    IMAGE_SIZE: int = 640
    CLASSES: List[str] = None
    NUM_CLASSES: int = 2
    
    # Training
    BATCH_SIZE: int = 16
    NUM_EPOCHS: int = 100
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 0.0005
    
    # Validation
    VAL_SPLIT: float = 0.2
    TEST_SPLIT: float = 0.0  # MVTec ya tiene test separado
    
    # MLflow
    MLFLOW_TRACKING_URI: str = "file:./mlruns"
    EXPERIMENT_NAME: str = "defect-detection-dinov2"
    
    # Deployment
    ONNX_MODEL_PATH: Path = MODELS_DIR / "model.onnx"
    TENSORRT_MODEL_PATH: Path = MODELS_DIR / "model.engine"
    QUANTIZED_MODEL_PATH: Path = MODELS_DIR / "model_int8.onnx"
    
    # Edge
    EDGE_INPUT_SIZE: int = 416  # Smaller for edge devices
    EDGE_CONF_THRESHOLD: float = 0.6
    EDGE_FRAME_SKIP: int = 2
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_CONF_THRESHOLD: float = 0.5
    
    def __post_init__(self):
        if self.CLASSES is None:
            self.CLASSES = ["good", "defect"]
        
        # Crear directorios si no existen
        for dir_path in [self.DATA_DIR, self.RAW_DATA_DIR, 
                         self.PROCESSED_DATA_DIR, self.MODELS_DIR, 
                         self.EXPERIMENTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

# Instancia global
config = ProjectConfig()
"""
Configuration File - OPTIMIZED for RTX 3060 6GB (CORRECTED)
"""
from pathlib import Path

# Base directories
BASE_DIR = Path(r'D:\asl_project')
DATA_DIR = BASE_DIR / 'data' / 'processed'
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw' / 'asl_alphabet_train'
REAL_WORLD_DIR = BASE_DIR / 'data' / 'real_world_test'

# Model directories
SAVED_MODELS_DIR = BASE_DIR / 'models' / 'saved_models'
LOGS_DIR = BASE_DIR / 'models' / 'training_logs'
DOCS_DIR = BASE_DIR / 'docs'

# Model configurations
IMG_SIZE = 224
BATCH_SIZE = 32  # تقليل الباتش قليلاً لضمان استقرار التحديثات
NUM_CLASSES = 27
SEED = 42

MODELS = ['EfficientNetB0', 'ResNet50', 'InceptionV3']

# Training parameters
INITIAL_LR_PHASE1 = 0.001
INITIAL_LR_PHASE2 = 0.0001


# Epochs (تم التعديل لتناسب التبسيط)
MAX_EPOCHS_PHASE1 = 20  # كافية جداً للـ Head المبسط
MAX_EPOCHS_PHASE2 = 15


# Callbacks
PATIENCE_ES_PHASE1 = 5
PATIENCE_ES_PHASE2 = 5


PATIENCE_LR_PHASE1 = 3
PATIENCE_LR_PHASE2 = 2


# Regularization
L2_REGULARIZATION = 0.0001 # تم تخفيفه لأننا بسطنا المودل
DROPOUT_RATE_1 = 0.5

# ✅✅ Data augmentation parameters - SOFT (تم التخفيف للبدء)
ROTATION_RANGE = 10          # تم التقليل من 25 [cite: 47]
WIDTH_SHIFT_RANGE = 0.1      # تم التقليل
HEIGHT_SHIFT_RANGE = 0.1     # تم التقليل
SHEAR_RANGE = 0.1            # تم التقليل
ZOOM_RANGE = 0.1             # تم التقليل
BRIGHTNESS_RANGE = [0.9, 1.1] # تم التضييق لتجنب الصور المظلمة جداً
CHANNEL_SHIFT_RANGE = 0.0    # تم الإلغاء مؤقتاً للحفاظ على الألوان الطبيعية [cite: 48]
HORIZONTAL_FLIP = True

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
"""
IMPROVED Model Building Module - OPTIMIZED for ASL
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0, InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from tensorflow.keras.regularizers import l2
from pathlib import Path

# ✅ استورد كل حاجة من config
from config import *


class ModelBuilder:
    """Class to build OPTIMIZED transfer learning models"""

    def __init__(self, model_name, num_classes=NUM_CLASSES, img_size=IMG_SIZE):
        self.model_name = model_name
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        self.unfreeze_layers = None

    def build_model(self):
        print(f"\n🏗️  Building SIMPLIFIED {self.model_name} Model")

        inputs = layers.Input(shape=(self.img_size, self.img_size, 3))

        # Load pretrained base model
        if self.model_name == 'ResNet50':
            base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
            self.unfreeze_layers = 50
        elif self.model_name == 'EfficientNetB0':
            base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
            self.unfreeze_layers = 80
        elif self.model_name == 'InceptionV3':
            base_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=inputs)
            self.unfreeze_layers = 100
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        # Freeze base model
        base_model.trainable = False

        # ✅✅✅ بناء رأس (Head) بسيط وفعال بدلاً من التعقيد السابق
        x = base_model.output

        # التأكد من وجود Global Pooling (مهم جداً للربط)
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

        # [cite_start]طبقة Dense واحدة قوية تكفي في البداية [cite: 8, 9, 10, 11]
        x = layers.Dense(512, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION), name='dense_hidden')(x)
        x = layers.BatchNormalization(name='bn_hidden')(x)
        x = layers.Dropout(DROPOUT_RATE_1, name='dropout_hidden')(x)

        # Output Layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)

        self.model = models.Model(inputs=inputs, outputs=outputs, name=self.model_name)

        # Compile
        self.model.compile(
            optimizer=Adam(learning_rate=INITIAL_LR_PHASE1),
            loss='categorical_crossentropy',  # عدنا للـ Loss العادي مؤقتاً للتأكد
            metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
        )

        print(f"✅ {self.model_name} built successfully with SIMPLIFIED Head!")
        return self.model

    def unfreeze_for_finetuning(self, phase=2):
        """Unfreeze layers for fine-tuning with optimal strategy"""
        if self.model is None:
            raise ValueError("Model not built yet!")

        print(f"\n{'=' * 60}")
        print(f"🔓 Phase {phase}: Unfreezing layers for fine-tuning")
        print(f"{'=' * 60}")

        # ✅ استخدم القيم من config
        # Only one fine-tuning phase retained
        # Unfreeze top layers
        for layer in self.model.layers[-self.unfreeze_layers:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

        new_lr = INITIAL_LR_PHASE2
        optimizer = Adam(learning_rate=new_lr, clipnorm=1.0)  # ✅ أضف clipnorm

        # ✅ تعريف Label Smoothing (نفس التعريف)
        def label_smoothed_categorical_crossentropy(y_true, y_pred):
            label_smoothing = 0.1
            y_true = y_true * (1.0 - label_smoothing) + label_smoothing / self.num_classes
            return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        # Recompile model بنفس الإعدادات
        self.model.compile(
            optimizer=optimizer,
            loss=label_smoothed_categorical_crossentropy,  # ✅ نفس الـ Loss
            metrics=['accuracy',
                     keras.metrics.Precision(name='precision'),
                     keras.metrics.Recall(name='recall'),
                     keras.metrics.AUC(name='auc')]  # ✅ أضف AUC
        )

        trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        print(f"✅ Learning rate: {new_lr}")
        print(f"✅ Trainable parameters: {trainable_params:,}")
        print(f"✅ Phase {phase} fine-tuning ready!")
        print(f"{'=' * 60}\n")

    def get_callbacks(self, phase=1):
        """Get OPTIMIZED training callbacks for each phase"""
        # Create directories
        Path(SAVED_MODELS_DIR).mkdir(parents=True, exist_ok=True)
        Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

        # ✅ استخدم القيم من config
        if phase == 1:
            patience_es = PATIENCE_ES_PHASE1
            patience_lr = PATIENCE_LR_PHASE1
        else:  # phase 2 fine-tune fallback
            patience_es = PATIENCE_ES_PHASE2
            patience_lr = PATIENCE_LR_PHASE2

        callbacks = [
            ModelCheckpoint(
                filepath=str(SAVED_MODELS_DIR / f'{self.model_name}_phase{phase}_best.weights.h5'),
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            ),

            EarlyStopping(
                monitor='val_accuracy',
                patience=patience_es,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.002
            ),

            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience_lr,
                min_lr=1e-7,
                verbose=1
            ),

            CSVLogger(
                filename=str(LOGS_DIR / f'{self.model_name}_phase{phase}_log.csv'),
                separator=',',
                append=False
            )
        ]

        return callbacks


def create_all_models():
    """Create all three models"""
    models_dict = {}

    for model_name in MODELS:
        builder = ModelBuilder(model_name)
        model = builder.build_model()
        models_dict[model_name] = {
            'model': model,
            'builder': builder
        }

    return models_dict


if __name__ == "__main__":
    # Test model building
    print("\n" + "🚀" * 30)
    print("Testing OPTIMIZED Model Building")
    print("🚀" * 30 + "\n")

    models_dict = create_all_models()

    for model_name, model_info in models_dict.items():
        print(f"\n{'=' * 60}")
        print(f"📋 {model_name} Summary:")
        print(f"{'=' * 60}")
        model_info['model'].summary()

    print("\n" + "✅" * 30)
    print("All OPTIMIZED Models Built Successfully!")
    print("✅" * 30 + "\n")
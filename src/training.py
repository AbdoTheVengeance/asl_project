
"""
OPTIMIZED 3-Phase Training Module for ASL Recognition - TRAINING ONLY
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import numpy as np

# ✅ استورد من config
from config import *
from model_builder import ModelBuilder
from utils import ensure_dir
from data_preprocessing import load_class_weights

# Set memory growth for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Memory Growth Enabled: {len(gpus)} GPU(s) detected\n")
    except RuntimeError as e:
        print(e)


class OptimizedModelTrainer:
    """OPTIMIZED trainer with 3-phase training strategy"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.model_builder = ModelBuilder(model_name)
        self.model = None
        self.history = None

        # Fix class weights loading
        weights_path = DATA_DIR / 'class_weights.json'
        if weights_path.exists():
            with open(weights_path, 'r') as f:
                class_weights_str = json.load(f)
            # Convert string keys to integers and ensure all values are Python floats
            self.class_weights = {}
            for key, value in class_weights_str.items():
                # Convert value to Python float if it's a tensor
                if hasattr(value, 'numpy'):
                    float_value = float(value.numpy())
                else:
                    float_value = float(value)
                self.class_weights[int(key)] = float_value
        else:
            self.class_weights = None

    def prepare_data_generators(self):
        """Prepare data generators using the external preprocessing module."""
        print("📦 Preparing CORRECTED Data Generators")

        # ✅✅ التصحيح: استدعاء دالة إنشاء المولدات من ملف data_preprocessing.py
        from data_preprocessing import create_data_generators

        # يجب أن تقوم هذه الدالة بإرجاع المولدات الثلاثة بشكل صحيح
        train_generator, val_generator, test_generator = create_data_generators(self.model_name)

        # ✅ تأكد من تعريف المتغيرات قبل الإرجاع
        self.steps_per_epoch_train = train_generator.samples // train_generator.batch_size
        self.steps_per_epoch_val = val_generator.samples // val_generator.batch_size

        return train_generator, val_generator, test_generator  # الآن test_generator مُعرَّف

    def train_3_phase(self, train_gen, val_gen):
        """3-Phase OPTIMIZED Training Strategy"""
        print(f"\n{'=' * 60}")
        print(f"🎯 Starting 3-Phase Training: {self.model_name}")
        print(f"{'=' * 60}")

        # Phase 1: Feature Extraction
        print(f"\n📚 PHASE 1: Feature Extraction (Frozen Base)")
        print(f"{'-' * 50}")

        # Build model (base frozen by default)
        self.model = self.model_builder.build_model()

        # ✅ استخدم MAX_EPOCHS_PHASE1 من config
        phase1_history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=MAX_EPOCHS_PHASE1,
            callbacks=self.model_builder.get_callbacks(phase=1),
            verbose=1,
            class_weight=self.class_weights
        )

        # Phase 2: Fine-tuning top layers
        print(f"\n🔧 PHASE 2: Fine-tuning Top Layers")
        print(f"{'-' * 50}")

        self.model_builder.unfreeze_for_finetuning(phase=2)

        # ✅ استخدم MAX_EPOCHS_PHASE2 من config
        phase2_history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=MAX_EPOCHS_PHASE2,
            callbacks=self.model_builder.get_callbacks(phase=2),
            verbose=1,
            class_weight=self.class_weights
        )

        # # Phase 3: Full fine-tuning
        # print(f"\n🎯 PHASE 3: Full Model Fine-tuning")
        # print(f"{'-' * 50}")
        #
        # self.model_builder.unfreeze_for_finetuning(phase=3)
        #
        # # ✅ استخدم MAX_EPOCHS_PHASE3 من config
        # phase3_history = self.model.fit(
        #     train_gen,
        #     validation_data=val_gen,
        #     epochs=MAX_EPOCHS_PHASE3,
        #     callbacks=self.model_builder.get_callbacks(phase=3),
        #     verbose=1,
        #     class_weight=self.class_weights
        # )

        # Combine histories
        self.history = self._combine_histories([phase1_history, phase2_history])#phase3_history

        print(f"\n✅ 3-Phase Training Complete!")

        # 💡💡 التصحيح النهائي والأكثر قوة: حفظ الهيكلة والأوزان مع معالجة استثنائية
        
        # 1. حفظ هيكلة المودل (Architecture) إلى JSON
        try:
            model_json = self.model.to_json()
            arch_path = SAVED_MODELS_DIR / f'{self.model_name}_architecture.json'
            with open(arch_path, "w") as json_file:
                json_file.write(model_json)
            print(f"✅ Model architecture saved to JSON: {arch_path}")
        except Exception as e:
            # معالجة الخطأ والسماح للبرنامج بالاستمرار
            print(f"❌ WARNING: Failed to save model architecture to JSON due to {type(e).__name__}.")
            print("⚠️ السبب: كائن TensorFlow Tensor عالق في إعدادات المودل (مثل L2 Regularization).")
            print("⚠️ سيستمر البرنامج بحفظ الأوزان، ويمكن إعادة تحميل المودل باستخدام ModelBuilder + load_weights().")
            print(f"   Error details: {e}")
        
        # 2. حفظ أوزان المودل (Weights Only) إلى H5 (هذا يجب أن يعمل بشكل مؤكد)
        try:
            weights_path = SAVED_MODELS_DIR / f'{self.model_name}_weights_only.h5'
            self.model.save_weights(weights_path)
            print(f"✅ Final model weights saved to H5: {weights_path}")
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Failed to save model weights! {e}")

        print(f"{'=' * 60}\n")

        return self.history

    def _combine_histories(self, histories):
        """Combine multiple history objects"""
        combined_history = {}

        for key in histories[0].history.keys():
            combined_history[key] = []
            for history in histories:
                combined_history[key].extend(history.history[key])

        return type('History', (), {'history': combined_history})()

    # ❌ تم إزالة دالة evaluate_on_test
    # ❌ تم إزالة دالة evaluate_on_real_world

    def save_training_history(self):
        """Save training history to CSV - FIXED AGAINST KEY ERROR"""
        if self.history is None or not self.history.history: # إضافة تحقق من أن القاموس ليس فارغاً
            print("⚠️ No valid training history to save!")
            return

        ensure_dir(LOGS_DIR)
        
        history_data = self.history.history

        # 💡 التحقق من وجود مفتاح 'loss' قبل استخدامه لتحديد طول الـ Epochs
        if 'loss' not in history_data:
            print("❌ CRITICAL ERROR: 'loss' metric not found in combined history. Cannot save history.")
            print(f"   Available keys in history: {list(history_data.keys())}")
            return

        # Convert to DataFrame
        num_epochs = len(history_data['loss'])
        history_dict = {
            'epoch': list(range(1, num_epochs + 1)),
            
            # استخدام .get() مع قائمة من NaN لضمان عدم حدوث KeyError
            'loss': history_data['loss'],
            'accuracy': history_data.get('accuracy', [np.nan] * num_epochs),
            'precision': history_data.get('precision', [np.nan] * num_epochs),
            'recall': history_data.get('recall', [np.nan] * num_epochs),
            'val_loss': history_data.get('val_loss', [np.nan] * num_epochs),
            'val_accuracy': history_data.get('val_accuracy', [np.nan] * num_epochs),
            'val_precision': history_data.get('val_precision', [np.nan] * num_epochs),
            'val_recall': history_data.get('val_recall', [np.nan] * num_epochs),
        }

        # Add learning rate if available
        if 'lr' in history_data:
            history_dict['learning_rate'] = history_data['lr']

        df = pd.DataFrame(history_dict)

        # Save to CSV
        csv_path = LOGS_DIR / f'{self.model_name}_optimized_history.csv'
        df.to_csv(csv_path, index=False)
        print(f"✅ Training history saved to CSV: {csv_path}")

        # Also save summary stats
        summary = {
            'model_name': self.model_name,
            'total_epochs': len(df),
            'best_val_accuracy': float(df['val_accuracy'].max()),
            'best_val_accuracy_epoch': int(df['val_accuracy'].idxmax() + 1),
            'final_val_accuracy': float(df['val_accuracy'].iloc[-1]),
            'final_train_accuracy': float(df['accuracy'].iloc[-1]),
            'best_val_loss': float(df['val_loss'].min()),
            'training_strategy': '3-Phase Optimized',
            'class_weights_used': bool(self.class_weights),
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        summary_path = LOGS_DIR / f'{self.model_name}_optimized_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"✅ Training summary saved to: {summary_path}")

        return df

    def plot_training_history(self):
        """Plot training history with phase markers"""
        if self.history is None:
            print("⚠️ No training history to plot!")
            return

        ensure_dir(DOCS_DIR)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        epochs = range(1, len(self.history.history['loss']) + 1)

        # ✅ Phase boundaries (استخدم القيم من config)
        phase1_end = MAX_EPOCHS_PHASE1
        phase2_end = MAX_EPOCHS_PHASE1 + MAX_EPOCHS_PHASE2

        # Accuracy
        axes[0, 0].plot(epochs, self.history.history.get('accuracy', [np.nan]*len(epochs)),
                        'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history.history.get('val_accuracy', [np.nan]*len(epochs)),
                        'r-', label='Validation', linewidth=2)
        axes[0, 0].axvline(x=phase1_end, color='g', linestyle='--', alpha=0.7, label='Phase 1 End')
        axes[0, 0].axvline(x=phase2_end, color='orange', linestyle='--', alpha=0.7, label='Phase 2 End')
        axes[0, 0].set_title(f'{self.model_name} - Accuracy (3-Phase Training)',
                             fontweight='bold', fontsize=12)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Loss
        axes[0, 1].plot(epochs, self.history.history.get('loss', [np.nan]*len(epochs)),
                        'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.history.history.get('val_loss', [np.nan]*len(epochs)),
                        'r-', label='Validation', linewidth=2)
        axes[0, 1].axvline(x=phase1_end, color='g', linestyle='--', alpha=0.7, label='Phase 1 End')
        axes[0, 1].axvline(x=phase2_end, color='orange', linestyle='--', alpha=0.7, label='Phase 2 End')
        axes[0, 1].set_title(f'{self.model_name} - Loss (3-Phase Training)',
                             fontweight='bold', fontsize=12)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Precision
        axes[1, 0].plot(epochs, self.history.history.get('precision', [np.nan]*len(epochs)),
                        'b-', label='Train', linewidth=2)
        axes[1, 0].plot(epochs, self.history.history.get('val_precision', [np.nan]*len(epochs)),
                        'r-', label='Validation', linewidth=2)
        axes[1, 0].set_title(f'{self.model_name} - Precision',
                             fontweight='bold', fontsize=12)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Recall
        axes[1, 1].plot(epochs, self.history.history.get('recall', [np.nan]*len(epochs)),
                        'b-', label='Train', linewidth=2)
        axes[1, 1].plot(epochs, self.history.history.get('val_recall', [np.nan]*len(epochs)),
                        'r-', label='Validation', linewidth=2)
        axes[1, 1].set_title(f'{self.model_name} - Recall',
                             fontweight='bold', fontsize=12)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = DOCS_DIR / f'{self.model_name}_3phase_training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ 3-Phase training curves saved to: {plot_path}")
        plt.close()


def train_single_model_optimized(model_name):
    """Train a single model with OPTIMIZED 3-phase strategy (Training Only)"""
    print(f"\n{'🚀' * 30}")
    print(f"Starting OPTIMIZED 3-Phase Training: {model_name}")
    print(f"{'🚀' * 30}")

    trainer = OptimizedModelTrainer(model_name)

    # Prepare data (test_gen is still created but not used for evaluation here)
    train_gen, val_gen, test_gen = trainer.prepare_data_generators()

    # Train with 3-phase strategy
    trainer.train_3_phase(train_gen, val_gen)

    # Save history
    df_history = trainer.save_training_history()

    # Plot
    trainer.plot_training_history()

    print(f"\n{'✅' * 30}")
    print(f"{model_name} - 3-Phase Training Complete!")
    # نطبع فقط دقة التحقق (Validation) لأنها متوفرة من سجل التدريب
    if df_history is not None:
        print(f"Best Val Accuracy: {df_history['val_accuracy'].max():.4f}")
    
    print(f"{'✅' * 30}\n")

    return trainer, df_history


def choose_and_train_model():
    """اختيار وتدريب مودل واحد فقط (تدريب فقط)"""
    print("\n" + "🤖" * 30)
    print("نظام اختيار المودل للتدريب")
    print("🤖" * 30)

    # عرض المودلات المتاحة
    print("\n📋 المودلات المتاحة:")
    available_models = ["ResNet50", "EfficientNetB0", "InceptionV3"]
    for i, model_name in enumerate(available_models, 1):
        print(f"   {i}. {model_name}")

    # اختيار المودل
    while True:
        try:
            choice = input(f"\n🔢 اختر رقم المودل الذي تريد تدريبه (1-{len(available_models)}): ").strip()
            if not choice:
                print("⚠️  الرجاء إدخال رقم!")
                continue

            choice_num = int(choice)
            if 1 <= choice_num <= len(available_models):
                selected_model = available_models[choice_num - 1]
                break
            else:
                print(f"❌ الرجاء اختيار رقم بين 1 و {len(available_models)}")
        except ValueError:
            print("❌ الرجاء إدخال رقم صحيح!")

    print(f"\n🎯 تم اختيار المودل: {selected_model}")
    print("⏳ جاري البدء في التدريب...\n")

    # تدريب المودل المختار
    trainer, history_df = train_single_model_optimized(selected_model)

    return trainer, history_df


if __name__ == "__main__":
    print("\n" + "🎓" * 30)
    print("ASL - OPTIMIZED 3-Phase Training Strategy (Training Only)")
    print("🎓" * 30 + "\n")

    # اختيار طريقة التدريب
    print("🔧 طرق التدريب المتاحة:")
    print("   1. تدريب مودل واحد (مستحسن)")
    print("   2. تدريب جميع المودلات")

    while True:
        choice = input("\n🔢 اختر طريقة التدريب (1 أو 2): ").strip()
        if choice in ['1', '2']:
            break
        print("❌ الرجاء اختيار 1 أو 2!")

    if choice == '1':
        # تدريب مودل واحد فقط
        trainer, history_df = choose_and_train_model()
    else:
        # تدريب جميع المودلات
        print("\n🚀 جاري تدريب جميع المودلات...")
        results = {}
        for model_name in ["ResNet50", "EfficientNetB0", "InceptionV3"]:
            trainer, history_df = train_single_model_optimized(model_name)
            results[model_name] = {
                'trainer': trainer,
                'history_df': history_df,
            }

        print("\n🎉 تم تدريب جميع المودلات بنجاح!")

        # عرض ملخص التدريب فقط
        print("\n📊 ملخص التدريب (اعتماداً على بيانات التحقق فقط):")
        print("=" * 65)
        print(f"{'Model':<20} {'Best Val Accuracy':<25} {'Epochs':<10}")
        print("=" * 65)
        for model_name, result in results.items():
            df = result['history_df']
            # تحقق من أن df ليست None قبل محاولة الوصول إلى البيانات
            if df is not None:
                print(f"{model_name:<20} {df['val_accuracy'].max():<25.4f} {len(df):<10}")
            else:
                 print(f"{model_name:<20} {'N/A':<25} {'N/A':<10}")
        print("=" * 65)

    print("\n" + "✅" * 20)
    print("تم إنهاء العملية بنجاح!")
    print("✅" * 20)

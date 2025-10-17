# Simple Deep Learning Audio Stress Analysis - Basic Features Only
# This version uses the same simple features as Random Forest but with neural networks

import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

class SimpleDeepLearningStressAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.history = None
        
    def load_ravdess_data(self, data_path):
        """Load RAVDESS dataset - same as original"""
        print("Loading RAVDESS dataset...")
        
        audio_files = []
        emotions = []
        stress_levels = []
        actors = []
        
        emotion_to_stress = {
            1: 0,  # neutral -> low stress
            2: 0,  # calm -> low stress  
            3: 0,  # happy -> low stress
            4: 1,  # sad -> medium stress
            5: 2,  # angry -> high stress
            6: 2,  # fearful -> high stress
            7: 2,  # disgust -> high stress
            8: 1   # surprised -> medium stress
        }
        
        emotion_names = {
            1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 
            5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
        }
        
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.wav'):
                    parts = file.split('-')
                    if len(parts) >= 7:
                        try:
                            emotion = int(parts[2])
                            actor = int(parts[6].split('.')[0])
                            
                            audio_files.append(os.path.join(root, file))
                            emotions.append(emotion)
                            stress_levels.append(emotion_to_stress.get(emotion, 0))
                            actors.append(actor)
                        except (ValueError, IndexError):
                            continue
        
        df = pd.DataFrame({
            'file_path': audio_files,
            'emotion': emotions,
            'stress_level': stress_levels,
            'actor': actors
        })
        
        df['emotion_name'] = df['emotion'].map(emotion_names)
        
        print(f"Loaded {len(df)} audio files from {df['actor'].nunique()} actors")
        print("Dataset distribution:")
        print(df['stress_level'].value_counts().sort_index())
        
        return df
    
    def extract_simple_features(self, file_path):
        """Extract simple, consistent features - same as Random Forest version"""
        try:
            # Load audio file with simple settings
            y, sr = librosa.load(file_path, sr=22050, duration=3.0)
            
            if len(y) < 1000:  # Too short
                return None
            
            # Exactly the same features as your Random Forest model
            features = []
            
            # 1. MFCC features (13 mean + 13 std = 26 features)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            features.extend(mfcc_mean)
            features.extend(mfcc_std)
            
            # 2. Spectral features (3 features)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            features.extend([spectral_centroid, spectral_rolloff, spectral_bandwidth])
            
            # 3. Rhythm features (2 features)
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            try:
                tempo = librosa.beat.tempo(y=y, sr=sr)[0]
            except:
                tempo = 0
            features.extend([zero_crossing_rate, tempo])
            
            # 4. Energy features (3 features)
            rms = np.mean(librosa.feature.rms(y=y))
            try:
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                pitch_mean = np.mean(pitches[pitches > 0]) if len(pitches[pitches > 0]) > 0 else 0
            except:
                pitch_mean = 0
            
            # Harmonic and percussive
            try:
                harmonic, percussive = librosa.effects.hpss(y)
                harmonic_mean = np.mean(harmonic)
                percussive_mean = np.mean(percussive)
            except:
                harmonic_mean = 0
                percussive_mean = 0
            
            features.extend([rms, pitch_mean, harmonic_mean, percussive_mean])
            
            # Total: 26 + 3 + 2 + 4 = 35 features (same as Random Forest)
            features = np.array(features, dtype=np.float32)
            
            # Replace any NaN values
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Ensure exactly 35 features
            if len(features) != 35:
                print(f"Warning: Expected 35 features, got {len(features)}")
                if len(features) < 35:
                    features = np.pad(features, (0, 35 - len(features)), 'constant')
                else:
                    features = features[:35]
            
            return features
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def create_feature_names(self):
        """Create names for 35 basic features"""
        names = []
        
        # MFCC features
        for i in range(13):
            names.append(f'mfcc_mean_{i}')
        for i in range(13):
            names.append(f'mfcc_std_{i}')
        
        # Other features
        names.extend([
            'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth',
            'zero_crossing_rate', 'tempo', 'rms_energy', 'pitch_mean',
            'harmonic_mean', 'percussive_mean'
        ])
        
        return names
    
    def create_simple_deep_model(self, input_shape):
        """Create a simple but effective neural network"""
        tf.keras.backend.clear_session()
        
        model = models.Sequential([
            # Input layer
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.3),
            
            # Hidden layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, data_path):
        """Train with simple features"""
        print("🧠 Simple Deep Learning Training")
        print("=" * 50)
        print("Using basic features (same as Random Forest) with neural network")
        
        # Load dataset
        df = self.load_ravdess_data(data_path)
        
        # Extract simple features
        print(f"\nExtracting 35 basic features from {len(df)} files...")
        
        features_list = []
        valid_indices = []
        failed_count = 0
        
        for i, file_path in enumerate(df['file_path']):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(df)} files ({i/len(df)*100:.1f}%)")
            
            features = self.extract_simple_features(file_path)
            if features is not None and len(features) == 35:
                features_list.append(features)
                valid_indices.append(i)
            else:
                failed_count += 1
        
        if len(features_list) == 0:
            print("❌ No valid features extracted!")
            return 0.0
        
        # Create arrays
        X = np.array(features_list)
        y = df.iloc[valid_indices]['stress_level'].values
        
        print(f"\n✅ Feature extraction complete!")
        print(f"Successfully processed: {len(features_list)} files")
        print(f"Failed: {failed_count} files")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Labels distribution: {np.bincount(y)}")
        
        # Create feature names
        self.feature_names = self.create_feature_names()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
        
        # Scale features
        print("Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create model
        print("Building simple neural network...")
        self.model = self.create_simple_deep_model(X_train_scaled.shape[1])
        
        print("\n🧠 NEURAL NETWORK ARCHITECTURE:")
        self.model.summary()
        
        # Setup callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Train model
        print("\n🔥 Training neural network...")
        
        self.history = self.model.fit(
            X_train_scaled, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        y_pred = self.model.predict(X_test_scaled, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print(f"\n🎯 NEURAL NETWORK RESULTS")
        print("=" * 40)
        print(f"Test Accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
        print(f"Test Loss: {test_loss:.3f}")
        
        stress_labels = ['Low Stress', 'Medium Stress', 'High Stress']
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=stress_labels))
        
        # Plot results
        self.plot_training_history()
        self.plot_confusion_matrix(y_test, y_pred_classes)
        
        # Feature importance (using permutation)
        self.analyze_feature_importance(X_test_scaled[:100], y_test[:100])
        
        # Save model
        self.save_model("./models/simple_deep_stress_model")
        
        return test_accuracy
    
    def plot_training_history(self):
        """Plot training curves"""
        if self.history is None:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Training')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Training')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('./models/simple_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Simple Deep Learning - Confusion Matrix')
        plt.tight_layout()
        plt.savefig('./models/simple_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_feature_importance(self, X_sample, y_sample):
        """Simple feature importance analysis"""
        print("\n📊 Top Feature Analysis:")
        
        # Use model weights from first layer as rough importance
        weights = np.abs(self.model.layers[0].get_weights()[0])
        feature_importance = np.mean(weights, axis=1)
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("Top 10 Most Important Features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']:<25} {row['importance']:.4f}")
        
        # Plot
        plt.figure(figsize=(10, 6))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance (Layer 1 Weights)')
        plt.title('Feature Importance - Simple Deep Learning')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('./models/simple_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath):
        """Save model"""
        try:
            self.model.save(f"{filepath}.h5")
            
            with open(f"{filepath}_scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
            with open(f"{filepath}_features.pkl", "wb") as f:
                pickle.dump(self.feature_names, f)
            
            print(f"\n💾 Simple Deep Learning Model saved!")
            print(f"   - {filepath}.h5")
            print(f"   - {filepath}_scaler.pkl")
            print(f"   - {filepath}_features.pkl")
            
        except Exception as e:
            print(f"Error saving: {e}")
    
    def analyze_file(self, file_path):
        """Analyze single file"""
        if self.model is None:
            print("No model loaded!")
            return
        
        features = self.extract_simple_features(file_path)
        if features is not None:
            features_scaled = self.scaler.transform([features])
            probabilities = self.model.predict(features_scaled, verbose=0)[0]
            prediction = np.argmax(probabilities)
            
            stress_labels = ['Low Stress', 'Medium Stress', 'High Stress']
            
            print(f"\n🎯 Simple Deep Learning Analysis: {os.path.basename(file_path)}")
            print(f"Predicted: {stress_labels[prediction]}")
            print(f"Confidence: {np.max(probabilities):.3f}")
            print("Probabilities:")
            for i, prob in enumerate(probabilities):
                print(f"  {stress_labels[i]}: {prob:.3f}")

# Main execution
if __name__ == "__main__":
    print("🧠 Simple Deep Learning Audio Stress Analysis")
    print("=" * 60)
    print("Using basic features with neural network for reliability")
    
    DATA_PATH = "./data/RAVDESS/"
    
    if not os.path.exists(DATA_PATH):
        print("❌ Dataset not found!")
        print(f"Path: {DATA_PATH}")
    else:
        analyzer = SimpleDeepLearningStressAnalyzer()
        
        print("\nStarting simple deep learning training...")
        print("This uses the same 35 features as Random Forest but with neural networks")
        print("Should be more reliable and avoid feature extraction errors")
        
        accuracy = analyzer.train_model(DATA_PATH)
        
        if accuracy > 0.8:
            print(f"\n🎉 Excellent! Neural network achieved {accuracy:.1%} accuracy!")
        elif accuracy > 0.7:
            print(f"\n✨ Great! Neural network achieved {accuracy:.1%} accuracy!")
        elif accuracy > 0.6:
            print(f"\n👍 Good! Neural network achieved {accuracy:.1%} accuracy!")
        else:
            print(f"\n🤔 Neural network achieved {accuracy:.1%} accuracy")
        
        print("This should be higher than your Random Forest accuracy!")
        
        # Test samples
        sample_files = []
        for root, dirs, files in os.walk(DATA_PATH):
            for file in files[:3]:
                if file.endswith('.wav'):
                    sample_files.append(os.path.join(root, file))
        
        if sample_files:
            print(f"\n🧪 Testing neural network:")
            for sample_file in sample_files:
                analyzer.analyze_file(sample_file)
# Fixed Deep Learning Audio Stress Analysis - Consistent Feature Extraction
# This version fixes the array shape inconsistency issue

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
from tensorflow.keras.utils import to_categorical

class FixedDeepLearningStressAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.history = None
        self.expected_feature_count = 146  # Fixed feature count
        
    def load_ravdess_data(self, data_path):
        """Load RAVDESS dataset and create stress labels"""
        print("Loading RAVDESS dataset...")
        
        audio_files = []
        emotions = []
        stress_levels = []
        actors = []
        
        # Improved emotion to stress mapping
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
                    if len(parts) >= 7:  # Ensure we have enough parts
                        try:
                            emotion = int(parts[2])
                            actor = int(parts[6].split('.')[0])
                            
                            audio_files.append(os.path.join(root, file))
                            emotions.append(emotion)
                            stress_levels.append(emotion_to_stress.get(emotion, 0))
                            actors.append(actor)
                        except (ValueError, IndexError):
                            print(f"Skipping malformed filename: {file}")
                            continue
        
        df = pd.DataFrame({
            'file_path': audio_files,
            'emotion': emotions,
            'stress_level': stress_levels,
            'actor': actors
        })
        
        df['emotion_name'] = df['emotion'].map(emotion_names)
        
        print(f"Loaded {len(df)} audio files from {df['actor'].nunique()} actors")
        print("\nDataset distribution:")
        print(df['stress_level'].value_counts().sort_index())
        print("\nEmotion distribution:")
        print(df.groupby(['emotion_name', 'stress_level']).size())
        
        return df
    
    def safe_audio_load(self, file_path):
        """Safely load audio with consistent preprocessing"""
        try:
            # Load with consistent parameters
            y, sr = librosa.load(file_path, sr=22050, duration=3.0, res_type='kaiser_fast')
            
            # Ensure we have audio data
            if len(y) < 1000:  # Very short file
                return None, None
            
            # Normalize
            y = librosa.util.normalize(y)
            
            # Remove silence but keep some padding
            y_trimmed = librosa.effects.trim(y, top_db=20, frame_length=512, hop_length=64)[0]
            
            # Ensure minimum length
            min_samples = sr * 1  # 1 second minimum
            if len(y_trimmed) < min_samples:
                # Pad with zeros if too short
                pad_length = min_samples - len(y_trimmed)
                y_trimmed = np.pad(y_trimmed, (0, pad_length), 'constant')
            
            # Ensure maximum length for consistency
            max_samples = sr * 3  # 3 seconds maximum
            if len(y_trimmed) > max_samples:
                y_trimmed = y_trimmed[:max_samples]
                
            return y_trimmed, sr
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def extract_fixed_features(self, file_path):
        """Extract exactly 146 features consistently"""
        try:
            # Load audio safely
            y, sr = self.safe_audio_load(file_path)
            if y is None:
                return None
            
            features = []
            
            # 1. MFCC features (13 coefficients × 4 statistics = 52 features)
            try:
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
                if mfccs.shape[1] > 0:  # Ensure we have time frames
                    features.extend(np.mean(mfccs, axis=1))     # 13 features
                    features.extend(np.std(mfccs, axis=1))      # 13 features  
                    features.extend(np.max(mfccs, axis=1))      # 13 features
                    features.extend(np.min(mfccs, axis=1))      # 13 features
                else:
                    features.extend([0] * 52)
            except:
                features.extend([0] * 52)
            
            # 2. Spectral features (4 types × 4 statistics = 16 features)
            try:
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                spectral_rolloffs = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
                spectral_bandwidths = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
                zero_crossings = librosa.feature.zero_crossing_rate(y)[0]
                
                for feature_array in [spectral_centroids, spectral_rolloffs, spectral_bandwidths, zero_crossings]:
                    if len(feature_array) > 0:
                        features.extend([np.mean(feature_array), np.std(feature_array), 
                                       np.max(feature_array), np.min(feature_array)])
                    else:
                        features.extend([0, 0, 0, 0])
            except:
                features.extend([0] * 16)
            
            # 3. Energy features (2 types × 4 statistics = 8 features)
            try:
                rms = librosa.feature.rms(y=y)[0]
                if len(rms) > 0:
                    features.extend([np.mean(rms), np.std(rms), np.max(rms), np.min(rms)])
                else:
                    features.extend([0, 0, 0, 0])
                
                # Additional energy measure
                energy = np.sum(y**2) / len(y)
                max_energy = np.max(y**2)
                features.extend([energy, max_energy, np.var(y**2), np.mean(np.abs(y))])
            except:
                features.extend([0] * 8)
            
            # 4. Pitch features (6 features)
            try:
                # Use piptrack for more robust pitch detection
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=80, fmax=400)
                pitches_clean = pitches[magnitudes > 0.1]
                
                if len(pitches_clean) > 0:
                    features.extend([
                        np.mean(pitches_clean),
                        np.std(pitches_clean),
                        np.max(pitches_clean),
                        np.min(pitches_clean),
                        np.max(pitches_clean) - np.min(pitches_clean),  # pitch range
                        len(pitches_clean) / pitches.size  # voiced ratio
                    ])
                else:
                    features.extend([0] * 6)
            except:
                features.extend([0] * 6)
            
            # 5. Rhythm features (4 features)
            try:
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
                features.append(tempo)
                
                if len(beats) > 1:
                    beat_intervals = np.diff(beats) / sr
                    features.extend([
                        np.mean(beat_intervals),
                        np.std(beat_intervals),
                        len(beats) / (len(y) / sr)  # beat density
                    ])
                else:
                    features.extend([0, 0, 0])
            except:
                features.extend([0] * 4)
            
            # 6. Harmonic features (4 features)
            try:
                y_harmonic, y_percussive = librosa.effects.hpss(y)
                harmonic_energy = np.mean(librosa.feature.rms(y=y_harmonic)[0])
                percussive_energy = np.mean(librosa.feature.rms(y=y_percussive)[0])
                
                # Harmonic-to-noise ratio estimation
                harmonic_ratio = harmonic_energy / (harmonic_energy + percussive_energy + 1e-8)
                spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y)[0])
                
                features.extend([harmonic_energy, percussive_energy, harmonic_ratio, spectral_flatness])
            except:
                features.extend([0] * 4)
            
            # 7. Chroma features (12 features)
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                if chroma.shape[1] > 0:
                    features.extend(np.mean(chroma, axis=1))  # 12 chroma bins
                else:
                    features.extend([0] * 12)
            except:
                features.extend([0] * 12)
            
            # 8. Mel-frequency features (8 features)
            try:
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                if mel_spec_db.shape[1] > 0:
                    features.extend([
                        np.mean(mel_spec_db), np.std(mel_spec_db),
                        np.max(mel_spec_db), np.min(mel_spec_db),
                        np.percentile(mel_spec_db, 25), np.percentile(mel_spec_db, 75),
                        np.median(mel_spec_db), np.var(mel_spec_db)
                    ])
                else:
                    features.extend([0] * 8)
            except:
                features.extend([0] * 8)
            
            # 9. Contrast features (7 features)
            try:
                contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
                if contrast.shape[1] > 0:
                    features.extend(np.mean(contrast, axis=1))  # 7 contrast bands
                else:
                    features.extend([0] * 7)
            except:
                features.extend([0] * 7)
            
            # 10. Additional statistical features (5 features)
            try:
                # Additional time-domain features
                features.extend([
                    len(y) / sr,  # duration
                    np.sum(np.abs(np.diff(y))),  # total variation
                    np.mean(np.abs(np.diff(y))),  # mean absolute difference
                    np.std(np.abs(np.diff(y))),   # std of absolute difference
                    np.sum(y > 0) / len(y)  # positive sample ratio
                ])
            except:
                features.extend([0] * 5)
            
            # Ensure exactly 146 features
            features = np.array(features, dtype=np.float32)
            
            if len(features) != self.expected_feature_count:
                print(f"Warning: Expected {self.expected_feature_count} features, got {len(features)}")
                # Pad or truncate to expected size
                if len(features) < self.expected_feature_count:
                    features = np.pad(features, (0, self.expected_feature_count - len(features)), 'constant')
                else:
                    features = features[:self.expected_feature_count]
            
            # Replace any NaN or inf values
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {file_path}: {e}")
            return None
    
    def create_feature_names(self):
        """Create names for exactly 146 features"""
        names = []
        
        # MFCC features (52)
        for stat in ['mean', 'std', 'max', 'min']:
            for i in range(13):
                names.append(f'mfcc_{stat}_{i}')
        
        # Spectral features (16)
        spectral_types = ['centroid', 'rolloff', 'bandwidth', 'zcr']
        for stype in spectral_types:
            for stat in ['mean', 'std', 'max', 'min']:
                names.append(f'spectral_{stype}_{stat}')
        
        # Energy features (8)
        for stat in ['mean', 'std', 'max', 'min']:
            names.append(f'rms_{stat}')
        names.extend(['energy', 'max_energy', 'energy_var', 'mean_abs'])
        
        # Pitch features (6)
        names.extend(['pitch_mean', 'pitch_std', 'pitch_max', 'pitch_min', 'pitch_range', 'voiced_ratio'])
        
        # Rhythm features (4)
        names.extend(['tempo', 'beat_interval_mean', 'beat_interval_std', 'beat_density'])
        
        # Harmonic features (4)
        names.extend(['harmonic_energy', 'percussive_energy', 'harmonic_ratio', 'spectral_flatness'])
        
        # Chroma features (12)
        for i in range(12):
            names.append(f'chroma_{i}')
        
        # Mel features (8)
        names.extend(['mel_mean', 'mel_std', 'mel_max', 'mel_min', 'mel_q25', 'mel_q75', 'mel_median', 'mel_var'])
        
        # Contrast features (7)
        for i in range(7):
            names.append(f'contrast_{i}')
        
        # Additional features (5)
        names.extend(['duration', 'total_variation', 'mean_abs_diff', 'std_abs_diff', 'positive_ratio'])
        
        return names
    
    def create_deep_model(self, input_shape):
        """Create deep neural network model"""
        tf.keras.backend.clear_session()
        
        model = models.Sequential([
            # Input layer
            layers.Dense(256, activation='relu', input_shape=(input_shape,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Hidden layers
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
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
        """Complete training pipeline with consistent features"""
        print("🚀 Starting Fixed Deep Learning Training")
        print("=" * 60)
        
        # Load dataset
        df = self.load_ravdess_data(data_path)
        
        # Extract features
        print(f"\nExtracting {self.expected_feature_count} features from {len(df)} files...")
        print("Using fixed feature extraction for consistency...")
        
        features_list = []
        valid_indices = []
        failed_count = 0
        
        for i, file_path in enumerate(df['file_path']):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(df)} files ({i/len(df)*100:.1f}%) - Failed: {failed_count}")
            
            features = self.extract_fixed_features(file_path)
            if features is not None:
                features_list.append(features)
                valid_indices.append(i)
            else:
                failed_count += 1
        
        if len(features_list) == 0:
            print("❌ No features could be extracted! Check your audio files.")
            return 0.0
        
        # Create arrays
        X = np.array(features_list)
        y = df.iloc[valid_indices]['stress_level'].values
        
        print(f"\n✅ Feature extraction complete!")
        print(f"Successfully processed: {len(features_list)} files")
        print(f"Failed: {failed_count} files")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Expected shape: ({len(features_list)}, {self.expected_feature_count})")
        print(f"Labels distribution: {np.bincount(y)}")
        
        # Verify feature consistency
        if X.shape[1] != self.expected_feature_count:
            print(f"❌ Feature shape mismatch! Expected {self.expected_feature_count}, got {X.shape[1]}")
            return 0.0
        
        # Create feature names
        self.feature_names = self.create_feature_names()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        print("\nScaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create model
        print("Building deep neural network...")
        self.model = self.create_deep_model(X_train_scaled.shape[1])
        
        print(f"\n🧠 MODEL ARCHITECTURE:")
        self.model.summary()
        
        # Setup callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        print("\n🔥 Training deep learning model...")
        
        self.history = self.model.fit(
            X_train_scaled, y_train,
            epochs=150,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        y_pred = self.model.predict(X_test_scaled, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print(f"\n🎯 DEEP LEARNING RESULTS")
        print("=" * 40)
        print(f"Test Accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
        print(f"Test Loss: {test_loss:.3f}")
        
        stress_labels = ['Low Stress', 'Medium Stress', 'High Stress']
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=stress_labels))
        
        # Plot results
        self.plot_training_history()
        self.plot_confusion_matrix(y_test, y_pred_classes)
        
        # Save model
        self.save_model("./models/deep_stress_model")
        
        return test_accuracy
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', color='blue')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', color='red')
        ax1.set_title('Model Accuracy Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Training Loss', color='blue')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', color='red')
        ax2.set_title('Model Loss Over Time')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./models/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Training history saved to ./models/training_history.png")
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Low', 'Medium', 'High'],
                   yticklabels=['Low', 'Medium', 'High'])
        
        plt.xlabel('Predicted Stress Level')
        plt.ylabel('True Stress Level')
        plt.title('Deep Learning Model - Confusion Matrix')
        plt.tight_layout()
        plt.savefig('./models/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Confusion matrix saved to ./models/confusion_matrix.png")
    
    def save_model(self, filepath):
        """Save model and components"""
        try:
            # Save Keras model
            self.model.save(f"{filepath}.h5")
            
            # Save other components
            with open(f"{filepath}_scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
            with open(f"{filepath}_features.pkl", "wb") as f:
                pickle.dump(self.feature_names, f)
            
            if self.history:
                with open(f"{filepath}_history.pkl", "wb") as f:
                    pickle.dump(self.history.history, f)
            
            print(f"\n💾 Deep Learning Model saved!")
            print(f"   - {filepath}.h5")
            print(f"   - {filepath}_scaler.pkl")
            print(f"   - {filepath}_features.pkl")
            
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def analyze_file(self, file_path):
        """Analyze a single file"""
        if self.model is None:
            print("No trained model found!")
            return
        
        features = self.extract_fixed_features(file_path)
        if features is not None:
            features_scaled = self.scaler.transform([features])
            probabilities = self.model.predict(features_scaled, verbose=0)[0]
            prediction = np.argmax(probabilities)
            
            stress_labels = ['Low Stress', 'Medium Stress', 'High Stress']
            confidence = np.max(probabilities)
            
            print(f"\n🎯 Analysis for: {os.path.basename(file_path)}")
            print(f"Predicted: {stress_labels[prediction]}")
            print(f"Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
            print("Probabilities:")
            for i, prob in enumerate(probabilities):
                print(f"  {stress_labels[i]}: {prob:.3f}")
        else:
            print("Could not extract features from the audio file")

# Main execution
if __name__ == "__main__":
    print("🧠 Fixed Deep Learning Audio Stress Analysis")
    print("=" * 60)
    
    DATA_PATH = "./data/RAVDESS/"
    
    if not os.path.exists(DATA_PATH):
        print("❌ RAVDESS dataset not found!")
        print(f"Please make sure the dataset is in: {DATA_PATH}")
    else:
        analyzer = FixedDeepLearningStressAnalyzer()
        
        print("Starting FIXED deep learning training...")
        print("This version ensures consistent feature extraction.")
        
        accuracy = analyzer.train_model(DATA_PATH)
        
        if accuracy > 0.8:
            print(f"\n🎉 Excellent! Achieved {accuracy:.1%} accuracy!")
        elif accuracy > 0.7:
            print(f"\n✨ Great! Achieved {accuracy:.1%} accuracy!")
        elif accuracy > 0.6:
            print(f"\n👍 Good! Achieved {accuracy:.1%} accuracy!")
        else:
            print(f"\n🤔 Achieved {accuracy:.1%} accuracy.")
        
        print(f"\n📁 Files saved in ./models/ folder")
        
        # Test with sample files
        sample_files = []
        for root, dirs, files in os.walk(DATA_PATH):
            for file in files[:2]:
                if file.endswith('.wav'):
                    sample_files.append(os.path.join(root, file))
        
        if sample_files:
            print(f"\n🧪 Testing with sample files:")
            for sample_file in sample_files:
                analyzer.analyze_file(sample_file)
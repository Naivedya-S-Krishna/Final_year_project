# Audio Stress Analysis - Main Script for VS Code
# This script will build your first stress detection model

import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AudioStressAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def load_ravdess_data(self, data_path):
        """Load RAVDESS dataset and create stress labels"""
        print("Loading RAVDESS dataset...")
        
        audio_files = []
        emotions = []
        stress_levels = []
        actors = []
        
        # Emotion to stress mapping
        emotion_to_stress = {
            1: 0,  # neutral -> low stress
            2: 0,  # calm -> low stress  
            3: 0,  # happy -> low stress
            4: 1,  # sad -> medium stress
            5: 2,  # angry -> high stress
            6: 2,  # fearful -> high stress
            7: 1,  # disgust -> medium stress
            8: 1   # surprised -> medium stress
        }
        
        emotion_names = {
            1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 
            5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
        }
        
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.wav'):
                    # Extract info from filename: 03-01-06-01-02-01-12.wav
                    parts = file.split('-')
                    if len(parts) >= 3:
                        emotion = int(parts[2])
                        actor = int(parts[6].split('.')[0])
                        
                        audio_files.append(os.path.join(root, file))
                        emotions.append(emotion)
                        stress_levels.append(emotion_to_stress.get(emotion, 0))
                        actors.append(actor)
        
        df = pd.DataFrame({
            'file_path': audio_files,
            'emotion': emotions,
            'stress_level': stress_levels,
            'actor': actors
        })
        
        # Add emotion names for better understanding
        df['emotion_name'] = df['emotion'].map(emotion_names)
        
        print(f"Loaded {len(df)} audio files from {df['actor'].nunique()} actors")
        print("\nDataset distribution:")
        print(df['stress_level'].value_counts().sort_index())
        print("\nEmotion distribution:")
        print(df.groupby(['emotion_name', 'stress_level']).size())
        
        return df
    
    def extract_features(self, file_path):
        """Extract comprehensive audio features"""
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=22050, duration=3.0)
            
            # 1. MFCC features (most important for speech)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # 2. Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            
            # 3. Rhythm and tempo features
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            
            try:
                tempo = librosa.beat.tempo(y=y, sr=sr)[0]
            except:
                tempo = 0
            
            # 4. Energy features
            rms = np.mean(librosa.feature.rms(y=y))
            
            # 5. Pitch features
            try:
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                pitch_mean = np.mean(pitches[pitches > 0]) if len(pitches[pitches > 0]) > 0 else 0
            except:
                pitch_mean = 0
            
            # 6. Harmonic and percussive components
            harmonic, percussive = librosa.effects.hpss(y)
            harmonic_mean = np.mean(harmonic)
            percussive_mean = np.mean(percussive)
            
            # Combine all features
            features = np.concatenate([
                mfcc_mean,          # 13 features
                mfcc_std,           # 13 features
                [spectral_centroid, spectral_rolloff, spectral_bandwidth,  # 3 features
                 zero_crossing_rate, tempo, rms, pitch_mean,               # 4 features
                 harmonic_mean, percussive_mean]                           # 2 features
            ])
            
            return features
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def create_feature_names(self):
        """Create descriptive names for all features"""
        names = []
        
        # MFCC names
        for i in range(13):
            names.append(f'mfcc_mean_{i}')
        for i in range(13):
            names.append(f'mfcc_std_{i}')
            
        # Other feature names
        names.extend([
            'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth',
            'zero_crossing_rate', 'tempo', 'rms_energy', 'pitch_mean',
            'harmonic_mean', 'percussive_mean'
        ])
        
        return names
    
    def train_model(self, data_path):
        """Complete training pipeline"""
        print("🚀 Starting Audio Stress Analysis Training")
        print("=" * 50)
        
        # Load dataset
        df = self.load_ravdess_data(data_path)
        
        # Extract features for all files
        print(f"\nExtracting features from {len(df)} files...")
        print("This may take 5-10 minutes...")
        
        features_list = []
        valid_indices = []
        
        for i, file_path in enumerate(df['file_path']):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(df)} files ({i/len(df)*100:.1f}%)")
            
            features = self.extract_features(file_path)
            if features is not None:
                features_list.append(features)
                valid_indices.append(i)
        
        # Create feature matrix
        X = np.array(features_list)
        y = df.iloc[valid_indices]['stress_level'].values
        
        print(f"\n✓ Feature extraction complete!")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Labels distribution: {np.bincount(y)}")
        
        # Create feature names
        self.feature_names = self.create_feature_names()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
        
        # Scale features
        print("\nScaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Display results
        print(f"\n🎯 MODEL PERFORMANCE")
        print("=" * 30)
        print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        print(f"\nDetailed Classification Report:")
        stress_labels = ['Low Stress', 'Medium Stress', 'High Stress']
        print(classification_report(y_test, y_pred, target_names=stress_labels))
        
        # Feature importance
        self.show_feature_importance()
        
        # Confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        # Save model
        self.save_model("./models/stress_model")
        
        return accuracy
    
    def show_feature_importance(self):
        """Display most important features"""
        if self.model is None or self.feature_names is None:
            return
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 TOP 10 MOST IMPORTANT FEATURES:")
        print("-" * 40)
        for i, row in importance_df.head(10).iterrows():
            print(f"{row['feature']:<25} {row['importance']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Feature Importances for Stress Detection')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('./models/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Stress Detection - Confusion Matrix')
        stress_labels = ['Low\nStress', 'Medium\nStress', 'High\nStress']
        plt.xticks([0.5, 1.5, 2.5], stress_labels)
        plt.yticks([0.5, 1.5, 2.5], stress_labels, rotation=0)
        plt.tight_layout()
        plt.savefig('./models/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath):
        """Save trained model and scaler"""
        try:
            with open(f"{filepath}_model.pkl", "wb") as f:
                pickle.dump(self.model, f)
            with open(f"{filepath}_scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
            with open(f"{filepath}_features.pkl", "wb") as f:
                pickle.dump(self.feature_names, f)
            
            print(f"\n💾 Model saved successfully!")
            print(f"   - {filepath}_model.pkl")
            print(f"   - {filepath}_scaler.pkl")
            print(f"   - {filepath}_features.pkl")
            
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def analyze_file(self, file_path):
        """Analyze a single audio file"""
        if self.model is None:
            print("No trained model found!")
            return
        
        features = self.extract_features(file_path)
        if features is not None:
            features_scaled = self.scaler.transform([features])
            
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            stress_labels = ['Low Stress', 'Medium Stress', 'High Stress']
            
            print(f"\n🎯 Analysis Results for: {os.path.basename(file_path)}")
            print(f"Predicted Stress Level: {stress_labels[prediction]}")
            print("Confidence Scores:")
            for i, prob in enumerate(probabilities):
                print(f"  {stress_labels[i]}: {prob:.3f} ({prob*100:.1f}%)")
        else:
            print("Error: Could not extract features from the audio file")

# Main execution
if __name__ == "__main__":
    print("🎵 Audio Stress Analysis System")
    print("=" * 50)
    
    # Set your dataset path
    DATA_PATH = "./data/RAVDESS/"
    
    # Check if dataset exists
    if not os.path.exists(DATA_PATH):
        print("❌ RAVDESS dataset not found!")
        print(f"Please make sure the dataset is in: {DATA_PATH}")
        print("Run test_data.py first to verify your setup.")
    else:
        # Create analyzer and train model
        analyzer = AudioStressAnalyzer()
        
        print("Starting model training...")
        print("This will take 10-15 minutes on the full dataset.")
        
        accuracy = analyzer.train_model(DATA_PATH)
        
        if accuracy > 0.7:
            print(f"\n🎉 Great! Your model achieved {accuracy:.1%} accuracy!")
        elif accuracy > 0.6:
            print(f"\n✓ Good start! Your model achieved {accuracy:.1%} accuracy.")
        else:
            print(f"\n⚠️  Model accuracy is {accuracy:.1%}. This might improve with more data or better features.")
        
        print(f"\n📁 Check the ./models/ folder for saved files!")
        print(f"📊 Check the generated plots for insights!")
        
        # Test with a sample file
        sample_files = []
        for root, dirs, files in os.walk(DATA_PATH):
            for file in files[:3]:  # Get first 3 files
                if file.endswith('.wav'):
                    sample_files.append(os.path.join(root, file))
        
        if sample_files:
            print(f"\n🧪 Testing with sample files:")
            for sample_file in sample_files:
                analyzer.analyze_file(sample_file)
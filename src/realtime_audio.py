"""
Real-time Audio Stress Analysis for VS Code
This script provides real-time stress detection from microphone input
"""

import pyaudio
import wave
import numpy as np
import librosa
import threading
import time
import pickle
import os
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime

class RealTimeStressAnalyzer:
    def __init__(self, model_path="../models/stress_model"):
        print("🎤 Initializing Real-time Stress Analyzer...")
        
        # Audio parameters
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 22050
        self.RECORD_SECONDS = 3  # Analyze every 3 seconds
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.recording = False
        
        # Load pre-trained model
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_model(model_path)
        
        # Store recent predictions for smoothing
        self.stress_history = deque(maxlen=10)
        self.timestamps = deque(maxlen=10)
        
        # Statistics
        self.total_predictions = 0
        self.session_start = datetime.now()
    
    def load_model(self, model_path):
        """Load pre-trained model and scaler"""
        try:
            with open(f"{model_path}_model.pkl", "rb") as f:
                self.model = pickle.load(f)
            with open(f"{model_path}_scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            with open(f"{model_path}_features.pkl", "rb") as f:
                self.feature_names = pickle.load(f)
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Make sure you've run main_analysis.py first!")
            return False
    
    def extract_features(self, audio_data, sr):
        """Extract features from audio data (same as training)"""
        try:
            # Ensure audio is 1D and has sufficient length
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()
            
            if len(audio_data) < sr * 0.5:  # Less than 0.5 seconds
                return None
            
            # Extract features (exactly same as training)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_data))
            
            try:
                tempo = librosa.beat.tempo(y=audio_data, sr=sr)[0]
            except:
                tempo = 0
            
            rms = np.mean(librosa.feature.rms(y=audio_data))
            
            try:
                pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
                pitch_mean = np.mean(pitches[pitches > 0]) if len(pitches[pitches > 0]) > 0 else 0
            except:
                pitch_mean = 0
            
            harmonic, percussive = librosa.effects.hpss(audio_data)
            harmonic_mean = np.mean(harmonic)
            percussive_mean = np.mean(percussive)
            
            # Combine features
            features = np.concatenate([
                mfcc_mean, mfcc_std,
                [spectral_centroid, spectral_rolloff, spectral_bandwidth,
                 zero_crossing_rate, tempo, rms, pitch_mean,
                 harmonic_mean, percussive_mean]
            ])
            
            return features
        
        except Exception as e:
            print(f"⚠️  Feature extraction error: {e}")
            return None
    
    def predict_stress(self, features):
        """Predict stress level from features"""
        if self.model is None or self.scaler is None:
            return None, None
        
        try:
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            return prediction, probabilities
        
        except Exception as e:
            print(f"⚠️  Prediction error: {e}")
            return None, None
    
    def detect_voice_activity(self, audio_data):
        """Simple voice activity detection"""
        # Calculate RMS energy
        rms_energy = np.sqrt(np.mean(audio_data**2))
        
        # Simple threshold-based VAD
        voice_threshold = 0.01
        return rms_energy > voice_threshold
    
    def record_and_analyze(self):
        """Record audio and analyze stress level"""
        frames = []
        
        # Record for specified duration
        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            if not self.recording:
                break
            
            try:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)
            except Exception as e:
                print(f"⚠️  Recording error: {e}")
                break
        
        if not frames:
            return None
        
        # Convert to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
        
        # Check for voice activity
        if not self.detect_voice_activity(audio_data):
            print("🔇 No voice detected, skipping analysis...")
            return None
        
        # Extract features
        features = self.extract_features(audio_data, self.RATE)
        
        if features is not None:
            # Predict stress
            prediction, probabilities = self.predict_stress(features)
            
            if prediction is not None:
                # Add to history
                current_time = datetime.now()
                self.stress_history.append(prediction)
                self.timestamps.append(current_time)
                self.total_predictions += 1
                
                # Calculate smoothed prediction
                recent_predictions = list(self.stress_history)
                smoothed_prediction = np.mean(recent_predictions)
                
                # Display results
                self.display_results(prediction, probabilities, smoothed_prediction, current_time)
                
                return prediction, probabilities
        
        return None
    
    def display_results(self, prediction, probabilities, smoothed, timestamp):
        """Display analysis results in a nice format"""
        stress_labels = ['Low Stress', 'Medium Stress', 'High Stress']
        colors = ['🟢', '🟡', '🔴']
        
        # Clear screen (works in most terminals)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 60)
        print("🎤 REAL-TIME STRESS ANALYSIS")
        print("=" * 60)
        
        print(f"⏰ Time: {timestamp.strftime('%H:%M:%S')}")
        print(f"📊 Session Duration: {str(timestamp - self.session_start).split('.')[0]}")
        print(f"🔢 Total Predictions: {self.total_predictions}")
        
        print(f"\n{colors[prediction]} CURRENT PREDICTION: {stress_labels[prediction]}")
        print(f"📈 SMOOTHED LEVEL: {smoothed:.2f}")
        
        print(f"\n📋 CONFIDENCE SCORES:")
        for i, (label, prob) in enumerate(zip(stress_labels, probabilities)):
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"  {colors[i]} {label:<13}: {prob:.3f} |{bar}| {prob*100:.1f}%")
        
        # Recent history
        if len(self.stress_history) > 1:
            print(f"\n📈 RECENT TREND:")
            recent_5 = list(self.stress_history)[-5:]
            trend_chars = [colors[p] for p in recent_5]
            print(f"  {''.join(trend_chars)} (last 5 predictions)")
        
        print("\n" + "=" * 60)
        print("🗣️  Keep speaking... Press Ctrl+C to stop")
        print("=" * 60)
    
    def start_recording(self):
        """Start real-time recording and analysis"""
        if self.model is None:
            print("❌ No model loaded! Cannot start real-time analysis.")
            return
        
        try:
            # Test microphone
            print("🎤 Testing microphone...")
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            # Test recording
            test_data = self.stream.read(self.CHUNK)
            print("✅ Microphone working!")
            
            print("\n🚀 Starting real-time stress analysis...")
            print("📣 Speak naturally into your microphone")
            print("🔄 Analysis updates every 3 seconds")
            print("⏹️  Press Ctrl+C to stop\n")
            
            time.sleep(2)  # Give user time to read
            
            self.recording = True
            self.session_start = datetime.now()
            
            while self.recording:
                result = self.record_and_analyze()
                if result is None:
                    time.sleep(0.5)  # Wait a bit if no voice detected
                else:
                    time.sleep(1)  # Small pause between analyses
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Recording stopped by user")
            self.show_session_summary()
        except Exception as e:
            print(f"❌ Recording error: {e}")
            print("Make sure your microphone is connected and working.")
        finally:
            self.stop_recording()
    
    def show_session_summary(self):
        """Show summary of the session"""
        if self.total_predictions == 0:
            return
        
        print("\n📊 SESSION SUMMARY")
        print("=" * 30)
        
        duration = datetime.now() - self.session_start
        predictions_array = np.array(list(self.stress_history))
        
        print(f"⏱️  Duration: {str(duration).split('.')[0]}")
        print(f"🔢 Total Predictions: {self.total_predictions}")
        
        if len(predictions_array) > 0:
            stress_counts = np.bincount(predictions_array, minlength=3)
            stress_labels = ['Low Stress', 'Medium Stress', 'High Stress']
            colors = ['🟢', '🟡', '🔴']
            
            print(f"\n📈 STRESS LEVEL DISTRIBUTION:")
            for i, (label, count) in enumerate(zip(stress_labels, stress_counts)):
                percentage = count / len(predictions_array) * 100
                print(f"  {colors[i]} {label:<13}: {count:2d} ({percentage:4.1f}%)")
            
            avg_stress = np.mean(predictions_array)
            print(f"\n📊 Average Stress Level: {avg_stress:.2f}")
            
            if avg_stress < 0.5:
                print("😌 Overall: You seemed quite calm during this session!")
            elif avg_stress < 1.5:
                print("😐 Overall: You had moderate stress levels during this session.")
            else:
                print("😰 Overall: You seemed quite stressed during this session.")
    
    def stop_recording(self):
        """Stop recording and clean up"""
        self.recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("✅ Audio stream closed")
    
    def record_test_file(self, filename="test_recording.wav", duration=5):
        """Record a test file to verify everything works"""
        print(f"🎤 Recording {duration} seconds to {filename}...")
        print("Speak into your microphone now!")
        
        frames = []
        
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        for i in range(0, int(self.RATE / self.CHUNK * duration)):
            print(f"Recording... {i/(self.RATE/self.CHUNK):.1f}s", end='\r')
            data = stream.read(self.CHUNK)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        # Save as WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print(f"\n✅ Audio saved to {filename}")
        
        # Analyze the recorded file
        self.analyze_file(filename)
    
    def analyze_file(self, filename):
        """Analyze a pre-recorded audio file"""
        if not os.path.exists(filename):
            print(f"❌ File {filename} not found!")
            return
        
        print(f"🎵 Analyzing file: {filename}")
        
        try:
            # Load audio file
            y, sr = librosa.load(filename, sr=self.RATE, duration=5.0)
            
            # Extract features
            features = self.extract_features(y, sr)
            
            if features is not None:
                # Predict
                prediction, probabilities = self.predict_stress(features)
                
                if prediction is not None:
                    stress_labels = ['Low Stress', 'Medium Stress', 'High Stress']
                    colors = ['🟢', '🟡', '🔴']
                    
                    print(f"\n🎯 ANALYSIS RESULT:")
                    print(f"{colors[prediction]} Predicted Stress Level: {stress_labels[prediction]}")
                    
                    print(f"\n📊 Confidence Scores:")
                    for i, (label, prob) in enumerate(zip(stress_labels, probabilities)):
                        bar = "█" * int(prob * 20)
                        print(f"  {colors[i]} {label:<13}: {prob:.3f} |{bar:<20}| {prob*100:.1f}%")
                else:
                    print("❌ Could not make prediction")
            else:
                print("❌ Could not extract features from audio")
                
        except Exception as e:
            print(f"❌ Error analyzing file: {e}")

def main():
    """Main function with user menu"""
    print("🎵 Real-time Audio Stress Analyzer")
    print("=" * 40)
    
    # Create analyzer
    analyzer = RealTimeStressAnalyzer()
    
    if analyzer.model is None:
        print("❌ Could not load model. Please run main_analysis.py first!")
        return
    
    while True:
        print("\nChoose an option:")
        print("1. 🎤 Start real-time stress analysis")
        print("2. 📁 Analyze an audio file")
        print("3. 🎙️  Record and analyze test file (5 seconds)")
        print("4. 📊 Test microphone")
        print("5. ❌ Exit")
        
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                print("\n🚀 Starting real-time analysis...")
                analyzer.start_recording()
                
            elif choice == "2":
                filename = input("Enter audio file path: ").strip()
                if filename:
                    analyzer.analyze_file(filename)
                
            elif choice == "3":
                filename = input("Enter filename (or press Enter for 'test.wav'): ").strip()
                if not filename:
                    filename = "test.wav"
                analyzer.record_test_file(filename, duration=5)
                
            elif choice == "4":
                test_microphone()
                
            elif choice == "5":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def test_microphone():
    """Test if microphone is working"""
    print("🎤 Testing microphone...")
    
    try:
        audio = pyaudio.PyAudio()
        
        # Test audio input
        stream = audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=22050,
            input=True,
            frames_per_buffer=1024
        )
        
        print("🔊 Recording 2 seconds for microphone test...")
        frames = []
        for i in range(0, int(22050 / 1024 * 2)):
            data = stream.read(1024)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Analyze the recorded data
        audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
        max_amplitude = np.max(np.abs(audio_data))
        rms_level = np.sqrt(np.mean(audio_data**2))
        
        print(f"✅ Microphone test complete!")
        print(f"📊 Max amplitude: {max_amplitude:.4f}")
        print(f"📊 RMS level: {rms_level:.4f}")
        
        if max_amplitude > 0.01:
            print("🟢 Microphone is working well!")
        elif max_amplitude > 0.001:
            print("🟡 Microphone is working but signal is weak. Try speaking louder.")
        else:
            print("🔴 Microphone signal is very weak or not working.")
            print("   Check your microphone connection and permissions.")
            
    except Exception as e:
        print(f"❌ Microphone test failed: {e}")
        print("   Make sure your microphone is connected and try again.")

if __name__ == "__main__":
    main()
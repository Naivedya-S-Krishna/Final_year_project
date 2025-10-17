"""
Updated Web Interface for Simple Deep Learning Audio Stress Analysis
Run with: streamlit run web_interface.py
"""

import streamlit as st
import librosa
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tempfile
import pandas as pd

# Import TensorFlow for deep learning model
import tensorflow as tf

# Configure Streamlit page
st.set_page_config(
    page_title="Deep Learning Audio Stress Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_deep_learning_model():
    """Load the simple deep learning model"""
    model_path = "./models/simple_deep_stress_model"
    
    try:
        # Load Keras model (.h5 file)
        model = tf.keras.models.load_model(f"{model_path}.h5")
        
        # Load scaler and feature names (pickle files)
        with open(f"{model_path}_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(f"{model_path}_features.pkl", "rb") as f:
            feature_names = pickle.load(f)
        
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading deep learning model: {e}")
        st.error("Make sure you've run simple_deep_learning.py first!")
        return None, None, None

class DeepLearningWebAnalyzer:
    def __init__(self):
        # Load model using the cached function
        self.model, self.scaler, self.feature_names = load_deep_learning_model()
    
    def extract_simple_features(self, audio_data, sr):
        """Extract the same 35 basic features as training (same as Random Forest)"""
        try:
            features = []
            
            # 1. MFCC features (13 mean + 13 std = 26 features)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            features.extend(mfcc_mean)
            features.extend(mfcc_std)
            
            # 2. Spectral features (3 features)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sr))
            features.extend([spectral_centroid, spectral_rolloff, spectral_bandwidth])
            
            # 3. Rhythm features (2 features)
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_data))
            try:
                tempo = librosa.beat.tempo(y=audio_data, sr=sr)[0]
            except:
                tempo = 0
            features.extend([zero_crossing_rate, tempo])
            
            # 4. Energy features (4 features)
            rms = np.mean(librosa.feature.rms(y=audio_data))
            
            try:
                pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
                pitch_mean = np.mean(pitches[pitches > 0]) if len(pitches[pitches > 0]) > 0 else 0
            except:
                pitch_mean = 0
            
            try:
                harmonic, percussive = librosa.effects.hpss(audio_data)
                harmonic_mean = np.mean(harmonic)
                percussive_mean = np.mean(percussive)
            except:
                harmonic_mean = 0
                percussive_mean = 0
            
            features.extend([rms, pitch_mean, harmonic_mean, percussive_mean])
            
            # Convert to array and handle NaN values
            features = np.array(features, dtype=np.float32)
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Ensure exactly 35 features
            if len(features) != 35:
                if len(features) < 35:
                    features = np.pad(features, (0, 35 - len(features)), 'constant')
                else:
                    features = features[:35]
            
            return features
            
        except Exception as e:
            st.error(f"Feature extraction error: {e}")
            return None
    
    def predict_stress_deep(self, features):
        """Predict stress level using deep learning model"""
        if self.model is None or self.scaler is None:
            return None, None, None
        
        try:
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Get prediction probabilities from neural network
            probabilities = self.model.predict(features_scaled, verbose=0)[0]
            prediction = np.argmax(probabilities)
            
            # Calculate confidence
            max_prob = np.max(probabilities)
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
            confidence = max_prob * (1 - entropy/1.5)  # Simple confidence measure
            confidence = np.clip(confidence, 0, 1)
            
            return prediction, probabilities, confidence
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, None, None
    
    def visualize_audio(self, audio_data, sr):
        """Create audio visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Waveform
        time_axis = np.linspace(0, len(audio_data)/sr, len(audio_data))
        axes[0,0].plot(time_axis, audio_data)
        axes[0,0].set_title('Waveform')
        axes[0,0].set_xlabel('Time (seconds)')
        axes[0,0].set_ylabel('Amplitude')
        axes[0,0].grid(True, alpha=0.3)
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=axes[0,1])
        axes[0,1].set_title('Spectrogram')
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        librosa.display.specshow(mfccs, x_axis='time', ax=axes[1,0])
        axes[1,0].set_title('MFCC Features')
        
        # Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr)
        mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, ax=axes[1,1])
        axes[1,1].set_title('Mel Spectrogram')
        
        plt.tight_layout()
        return fig

def main():
    """Main deep learning web interface"""
    
    # Header
    st.title("🧠 Deep Learning Audio Stress Analyzer")
    st.markdown("**Neural Network model for enhanced stress detection**")
    
    # Initialize analyzer
    analyzer = DeepLearningWebAnalyzer()
    
    if analyzer.model is None:
        st.error("❌ Deep learning model not loaded!")
        st.markdown("""
        ### To train the deep learning model:
        1. Open VS Code terminal
        2. Run: `python simple_deep_learning.py`
        3. Wait for training to complete
        4. Refresh this page
        """)
        st.stop()
    
    # Success message
    st.success("✅ Deep learning model loaded successfully!")
    
    # Model info
    with st.expander("🔍 Model Information", expanded=False):
        st.markdown("""
        **Model Type:** Neural Network (TensorFlow/Keras)
        **Architecture:** 3-layer fully connected network
        **Features:** 35 audio features (same as Random Forest but processed by neural network)
        **Training:** Enhanced with deep learning pattern recognition
        **Expected Accuracy:** 75-80%+ (higher than Random Forest)
        """)
    
    # Sidebar
    st.sidebar.title("📊 Analysis Options")
    analysis_mode = st.sidebar.radio(
        "Choose Analysis Mode:",
        ["Upload Audio File", "Demo Analysis"]
    )
    
    # Main content
    if analysis_mode == "Upload Audio File":
        upload_analysis(analyzer)
    else:
        demo_analysis(analyzer)

def upload_analysis(analyzer):
    """Handle file upload analysis"""
    st.header("📁 Upload Audio File")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="Upload any audio file for deep learning analysis"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            # Load audio
            audio_data, sr = librosa.load(tmp_file_path, sr=22050, duration=3.0)
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duration", f"{len(audio_data)/sr:.2f} sec")
            with col2:
                st.metric("Sample Rate", f"{sr} Hz")
            with col3:
                st.metric("Model", "Neural Network")
            
            # Audio player
            st.audio(uploaded_file, format='audio/wav')
            
            # Analyze
            with st.spinner("🧠 Deep learning model analyzing..."):
                features = analyzer.extract_simple_features(audio_data, sr)
                
                if features is not None:
                    prediction, probabilities, confidence = analyzer.predict_stress_deep(features)
                    
                    if prediction is not None:
                        display_deep_results(prediction, probabilities, confidence)
                        
                        # Show visualizations
                        if st.checkbox("Show Audio Visualizations", value=True):
                            with st.spinner("Creating visualizations..."):
                                fig = analyzer.visualize_audio(audio_data, sr)
                                st.pyplot(fig)
                        
                        # Show feature details
                        if st.checkbox("Show Feature Analysis"):
                            show_feature_details(features, analyzer.feature_names)
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass

def demo_analysis(analyzer):
    """Demo analysis with sample files"""
    st.header("🧪 Demo Analysis")
    
    # Check if RAVDESS data exists
    demo_path = "./data/RAVDESS/"
    
    if not os.path.exists(demo_path):
        st.warning("Demo data not found. Please upload your own file.")
        return
    
    # Find sample files
    sample_files = []
    emotion_names = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 
                    5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}
    
    for root, dirs, files in os.walk(demo_path):
        for file in files[:15]:  # First 15 files
            if file.endswith('.wav'):
                parts = file.split('-')
                if len(parts) >= 3:
                    emotion = int(parts[2])
                    emotion_name = emotion_names.get(emotion, 'unknown')
                    sample_files.append((os.path.join(root, file), emotion_name, file))
    
    if sample_files:
        st.write("Select a demo file to analyze with the neural network:")
        
        selected_file = st.selectbox(
            "Choose a sample file:",
            sample_files,
            format_func=lambda x: f"{x[1].title()} - {x[2]}"
        )
        
        if st.button("🧠 Analyze with Neural Network", type="primary"):
            file_path, emotion_name, filename = selected_file
            
            try:
                # Load audio
                audio_data, sr = librosa.load(file_path, sr=22050, duration=3.0)
                
                # Display info
                st.write(f"**File:** {filename}")
                st.write(f"**Original Emotion:** {emotion_name.title()}")
                
                # Audio player
                st.audio(file_path)
                
                # Analyze
                with st.spinner("🧠 Neural network processing..."):
                    features = analyzer.extract_simple_features(audio_data, sr)
                    
                    if features is not None:
                        prediction, probabilities, confidence = analyzer.predict_stress_deep(features)
                        
                        if prediction is not None:
                            display_deep_results(prediction, probabilities, confidence)
                            
                            # Compare with expected
                            emotion_to_stress = {
                                'neutral': 0, 'calm': 0, 'happy': 0,
                                'sad': 1, 'angry': 2, 'fearful': 2,
                                'disgust': 2, 'surprised': 1
                            }
                            expected_stress = emotion_to_stress.get(emotion_name, 0)
                            stress_labels = ['Low Stress', 'Medium Stress', 'High Stress']
                            
                            st.write("**Neural Network vs Expected:**")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Expected", stress_labels[expected_stress])
                            with col2:
                                st.metric("Neural Network", stress_labels[prediction])
                            with col3:
                                match = "✅ Match" if prediction == expected_stress else "❌ Different"
                                st.metric("Result", match)
                            
                            if prediction == expected_stress:
                                st.success("🎉 Perfect! Neural network correctly identified the stress level!")
                            else:
                                st.info("🤔 Different prediction. Neural networks can sometimes detect patterns humans might miss.")
            
            except Exception as e:
                st.error(f"Error analyzing demo file: {e}")

def display_deep_results(prediction, probabilities, confidence):
    """Display deep learning analysis results"""
    stress_labels = ['Low Stress', 'Medium Stress', 'High Stress']
    colors = ['#28a745', '#ffc107', '#dc3545']  # Green, Yellow, Red
    emojis = ['😌', '😐', '😰']
    
    st.header("🎯 Neural Network Analysis Results")
    
    # Main prediction
    predicted_label = stress_labels[prediction]
    predicted_color = colors[prediction]
    predicted_emoji = emojis[prediction]
    
    st.markdown(f"""
    <div style="padding: 25px; background: linear-gradient(135deg, {predicted_color}20, {predicted_color}10); 
                border-left: 6px solid {predicted_color}; border-radius: 10px; margin: 20px 0;">
        <h2 style="color: {predicted_color}; margin: 0; display: flex; align-items: center;">
            <span style="font-size: 2em; margin-right: 15px;">{predicted_emoji}</span>
            Neural Network Prediction: {predicted_label}
        </h2>
        <p style="font-size: 1.1em; margin: 10px 0 0 0; color: #666;">
            Deep Learning Confidence: <strong>{confidence:.1%}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed probabilities
    st.subheader("📊 Detailed Neural Network Output")
    
    for i, (label, prob) in enumerate(zip(stress_labels, probabilities)):
        col1, col2, col3 = st.columns([2, 3, 1])
        
        with col1:
            st.write(f"**{emojis[i]} {label}**")
        with col2:
            st.progress(float(prob))
        with col3:
            st.write(f"**{prob:.1%}**")
    
    # Neural network interpretation
    st.subheader("🧠 Neural Network Analysis")
    
    if confidence > 0.8:
        confidence_text = "Very High Confidence"
        confidence_color = "#28a745"
        conf_desc = "The neural network is very certain about this prediction."
    elif confidence > 0.6:
        confidence_text = "High Confidence"
        confidence_color = "#17a2b8"
        conf_desc = "The neural network has strong confidence in this prediction."
    elif confidence > 0.4:
        confidence_text = "Moderate Confidence"
        confidence_color = "#ffc107"
        conf_desc = "The neural network has moderate confidence."
    else:
        confidence_text = "Lower Confidence"
        confidence_color = "#dc3545"
        conf_desc = "The neural network shows some uncertainty."
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="padding: 15px; background-color: {confidence_color}20; 
                    border: 2px solid {confidence_color}; border-radius: 10px;">
            <h4 style="color: {confidence_color}; margin: 0;">{confidence_text}</h4>
            <p style="margin: 5px 0 0 0;">{conf_desc}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        interpretation = {
            0: "**Relaxed State:** Neural network detected calm voice patterns with minimal stress indicators.",
            1: "**Moderate Tension:** Neural network identified some stress patterns in voice characteristics.",
            2: "**High Stress:** Neural network detected strong stress indicators in voice patterns and emotional markers."
        }
        st.markdown(interpretation[prediction])

def show_feature_details(features, feature_names):
    """Show feature analysis"""
    st.subheader("🔧 Neural Network Features")
    st.markdown("*The same 35 features as Random Forest, but processed by neural network*")
    
    # Create feature DataFrame
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': features
    })
    
    # Group features
    feature_groups = {
        'MFCC Features': feature_df[feature_df['Feature'].str.contains('mfcc')],
        'Spectral Features': feature_df[feature_df['Feature'].str.contains('spectral')],
        'Other Features': feature_df[~feature_df['Feature'].str.contains('mfcc|spectral')]
    }
    
    tabs = st.tabs(list(feature_groups.keys()))
    
    for tab, (group_name, group_data) in zip(tabs, feature_groups.items()):
        with tab:
            if not group_data.empty:
                # Show chart
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(range(len(group_data)), group_data['Value'])
                ax.set_xticks(range(len(group_data)))
                ax.set_xticklabels(group_data['Feature'], rotation=45, ha='right')
                ax.set_title(f'{group_name} - Neural Network Input')
                ax.set_ylabel('Feature Value')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show data table
                st.dataframe(group_data, use_container_width=True)

if __name__ == "__main__":
    main()
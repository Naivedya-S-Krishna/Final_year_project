import os
import librosa

print("Testing Audio Stress Analysis Setup...")
print("=" * 40)

# Test if dataset is properly loaded
data_path = "./data/RAVDESS/"

if os.path.exists(data_path):
    print("✓ RAVDESS folder found!")
    
    # Count total audio files
    audio_count = 0
    actors_found = []
    
    for root, dirs, files in os.walk(data_path):
        if "Actor_" in root:
            actor_name = os.path.basename(root)
            actors_found.append(actor_name)
        
        for file in files:
            if file.endswith('.wav'):
                audio_count += 1
    
    print(f"✓ Found {len(actors_found)} actors: {sorted(actors_found)[:5]}...")
    print(f"✓ Found {audio_count} total audio files")
    
    if audio_count > 0:
        # Test loading one audio file
        test_file = None
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.wav'):
                    test_file = os.path.join(root, file)
                    break
            if test_file:
                break
        
        if test_file:
            try:
                print(f"\nTesting audio loading...")
                y, sr = librosa.load(test_file, duration=1.0)  # Load just 1 second for test
                print(f"✓ Successfully loaded: {os.path.basename(test_file)}")
                print(f"  Sample rate: {sr} Hz")
                print(f"  Audio length: {len(y)} samples ({len(y)/sr:.2f} seconds)")
                print(f"  Audio range: {y.min():.3f} to {y.max():.3f}")
                
                # Test MFCC extraction
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                print(f"✓ MFCC extraction successful: {mfccs.shape}")
                
                print("\n🎉 Everything is working perfectly!")
                print("You're ready to start building your stress analyzer!")
                
            except Exception as e:
                print(f"✗ Error loading audio file: {e}")
                print("Try: pip install soundfile")
    else:
        print("✗ No audio files found! Check your dataset extraction.")
        
else:
    print("✗ RAVDESS folder not found!")
    print("Make sure you've extracted the dataset to:")
    print("  ./data/RAVDESS/Actor_01/")
    print("  ./data/RAVDESS/Actor_02/")
    print("  etc.")

print("\n" + "=" * 40)
print("Next step: Run main_analysis.py")
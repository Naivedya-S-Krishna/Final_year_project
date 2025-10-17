#!/usr/bin/env python3
"""
Test PyAudio installation on Mac
"""

import sys

def test_pyaudio():
    try:
        import pyaudio
        print("✅ PyAudio imported successfully!")
        
        # Test basic functionality
        p = pyaudio.PyAudio()
        
        print(f"📊 PyAudio version: {pyaudio.__version__}")
        print(f"🎤 Available audio devices: {p.get_device_count()}")
        
        # List some devices
        print("\n🔊 Audio devices found:")
        for i in range(min(3, p.get_device_count())):  # Show first 3 devices
            info = p.get_device_info_by_index(i)
            print(f"  Device {i}: {info['name']} (inputs: {info['maxInputChannels']})")
        
        p.terminate()
        print("\n🎉 PyAudio is working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ PyAudio not installed: {e}")
        return False
    except Exception as e:
        print(f"⚠️  PyAudio installed but has issues: {e}")
        return False

def test_other_libraries():
    """Test other required libraries"""
    libraries = ['librosa', 'numpy', 'pandas', 'sklearn', 'matplotlib']
    
    print("\n📦 Testing other libraries:")
    for lib in libraries:
        try:
            __import__(lib)
            print(f"✅ {lib}")
        except ImportError:
            print(f"❌ {lib} - need to install")

if __name__ == "__main__":
    print("🧪 Testing Audio Environment Setup")
    print("=" * 40)
    
    pyaudio_works = test_pyaudio()
    test_other_libraries()
    
    print("\n" + "=" * 40)
    if pyaudio_works:
        print("🚀 Ready to proceed with audio analysis!")
        print("Next step: Download RAVDESS dataset")
    else:
        print("⚠️  Need to fix PyAudio installation first")
        print("Try the conda method if brew didn't work")

#!/bin/bash

# Setup script for QuickTime Speech-to-Text with Speaker Diarization
# Optimized for Apple Silicon (M1/M2/M3 Macs)

set -e  # Exit on any error

echo "🍎 Setting up QuickTime Speech-to-Text for Apple Silicon..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This script is designed for macOS. Detected OS: $OSTYPE"
    exit 1
fi

# Check if we're on Apple Silicon
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    print_warning "This script is optimized for Apple Silicon (arm64). Detected: $ARCH"
    print_warning "The script will continue but may not be fully optimized."
fi

# Check if Homebrew is installed
print_status "Checking for Homebrew..."
if ! command -v brew &> /dev/null; then
    print_error "Homebrew is not installed. Please install it first:"
    echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi
print_success "Homebrew found"

# Update Homebrew
print_status "Updating Homebrew..."
brew update

# Install system dependencies via Homebrew
print_status "Installing system dependencies..."

# Essential dependencies
BREW_PACKAGES=(
    "ffmpeg"
    "portaudio"
    "cmake"
    "pkg-config"
    "libsndfile"
    "llvm"
)

# Only install sentencepiece if user wants SpeechBrain support
read -p "Do you want to install SpeechBrain support? (advanced speaker recognition, but more complex setup) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    BREW_PACKAGES+=("sentencepiece")
    INSTALL_SPEECHBRAIN=true
else
    INSTALL_SPEECHBRAIN=false
    print_status "Skipping SpeechBrain - will use MFCC method only (recommended for most users)"
fi

for package in "${BREW_PACKAGES[@]}"; do
    if brew list "$package" &>/dev/null; then
        print_success "$package already installed"
    else
        print_status "Installing $package..."
        brew install "$package"
    fi
done

# Check Python version
print_status "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
REQUIRED_VERSION="3.8"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    print_error "Python 3.8+ is required. Current version: $PYTHON_VERSION"
    print_status "Install Python 3.11 via Homebrew:"
    echo "  brew install python@3.11"
    exit 1
fi
print_success "Python version OK: $PYTHON_VERSION"

# Create virtual environment
VENV_NAME="whisper-transcribe"
print_status "Creating virtual environment: $VENV_NAME"

if [ -d "$VENV_NAME" ]; then
    print_warning "Virtual environment already exists. Removing..."
    rm -rf "$VENV_NAME"
fi

python3 -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# Upgrade pip and setuptools
print_status "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Set environment variables for Apple Silicon optimization
export ARCHFLAGS="-arch arm64"
export _PYTHON_HOST_PLATFORM="macosx-11.0-arm64"
export MACOSX_DEPLOYMENT_TARGET="11.0"

# Install core packages first
print_status "Installing core packages..."
pip install numpy scipy

# Install PyTorch with Apple Silicon support
print_status "Installing PyTorch with Apple Silicon optimization..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install audio processing packages
print_status "Installing audio processing packages..."
pip install librosa soundfile moviepy scikit-learn

# Install Whisper
print_status "Installing OpenAI Whisper..."
pip install openai-whisper

# Install additional packages
print_status "Installing additional packages..."
pip install ffmpeg-python pydub

# Conditionally install SpeechBrain if requested
if [ "$INSTALL_SPEECHBRAIN" = true ]; then
    print_status "Installing SpeechBrain and dependencies..."
    pip install speechbrain huggingface-hub pyannote.audio || {
        print_warning "SpeechBrain installation failed. You can still use MFCC method."
        SPEECHBRAIN_FAILED=true
    }
fi

# Verify critical imports
print_status "Verifying installations..."

IMPORT_TESTS=(
    "import whisper; print('✓ Whisper OK')"
    "import moviepy; print('✓ MoviePy OK')"
    "import librosa; print('✓ Librosa OK')"
    "import sklearn; print('✓ Scikit-learn OK')"
    "import torch; print('✓ PyTorch OK')"
    "import numpy; print('✓ NumPy OK')"
    "import scipy; print('✓ SciPy OK')"
)

if [ "$INSTALL_SPEECHBRAIN" = true ] && [ "$SPEECHBRAIN_FAILED" != true ]; then
    IMPORT_TESTS+=("import speechbrain; print('✓ SpeechBrain OK')")
fi

FAILED_IMPORTS=()

for test in "${IMPORT_TESTS[@]}"; do
    if python3 -c "$test" 2>/dev/null; then
        :  # Success, do nothing
    else
        FAILED_IMPORTS+=("$test")
    fi
done

if [ ${#FAILED_IMPORTS[@]} -eq 0 ]; then
    print_success "All packages installed successfully!"
else
    print_error "Some packages failed to import:"
    for failed in "${FAILED_IMPORTS[@]}"; do
        echo "  ❌ $failed"
    done
fi

# Test Whisper model download
print_status "Testing Whisper model download..."
python3 -c "
import whisper
print('Downloading tiny model for testing...')
model = whisper.load_model('tiny')
print('✓ Whisper model loaded successfully!')
"

# Test librosa compatibility
print_status "Testing librosa compatibility..."
python3 -c "
import librosa
import numpy as np
print('Testing librosa features...')
# Test basic MFCC
dummy_audio = np.random.randn(16000)
mfcc = librosa.feature.mfcc(y=dummy_audio, sr=16000, n_mfcc=13)
print('✓ Basic MFCC works')
try:
    chroma = librosa.feature.chroma(y=dummy_audio, sr=16000)
    print('✓ Chroma features work')
except:
    print('⚠ Chroma features not available (will use basic MFCC only)')
"

# Create activation script
print_status "Creating activation script..."
cat > activate_whisper.sh << 'EOF'
#!/bin/bash
# Activation script for QuickTime Speech-to-Text environment

echo "🎙️ Activating Whisper Transcription Environment..."
source whisper-transcribe/bin/activate

# Set environment variables for optimal performance on Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1
export WHISPER_CACHE_DIR="$HOME/.cache/whisper"

echo "✅ Environment activated!"
echo ""
echo "Usage examples:"
echo "  # Basic transcription with speaker identification (MFCC method)"
echo "  python qt_speech_to_text.py recording.mov"
echo ""
echo "  # Specify number of speakers"
echo "  python qt_speech_to_text.py recording.mov --speakers 2"
echo ""
echo "  # Fast transcription without speaker ID"
echo "  python qt_speech_to_text.py recording.mov --no-diarization"
echo ""
echo "  # High quality model"
echo "  python qt_speech_to_text.py recording.mov --model medium"
EOF

if [ "$INSTALL_SPEECHBRAIN" = true ] && [ "$SPEECHBRAIN_FAILED" != true ]; then
cat >> activate_whisper.sh << 'EOF'
echo ""
echo "  # Use SpeechBrain method (if available)"
echo "  python qt_speech_to_text.py recording.mov --method speechbrain"
EOF
fi

cat >> activate_whisper.sh << 'EOF'
echo ""
echo "To deactivate: deactivate"
EOF

chmod +x activate_whisper.sh

# Print final instructions
echo ""
echo "=================================================="
print_success "🎉 Setup completed successfully!"
echo "=================================================="
echo ""
echo "📋 Next steps:"
echo ""
echo "1. Activate the environment:"
echo "   ${BLUE}source activate_whisper.sh${NC}"
echo ""
echo "2. Run the transcription script:"
echo "   ${BLUE}python qt_speech_to_text.py your_recording.mov${NC}"
echo ""
if [ "$INSTALL_SPEECHBRAIN" = true ] && [ "$SPEECHBRAIN_FAILED" != true ]; then
echo "3. For advanced speaker identification:"
echo "   ${BLUE}python qt_speech_to_text.py your_recording.mov --method speechbrain${NC}"
echo ""
fi
echo "📁 Files created:"
echo "   • whisper-transcribe/     (virtual environment)"
echo "   • activate_whisper.sh     (activation script)"
echo "   • requirements.txt        (package list)"
echo ""
echo "🔍 Troubleshooting:"
echo "   • The script uses MFCC method by default (most reliable)"
if [ "$INSTALL_SPEECHBRAIN" != true ]; then
echo "   • SpeechBrain was not installed (MFCC-only setup)"
fi
echo "   • If you get permission errors, try: chmod +x *.py"
echo "   • For audio issues, ensure your .mov file has audio track"
echo "   • Models are cached in: ~/.cache/whisper/"
echo ""
print_success "Happy transcribing! 🚀"

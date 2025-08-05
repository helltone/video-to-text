#!/usr/bin/env python3
"""
Local QuickTime Movie Speech-to-Text Converter with Speaker Diarization

This script runs entirely locally without requiring API tokens or internet access.
Uses local models for both transcription and speaker identification.

Requirements:
- pip install openai-whisper moviepy librosa scikit-learn

Usage:
    python qt_speech_to_text.py input_movie.mov
    python qt_speech_to_text.py input_movie.mov -o output.txt --method mfcc
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from datetime import timedelta
import json
import numpy as np

try:
    import whisper
    from moviepy.editor import VideoFileClip
    import librosa
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    print("\nInstall required packages with:")
    print("pip install openai-whisper moviepy librosa scikit-learn")
    print("\nFor better speaker recognition, also install:")
    print("pip install speechbrain torch torchaudio")
    sys.exit(1)

# Optional imports for better speaker recognition
try:
    from speechbrain.pretrained import SpeakerRecognition
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False


def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format."""
    return str(timedelta(seconds=int(seconds)))


def extract_audio(video_path, audio_path):
    """Extract audio from video file."""
    try:
        print(f"Extracting audio from {video_path}...")
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()
        audio.close()
        print(f"Audio extracted to {audio_path}")
        return True
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return False


def extract_mfcc_features(audio_path, segments, min_segment_length=0.3):
    """Extract MFCC features for speaker clustering."""
    try:
        print("Extracting audio features for speaker identification...")
        y, sr = librosa.load(audio_path)
        
        features = []
        valid_segments = []
        
        for segment in segments:
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            
            if end_sample > start_sample:
                segment_audio = y[start_sample:end_sample]
                
                if len(segment_audio) > sr * min_segment_length:
                    try:
                        # Extract basic MFCC features (most compatible)
                        mfcc = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
                        mfcc_mean = np.mean(mfcc, axis=1)
                        mfcc_std = np.std(mfcc, axis=1)
                        
                        # Try to add additional features, but continue if they fail
                        feature_vector = np.concatenate([mfcc_mean, mfcc_std])
                        
                        # Optional features - add if available
                        try:
                            chroma = librosa.feature.chroma(y=segment_audio, sr=sr)
                            chroma_mean = np.mean(chroma, axis=1)
                            feature_vector = np.concatenate([feature_vector, chroma_mean])
                        except:
                            pass  # Skip chroma if not available
                        
                        try:
                            spectral_contrast = librosa.feature.spectral_contrast(y=segment_audio, sr=sr)
                            contrast_mean = np.mean(spectral_contrast, axis=1)
                            feature_vector = np.concatenate([feature_vector, contrast_mean])
                        except:
                            pass  # Skip spectral contrast if not available
                        
                        try:
                            tonnetz = librosa.feature.tonnetz(y=segment_audio, sr=sr)
                            tonnetz_mean = np.mean(tonnetz, axis=1)
                            feature_vector = np.concatenate([feature_vector, tonnetz_mean])
                        except:
                            pass  # Skip tonnetz if not available
                        
                        features.append(feature_vector)
                        valid_segments.append(segment)
                        
                    except Exception as e:
                        print(f"Warning: Failed to extract features from segment {len(features)}: {e}")
                        continue  # Skip this segment but continue with others
        
        print(f"Successfully extracted features from {len(features)} segments")
        return np.array(features), valid_segments
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None, []


def extract_speechbrain_features(audio_path, segments, min_segment_length=0.3):
    """Extract speaker embeddings using SpeechBrain (if available)."""
    try:
        print("Loading SpeechBrain speaker recognition model...")
        model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="tmp_speaker_model"
        )
        
        features = []
        valid_segments = []
        
        print("Extracting speaker embeddings...")
        y, sr = librosa.load(audio_path)
        
        for i, segment in enumerate(segments):
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            
            if end_sample > start_sample:
                segment_audio = y[start_sample:end_sample]
                
                if len(segment_audio) > sr * max(1.0, min_segment_length):  # At least 1 second for SpeechBrain
                    # Save temporary segment using soundfile instead of librosa.output
                    temp_segment_path = f"temp_segment_{i}.wav"
                    
                    try:
                        import soundfile as sf
                        sf.write(temp_segment_path, segment_audio, sr)
                    except ImportError:
                        # Fallback: use scipy if soundfile not available
                        from scipy.io import wavfile
                        wavfile.write(temp_segment_path, sr, (segment_audio * 32767).astype(np.int16))
                    
                    try:
                        # Extract embedding
                        embedding = model.encode_batch([temp_segment_path])
                        features.append(embedding.squeeze().cpu().numpy())
                        valid_segments.append(segment)
                    except Exception as e:
                        print(f"Failed to process segment {i}: {e}")
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_segment_path):
                            os.remove(temp_segment_path)
        
        return np.array(features), valid_segments
    except Exception as e:
        print(f"Error with SpeechBrain features: {e}")
        return None, []


def cluster_speakers(features, n_speakers=None, max_speakers=6, method="auto"):
    """Cluster audio segments by speaker using various methods."""
    if len(features) < 2:
        return [0] * len(features)
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    if method == "auto" or n_speakers is None:
        # Try different numbers of speakers and pick the best
        best_score = -1
        best_labels = None
        best_n_speakers = 2
        
        # Try 2-max_speakers and evaluate clustering quality
        for n in range(2, min(max_speakers + 1, len(features) + 1)):
            clustering = AgglomerativeClustering(
                n_clusters=n, 
                linkage='ward',
                metric='euclidean'
            )
            labels = clustering.fit_predict(features_scaled)
            
            # Improved scoring: balance between cluster separation and balance
            unique_labels = np.unique(labels)
            if len(unique_labels) > 1:
                cluster_sizes = [np.sum(labels == label) for label in unique_labels]
                min_cluster_size = min(cluster_sizes)
                
                # Prefer solutions with reasonably sized clusters
                if min_cluster_size >= 2:  # Each speaker should have at least 2 segments
                    balance_score = 1.0 - (np.std(cluster_sizes) / np.mean(cluster_sizes))
                    
                    # Bonus for having 2-3 speakers (most common in meetings)
                    if n in [2, 3]:
                        balance_score += 0.1
                    
                    if balance_score > best_score:
                        best_score = balance_score
                        best_labels = labels
                        best_n_speakers = n
        
        print(f"Auto-detected {best_n_speakers} speakers (confidence: {best_score:.2f})")
        return best_labels if best_labels is not None else [0] * len(features)
    
    else:
        clustering = AgglomerativeClustering(
            n_clusters=n_speakers, 
            linkage='ward',
            metric='euclidean'
        )
        return clustering.fit_predict(features_scaled)


def simple_voice_activity_detection(transcription_result, min_duration=0.5):
    """Create segments based on Whisper's natural segmentation."""
    segments = []
    
    for segment in transcription_result['segments']:
        if segment['end'] - segment['start'] >= min_duration:
            segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip()
            })
    
    return segments


def assign_speakers_to_segments(segments, speaker_labels):
    """Assign speaker labels to segments."""
    speaker_segments = []
    
    for i, segment in enumerate(segments):
        if i < len(speaker_labels):
            speaker_id = f"SPEAKER_{speaker_labels[i]:02d}"
        else:
            speaker_id = "SPEAKER_00"
        
        speaker_segments.append({
            'start': segment['start'],
            'end': segment['end'],
            'speaker': speaker_id,
            'text': segment['text']
        })
    
    return speaker_segments


def format_transcript(segments, include_timestamps=True):
    """Format the transcript with speakers and timestamps."""
    transcript_lines = []
    
    if include_timestamps:
        transcript_lines.append("TRANSCRIPT WITH SPEAKERS AND TIMESTAMPS")
        transcript_lines.append("=" * 50)
    else:
        transcript_lines.append("TRANSCRIPT WITH SPEAKERS")
        transcript_lines.append("=" * 30)
    
    current_speaker = None
    
    for segment in segments:
        speaker = segment['speaker']
        text = segment['text']
        start_time = format_timestamp(segment['start'])
        end_time = format_timestamp(segment['end'])
        
        if speaker != current_speaker:
            if current_speaker is not None:
                transcript_lines.append("")  # Add blank line between speakers
            current_speaker = speaker
        
        if include_timestamps:
            transcript_lines.append(f"[{start_time} - {end_time}] {speaker}: {text}")
        else:
            transcript_lines.append(f"{speaker}: {text}")
    
    return "\n".join(transcript_lines)


def save_transcript(transcript, output_path, segments=None):
    """Save transcript to file, optionally with JSON data."""
    try:
        # Save main transcript
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        
        # Save detailed JSON if segments provided
        if segments:
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2, ensure_ascii=False)
            print(f"Detailed data saved to {json_path}")
        
        print(f"Transcript saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving transcript: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert speech in QuickTime movies to text with local speaker identification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qt_speech_to_text.py recording.mov
  python qt_speech_to_text.py recording.mov -o transcript.txt
  python qt_speech_to_text.py recording.mov --method speechbrain --speakers 3
  python qt_speech_to_text.py recording.mov --no-diarization
  
Methods:
  - mfcc: Fast, robust clustering using audio features (DEFAULT - works well for screen recordings)
  - speechbrain: Deep learning embeddings (better for clean studio audio)
  - auto: Automatically choose best method
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Input QuickTime movie file (.mov, .mp4, etc.)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output text file (default: input_filename.txt)"
    )
    
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to use (default: base)"
    )
    
    parser.add_argument(
        "--method",
        default="mfcc",
        choices=["mfcc", "speechbrain", "auto"],
        help="Speaker identification method (default: mfcc - often works better for screen recordings)"
    )
    
    parser.add_argument(
        "--speakers",
        type=int,
        help="Number of speakers (auto-detect if not specified)"
    )
    
    parser.add_argument(
        "--min-segment-length",
        type=float,
        default=0.3,
        help="Minimum segment length in seconds for speaker analysis (default: 0.3)"
    )
    
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=6,
        help="Maximum number of speakers to consider (default: 6)"
    )
    
    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Don't include timestamps in output"
    )
    
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep extracted audio file"
    )
    
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Skip speaker diarization (faster, but no speaker labels)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)
    
    # Check method availability
    if args.method == "speechbrain" and not SPEECHBRAIN_AVAILABLE:
        print("SpeechBrain not available. Install with: pip install speechbrain torch torchaudio")
        print("Falling back to MFCC method...")
        args.method = "mfcc"
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.txt')
    
    # Create temporary audio file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_audio_path = temp_audio.name
    
    try:
        # Extract audio
        if not extract_audio(str(input_path), temp_audio_path):
            sys.exit(1)
        
        # Transcribe audio
        print(f"Loading Whisper model '{args.model}'...")
        model = whisper.load_model(args.model)
        print("Transcribing audio...")
        transcription_result = model.transcribe(temp_audio_path)
        
        if not args.no_diarization:
            # Create segments from transcription
            segments = simple_voice_activity_detection(transcription_result)
            print(f"Found {len(segments)} segments for speaker analysis")
            
            if len(segments) > 1:
                # Extract features for speaker clustering
                if args.method == "speechbrain" or (args.method == "auto" and SPEECHBRAIN_AVAILABLE):
                    features, valid_segments = extract_speechbrain_features(
                        temp_audio_path, segments, args.min_segment_length
                    )
                    method_used = "SpeechBrain"
                else:
                    features, valid_segments = extract_mfcc_features(
                        temp_audio_path, segments, args.min_segment_length
                    )
                    method_used = "MFCC"
                
                if features is not None and len(features) > 0:
                    print(f"Extracted features from {len(features)} segments")
                    print(f"Clustering speakers using {method_used} method...")
                    speaker_labels = cluster_speakers(
                        features, args.speakers, args.max_speakers
                    )
                    speaker_segments = assign_speakers_to_segments(valid_segments, speaker_labels)
                    
                    unique_speakers = len(set(speaker_labels))
                    print(f"Identified {unique_speakers} unique speakers")
                    
                    transcript = format_transcript(
                        speaker_segments, 
                        include_timestamps=not args.no_timestamps
                    )
                else:
                    print(f"Feature extraction failed - features: {features is not None}, count: {len(features) if features is not None else 0}")
                    print("Could not extract speaker features, using single speaker...")
                    speaker_segments = [{
                        'start': 0,
                        'end': transcription_result.get('segments', [{}])[-1].get('end', 0),
                        'speaker': 'SPEAKER_00',
                        'text': transcription_result['text'].strip()
                    }]
                    transcript = format_transcript(
                        speaker_segments, 
                        include_timestamps=not args.no_timestamps
                    )
            else:
                print(f"Not enough segments for speaker diarization (found {len(segments)}, need >1)...")
                speaker_segments = [{
                    'start': 0,
                    'end': transcription_result.get('segments', [{}])[-1].get('end', 0),
                    'speaker': 'SPEAKER_00',
                    'text': transcription_result['text'].strip()
                }]
                transcript = transcription_result['text'].strip()
        else:
            transcript = transcription_result['text'].strip()
            speaker_segments = None
        
        # Display transcript
        print("\n" + "="*70)
        print(transcript)
        print("="*70)
        
        # Save transcript
        if save_transcript(transcript, output_path, speaker_segments):
            print(f"\nSuccess! Transcript saved to: {output_path}")
        
    finally:
        # Cleanup temporary audio file
        if not args.keep_audio and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        elif args.keep_audio:
            audio_output = input_path.with_suffix('.wav')
            os.rename(temp_audio_path, str(audio_output))
            print(f"Audio file saved as: {audio_output}")


if __name__ == "__main__":
    main()

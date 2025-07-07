import os
import librosa
import numpy as np
# import torchaudio  # Removed because it is not used

def load_audio(file_path, sr=16000):
    """Load an audio file as a waveform and resample."""
    waveform, sample_rate = librosa.load(file_path, sr=sr)
    return waveform, sample_rate

def extract_mfcc(waveform, sr, n_mfcc=13):
    """Extract MFCC features from a waveform."""
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def save_features(features, out_path):
    np.save(out_path, features)

def process_directory(input_dir, output_dir, sr=16000, n_mfcc=13):
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, files in os.walk(input_dir):
        rel_dir = os.path.relpath(root, input_dir)
        out_subdir = os.path.join(output_dir, rel_dir) if rel_dir != '.' else output_dir
        os.makedirs(out_subdir, exist_ok=True)
        for fname in files:
            if fname.lower().endswith('.wav'):
                in_path = os.path.join(root, fname)
                out_path = os.path.join(out_subdir, fname.replace('.wav', '.npy'))
                waveform, sample_rate = load_audio(in_path, sr=sr)
                mfcc = extract_mfcc(waveform, sample_rate, n_mfcc=n_mfcc)
                save_features(mfcc, out_path)
                print(f"Processed {in_path} -> {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audio Preprocessing")
    parser.add_argument('--input', type=str, help='Path to input audio file')
    parser.add_argument('--output', type=str, help='Path to save features (.npy)')
    parser.add_argument('--sr', type=int, default=16000, help='Target sample rate')
    parser.add_argument('--input_dir', type=str, help='Path to input directory of audio files')
    parser.add_argument('--output_dir', type=str, help='Path to save features for all files')
    parser.add_argument('--n_mfcc', type=int, default=13, help='Number of MFCCs to extract')
    args = parser.parse_args()

    if args.input_dir and args.output_dir:
        process_directory(args.input_dir, args.output_dir, sr=args.sr, n_mfcc=args.n_mfcc)
    elif args.input and args.output:
        waveform, sr = load_audio(args.input, sr=args.sr)
        mfcc = extract_mfcc(waveform, sr, n_mfcc=args.n_mfcc)
        save_features(mfcc, args.output)
        print(f"Saved MFCC features to {args.output}")
    else:
        parser.error('You must specify either --input and --output, or --input_dir and --output_dir.') 
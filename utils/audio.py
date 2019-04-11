from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import librosa
import numpy as np


DELTA_WIDTH = 9
HOP_LENGTH = 512
MIN_SAMPLE_MS = 100
MSE_FRAME_LENGTH = 2048


def get_db(y):
    # Calculate MSE energy per frame
    mse = librosa.feature.rmse(y=y,
                               frame_length=MSE_FRAME_LENGTH,
                               hop_length=HOP_LENGTH) ** 2

    # Convert power to decibels
    return librosa.power_to_db(mse.squeeze(), ref=-100)


def is_silent(y, threshold_db):
    return np.max(get_db(y)) < threshold_db


def pca(features, components=2):
    """Dimension reduction via Principal Component Analysis (PCA)"""
    pca = PCA(n_components=components)
    transformed = pca.fit(features).transform(features)

    variance = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(transformed)

    return scaler.transform(transformed), pca, scaler, variance


def mfcc_features(y, sr, n_mels=128, n_mfcc=13):
    """Extract MFCCs (Mel-Frequency Cepstral Coefficients)"""
    # Analyze only first second
    y = y[0:sr]

    # Calculate MFCCs (Mel-Frequency Cepstral Coefficients)
    mel_spectrum = librosa.feature.melspectrogram(y,
                                                  sr=sr,
                                                  n_mels=n_mels)
    log_spectrum = librosa.amplitude_to_db(mel_spectrum,
                                           ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_spectrum,
                                sr=sr,
                                n_mfcc=n_mfcc)

    if mfcc.shape[-1] < DELTA_WIDTH:
        raise ValueError('MFCC vector does not contain enough time steps')

    if not mfcc.any():
        return np.zeros(n_mfcc * 3)

    # Standardize feature for equal variance
    delta_mfcc = librosa.feature.delta(mfcc, width=DELTA_WIDTH)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2, width=DELTA_WIDTH)
    feature_vector = np.concatenate((
        np.mean(mfcc, 1),
        np.mean(delta_mfcc, 1),
        np.mean(delta2_mfcc, 1)))
    feature_vector = (
        feature_vector - np.mean(feature_vector)
    ) / np.std(feature_vector)

    return feature_vector


def slice_audio(y, onsets, sr=44100, offset=0, top_db=20):
    frames = []
    min_frames = (sr // 1000) * MIN_SAMPLE_MS
    
    for i in range(len(onsets) - 1):
        # Take audio from onset start to next onset
        onset_start = onsets[i] * HOP_LENGTH
        onset_end = onsets[i + 1] * HOP_LENGTH
        
        # Ignore too short samples
        if onset_end - onset_start < min_frames:
            continue
        
        # Trim silence
        y_trim, trim_indexes = librosa.effects.trim(y[onset_start:onset_end],
                                                    ref=np.mean,
                                                    top_db=top_db)

        if len(y_trim) < min_frames:
            continue
        
        # Set new slice relative to onset position
        start = onset_start + trim_indexes[0]
        end = onset_start + trim_indexes[1]

        if end - start < min_frames:
            continue
        
        frames.append([y[start:end], start + offset, end + offset])
            
    return frames


def detect_onsets(y, sr=441000):
    # Get the frame->beat strength profile
    onset_envelope = librosa.onset.onset_strength(y=y,
                                                  sr=sr,
                                                  hop_length=HOP_LENGTH,
                                                  aggregate=np.median)

    # Locate note onset events
    onsets = librosa.onset.onset_detect(y=y,
                                        sr=sr,
                                        onset_envelope=onset_envelope,
                                        hop_length=HOP_LENGTH,
                                        backtrack=True)

    return onsets
import os
import glob
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler

# --- 1. Cấu hình đường dẫn ---
# Giả sử bạn đã giải nén file zip vào thư mục này
DATA_PATH = "C:\JAVA\khaiphadulieu\BTL_KhaiPhaDuLieu\AudioWAV" 
# DATA_PATH = os.path.join(current_dir, "AudioWAV")

# --- 2. Định nghĩa các bảng mã (Mapping) theo RAVDESS ---
# Modality (01 = full-AV, 02 = video-only, 03 = audio-only)
modality_map = {'01': 'full-AV', '02': 'video-only', '03': 'audio-only'}

# Vocal channel (01 = speech, 02 = song)
vocal_channel_map = {'01': 'speech', '02': 'song'}

# Emotion (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised)
emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# Emotional intensity (01 = normal, 02 = strong)
intensity_map = {'01': 'normal', '02': 'strong'}

# Statement
statement_map = {'01': "Kids are talking by the door", '02': "Dogs are sitting by the door"}

# Repetition
repetition_map = {'01': '1st', '02': '2nd'}

def extract_audio_features(file_path):
    """
    Hàm trích xuất đặc trưng từ file âm thanh
    Trả về: MFCC (mean), Pitch (mean), Energy (mean), ZCR (mean)
    """
    try:
        # Load file âm thanh (bỏ qua khoảng lặng đầu/cuối để tối ưu)
        y, sr = librosa.load(file_path, duration=3, offset=0.5)

        # 1. MFCC (Lấy 13 hệ số, tính trung bình theo trục thời gian)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        # 2. Energy - Năng lượng (Sử dụng RMS - Root Mean Square)
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)

        # 3. Zero Crossing Rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = np.mean(zcr)

        # 4. Pitch (Cao độ) - Sử dụng thuật toán PYIN để ước lượng F0
        # Lưu ý: pyin chạy khá chậm. Nếu muốn nhanh hơn có thể dùng librosa.yin hoặc bỏ qua.
        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        # Lấy trung bình cao độ (bỏ qua các giá trị NaN - không có tiếng)
        pitch_mean = np.nanmean(f0) if np.any(np.isfinite(f0)) else 0

        # Kết hợp các đặc trưng thành một mảng
        # Cấu trúc: [MFCC_1, ..., MFCC_13, Pitch, Energy, ZCR]
        features = np.hstack([mfcc_mean, pitch_mean, rms_mean, zcr_mean])
        return features

    except Exception as e:
        print(f"Loi khi xu ly file {file_path}: {e}")
        return None

# --- 3. Chương trình chính ---
data = []
print("Dang bat dau xu ly")

# Duyệt qua tất cả các file trong thư mục (bao gồm cả thư mục con Actor_01, Actor_02...)
# Cấu trúc RAVDESS thường là: AudioWAV/Actor_01/*.wav
for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            
            # --- Phân tích tên file ---
            # Ví dụ: 03-01-06-01-02-01-12.wav
            parts = file.split('.')[0].split('-')
            
            if len(parts) != 7:
                continue # Bỏ qua nếu tên file không đúng định dạng
            
            modality = modality_map.get(parts[0])
            vocal_channel = vocal_channel_map.get(parts[1])
            emotion = emotion_map.get(parts[2])
            intensity = intensity_map.get(parts[3])
            statement = statement_map.get(parts[4])
            repetition = repetition_map.get(parts[5])
            actor_id = int(parts[6])
            
            # Xác định giới tính: Lẻ = Nam, Chẵn = Nữ
            gender = 'Male' if actor_id % 2 != 0 else 'Female'

            # --- Trích xuất đặc trưng ---
            features = extract_audio_features(file_path)
            
            if features is not None:
                # Tạo dictionary cho dòng dữ liệu này
                row = {
                    'Modality': modality,
                    'Vocal_Channel': vocal_channel,
                    'Emotion': emotion,
                    'Intensity': intensity,
                    'Statement': statement,
                    'Repetition': repetition,
                    'Actor_ID': actor_id,
                    'Gender': gender,
                    # Đặc trưng số học
                    'Pitch': features[13], # Phần tử thứ 14 (sau 13 MFCC)
                    'Energy': features[14],
                    'ZCR': features[15]
                }
                # Thêm 13 cột MFCC
                for i in range(13):
                    row[f'MFCC_{i+1}'] = features[i]
                
                data.append(row)

# Tạo DataFrame
df = pd.DataFrame(data)

print(f"Da trich xuat {len(df)} file am thanh.")

# --- 4. Chuẩn hóa dữ liệu (StandardScaler) ---
if not df.empty:
    # Chỉ chuẩn hóa các cột đặc trưng số (MFCC, Pitch, Energy, ZCR)
    feature_cols = [col for col in df.columns if col.startswith('MFCC') or col in ['Pitch', 'Energy', 'ZCR']]
    
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # --- 5. Lưu file CSV ---
    output_file = 'RAVDESS_features.csv'
    df.to_csv(output_file, index=False)
    print(f"Du lieu da duoc chuan hoa va luu vao '{output_file}'")
    
    # Hiển thị 5 dòng đầu
    print(df.head())
else:
    print("Khong thay du lieu hoac co loi xay ra")
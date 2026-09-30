# In the name of Allah, the most compassionate, the most merciful

import os
import whisper
from whisper.normalizers import EnglishTextNormalizer
from jiwer import wer, cer
import torch
import pandas as pd
import librosa

# --- Paths ---
ENHANCED_AUDIO_PATH = r"path...\generated_files\g_best"
TRANSCRIPTION_PATH = r"path...\VoiceBank+DEMAND\testset_txt\testset_txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Whisper base on {DEVICE}...")
model = whisper.load_model("base", device=DEVICE)
normalizer = EnglishTextNormalizer()

def run_evaluation():
    audio_files = sorted([f for f in os.listdir(ENHANCED_AUDIO_PATH) if f.endswith(".wav")])
    detailed_results = []
    
    print(f"Processing {len(audio_files)} files (Calculating WER & CER)...")

    for filename in audio_files:
        txt_filename = filename.lower().replace(".wav", ".txt")
        txt_path = os.path.join(TRANSCRIPTION_PATH, txt_filename)
        
        if not os.path.exists(txt_path):
            continue
            
        with open(txt_path, "r", encoding="utf-8") as f:
            raw_ref = f.read().strip()
            reference = normalizer(raw_ref)

        # ۱. Loading audio
        audio_full_path = os.path.join(ENHANCED_AUDIO_PATH, filename)
        audio_array, _ = librosa.load(audio_full_path, sr=16000)

        # ۲. Convert to text
        result = model.transcribe(audio_array, language="en", temperature=0)
        hypothesis = normalizer(result["text"])

        # ۳. Calculating WER and CER

        if len(reference) > 0:

            # current_wer = wer(reference, hypothesis)
            current_cer = cer(reference, hypothesis)
            
            detailed_results.append({
                "File_Name": filename,
                "Reference": reference,
                "Hypothesis": hypothesis,
                # "WER": current_wer,
                "CER": current_cer
            })
        
        if len(detailed_results) % 50 == 0:
            # avg_c = (sum(d['WER'] for d in detailed_results) / len(detailed_results)) * 100
            avg_c = (sum(d['CER'] for d in detailed_results) / len(detailed_results)) * 100
            # print(f"✅ {len(detailed_results)} files processed... Current Avg WER: {avg_c:.2f}%")
            print(f"✅ {len(detailed_results)} files processed... Current Avg CER: {avg_c:.2f}%")

    # ۴. Save and print final results
    df = pd.DataFrame(detailed_results)
    df.to_csv("ASR_CER_VB_EN_Model12_Base.csv", index=False, encoding="utf-8-sig")
    
    # final_avg_wer = df["WER"].mean() * 100
    final_avg_cer = df["CER"].mean() * 100
    
    print("\n" + "="*40)
    print(f"📊 EVALUATION COMPLETE")
    # print(f"Final Average WER: {final_avg_wer:.2f}%")
    print(f"Final Average CER: {final_avg_cer:.2f}%")
    print(f"Results saved to: ASR_CER_VB_EN_Model12_Base.csv")
    print("="*40)

if __name__ == "__main__":
    run_evaluation()

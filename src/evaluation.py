# In the name of Allah, the most compassionate, the most merciful

# Evaluation

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import argparse
import librosa
import numpy as np
from tools.compute_metrics import compute_metrics
from rich.progress import track
import json
from env import AttrDict

def get_dataset_filelist(input_test_file):
    with open(input_test_file, 'r', encoding='utf-8') as fi:
        indexes = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]
    return indexes

def main():
    print('Initializing Evaluation Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_clean_wavs_dir', default='/content/VoiceBank+DEMAND/wavs_clean')
    parser.add_argument('--input_enhanced_wavs_dir', default='generated_files/g_best')
    parser.add_argument('--input_test_file', default='/content/VoiceBank+DEMAND/test.txt')
    parser.add_argument('--config_file', default='')
    
    a = parser.parse_args()

    with open(a.config_file) as f:
        data = f.read()
    
    json_config = json.loads(data)
    h = AttrDict(json_config)

    indexes = get_dataset_filelist(a.input_test_file)
    num = len(indexes)
    print(f"Found {num} files to evaluate")
    
    metrics_total = np.zeros(6)
    for index in track(indexes):
        clean_wav = os.path.join(a.input_clean_wavs_dir, index + '.wav')
        enhanced_wav = os.path.join(a.input_enhanced_wavs_dir, index + '.wav')
        
        clean, sr = librosa.load(clean_wav, sr=h.sampling_rate)
        enhanced, sr = librosa.load(enhanced_wav, sr=h.sampling_rate)

        metrics = compute_metrics(clean, enhanced, sr, 0)
        metrics = np.array(metrics)
        metrics_total += metrics

    metrics_avg = metrics_total / num
    print('\nEvaluation Results:')
    print('##################################################################\n')
    print(f"PESQ: {metrics_avg[0]:.4f}\n"
          f"CSIG: {metrics_avg[1]:.4f}\n"
          f"CBAK: {metrics_avg[2]:.4f}\n"
          f"COVL: {metrics_avg[3]:.4f}\n"
          f"SSNR: {metrics_avg[4]:.4f}\n"
          f"STOI: {metrics_avg[5]:.4f}")

    # Save results to file
    output_file = os.path.join(a.input_enhanced_wavs_dir, 'evaluation_results.txt')
    with open(output_file, 'w') as f:
        f.write('Evaluation Results:\n')
        f.write('##################################################################\n\n')
        f.write(f"PESQ: {metrics_avg[0]:.4f}\n")
        f.write(f"CSIG: {metrics_avg[1]:.4f}\n")
        f.write(f"CBAK: {metrics_avg[2]:.4f}\n")
        f.write(f"COVL: {metrics_avg[3]:.4f}\n")
        f.write(f"SSNR: {metrics_avg[4]:.4f}\n")
        f.write(f"STOI: {metrics_avg[5]:.4f}\n")

if __name__ == '__main__':
    main()

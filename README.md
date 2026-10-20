# LUDAT-Net: A Layer-wise U-Shaped Architecture Using Depthwise-Augmented Transformers for Single-Channel Speech Enhancement

## Online Demo

An interactive demo of LUDAT-Net is available on GitHub Pages, presenting an ablation study on the effect of LUE hierarchy depth.

**[Visit the LUDAT-Net Demo](https://alkata-ai.github.io/LUDAT-Net/)**


**Abstract:** Speech enhancement is an important task in speech processing that aims to reduce noise and improve speech quality and intelligibility. This task plays an important role in improving the performance of communication systems, hearing-assistive devices, and automatic speech recognition (ASR) systems in noisy environments. One of the fundamental challenges in speech enhancement is achieving high perceptual quality and intelligibility simultaneously, as noise removal, particularly at signal-to-noise ratios (SNRs) ranging from 0 to 15 dB, may distort the speech structure and consequently reduce intelligibility. Therefore, in this research, a novel architecture, namely LUDAT-Net, is proposed for single-channel speech enhancement, with the aim of achieving a balance between noise suppression and preservation of speech structure. The proposed method is based on the simultaneous estimation and reconstruction of magnitude and phase components in the time–frequency domain. The LUDAT-Net architecture is an encoder–decoder structure consisting of a Layer-wise U-shaped Enhancement (LUE) block based on a Time–Frequency Depthwise-Augmented Transformer (TF-DAT), a channel-attention-based encoder and decoders, and targeted skip connections. By exploiting hierarchical processing, modeling global dependencies, and extracting local details, this structure enables effective separation of noise patterns from speech structure and more accurate signal reconstruction.
The model was evaluated using the VoiceBank+DEMAND dataset in terms of perceptual quality, intelligibility, convergence, and computational efficiency, as well as through ablation studies. The results indicate that the proposed model achieved a PESQ score of 3.47, representing a 16.84% increase over the PESQ score recorded at the beginning of training (2.97). Furthermore, evaluation using ASR systems over the SNR range of 0–15 dB showed that, compared with the noisy input signals, the proposed model reduced the word error rate (WER) and character error rate (CER) by an average of 0.87 and 0.46 percentage points, respectively. Comparisons with a set of advanced methods showed that, among signal reconstruction-based methods and methods employing augmented transformers, the proposed model achieved the best results in PESQ, CSIG, CBAK, and COVL, while attaining the highest reported STOI score of 96.1%. In terms of computational efficiency, LUDAT-Net, with 1.60 million parameters, 15.534 GFLOPs, and an RTF of 0.037, provides a relatively compact and fast architecture. Overall, the results indicate that LUDAT-Net effectively suppresses noise while preventing distortion of speech structure, thereby enabling simultaneous improvements in perceptual quality and intelligibility under the conditions considered.

## Note
Pre-trained model weights are not provided. The model can be trained from scratch using the provided source code.

## Pre-requisites

**<p>1. Python >= 3.6.</p>**
   
**<p>2. Clone this repository:</p>**

```bash
git clone https://github.com/ALKATA-AI/LUDAT-Net.git
```

**<p>3. Install PyTorch with CUDA 11.8:</p>**

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
```

**<p>4. Install the Python requirements:</p>**
   
Please refer to [requirements.txt](https://github.com/ALKATA-AI/LUDAT-Net/blob/main/requirements.txt).

```bash
pip install -r requirements.txt
```

**<p>5. Download and extract the <a href="https://datashare.ed.ac.uk/items/e3eb2725-46b6-40c6-88cf-3ccad62cd419">VoiceBank+DEMAND dataset</a>:</p>**

Resample all wav files to 16 kHz, and move the clean and noisy WAV files to <code>VoiceBank+DEMAND/wavs_clean</code> and <code>VoiceBank+DEMAND/wavs_noisy</code>, respectively. Alternatively, you can directly download the downsampled 16 kHz dataset <a href="https://drive.google.com/drive/folders/19I_thf6F396y5gZxLTxYIojZXC0Ywm8l">here</a>.

## Training

For single GPU (Recommend), LUDAT-Net needs at least 13 GB GPU memery.

Training parameters can be configured directly in `train.py` or provided as command-line arguments when running the script.

```bash
python train.py `
 --input_clean_wavs_dir "VoiceBank+DEMAND/wavs_clean" `
 --input_noisy_wavs_dir "VoiceBank+DEMAND/wavs_noisy" `
 --checkpoint_path "checkpoints" `
 --config "config.json" `
 --input_training_file "VoiceBank+DEMAND/training.txt" `
 --input_validation_file "VoiceBank+DEMAND/test.txt"
```

## Inference

Inference parameters can be specified either directly in `inference.py` or provided as command-line arguments.

```bash
python inference.py `
 --checkpoint_file=checkpoints\g_best `
 --input_clean_wavs_dir VoiceBank+DEMAND/wavs_clean `
 --input_noisy_wavs_dir VoiceBank+DEMAND/wavs_noisy `
 --input_test_file VoiceBank+DEMAND/test.txt `
 --output_dir generated_files/g_best
```

## Inference Speed

Install the required packages:

```bash
pip install thop
pip install attrdict
```

Then, run the following command:

```bash
python inference_speed.py --checkpoint_file "checkpoints/g_best"
```

## Evaluation

Evaluation parameters can be specified either directly in `evaluation.py` or provided as command-line arguments.

```bash
python evaluation.py `
 --config_file config.json `
 --input_clean_wavs_dir VoiceBank+DEMAND/wavs_clean `
 --input_test_file VoiceBank+DEMAND/test.txt `
 --input_enhanced_wavs_dir generated_files/g_best
```

## WER and CER Evaluation

**1. Install the Required Libraries**

Install [Whisper](https://github.com/openai/whisper) and the required audio processing libraries:

```bash
pip install openai-whisper librosa
```

Install [jiwer](https://jitsi.github.io/jiwer/) for calculating WER and CER:

```bash
pip install jiwer
```

**2. Test CUDA and Load the Model**

Run the following command to verify CUDA availability and load the Whisper model:

```bash
python test_cuda_and_load_model.py
```

**3. Calculate WER and CER**

```bash
python calculate_wer_cer.py
```

## Model Structure

## Comparison with other SE models

## Acknowledgements 
We referred to [GSE-Net](https://github.com/ALKATA-AI/GSE-Net), [IDTransformer](https://github.com/szw811/IDTransformer), [PrimeK-Net](https://github.com/huaidanquede/PrimeK-Net), [CBAM](https://github.com/Peachypie98/CBAM) and [CMGAN](https://github.com/ruizhecao96/CMGAN) to implement this.

## Citation

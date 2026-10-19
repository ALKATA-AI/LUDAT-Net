# LUDAT-Net: A Layer-wise U-Shaped Architecture Using Depthwise-Augmented Transformers for Single-Channel Speech Enhancement

**Abstract:**

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

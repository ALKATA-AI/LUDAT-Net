# LUDAT-Net
### Alireza KayedNezami, Laleh Tariveh, Rahil Mahdian Toroghi, Hassan Zareian
**Official implementation of LUDAT-Net, accepted at ... 2027.**

**Abstract:**

## Note
Pre-trained model weights are not provided. The model can be trained from scratch using the provided source code.

## Pre-requisites

<p>1. Python >= 3.6.</p>
   
<p>2. Clone this repository:</p>

```bash
git clone https://github.com/ALKATA-AI/LUDAT-Net.git
```

<p>3. Install PyTorch with CUDA 11.8:</p>

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
```

<p>4. Install the Python requirements:</p>
   
Please refer to [requirements.txt](https://github.com/ALKATA-AI/LUDAT-Net/blob/main/requirements.txt).

```bash
pip install -r requirements.txt
```

<p>5. Download and extract the <a href="https://datashare.ed.ac.uk/items/e3eb2725-46b6-40c6-88cf-3ccad62cd419">VoiceBank+DEMAND dataset</a>:</p>

Resample all wav files to 16 kHz, and move the clean and noisy WAV files to <code>VoiceBank+DEMAND/wavs_clean</code> and <code>VoiceBank+DEMAND/wavs_noisy</code>, respectively. Alternatively, you can directly download the downsampled 16 kHz dataset <a href="https://drive.google.com/drive/folders/19I_thf6F396y5gZxLTxYIojZXC0Ywm8l">here</a>.

## Training

## Inference

## Model Structure

## Comparison with other SE models
<p align="center">
  <img src="Figures/Table2 - GSE-Net.png" alt="LUDAT-Net Architecture">
</p>

## Acknowledgements 
We referred to [GSE-Net](https://github.com/ALKATA-AI/GSE-Net), [IDTransformer](https://github.com/szw811/IDTransformer), [PrimeK-Net](https://github.com/huaidanquede/PrimeK-Net), [CBAM](https://github.com/Peachypie98/CBAM) and [CMGAN](https://github.com/ruizhecao96/CMGAN) to implement this.

## Citation

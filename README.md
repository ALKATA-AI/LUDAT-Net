# LUDAT-Net
### Alireza KayedNezami, Laleh Tariveh, Rahil Mahdian Toroghi, Hassan Zareian
**Official implementation of LUDAT-Net, accepted at ... 2027.**

**Abstract:**

## Note
Pre-trained model weights are not provided. The model can be trained from scratch using the provided source code.

## Pre-requisites
1. Python >= 3.6.
2. Clone this repository.
```bash
git clone 
```
3. Install python requirements. Please refer [requirements.txt](https://datashare.ed.ac.uk/items/e3eb2725-46b6-40c6-88cf-3ccad62cd419).
```bash
pip install -r requirements.txt 
```
4. Download and extract the [VoiceBank+DEMAND dataset](https://datashare.ed.ac.uk/items/e3eb2725-46b6-40c6-88cf-3ccad62cd419). Resample all wav files to 16kHz, and move the clean and noisy wavs to VoiceBank+DEMAND/wavs_clean and VoiceBank+DEMAND/wavs_noisy, respectively. You can also directly download the downsampled 16kHz dataset [here](https://drive.google.com/drive/folders/19I_thf6F396y5gZxLTxYIojZXC0Ywm8l).

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

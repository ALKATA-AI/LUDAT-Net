# In the name of Allah, the most compassionate, the most merciful

# LUDAT-Net Training

#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")


# Standard Libraries
import os
import time
import argparse
import json
import torch
import logging
import sys
import glob
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.distributed as dist
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from pesq import pesq
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

# Custom Modules
from env import AttrDict, build_env
from datasets.dataset import Val_Dataset, Dataset, mag_pha_stft, mag_pha_istft, get_dataset_filelist
from models.network import DAT_UNet, pesq_score, phase_losses
from utils import scan_checkpoint, load_checkpoint, save_checkpoint as original_save_checkpoint

# Extra Libraries
import pickle
from torchinfo import summary as torchinfo_summary

# Enable CuDNN benchmark for faster training
torch.backends.cudnn.benchmark = True

# Improved Logging Setup
def setup_logging(log_file_path=None, log_level=logging.INFO):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s",
                                    datefmt='%Y-%m-%d %H:%M:%S')
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(log_level)
    if log_file_path:
        try:
            file_handler = logging.FileHandler(log_file_path, mode='a')
            file_handler.setFormatter(log_formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"Error creating file handler for {log_file_path}: {e}", flush=True)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

# Improved Checkpoint Handling
def save_checkpoint(filepath, obj):
    try:
        logging.info(f"\n{'='*50}\nSaving checkpoint to {filepath}")
        
        # 1. Save to a temporary file first
        tmp_filepath = f"{filepath}.tmp"
        torch.save(obj, tmp_filepath)
        
        # 2. Use os.replace to overwrite the existing file safely
        # os.replace works on Windows and Linux and handles overwriting automatically
        os.replace(tmp_filepath, filepath)
        
        logging.info(f"Checkpoint saved successfully.\n{'='*50}")
    except Exception as e:
        logging.error(f"Could not save checkpoint to {filepath}: {e}")
        # If the replacement failed, try to remove the temp file to keep things clean
        if os.path.exists(f"{filepath}.tmp"):
             try: os.remove(f"{filepath}.tmp")
             except: pass
        raise

# =======================================================================================
# SECTION: Main Training Function
# =======================================================================================
def train(rank, a, h):
    # Initialize logging for each process
    log_file = os.path.join(a.checkpoint_path, 'training.log')
    setup_logging(log_file if rank == 0 else None, logging.INFO)

    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device(f'cuda:{rank}')

    # MODEL: Initialize Model
    model = DAT_UNet(h).to(device)
    if hasattr(torch, 'compile') and h.get('compile_model', False):
        model = torch.compile(model)
        #print("[+] Model compiled with torch.compile()")
        logging.info("[+] Model compiled with torch.compile()")

    # Logging and model summary on rank 0
    if rank == 0:
        #print(f"Model architecture:\n{model}")
        #logging.info(f"Model architecture:\n{model}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #print(f"Total Parameters: {total_params:,}")
        #print(f"Trainable Parameters: {trainable_params:,}")
        logging.info(f"Total Parameters: {total_params:,}")
        logging.info(f"Trainable Parameters: {trainable_params:,}")
        os.makedirs(a.checkpoint_path, exist_ok=True)
        os.makedirs(os.path.join(a.checkpoint_path, 'logs'), exist_ok=True)
        #print(f"Checkpoints directory: {a.checkpoint_path}")
        logging.info(f"Checkpoints directory: {a.checkpoint_path}")

    # Enhanced model summary
    try:
        # Create dummy inputs
        dummy_mag = torch.randn(1, h.n_fft // 2 + 1, h.segment_size // h.hop_size + 1).to(device)
        dummy_pha = torch.randn(1, h.n_fft // 2 + 1, h.segment_size // h.hop_size + 1).to(device)
            
        # Generate summary
        summary = torchinfo_summary(
            model,
            input_data=(dummy_mag, dummy_pha),
            col_names=[
                "input_size", 
                "output_size", 
                "num_params",
                "params_percent",
                "kernel_size",
                "mult_adds",
                "trainable"
            ],
            depth=5,
            verbose=0,
            device=device,
        )
            
        # Calculate parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
        # Create formatted summary
        model_summary = f"""
    {'='*80}
    {'MODEL SUMMARY':^80}
    {'='*80}
    {'Model Name:':<20} ALSENet
    {'Input Shape (Mag):':<20} {tuple(dummy_mag.shape)}
    {'Input Shape (Pha):':<20} {tuple(dummy_pha.shape)}
    {'Total Parameters:':<20} {total_params:,}
    {'Trainable Parameters:':<20} {trainable_params:,}
    {'Non-trainable Parameters:':<20} {total_params - trainable_params:,}
    {'='*80}

    {str(summary)}

    {'='*80}
    {'ESTIMATIONS':^80}
    {'='*80}
    {'Total Mult-Adds:':<30} {summary.total_mult_adds/1e9:.2f} G
    {'Params Size:':<30} {total_params * 4 / 1e6:.2f} MB (estimated)
    {'='*80}
    """

        #print(model_summary)
        logging.info(model_summary)
            
        # Save to file
        summary_file = os.path.join(a.checkpoint_path, 'model_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(model_summary)
        #print(f"Model summary saved to {summary_file}")
        logging.info(f"Model summary saved to {summary_file}")

    except Exception as e:
        #print(f"Error generating model summary: {str(e)}")
        logging.error(f"Error generating model summary: {str(e)}")

    # DATASET & DATALOADER SETUP
    training_indexes, validation_indexes = get_dataset_filelist(a)
    trainset = Dataset(training_indexes, a.input_clean_wavs_dir, a.input_noisy_wavs_dir,
                       h.segment_size, h.n_fft, h.hop_size, h.win_size, h.sampling_rate,
                       h.compress_factor, split=True, n_cache_reuse=0,
                       shuffle=False if h.num_gpus > 1 else True, device=device)
    
    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None
    
    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler, batch_size=h.batch_size,
                              pin_memory=True, drop_last=True,
                              persistent_workers=False
                              )
    
    if rank == 0:
        #print(f"Training dataset initialized with {len(trainset)} samples.")
        logging.info(f"Training dataset initialized with {len(trainset)} samples.")
        num_steps_per_epoch = len(train_loader)
        #print(f"Steps per epoch: {num_steps_per_epoch}")
        logging.info(f"Steps per epoch: {num_steps_per_epoch}")

        validset = Val_Dataset(validation_indexes, a.input_clean_wavs_dir, a.input_noisy_wavs_dir,
                               h.segment_size, h.n_fft, h.hop_size, h.win_size, h.sampling_rate,
                               h.compress_factor, split=False, shuffle=False, n_cache_reuse=0, device=device)
        
        validation_loader = DataLoader(validset, num_workers=h.num_workers, shuffle=False, sampler=None,
                                       batch_size=1, pin_memory=True, drop_last=True,
                                       persistent_workers=False
                                       )
        #print(f"Validation dataset initialized with {len(validset)} samples.")
        logging.info(f"Validation dataset initialized with {len(validset)} samples.")
        
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))
    
    # OPTIMIZER & SCHEDULER SETUP
    optimizer = torch.optim.AdamW(model.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    
    # =======================================================================================
    # CHECKPOINT AND STATE INITIALIZATION
    # =======================================================================================
    start_epoch = 0
    global_steps = 0
    best_pesq = 0.0
    losses_history = {'train': [], 'val': []}
    cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
    if cp_g is None:
        if rank == 0:
            #print("Starting training from scratch.")
            logging.info("Starting training from scratch.")
    else:
        if rank == 0:
            #print(f"Resuming training from checkpoint {cp_g}")
            logging.info(f"Resuming training from checkpoint {cp_g}")
        state_dict_g = load_checkpoint(cp_g, device)
        
        model.load_state_dict(state_dict_g['model'])
        optimizer.load_state_dict(state_dict_g['optimizer'])
        
        # Accurate status retrieval
        global_steps = state_dict_g['steps'] + 1
        start_epoch = state_dict_g['epoch']
        best_pesq = state_dict_g.get('best_pesq', 0.0) # For backward compatibility
        if rank == 0:
            #print(f"Resumed from epoch {start_epoch + 1} at step {global_steps}.")
            logging.info(f"Resumed from epoch {start_epoch + 1} at step {global_steps}.")

    # last_epoch should be the epoch number *before* starting the new one.
    # If we saved at the end of epoch 2 (epoch_idx=2), the next start is epoch 3.
    # The scheduler needs last_epoch=2.
    # Our 'start_epoch' is the one we are about to run. So scheduler needs start_epoch - 1
    scheduler_last_epoch = start_epoch - 1 if start_epoch > 0 else -1
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=h.lr_decay, last_epoch=scheduler_last_epoch)

    if h.num_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[rank])

    # =======================================================================================
    # TRAINING LOOP
    # =======================================================================================
    model.train()
    
    for epoch in range(start_epoch, a.training_epochs):
        if rank == 0:
            epoch_start_time = time.time()
            #print(f"\n{'='*50}\nStarting epoch {epoch + 1}/{a.training_epochs} | LR: {optimizer.param_groups[0]['lr']:.6f}\n{'='*50}")
            logging.info(f"\n{'='*50}\nStarting epoch {epoch + 1}/{a.training_epochs} | LR: {optimizer.param_groups[0]['lr']:.6f}\n{'='*50}")
            #print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
            logging.info(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
            #print(f"Steps per epoch: {len(train_loader)}")
            logging.info(f"Steps per epoch: {len(train_loader)}")
            #print(f"{'='*50}")
            logging.info(f"{'='*50}")

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        steps_in_epoch = len(train_loader)
        # Calculate the batch index to resume from, based on completed steps.
        # This should only be done for the very first epoch after resuming.
        resume_batch_idx = global_steps % steps_in_epoch if epoch == start_epoch else 0

        for i, batch in enumerate(train_loader):
            # If resuming, skip batches that have already been processed in this epoch.
            if i < resume_batch_idx:
                continue
            
            if rank == 0:
                batch_start_time = time.time()
                
            # Move data to GPU
            clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha = batch # [B, 1, F, T], F = nfft // 2+ 1, T = nframes
            clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
            clean_mag = torch.autograd.Variable(clean_mag.to(device, non_blocking=True))
            clean_pha = torch.autograd.Variable(clean_pha.to(device, non_blocking=True))
            clean_com = torch.autograd.Variable(clean_com.to(device, non_blocking=True))
            noisy_mag = torch.autograd.Variable(noisy_mag.to(device, non_blocking=True))
            noisy_pha = torch.autograd.Variable(noisy_pha.to(device, non_blocking=True))

            #clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha = batch
            #clean_audio = clean_audio.to(device, non_blocking=True)
            #clean_mag = clean_mag.to(device, non_blocking=True)
            #clean_pha = clean_pha.to(device, non_blocking=True)
            #clean_com = clean_com.to(device, non_blocking=True)
            #noisy_mag = noisy_mag.to(device, non_blocking=True)
            #noisy_pha = noisy_pha.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass
            mag_g, pha_g, com_g = model(noisy_mag, noisy_pha)
            audio_g = mag_pha_istft(mag_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            mag_g_con, pha_g_con, com_g_con = mag_pha_stft(audio_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            #_, _, com_g_con = mag_pha_stft(audio_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)


            # Reference: https://github.com/huaidanquede/PrimeK-Net/blob/main/train.py

            # ... Loss calculation ...

            # L2 Magnitude Loss
            loss_mag = F.mse_loss(clean_mag, mag_g)
            # Anti-wrapping Phase Loss
            loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g, h)
            loss_pha = loss_ip + loss_gd + loss_iaf        
            # L2 Complex Loss
            loss_com = F.mse_loss(clean_com, com_g) * 2
            # Consis Complex Loss
            loss_consis_con = F.mse_loss(com_g, com_g_con)
            # Time Loss
            loss_time = F.l1_loss(clean_audio, audio_g)

            # Total Loss
            total_loss = (
                loss_mag * 0.9 +
                loss_pha * 0.3 +
                loss_com * 0.1 +
                loss_consis_con * 0.1
            )
       
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Logging, Checkpointing, and Validation (only on rank 0)
            if rank == 0:
                # STDOUT logging
                if global_steps % a.stdout_interval == 0:
                    #print(
                    #    f"Step {global_steps} | [Epoch:{epoch + 1}/{a.training_epochs}, Batch:{i+1}/{steps_in_epoch}] | "
                    #    f"Loss: {total_loss.item():.4f} [M: {loss_mag.item():.4f}, "
                    #    f"P: {loss_pha.item():.4f}, C: {loss_com.item():.4f}, "
                    #    f"Con: {loss_consis_con.item():.4f}, T: {loss_time.item():.4f}] | "
                    #    f"Time: {time.time() - batch_start_time:.2f}s"
                    #)
                    log_msg = (
                        f"Step {global_steps} | [Epoch:{epoch + 1}/{a.training_epochs}, Batch:{i+1}/{steps_in_epoch}] | "
                        f"Loss: {total_loss.item():.4f} [M: {loss_mag.item():.4f}, "
                        f"P: {loss_pha.item():.4f}, C: {loss_com.item():.4f}, "
                        f"Con: {loss_consis_con.item():.4f}, T: {loss_time.item():.4f}] | "
                        f"Time: {time.time() - batch_start_time:.2f}s"
                    )
                    logging.info(log_msg)

                # Tensorboard summary logging
                #if global_steps % a.summary_interval == 0:
                    #sw.add_scalar("Training/Total_Loss", total_loss, global_steps)
                    #sw.add_scalar("Training/Magnitude_Loss", loss_mag, global_steps)
                    #sw.add_scalar("Training/Phase_Loss", loss_pha, global_steps)
                    #sw.add_scalar("Training/Complex Loss", loss_com, global_steps)
                    #sw.add_scalar("Training/Con_Complex Loss", loss_consis_con, global_steps)
                    #sw.add_scalar("Training/Time Loss", loss_time, global_steps)

                # Checkpointing
                if global_steps % a.checkpoint_interval == 0 and global_steps != 0:
                    checkpoint_path = f"{a.checkpoint_path}/g_step_{global_steps:08d}"
                    save_checkpoint(checkpoint_path, {
                        'model': (model.module if h.num_gpus > 1 else model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'steps': global_steps,
                        'epoch': epoch, # Save the current epoch
                        'best_pesq': best_pesq
                    })
                
                # Validation
                if rank == 0: # new ......................................................
                    if global_steps % a.validation_interval == 0 and global_steps != 0:
                        model.eval()
                        torch.cuda.empty_cache()
                        #print(f"Starting validation...")
                        logging.info(f"Starting validation...")
                        val_start_time = time.time()
                        audios_r, audios_g = [], []
                        val_mag_err_tot = 0
                        val_pha_err_tot = 0
                        val_com_err_tot = 0

                        segment_size = h.segment_size
                        n_fft = h.n_fft
                        hop_size = h.hop_size
                        win_size = h.win_size
                        compress_factor = h.compress_factor

                        with torch.no_grad():
                            for j, batch in enumerate(validation_loader):
                                try:
                                    clean_audio, noisy_audio = batch # [B, 1, F, T], F = nfft // 2+ 1, T = nframes
                                            
                                    clean_mag, clean_pha, clean_com = mag_pha_stft(
                                        clean_audio, n_fft, hop_size, win_size, compress_factor
                                        )

                                    noisy_audio = torch.autograd.Variable(noisy_audio.to(device, non_blocking=True))
                                    clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
                                    clean_mag = torch.autograd.Variable(clean_mag.to(device, non_blocking=True))
                                    clean_pha = torch.autograd.Variable(clean_pha.to(device, non_blocking=True))
                                    clean_com = torch.autograd.Variable(clean_com.to(device, non_blocking=True))
                                            
                                    orig_size = noisy_audio.size(1)

                                    if noisy_audio.size(1) >= segment_size:
                                        num_segments = noisy_audio.size(1) // segment_size
                                        last_segment_size = noisy_audio.size(1) % segment_size
                                        if last_segment_size > 0:
                                            last_segment = noisy_audio[:, -segment_size:]
                                            noisy_audio = noisy_audio[:, :-last_segment_size]
                                            segments = torch.split(noisy_audio, segment_size, dim=1)
                                            segments = list(segments)
                                            segments.append(last_segment)
                                            reshapelast = 1
                                        else:
                                            segments = torch.split(noisy_audio, segment_size, dim=1)
                                            reshapelast = 0
                                    else:
                                        padded_zeros = torch.zeros(1, segment_size - noisy_audio.size(1)).to(device)
                                        noisy_audio = torch.cat((noisy_audio, padded_zeros), dim=1)
                                        segments = [noisy_audio]
                                        reshapelast = 0

                                    processed_segments = []
                                    audio_g = []

                                    for i, segment in enumerate(segments):

                                        noisy_amp, noisy_pha, noisy_com = mag_pha_stft(segment, n_fft, hop_size, win_size, compress_factor)
                                        amp_g, pha_g, com_g = model(noisy_amp.to(device, non_blocking=True),
                                                                    noisy_pha.to(device, non_blocking=True))
                                        audio_g = mag_pha_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor)
                                        audio_g = audio_g.squeeze()
                                        if reshapelast == 1 and i == len(segments) - 2:
                                            audio_g = audio_g[:-(segment_size - last_segment_size)]
                                            # print(orig_size)

                                        processed_segments.append(audio_g)

                                    processed_audio = torch.cat(processed_segments, dim=-1)


                                    audio_g = processed_audio[:orig_size]


                                    mag_g, pha_g, com_g = mag_pha_stft(audio_g, n_fft, hop_size, win_size,
                                                                                    compress_factor)

                                    mag_g = torch.autograd.Variable(mag_g.to(device, non_blocking=True))
                                    pha_g = torch.autograd.Variable(pha_g.to(device, non_blocking=True))
                                    com_g = torch.autograd.Variable(com_g.to(device, non_blocking=True))

                                    mag_g = mag_g.squeeze()
                                    pha_g = torch.unsqueeze(pha_g, dim=0)

                                    # com_g = com_g.squeeze()
                                    clean_mag = clean_mag.squeeze()
                                    # clean_pha = clean_pha.squeeze()

                                    clean_com = clean_com.squeeze()
                                    audios_r += torch.split(clean_audio, 1, dim=0) # [1, T] * B
                                    # print(clean_audio.size())
                                    # print(len(audios_r))
                                    audio_g = torch.unsqueeze(audio_g, dim=0)
                                    audios_g += torch.split(audio_g, 1, dim=0)
                                    # print(audio_g.size())
                                    # print(len(audios_g))
                                    val_mag_err_tot += F.mse_loss(clean_mag, mag_g).item()
                                    val_ip_err, val_gd_err, val_iaf_err = phase_losses(clean_pha, pha_g, h)
                                    val_pha_err_tot += (val_ip_err + val_gd_err + val_iaf_err).item()
                                    val_com_err_tot += F.mse_loss(clean_com, com_g).item()
                                except Exception as e:
                                    #print(f"Error processing validation batch {j}: {e}")
                                    logging.warning(f"Error processing validation batch {j}: {e}")
                                    continue            

                            # Calculate average metrics
                            val_mag_err = val_mag_err_tot / (j+1)
                            val_pha_err = val_pha_err_tot / (j+1)
                            val_com_err = val_com_err_tot / (j+1)

                            # Calculate PESQ
                            #print(f"Samples: {len(audios_g)}")
                            logging.info(f"Samples: {len(audios_g)}")
                            val_pesq_score = pesq_score(audios_r, audios_g, h).item()

                            # Log validation results
                            #print(
                            #    f"\n{'='*50}\nValidation at Step {global_steps} & Epoch {epoch + 1} | "
                            #    f"PESQ: {val_pesq_score:.3f} | "
                            #    f"Mag Loss: {val_mag_err:.4f} | "
                            #    f"Pha Loss: {val_pha_err:.4f} | "
                            #    f"Com Loss: {val_com_err:.4f} | "
                            #    f"Time: {time.time() - val_start_time:.2f}s"
                            #)
                            val_log_msg = (
                                f"\n{'='*50}\nValidation at Step {global_steps} & Epoch {epoch + 1} | "
                                f"PESQ: {val_pesq_score:.3f} | "
                                f"Mag Loss: {val_mag_err:.4f} | "
                                f"Pha Loss: {val_pha_err:.4f} | "
                                f"Com Loss: {val_com_err:.4f} | "
                                f"Time: {time.time() - val_start_time:.2f}s"
                            )
                            logging.info(val_log_msg)

                            #losses_history['val'].append({
                                #'step': global_steps,
                                #'pesq': val_pesq_score,
                                #'mag_loss': val_mag_err,
                                #'pha_loss': val_pha_err,
                                #'com_loss': val_com_err
                            #})

                            sw.add_scalar("Validation/PESQ Score", val_pesq_score, global_steps)
                            sw.add_scalar("Validation/Magnitude Loss", val_mag_err, global_steps)
                            sw.add_scalar("Validation/Phase Loss", val_pha_err, global_steps)
                            sw.add_scalar("Validation/Complex Loss", val_com_err, global_steps)

                            if val_pesq_score > best_pesq:
                                best_pesq = val_pesq_score
                                #print(f"\n{'='*50}\n[!] New Best PESQ: {best_pesq:.4f} at step {global_steps} & Epoch {epoch + 1}\n{'='*50}")
                                logging.info(f"\n{'='*50}\n[!] New Best PESQ: {best_pesq:.4f} at step {global_steps} & Epoch {epoch + 1}\n{'='*50}")
                                checkpoint_path_best = "{}/g_best".format(a.checkpoint_path)
                                save_checkpoint(checkpoint_path_best,
                                                {'model': (model.module if h.num_gpus > 1 else model).state_dict(),
                                                'optimizer': optimizer.state_dict(),
                                                'steps': global_steps,
                                                'epoch': epoch,
                                                'best_pesq': best_pesq
                                })
                            #logging.info(f"\n{'='*50}\nBest PESQ: {best_pesq:.4f} at step {best_pesq_step} & Epoch {best_epoch_idx + 1}\n{'='*50}")
                    
                    model.train()

            global_steps += 1 # Increment the total counter at the end of each batch.

        # End of epoch processing
        scheduler.step()
        
        if rank == 0:
            epoch_time = time.time() - epoch_start_time
            #print(f"Epoch {epoch + 1} completed in {int(epoch_time // 60)}m {int(epoch_time % 60)}s")
            logging.info(f"Epoch {epoch + 1} completed in {int(epoch_time // 60)}m {int(epoch_time % 60)}s")
            
            # Save end-of-epoch checkpoint
            checkpoint_path_epoch = f"{a.checkpoint_path}/g_epoch_{epoch + 1:04d}"
            save_checkpoint(checkpoint_path_epoch, {
                'model': (model.module if h.num_gpus > 1 else model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'steps': global_steps,
                'epoch': epoch + 1, # Because this epoch is over, we must start from the next epoch to start again.
                'best_pesq': best_pesq
            })
            
    # Final cleanup and saving on rank 0
    if rank == 0:
        
        # Save losses as CSV files
        df_train = pd.DataFrame(losses_history['train'])
        df_val = pd.DataFrame(losses_history['val'])

        train_csv_path = os.path.join(a.checkpoint_path, 'train_losses.csv')
        val_csv_path = os.path.join(a.checkpoint_path, 'val_losses.csv')

        df_train.to_csv(train_csv_path, index=False)
        df_val.to_csv(val_csv_path, index=False)

        logging.info(f"[+] Training loss history saved to {train_csv_path}")
        logging.info(f"[+] Validation loss history saved to {val_csv_path}")

        # Save final loss history
        with open(os.path.join(a.checkpoint_path, 'losses_history.pkl'), 'wb') as f:
            pickle.dump(losses_history, f)

        #print(f"\n{'='*50}")
        logging.info(f"\n{'='*50}")
        #print(f"Training finished and completed successfully.")
        logging.info(f"Training finished and completed successfully.")
        #print(f"Best PESQ: {best_pesq:.4f}")
        logging.info(f"Best PESQ: {best_pesq:.4f}")
        #print(f"Total steps: {global_steps -epoch}")
        logging.info(f"Total steps: {global_steps -epoch}")
        #print(f"Output directory: {a.checkpoint_path}")
        logging.info(f"Output directory: {a.checkpoint_path}")
        #print(f"{'='*50}")
        logging.info(f"{'='*50}")

    if h.num_gpus > 1:
        dist.destroy_process_group()


# =======================================================================================
# SECTION: Main Entry Point
# =======================================================================================
def main():
    print('Initializing Training Process...')
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_name', default=None)	
    parser.add_argument('--input_clean_wavs_dir', required=True, default='/content/VoiceBank+DEMAND/wavs_clean', help='Directory containing clean training wav files')
    parser.add_argument('--input_noisy_wavs_dir', required=True, default='/content/VoiceBank+DEMAND/wavs_noisy', help='Directory containing noisy training wav files')
    parser.add_argument('--input_training_file', required=True, default='/content/VoiceBank+DEMAND/training.txt', help='File containing training file list')
    parser.add_argument('--input_validation_file', required=True, default='/content/VoiceBank+DEMAND/test.txt', help='File containing validation file list')
    parser.add_argument('--checkpoint_path', required=True, default='checkpoints', help='Path to save checkpoints')
    parser.add_argument('--config', required=True, default='' , help='Path to config file')
    parser.add_argument('--training_epochs', default=130, type=int, help='Number of training epochs')
    parser.add_argument('--stdout_interval', default=10, type=int, help='Interval for printing training stats')
    parser.add_argument('--checkpoint_interval', default=5000, type=int, help='Interval for saving checkpoints')
    parser.add_argument('--summary_interval', default=1e9, type=int, help='Interval for saving TensorBoard summaries')
    parser.add_argument('--validation_interval', default=5000, type=int, help='Interval for running validation')
    a = parser.parse_args()
    
    # Load config
    with open(a.config) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    # Build environment
    build_env(a.config, 'config.json', a.checkpoint_path)
    #print(f"Config copied to {os.path.join(a.checkpoint_path, 'config.json')}")
    logging.info(f"Config copied to {os.path.join(a.checkpoint_path, 'config.json')}")

    # Set seeds
    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        if h.num_gpus > 0:
            h.batch_size = int(h.batch_size / h.num_gpus)
        #print(f'Using {h.num_gpus} GPUs, batch size per GPU: {h.batch_size}')
        logging.info(f'Using {h.num_gpus} GPUs, batch size per GPU: {h.batch_size}')
    else:
        #print('CUDA not available, using CPU')
        logging.warning('CUDA not available, using CPU')
        h.num_gpus = 0

    # Start training
    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
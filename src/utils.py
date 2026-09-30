# In the name of Allah, the most compassionate, the most merciful

import glob
import os
import torch
import logging
import torch.nn as nn

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (int((kernel_size[0]*dilation[0] - dilation[0])/2), int((kernel_size[1]*dilation[1] - dilation[1])/2))


def save_checkpoint(filepath, obj):
    """
    Saves a checkpoint dictionary to a file.
    
    Args:
        filepath (str): Path to save the checkpoint file.
        obj (dict): A dictionary containing model state, optimizer state, etc.
    """
    try:
        # It's safer to save to a temporary file and then rename
        temp_filepath = filepath + ".tmp"
        torch.save(obj, temp_filepath)
        os.rename(temp_filepath, filepath)
    except Exception as e:
        logging.error(f"Error saving checkpoint to {filepath}: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)


def load_checkpoint(filepath, device):
    """
    Loads a checkpoint dictionary from a file.
    
    Args:
        filepath (str): Path to the checkpoint file.
        device (torch.device): The device to map the loaded tensors to.
        
    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    logging.info(f"Loading checkpoint from {filepath}")
    checkpoint_dict = torch.load(filepath, map_location=device)
    logging.info("Checkpoint loaded successfully.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    """
    Scans a directory for checkpoints and returns the latest one based on modification time.
    This is generally more robust than alphabetical sorting.
    
    Args:
        cp_dir (str): The directory to scan.
        prefix (str): The prefix of the checkpoint files (e.g., 'g_').
        
    Returns:
        str or None: The path to the latest checkpoint, or None if no checkpoints are found.
    """
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    
    if len(cp_list) == 0:
        return None
        
    # Find the checkpoint that was most recently modified
    try:
        latest_file = max(cp_list, key=os.path.getmtime)
    except FileNotFoundError:
        # This can happen in a race condition if a file is deleted
        # between glob and max. We can just retry or return None.
        return None

    return latest_file


class LearnableSigmoid_1d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class LearnableSigmoid_2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)

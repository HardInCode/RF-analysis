import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
from scipy.signal import hilbert
import h5py
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# AES SBox for Hamming weight calculations
AES_SBOX = np.array([
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
], dtype=np.uint8)

def hamming_weight(byte):
    """Calculate the Hamming weight (number of '1' bits) in an integer."""
    return bin(byte).count('1')

def aes_state_byte0(plaintext, key_byte):
    """Return the value of the first byte after AES SubBytes."""
    return AES_SBOX[plaintext[0] ^ key_byte]

def generate_synthetic_traces(num_traces=1000, samples_per_trace=5000, fs=1000000, fc=100000, snr=10, plaintexts=None, enhanced_model=False):
    """
    Generate synthetic RF side-channel traces for AES with multiple leakage points.
    
    Parameters:
    - num_traces: Number of traces to generate
    - samples_per_trace: Number of samples in each trace
    - fs: Sampling frequency in Hz
    - fc: Carrier frequency in Hz
    - snr: Signal-to-noise ratio in dB
    - plaintexts: Optional array of plaintexts (will be randomly generated if None)
    - enhanced_model: Use enhanced Hamming Distance model if True, otherwise use classic Hamming Weight
    
    Returns:
    - traces: Array of shape (num_traces, samples_per_trace)
    - plaintexts: Array of shape (num_traces, 16) containing plaintexts
    - true_key: The key byte used for generating the traces (or array of key bytes)
    """
    # Generate random plaintexts if not provided
    if plaintexts is None:
        plaintexts = np.random.randint(0, 256, size=(num_traces, 16), dtype=np.uint8)
    else:
        num_traces = len(plaintexts)
    
    # Generate a random key
    true_key = np.random.randint(0, 256, size=16, dtype=np.uint8)
    
    # Time base
    t = np.linspace(0, samples_per_trace/fs, samples_per_trace)
    
    # Generate traces
    traces = np.zeros((num_traces, samples_per_trace))
    
    #hamming distance
    if enhanced_model:
        # Enhanced model with multiple leakage points
        # Define AES operation points (relative positions in the trace)
        aes_operations = {
            'key_addition_init': 0.1,      # Initial AddRoundKey
            'sub_bytes': 0.25,             # SubBytes operation
            'shift_rows': 0.35,            # ShiftRows operation
            'mix_columns': 0.5,            # MixColumns operation
            'key_addition_round': 0.65,    # AddRoundKey for a round
            'final_round': 0.8             # Final operations
        }
        
        # Define leakage weights for each operation (how much each operation leaks)
        leakage_weights = {
            'key_addition_init': 0.7,
            'sub_bytes': 1.0,              # SubBytes typically leaks the most
            'shift_rows': 0.3,
            'mix_columns': 0.6,
            'key_addition_round': 0.8,
            'final_round': 0.9
        }
        
        # Define width of the Gaussian for each operation (some operations take longer)
        burst_widths = {
            'key_addition_init': samples_per_trace // 30,
            'sub_bytes': samples_per_trace // 25,
            'shift_rows': samples_per_trace // 35,
            'mix_columns': samples_per_trace // 20,
            'key_addition_round': samples_per_trace // 30,
            'final_round': samples_per_trace // 22
        }
        
        for i in range(num_traces):
            # Create a clean trace for this iteration
            envelope = np.zeros(samples_per_trace)
            
            # Track previous state for Hamming distance calculation
            previous_state = np.copy(plaintexts[i])
            
            # Initialize state
            current_state = np.copy(plaintexts[i])
            
            # Process each AES operation and add its leakage
            for op_name, op_position in aes_operations.items():
                # Calculate the operation's burst center in the trace
                burst_center = int(op_position * samples_per_trace)
                burst_width = burst_widths[op_name]
                
                # Update state based on the operation (simplified AES operations)
                if op_name == 'key_addition_init':
                    # Initial AddRoundKey
                    for j in range(16):
                        current_state[j] = previous_state[j] ^ true_key[j]
                elif op_name == 'sub_bytes':
                    # SubBytes operation
                    for j in range(16):
                        current_state[j] = AES_SBOX[current_state[j]]
                elif op_name == 'shift_rows':
                    # Simplified ShiftRows (just swapping some bytes)
                    if len(current_state) >= 16:  # Make sure we have enough bytes
                        current_state[1], current_state[5], current_state[9], current_state[13] = \
                            current_state[5], current_state[9], current_state[13], current_state[1]
                elif op_name == 'mix_columns':
                    # Simplified MixColumns (just some byte manipulation)
                    if len(current_state) >= 16:
                        for col in range(4):
                            idx = col * 4
                            # Simple mixing operation (not actual AES MixColumns)
                            temp = current_state[idx:idx+4].copy()
                            current_state[idx] = (temp[0] ^ temp[1]) & 0xFF
                            current_state[idx+1] = (temp[1] ^ temp[2]) & 0xFF
                            current_state[idx+2] = (temp[2] ^ temp[3]) & 0xFF
                            current_state[idx+3] = (temp[3] ^ temp[0]) & 0xFF
                elif op_name == 'key_addition_round':
                    # Round AddRoundKey (simplified)
                    for j in range(16):
                        # Using a derived round key (simplified)
                        round_key_byte = (true_key[j] ^ j) & 0xFF
                        current_state[j] = current_state[j] ^ round_key_byte
                elif op_name == 'final_round':
                    # Final round operations (simplified)
                    for j in range(16):
                        current_state[j] = AES_SBOX[current_state[j]] ^ true_key[j % len(true_key)]
                
                # Calculate Hamming distance between previous and current state
                hd = sum(hamming_weight(prev ^ curr) for prev, curr in zip(previous_state, current_state))
                # Normalize by maximum possible Hamming distance
                hd_normalized = hd / (16 * 8)  # 16 bytes * 8 bits maximum
                
                # Create a Gaussian envelope for this operation
                op_envelope = np.exp(-0.5 * ((t*fs - burst_center) / burst_width)**2)
                
                # Scale the envelope by the Hamming distance and operation weight
                weight = leakage_weights[op_name]
                envelope += weight * (0.2 + 0.8 * hd_normalized) * op_envelope
                
                # Update previous state for next operation
                previous_state = np.copy(current_state)
            
            # Normalize the envelope to [0, 1] range
            if np.max(envelope) > 0:
                envelope = envelope / np.max(envelope)
                
            # Create the carrier signal
            carrier = np.sin(2 * np.pi * fc * t)
            
            # Amplitude modulate the carrier
            trace = envelope * carrier
            
            # Add noise
            signal_power = np.mean(trace**2)
            noise_power = signal_power / (10**(snr/10))
            noise = np.random.normal(0, np.sqrt(noise_power), samples_per_trace)
            
            traces[i] = trace + noise
        
        # Return full key for enhanced model
        return traces, plaintexts, true_key
    
    else:
        # Classic model with single leakage point (original implementation)
        # Create Gaussian envelope template
        mid_point = samples_per_trace // 2
        width = samples_per_trace // 10
        envelope_template = np.exp(-0.5 * ((t*fs - mid_point) / width)**2)
        
        # Generate key byte
        key_byte = true_key[0]  # Just use the first byte for classic model
        
        for i in range(num_traces):
            # Create envelope for this trace
            pt_byte = plaintexts[i][0]  # Only use first byte of plaintext
            # Calculate Hamming weight of S-box output
            hw = hamming_weight(AES_SBOX[pt_byte ^ key_byte])
            # Scale envelope by Hamming weight
            envelope = (0.2 + 0.8 * hw / 8) * envelope_template
            
            # Create carrier
            carrier = np.sin(2 * np.pi * fc * t)
            
            # AM modulate
            trace = envelope * carrier
            
            # Add noise
            signal_power = np.mean(trace**2)
            noise_power = signal_power / (10**(snr/10))
            noise = np.random.normal(0, np.sqrt(noise_power), samples_per_trace)
            
            traces[i] = trace + noise
        
        # Return only key byte for classic model
        return traces, plaintexts, key_byte

def export_to_wav(traces, fs, path):
    """
    Export traces to a WAV file.
    
    Parameters:
    - traces: Array of shape (num_traces, samples_per_trace)
    - fs: Sampling frequency in Hz
    - path: Output file path
    """
    # Normalize to [-1, 1] for WAV format
    max_val = np.max(np.abs(traces))
    normalized_traces = traces / max_val if max_val > 0 else traces
    
    # Convert to 16-bit PCM
    scaled_traces = np.int16(normalized_traces * 32767)
    
    # Flatten all traces into a single continuous signal
    flattened_traces = scaled_traces.flatten()
    
    # Write to WAV file
    wav.write(path, int(fs), flattened_traces)
    
    return True


def run_cpa(traces, plaintexts, byte_index=0, vectorized=True):
    """
    Run Correlation Power Analysis to find the most likely key byte.
    
    Parameters:
    - traces: Array of shape (num_traces, samples_per_trace)
    - plaintexts: Array of shape (num_traces, 16) containing plaintexts
    - byte_index: Index of the target key byte (default is 0)
    - vectorized: Whether to use vectorized implementation (default is True)
    
    Returns:
    - correlations: Array of shape (256, samples_per_trace) containing correlation for each key guess
    - max_correlations: Array of shape (256,) containing max correlation for each key guess
    - best_key: The key byte with the highest correlation
    """
    num_traces, samples_per_trace = traces.shape
    
    # Ensure plaintexts are in the correct shape
    if plaintexts.ndim == 1:
        plaintexts = plaintexts.reshape(-1, 1)
    
    # Prepare plaintext bytes - handle array boundaries carefully
    if byte_index >= plaintexts.shape[1]:
        byte_index = 0  # Reset to first byte if out of bounds
    plaintext_bytes = plaintexts[:, byte_index]
    
    # Initialize correlation matrix
    correlations = np.zeros((256, samples_per_trace))
    
    # Center traces (subtract mean from each sample)
    traces_mean = np.mean(traces, axis=0)
    centered_traces = traces - traces_mean
    
    # Precompute trace standard deviations
    traces_std = np.std(traces, axis=0)
    # Avoid division by zero
    traces_std[traces_std == 0] = 1
    
    # For each key guess
    for key_guess in tqdm(range(256), desc="Running CPA"):
        try:
            # Calculate hypothetical hamming weights
            hw_values = np.array([hamming_weight(aes_state_byte0(np.array([p]), key_guess)) for p in plaintext_bytes])
            
            # Center the model
            hw_mean = np.mean(hw_values)
            hw_centered = hw_values - hw_mean
            
            # Compute standard deviation of hamming weights
            hw_std = np.std(hw_values)
            if hw_std == 0:
                hw_std = 1  # Avoid division by zero
            
            # Vectorized correlation computation across all time samples
            # Formula: corr = sum((x-x_mean)*(y-y_mean)) / (sqrt(sum((x-x_mean)^2) * sum((y-y_mean)^2)))
            
            # Numerator: sum of products of centered values
            # Reshape hw_centered to allow broadcasting
            hw_centered_reshaped = hw_centered.reshape(-1, 1)
            numerator = np.sum(hw_centered_reshaped * centered_traces, axis=0)
            
            # Denominator: product of standard deviations
            denominator = hw_std * traces_std * num_traces
            
            # Calculate correlation
            correlations[key_guess, :] = numerator / denominator
            
        except Exception as e:
            print(f"Error processing key guess {key_guess}: {e}")
            correlations[key_guess, :] = 0
    
    # Find the maximum correlation for each key guess
    max_correlations = np.max(np.abs(correlations), axis=1)
    
    # Find the key with the highest correlation
    best_key = np.argmax(max_correlations)
    
    return correlations, max_correlations, best_key

def plot_cpa_results(correlations, max_correlations, best_key, true_key=None):
    """
    Plot the results of the CPA analysis.
    
    Parameters:
    - correlations: Array of shape (256, samples_per_trace) containing correlation for each key guess
    - max_correlations: Array of shape (256,) containing max correlation for each key guess
    - best_key: The key byte with the highest correlation
    - true_key: The true key byte (if known)
    
    Returns:
    - fig: Matplotlib figure object
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot correlation vs time for the correct key
    axs[0].plot(correlations[best_key], 'b-', label=f'Best key: 0x{best_key:02x}')
    axs[0].set_title('Correlation vs Time (Best Key)')
    axs[0].set_xlabel('Sample')
    axs[0].set_ylabel('Correlation')
    axs[0].grid(True)
    axs[0].legend()
    
    # Plot correlation vs time for a wrong key
    wrong_key = (best_key + 1) % 256
    axs[1].plot(correlations[wrong_key], 'r-', label=f'Wrong key: 0x{wrong_key:02x}')
    axs[1].set_title('Correlation vs Time (Wrong Key)')
    axs[1].set_xlabel('Sample')
    axs[1].set_ylabel('Correlation')
    axs[1].grid(True)
    axs[1].legend()
    
    # Bar chart of peak correlations for all key guesses
    axs[2].bar(range(256), max_correlations)
    axs[2].set_title('Peak Correlation for All Key Guesses')
    axs[2].set_xlabel('Key Guess')
    axs[2].set_ylabel('Max Correlation')
    axs[2].grid(True)
    
    # Mark the best key
    axs[2].bar(best_key, max_correlations[best_key], color='green', label=f'Best Key: 0x{best_key:02x}')
    
    # Mark the true key if provided
    if true_key is not None:
        # Convert from numpy array if needed
        if isinstance(true_key, np.ndarray) and len(true_key) > 0:
            true_key = true_key[0]
        axs[2].bar(true_key, max_correlations[true_key], color='red', label=f'True Key: 0x{true_key:02x}')
    
    axs[2].legend()
    
    plt.tight_layout()
    
    return fig

def align_traces(traces, method='cross_correlation', reference='first_trace', window=None):
    """
    Align traces using cross-correlation or similar techniques.
    
    Parameters:
    - traces: Array of shape (num_traces, samples_per_trace)
    - method: Alignment method ('cross_correlation', 'sum_of_absolute_differences', or 'maximum_peak')
    - reference: Reference trace selection ('first_trace', 'average_of_all_traces', or 'trace_with_highest_snr')
    - window: Tuple of (start_idx, end_idx) for alignment window, or None to use the full trace
    
    Returns:
    - aligned_traces: Array of shape (num_traces, samples_per_trace) with aligned traces
    """
    from scipy.signal import correlate
    from scipy.ndimage import shift
    
    # Debug print to see what method name is being passed
    print(f"Method received: '{method}'")
    
    num_traces, samples_per_trace = traces.shape
    
    # Set default parameters if not provided
    max_shift = samples_per_trace // 10  # Default to 10% of trace length
    
    # Generate reference trace based on selected option
    if reference == 'first_trace':
        reference_trace = traces[0]
    elif reference == 'average_of_all_traces':
        reference_trace = np.mean(traces, axis=0)
    elif reference == 'trace_with_highest_snr':
        # Simple SNR estimation: variance / mean of absolute differences
        variances = np.var(traces, axis=1)
        mean_abs_diffs = np.mean(np.abs(np.diff(traces, axis=1)), axis=1)
        snrs = variances / (mean_abs_diffs + 1e-10)  # Avoid division by zero
        reference_trace = traces[np.argmax(snrs)]
    else:
        raise ValueError(f"Unknown reference selection: {reference}")
    
    # Determine window for alignment
    if window is None:
        # Use the entire trace
        start_idx, end_idx = 0, samples_per_trace
    else:
        start_idx, end_idx = window
        
    # Extract reference window
    ref_window = reference_trace[start_idx:end_idx]
    
    # Initialize array for aligned traces
    aligned_traces = np.zeros_like(traces)
    
    # Perform alignment for each trace
    for i in range(num_traces):
        current_trace = traces[i]
        
        # Extract current trace window with padding for potential shifts
        padded_start = max(0, start_idx - max_shift)
        padded_end = min(samples_per_trace, end_idx + max_shift)
        trace_window = current_trace[padded_start:padded_end]
        
        # Fix: standardize method names and handle various formats that might be passed
        method_lower = method.lower().replace('-', '_').replace(' ', '_')
        
        if method_lower in ['cross_correlation', 'xcorr', 'crosscorrelation']:
            # Cross-correlation based alignment
            xcorr = correlate(trace_window, ref_window, mode='valid')
            max_idx = np.argmax(xcorr)
            shift_value = max_idx - (padded_start - start_idx)
            
        elif method_lower in ['sum_of_absolute_differences', 'sad', 'absolute_differences']:
            # Sum of absolute differences (SAD) - sliding window approach
            sad_values = []
            for shift_idx in range(2 * max_shift):
                if padded_start + shift_idx + (end_idx - start_idx) <= padded_end:
                    comparison_window = trace_window[shift_idx:shift_idx + (end_idx - start_idx)]
                    sad = np.sum(np.abs(comparison_window - ref_window))
                    sad_values.append(sad)
                else:
                    sad_values.append(float('inf'))
            
            # Minimum SAD corresponds to best alignment
            min_idx = np.argmin(sad_values)
            shift_value = min_idx - max_shift
            
        elif method_lower in ['maximum_peak', 'peak', 'max_peak']:
            # Find and align based on maximum peak
            ref_peak_idx = start_idx + np.argmax(ref_window)
            trace_peak_idx = padded_start + np.argmax(trace_window)
            shift_value = trace_peak_idx - ref_peak_idx
            
        else:
            raise ValueError(f"Unknown alignment method: {method}")
        
        # Limit shift to max_shift
        shift_value = max(min(shift_value, max_shift), -max_shift)
        
        # Apply the shift
        shifted_trace = shift(current_trace, -shift_value, mode='constant', cval=0)
        aligned_traces[i] = shifted_trace
    
    return aligned_traces

def align_max_peak(traces, reference='trace_with_highest_snr'):
    """
    Align a batch of traces by finding each trace’s maximum peak
    relative to the highest‐SNR trace.
    """
    return align_traces(
        traces,
        method='maximum_peak',
        reference=reference,
        window=None
    )

def visualize_alignment(original_traces, aligned_traces, num_to_show=5):
    """
    Create visualization to compare original and aligned traces.
    
    Parameters:
    - original_traces: Array of original traces
    - aligned_traces: Array of aligned traces
    - num_to_show: Number of traces to display
    
    Returns:
    - fig: Matplotlib figure object with the visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot subset of original traces
    for i in range(min(num_to_show, len(original_traces))):
        ax1.plot(original_traces[i], alpha=0.7)
    
    # Plot subset of aligned traces
    for i in range(min(num_to_show, len(aligned_traces))):
        ax2.plot(aligned_traces[i], alpha=0.7)
    
    # Add titles and labels
    ax1.set_title("Original Traces")
    ax2.set_title("Aligned Traces")
    ax2.set_xlabel("Sample")
    ax1.set_ylabel("Amplitude")
    ax2.set_ylabel("Amplitude")
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    return fig


def compare_keys(recovered_key, true_key=None):
    """
    Compare recovered key with true key and format the result.
    
    Parameters:
    - recovered_key: Array of recovered key bytes
    - true_key: Array of true key bytes (if known) or single byte for synthetic case
    
    Returns:
    - fig: Matplotlib figure with comparison results
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create a figure for the visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Format recovered key for display
    recovered_key_hex = ' '.join([f"{b:02X}" for b in recovered_key])
    
    # Prepare data for the bar chart
    x = np.arange(16)
    bar_height = np.ones(16)
    
    # Set up the bar chart
    ax.bar(x, bar_height, color='lightgray', width=0.7, alpha=0.8)
    
    # Add recovered key bytes text
    for i, b in enumerate(recovered_key):
        ax.text(i, 0.7, f"{b:02X}", ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Handle true key comparison if provided
    if true_key is not None:
        # Check if it's a scalar (single byte) or array
        if isinstance(true_key, (int, np.integer)):
            # For synthetic data case with single key byte
            correct = (recovered_key[0] == true_key)
            ax.text(0, 0.3, f"True: {true_key:02X}", ha='center', va='center', 
                   color='green' if correct else 'red', fontsize=9)
            title = f"Key Recovery Results (Byte 0: {'Correct' if correct else 'Incorrect'})"
        else:
            # Convert true_key to numpy array if it's not already
            true_key_array = np.asarray(true_key)
            
            # Determine which bytes are correct
            # Only compare up to the length of true_key_array
            correct_count = 0
            for i in range(min(16, len(true_key_array))):
                color = 'green' if recovered_key[i] == true_key_array[i] else 'red'
                ax.text(i, 0.3, f"True: {true_key_array[i]:02X}", ha='center', va='center', 
                       color=color, fontsize=9)
                if recovered_key[i] == true_key_array[i]:
                    correct_count += 1
            
            title = f"Key Recovery Results ({correct_count}/16 bytes correct)"
    else:
        title = "Key Recovery Results (No true key for comparison)"
    
    # Set titles and labels
    ax.set_title(title)
    ax.set_xlabel("Key Byte Index")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}" for i in x])
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    return fig

def plot_full_key_recovery(correlations_all, max_correlations_all, recovered_key, true_key=None):
    """
    Plot the results of the full key CPA analysis.
    
    Parameters:
    - correlations_all: List of arrays, each of shape (256, samples_per_trace) for each key byte
    - max_correlations_all: List of arrays, each of shape (256,) for each key byte
    - recovered_key: Array of shape (16,) containing the recovered key bytes
    - true_key: The true key bytes (if known)
    
    Returns:
    - fig: Matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create a figure with subplots - 4x4 grid for 16 bytes
    fig, axs = plt.subplots(4, 4, figsize=(20, 15))
    axs = axs.flatten()
    
    # Plot results for each byte
    for byte_index in range(16):
        max_correlations = max_correlations_all[byte_index]
        best_key = recovered_key[byte_index]
        
        # Bar chart of peak correlations for all key guesses for this byte
        axs[byte_index].bar(range(256), max_correlations)
        axs[byte_index].set_title(f'Byte {byte_index} - Best: 0x{best_key:02x}')
        
        # Limit x-axis to improve readability
        axs[byte_index].set_xlim(-5, 260)
        
        # Mark the best key
        axs[byte_index].bar(best_key, max_correlations[best_key], color='green')
        
        # Mark the true key if provided
        if true_key is not None:
            # Check if it's a scalar (single byte) or array
            if isinstance(true_key, (int, np.integer)):
                # For scalar case (single key byte, typically in synthetic model)
                if byte_index == 0:  # Only mark for the first byte
                    true_byte = true_key
                    axs[byte_index].bar(true_byte, max_correlations[true_byte], color='red')
                    tick_color = 'green' if true_byte == best_key else 'red'
                    axs[byte_index].axvline(x=true_byte, color=tick_color, linestyle='--', alpha=0.6)
            else:
                # Convert to numpy array to ensure we can index properly
                true_key_array = np.asarray(true_key)
                if byte_index < len(true_key_array):
                    true_byte = true_key_array[byte_index]
                    axs[byte_index].bar(true_byte, max_correlations[true_byte], color='red')
                    tick_color = 'green' if true_byte == best_key else 'red'
                    axs[byte_index].axvline(x=true_byte, color=tick_color, linestyle='--', alpha=0.6)
    
    # Show the recovered key as hex string
    recovered_key_hex = ''.join(f'{byte:02x}' for byte in recovered_key)
    fig.suptitle(f'Full Key Recovery: 0x{recovered_key_hex}', fontsize=16)
    
    # Add true key if available
    if true_key is not None:
        if isinstance(true_key, (int, np.integer)):
            # Single byte case
            fig.text(0.5, 0.01, 
                    f'True Key Byte 0: 0x{true_key:02x}\nCorrect: {"Yes" if recovered_key[0] == true_key else "No"}', 
                    ha='center', fontsize=14)
        else:
            # Array case
            true_key_array = np.asarray(true_key)
            # Only create hex string up to the length we have
            true_key_hex = ''.join(f'{byte:02x}' for byte in true_key_array[:min(16, len(true_key_array))])
            # Count correct bytes
            correct_bytes = sum(1 for i in range(min(16, len(true_key_array))) 
                               if recovered_key[i] == true_key_array[i])
            fig.text(0.5, 0.01, 
                    f'True Key: 0x{true_key_hex}\nCorrect Bytes: {correct_bytes}/{min(16, len(true_key_array))} ({100*correct_bytes/min(16, len(true_key_array)):.1f}%)', 
                    ha='center', fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, bottom=0.06)
    
    return fig


from tqdm import tqdm
import pandas as pd

def get_fasta(fasta_path):
    """
    Read a FASTA file and store sequences.
    """
    fasta = {}
    with open(fasta_path) as file:
        sequence = ""
        for line in file:
            if line.startswith(">"):
                name = line[1:].rstrip()
                fasta[name] = ''
                continue
            fasta[name] += line.rstrip().upper()
    return fasta

def get_kmer_level_means(ref_sequence, model_file_path):
    """
    Calculate the mean signal levels for each k-mer in the reference sequence based on a model.
    """
    model_df = pd.read_csv(model_file_path, sep='\t')
    
    kmer_level_means = []
    for i in tqdm(range(len(ref_sequence) - 5), desc="Processing", unit="iteration"):
        kmer = ref_sequence[i:i+6]
        kmer_row = model_df[model_df['kmer'] == kmer]
        if not kmer_row.empty:
            kmer_level_mean = kmer_row['level_mean'].values[0]
            kmer_level_means.append(kmer_level_mean)
        else:
            kmer_level_means.append(None)
    
    return kmer_level_means

def load_data(filepath):
    # Speed up version
    df = pd.read_csv(filepath, sep='\t', usecols=['kmer', 'level_mean', 'level_stdv'])
    df.set_index('kmer', inplace=True)
    return df

def get_level_mean(df, kmer):
    # Speed up version
    try:
        return df.at[kmer, 'level_mean']
    except KeyError:
        return 0

def get_kmer_level_means_new(ref_sequence, df):
    # Speed up version
    kmer_level_means = []
    for i in tqdm(range(len(ref_sequence) - 5), desc="Processing", unit="iteration"):
        kmer = ref_sequence[i:i+6]
        kmer_level_mean=get_level_mean(df, kmer)
        kmer_level_means.append(kmer_level_mean)  
    return kmer_level_means

def read_fasta(file_path):
    """
    Read a FASTA file and return the sequence of the first record found.
    """
    fasta_data = get_fasta(file_path)
    for name, sequence in fasta_data.items():
        return sequence  # Return the first sequence found
    return ""  # Return an empty string if no sequences are found

def get_sequence_length(file_path):
    """
    Get the length of the sequence from the given FASTA file.
    """
    sequence = read_fasta(file_path)
    if sequence:
        return len(sequence)
    else:
        return 0

def get_kmer_properties(ref_seq, model_path):
    """
    Retrieve the mean and standard deviation of k-mers for a given reference sequence and model file path.
    :param ref_seq: The reference sequence
    :param model_path: The path to the model file
    :return: the normalized mean of k-mers
    """
    # kmer_level_means = get_kmer_level_means(ref_seq, model_path)
    kmer_level_means = get_kmer_level_means_new(ref_seq, model_path)
    kmer_level_means_shift = kmer_level_means[:]  
    for i in range(len(kmer_level_means)):
        kmer_level_means_shift[i] = (kmer_level_means[i] - 90.17)
        # kmer_level_means_shift[i] = (kmer_level_means[i] - 90.17)/12.83
    return kmer_level_means_shift

def replace_continuous_signal(arr, threshold=12):
    """
    12 same signal should all be mismatched in Analog CAM
    """

    if not arr:
        return arr
    
    new_arr = arr.copy()
    
    current_value = None
    current_count = 0
    start_index = 0

    for i in range(len(new_arr)):
        if new_arr[i] == current_value:
            current_count += 1
        else:
            if current_count >= threshold:
                new_arr[start_index:i] = [0] * current_count
            
            current_value = new_arr[i]
            current_count = 1
            start_index = i

    if current_count >= threshold:
        new_arr[start_index:] = [0] * current_count

    return new_arr

#Check replace_continuous_signal
def count_zeros(arr):
    return arr.count(0)

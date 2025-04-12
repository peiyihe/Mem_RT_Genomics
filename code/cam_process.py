from event_processor import read_event, filter_events
from cam_search import process_event,process_event_contamination, process_event_variation ,process_event_contamination_variation

def process_sample_variation(gon, goff, sample_number, sp, _read_id, fast5_file, random_matrix_tensor, reference_array_tensor, reference_array_comp_tensor, col, Threshold, sub_array_row, n_blocks, thresholds, device):


    event_original = read_event(sp, fast5_file, _read_id,sample_number)
    search_time = 0
    for threshold in thresholds:
        event_filtered = filter_events(event_original, threshold)
        event_length = min(13000, len(event_filtered))
        event = event_filtered[:2000]
        search_time = search_time + event_length - col
        if (event_length-col>0):
            final_location, final_location_comp, votes, votes_comp = process_event_variation(gon, goff, sample_number,random_matrix_tensor, event, reference_array_tensor, reference_array_comp_tensor, col, Threshold, sub_array_row, n_blocks, event_length, device)
        else:
            final_location = 'N'
            final_location_comp = 'N'
            votes = 'N'
            votes_comp = 'N'

        if final_location != 'N':
            return (final_location, '+', search_time, votes)
        elif final_location_comp != 'N':
            final_location_comp = n_blocks - final_location_comp
            return (final_location_comp, '-', search_time,votes_comp)
    
    return ('N', 'N', search_time, 'N')

def process_sample_contamination_variation(gon, goff, sample_number, sp, _read_id, fast5_file, random_matrix_tensor, reference_array_tensor, reference_array_comp_tensor, col, Threshold, sub_array_row, n_blocks, thresholds, device):
    """
    Processes a single sample's data, trying different thresholds, and returns the position and orientation.

    Parameters:
    - _read_id: The ID of the read
    - fast5_file: Path to the FAST5 file
    - random_matrix_tensor, reference_array_tensor, reference_array_comp_tensor: Tensors used
    - col, Threshold, sub_array_row, n_blocks: Parameters for the algorithm
    - thresholds: List of thresholds to try

    Returns:
    - (final_location, direction): The position and orientation
    """
    event_original = read_event(sp, fast5_file, _read_id,sample_number)
    search_time = 0

    for threshold in thresholds:
        event_filtered = filter_events(event_original, threshold)
        event_length = min(13000, len(event_filtered))
        event = event_filtered[:2000]
        search_time = search_time + event_length - col
        if event_length > col:
            final_location, final_location_comp, votes, votes_comp = process_event_contamination_variation(gon, goff, sample_number, random_matrix_tensor, event, reference_array_tensor, reference_array_comp_tensor, col, Threshold, sub_array_row, n_blocks, event_length, device)
        else:
            final_location = 'N'
            final_location_comp = 'N'
            votes = -1
            votes_comp = -1

        if final_location != 'N':
            return (final_location, '+', search_time, votes)
        elif final_location_comp != 'N':
            final_location_comp = n_blocks - final_location_comp
            return (final_location_comp, '-', search_time, votes_comp)
    
    return ('N', 'N', search_time,-1)

def process_sample(sample_number, sp, _read_id, fast5_file, random_matrix_tensor, reference_array_tensor, reference_array_comp_tensor, col, Threshold, sub_array_row, n_blocks, thresholds, device):
    """
    Processes a single sample's data, trying different thresholds, and returns the position and orientation.

    Parameters:
    - _read_id: The ID of the read
    - fast5_file: Path to the FAST5 file
    - random_matrix_tensor, reference_array_tensor, reference_array_comp_tensor: Tensors used
    - col, Threshold, sub_array_row, n_blocks: Parameters for the algorithm
    - thresholds: List of thresholds to try

    Returns:
    - (final_location, direction, search_time): The position and orientation, search time (evaluate latency)
    """
    event_original = read_event(sp, fast5_file, _read_id,sample_number)
    search_time = 0
    for threshold in thresholds:
        event_filtered = filter_events(event_original, threshold)
        event_length = min(13000, len(event_filtered))
        event = event_filtered[:2000]
        search_time = search_time + event_length - col
        if (event_length-col>0):
            final_location, final_location_comp, votes, votes_comp = process_event(sample_number,random_matrix_tensor, event, reference_array_tensor, reference_array_comp_tensor, col, Threshold, sub_array_row, n_blocks, event_length, device)
        else:
            final_location = 'N'
            final_location_comp = 'N'
            votes = 'N'
            votes_comp = 'N'

        if final_location != 'N':
            return (final_location, '+', search_time, votes)
        elif final_location_comp != 'N':
            final_location_comp = n_blocks - final_location_comp
            return (final_location_comp, '-', search_time,votes_comp)
    
    return ('N', 'N', search_time, 'N')

def process_sample_contamination(sample_number, sp, _read_id, fast5_file, random_matrix_tensor, reference_array_tensor, reference_array_comp_tensor, col, Threshold, sub_array_row, n_blocks, thresholds, device):
    """
    Processes a single sample's data, trying different thresholds, and returns the position and orientation.

    Parameters:
    - _read_id: The ID of the read
    - fast5_file: Path to the FAST5 file
    - random_matrix_tensor, reference_array_tensor, reference_array_comp_tensor: Tensors used
    - col, Threshold, sub_array_row, n_blocks: Parameters for the algorithm
    - thresholds: List of thresholds to try

    Returns:
    - (final_location, direction): The position and orientation
    """
    event_original = read_event(sp, fast5_file, _read_id,sample_number)
    search_time = 0

    for threshold in thresholds:
        event_filtered = filter_events(event_original, threshold)
        event_length = min(13000, len(event_filtered))
        event = event_filtered[:2000]
        search_time = search_time + event_length - col
        if event_length > col:
            final_location, final_location_comp, votes, votes_comp = process_event_contamination(sample_number, random_matrix_tensor, event, reference_array_tensor, reference_array_comp_tensor, col, Threshold, sub_array_row, n_blocks, event_length, device)
        else:
            final_location = 'N'
            final_location_comp = 'N'
            votes = 'N'
            votes_comp = 'N'

        if final_location != 'N':
            return (final_location, '+', search_time, votes)
        elif final_location_comp != 'N':
            final_location_comp = n_blocks - final_location_comp
            return (final_location_comp, '-', search_time, votes_comp)
    
    return ('N', 'N', search_time,'N')

from tqdm import tqdm

def update_position_variation(gon, goff, sample_number, position, direction, search_time, vote_location, sp, read_id, fast5_file, index, thresholds, random_matrix_tensor, reference_array_tensor, reference_array_comp_tensor, col, Threshold, sub_array_row, n_blocks, device):
    """
    Update position and orientation, try different filtering thresholds.

    Parameters:
    - read_id: List of read IDs
    - fast5_file: Path to the FAST5 file
    - index: List of indexes to process
    - thresholds: List of thresholds
    - random_matrix_tensor, reference_array_tensor, reference_array_comp_tensor: Tensors used
    - col, Threshold, sub_array_row, n_blocks: Algorithm parameters
    """
    for i in tqdm(index):
        _read_id = read_id[i]
        for threshold in thresholds:
            event_new = read_event(sp, fast5_file, _read_id, sample_number)
            event_new = filter_events(event_new, threshold)
            event_length = min(13000, len(event_new))
            event = event_new[:2000]

            if event_length < 2000 and event_length > col:
                update_search_time = event_length
                final_location, final_location_comp, votes, votes_comp = process_event_variation(gon, goff, sample_number, random_matrix_tensor, event, reference_array_tensor, reference_array_comp_tensor, col, Threshold, sub_array_row, n_blocks, event_length, device)
                search_time[i] = search_time[i] + update_search_time
                if final_location != 'N':
                    # print("found:", i)
                    position[i] = final_location
                    direction[i] = '+'
                    vote_location[i] = votes
                    break
                elif final_location_comp != 'N':
                    final_location_comp = n_blocks - final_location_comp
                    # print("found:", i)
                    position[i] = final_location_comp
                    direction[i] = '-'
                    vote_location[i] = votes_comp
                    break

def update_position_contamination_variation(gon, goff, sample_number, position, direction, search_time, vote_location, sp, read_id, fast5_file, index, thresholds, random_matrix_tensor, reference_array_tensor, reference_array_comp_tensor, col, Threshold, sub_array_row, n_blocks, device):
    """
    Update position and orientation, try different filtering thresholds.

    Parameters:
    - read_id: List of read IDs
    - fast5_file: Path to the FAST5 file
    - index: List of indexes to process
    - thresholds: List of thresholds
    - random_matrix_tensor, reference_array_tensor, reference_array_comp_tensor: Tensors used
    - col, Threshold, sub_array_row, n_blocks: Algorithm parameters
    """
    for i in tqdm(index, mininterval=300):
        _read_id = read_id[i]
        for threshold in thresholds:
            event_new = read_event(sp, fast5_file, _read_id, sample_number)
            event_new = filter_events(event_new, threshold)
            event_length = min(13000, len(event_new))
            event = event_new[:2000]

            if event_length < 2000 and event_length>col:    #some reads are extremely short
                update_search_time = event_length
                final_location, final_location_comp, votes, votes_comp = process_event_contamination_variation(gon, goff, sample_number, random_matrix_tensor, event, reference_array_tensor, reference_array_comp_tensor, col, Threshold, sub_array_row, n_blocks, event_length, device)
                search_time[i] = search_time[i] + update_search_time                
                if final_location != 'N':
                    # print("found:", i)
                    position[i] = final_location
                    direction[i] = '+'
                    vote_location[i] = votes
                    break
                elif final_location_comp != 'N':
                    final_location_comp = n_blocks - final_location_comp
                    # print("found:", i)
                    position[i] = final_location_comp
                    direction[i] = '-'
                    vote_location[i] = votes_comp
                    break

def update_position(sample_number, position, direction, search_time, vote_location, sp, read_id, fast5_file, index, thresholds, random_matrix_tensor, reference_array_tensor, reference_array_comp_tensor, col, Threshold, sub_array_row, n_blocks, device):
    """
    Update position and orientation, try different filtering thresholds.

    Parameters:
    - read_id: List of read IDs
    - fast5_file: Path to the FAST5 file
    - index: List of indexes to process
    - thresholds: List of thresholds
    - random_matrix_tensor, reference_array_tensor, reference_array_comp_tensor: Tensors used
    - col, Threshold, sub_array_row, n_blocks: Algorithm parameters
    """
    for i in tqdm(index, mininterval=300):
        _read_id = read_id[i]
        for threshold in thresholds:
            event_new = read_event(sp, fast5_file, _read_id, sample_number)
            event_new = filter_events(event_new, threshold)
            event_length = min(13000, len(event_new))
            event = event_new[:2000]

            if event_length < 2000 and event_length > col:
                update_search_time = event_length
                final_location, final_location_comp, votes, votes_comp = process_event(sample_number, random_matrix_tensor, event, reference_array_tensor, reference_array_comp_tensor, col, Threshold, sub_array_row, n_blocks, event_length, device)
                search_time[i] = search_time[i] + update_search_time
                if final_location != 'N':
                    # print("found:", i)
                    position[i] = final_location
                    direction[i] = '+'
                    vote_location[i] = votes
                    break
                elif final_location_comp != 'N':
                    final_location_comp = n_blocks - final_location_comp
                    # print("found:", i)
                    position[i] = final_location_comp
                    direction[i] = '-'
                    vote_location[i] = votes_comp
                    break

def update_position_contamination(sample_number, position, direction, search_time, vote_location, sp, read_id, fast5_file, index, thresholds, random_matrix_tensor, reference_array_tensor, reference_array_comp_tensor, col, Threshold, sub_array_row, n_blocks, device):
    """
    Update position and orientation, try different filtering thresholds.

    Parameters:
    - read_id: List of read IDs
    - fast5_file: Path to the FAST5 file
    - index: List of indexes to process
    - thresholds: List of thresholds
    - random_matrix_tensor, reference_array_tensor, reference_array_comp_tensor: Tensors used
    - col, Threshold, sub_array_row, n_blocks: Algorithm parameters
    """
    for i in tqdm(index, mininterval=300):
        _read_id = read_id[i]
        for threshold in thresholds:
            event_new = read_event(sp, fast5_file, _read_id, sample_number)
            event_new = filter_events(event_new, threshold)
            event_length = min(13000, len(event_new))
            event = event_new[:2000]

            if event_length < 2000 and event_length>col:    #some reads are extremely short
                update_search_time = event_length
                final_location, final_location_comp, votes, votes_comp = process_event_contamination(sample_number, random_matrix_tensor, event, reference_array_tensor, reference_array_comp_tensor, col, Threshold, sub_array_row, n_blocks, event_length, device)
                search_time[i] = search_time[i] + update_search_time                
                if final_location != 'N':
                    # print("found:", i)
                    position[i] = final_location
                    direction[i] = '+'
                    vote_location[i] = votes
                    break
                elif final_location_comp != 'N':
                    final_location_comp = n_blocks - final_location_comp
                    # print("found:", i)
                    position[i] = final_location_comp
                    direction[i] = '-'
                    vote_location[i] = votes_comp
                    break

def process_location(sample_number, sp, low_boundary, high_boundary, read_id, read_number, positions, fast5_file, sub_array_row):
    
    for i in tqdm(range(0,read_number), disable=True):
        _read_id = read_id[i]
        event_original = read_event(sp, fast5_file, _read_id, sample_number)
        event_4 = filter_events(event_original, 4)
        event_length = len(event_4)
        event_length = max(event_length, sub_array_row)
        
        if positions[i] != 'N':
            pos = int(positions[i])
            high = pos * sub_array_row + 3 * event_length
            
            if (pos * sub_array_row - 3 * event_length > 0):
                low = pos * sub_array_row - 3 * event_length
            elif (pos * sub_array_row - 2 * event_length > 0):
                low = pos * sub_array_row - 2 * event_length
            elif (pos * sub_array_row - 1 * event_length > 0):
                low = pos * sub_array_row - 1 * event_length
            else:
                low = 0  # Adjusted to ensure low is not negative
            
            low_boundary.append(low)
            high_boundary.append(high)
        
        else:
            low_boundary.append('*')
            high_boundary.append('*')
    
    return low_boundary, high_boundary
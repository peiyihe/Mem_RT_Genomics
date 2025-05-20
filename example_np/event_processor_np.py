#import UNCALLED4
from uncalled4 import PoreModel, Config, EventDetector, SignalProcessor
from ont_fast5_api.fast5_interface import get_fast5_file
from tqdm import tqdm
import h5py

def EventProcessor():
    # first parameter
    m = PoreModel("dna_r9.4.1_400bps_6mer")
    c = Config()
    c.event_detector = EventDetector.PRMS_450BPS #or EventDetector.PRMS_70BPS
    c.event_detector.max_mean = 4000
    sp = SignalProcessor(m, c)

    # second parameter
    m1 = PoreModel("dna_r9.4.1_400bps_6mer")
    c1 = Config()
    c1.event_detector = EventDetector.PRMS_450BPS #or EventDetector.PRMS_70BPS
    c1.event_detector.max_mean = 4000
    c1.event_detector.threshold1 = 4.30265
    c1.event_detector.threshold2 = 2.57058
    c1.event_detector.peak_height = 1
    sp1 = SignalProcessor(m1, c1)

    return sp, sp1

def read_event(sp, file, id, sample_number):
    """ Extracts and processes events from a given FAST5 file using the specified signal processor. """
    with get_fast5_file(file, mode='r') as f5:
        read_id = id
        _read = f5.get_read(read_id)
        signal = _read.get_raw_data(scale=True)  # Scale signal to unit amplitude
        signal = signal[0:sample_number]
    read = sp.process_signal(signal, normalize=True)  # Signal should be numpy array/list of raw sample values
    return read.events["mean"]

def filter_events(read_events, difference):
    """ Filters events based on a difference threshold between consecutive events. """
    test_events = []
    for i in range(1, len(read_events)-1):
        if abs(read_events[i] - read_events[i-1]) > difference:
            test_events.append(read_events[i])
    # test_events = [(x - 90.17)/12.83 for x in test_events]
    test_events = [(x - 90.17) for x in test_events]
    return test_events

def fast5_id_list(fast5_file):
    id = []
    with h5py.File(fast5_file, 'r') as f:
        first_level_names = list(f.keys())
        processed_names = [name.replace('read_', '') for name in first_level_names]
        for name in processed_names:
            id.append(name)
    return id
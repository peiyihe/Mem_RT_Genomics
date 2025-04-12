import numpy as np

class CAMArray:
    """
    Class to handle matrix initialization and processing for CAM operations.
    
    Parameters:
        col (int): Number of columns for the matrix.
        LSH_col (int): Number of columns for LSH.
        k (int): Parameter for matrix subdivision.
        sub_array_row (int): Number of sub-array rows.
        kmer_level_means_shift (array): Shift values for k-mer levels.
        kmer_level_means_shift_comp (array): Complementary shift values for k-mer levels.
        random_matrix (array): Random matrix for transformation.
        device (str): Device for CUDA operations (default: 'cuda').
    """
    def __init__(self, col, LSH_col, k, sub_array_row, kmer_level_means_shift, kmer_level_means_shift_comp, random_matrix, device='cuda'):
        self.col = col
        self.LSH_col = LSH_col
        self.k = k
        self.sub_array_row = sub_array_row
        self.kmer_level_means_shift = kmer_level_means_shift
        self.kmer_level_means_shift_comp = kmer_level_means_shift_comp
        self.random_matrix = random_matrix
        self.device = device
        self.reference_array = None
        self.reference_array_comp = None

    def initialize_matrices(self):
        length = self.col // self.k * self.sub_array_row
        row = (len(self.kmer_level_means_shift)) // (self.col // self.k) - self.k
        
        self.reference_array = np.full((row, self.col), 0, dtype=float)
        self.reference_array_comp = np.full((row, self.col), 0, dtype=float)
        
        for i in range(row):
            row_data = self.kmer_level_means_shift[i * (self.col // self.k): i * (self.col // self.k) + self.col]
            self.reference_array[i] = row_data
            row_data_comp = self.kmer_level_means_shift_comp[i * (self.col // self.k): i * (self.col // self.k) + self.col]
            self.reference_array_comp[i] = row_data_comp
            
        # np.random.seed(42)
        # self.random_matrix = np.random.normal(size=(self.col, self.LSH_col))

    def process_LSH(self):
        LSH_reference = np.dot(self.reference_array, self.random_matrix)
        LSH_reference = (LSH_reference > 0).astype(int)
        LSH_reference_comp = np.dot(self.reference_array_comp, self.random_matrix)
        LSH_reference_comp = (LSH_reference_comp > 0).astype(int)
        
        # reference_array_tensor = torch.tensor(LSH_reference, device=self.device).float()
        # reference_array_comp_tensor = torch.tensor(LSH_reference_comp, device=self.device).float() 
        return LSH_reference, LSH_reference_comp

    def print_shapes(self):
        print("Reference Array Shape:", self.reference_array.shape)
        print("Reference Comp Array Shape:", self.reference_array_comp.shape)

    def calculate_sub_arrays(self, tensor):
        n_blocks = tensor.shape[0] // self.sub_array_row
        return n_blocks
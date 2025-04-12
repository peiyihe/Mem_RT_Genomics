## NumPy implementation

This implementation is CPU-only. For faster performance, please use the GPU-accelerated version.

**Key Differences from GPU Version:**

1. Memristor variation effects are not modeled in this simulation.
2. Due to significantly slower speeds (~50x slower than GPU implementation), we limit the simulation to 100 read operations.

## Installation

Create CPU environment and install:

```shell
conda create --name test_lsh_np python=3.8
conda activate test_lsh_np
pip install -r requirements_np.txt
```

## Simulation results

Run `LSH_Raw_Signal_Alignment_np_v2.ipynb`

**Example:**

```shell
Summary: 100 reads, 94 mapped (94.00%)

Comparing to reference PAF
     P     N
T  93.00  0.00
F   1.00  6.00
NA: 0.00

Recall: 93.94

Precision: 98.94

F1: 96.37
```

**NA** means minimap2 doesn't report the location but our method does. It is considered as False Positives here.
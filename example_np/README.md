## NumPy implementation

This implementation is CPU-only. For faster performance, please use the GPU-accelerated version.

**Key Differences from GPU Version:**

1. Memristor variation effects are not modeled in this simulation.
2. Due to significantly slower speeds (~50x slower than GPU implementation), we limit the simulation to 1000 read operations.

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
Summary: 1000 reads, 956 mapped (95.60%)

Comparing to reference PAF
     P     N
T  94.80  0.00
F   0.80  4.40
NA: 0.00

Recall: 95.56

Precision: 99.16

F1: 97.33
```

**NA** means minimap2 doesn't report the location but our method does. It is considered as False Positives here. 
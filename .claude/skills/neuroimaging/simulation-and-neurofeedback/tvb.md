# The Virtual Brain (TVB)

## Overview

The Virtual Brain (TVB) is an open-source neuroinformatics platform for simulating large-scale brain network dynamics using empirically-derived structural connectivity. TVB bridges the gap between brain structure and function by enabling researchers to build personalized whole-brain computational models from individual neuroimaging data, simulate neural dynamics using biophysical models, and validate predictions against empirical fMRI, MEG, and EEG recordings.

**Website:** https://www.thevirtualbrain.org/
**Platform:** Python (Windows/macOS/Linux)
**Language:** Python
**License:** GPL v3

## Key Features

- Whole-brain network simulation with multiple biophysical models
- Integration of structural connectivity from DTI/tractography
- Neural mass and neural field models (Wilson-Cowan, Kuramoto, Jansen-Rit, Epileptor)
- Forward modeling for fMRI BOLD, MEG, and EEG signals
- Interactive web-based GUI and Python scripting API
- Parameter space exploration and optimization
- Personalized brain models from individual connectivity
- Disease modeling (epilepsy, stroke, tumors, Alzheimer's)
- Virtual lesion studies and surgical planning
- Multi-scale modeling from regions to neurons
- Built-in atlases and connectivity datasets
- 3D visualization of brain dynamics

## Installation

### Using pip

```bash
# Install TVB framework (Python API)
pip install tvb-framework

# Install TVB library (core simulation engine)
pip install tvb-library

# Install TVB data (atlases and connectivity)
pip install tvb-data

# Install all TVB components
pip install tvb-framework tvb-library tvb-data tvb-contrib
```

### Using conda

```bash
# Create TVB environment
conda create -n tvb python=3.10
conda activate tvb

# Install TVB
conda install -c conda-forge tvb-framework tvb-library tvb-data
```

### Using Docker

```bash
# Pull TVB Docker image
docker pull thevirtualbrain/tvb-run

# Run TVB container with web interface
docker run -d -p 8080:8080 -v ~/TVB:/home/tvb_user/TVB thevirtualbrain/tvb-run

# Access TVB at http://localhost:8080
```

### Verify Installation

```python
import tvb
print(f"TVB version: {tvb.__version__}")

from tvb.simulator.lab import *
print("TVB simulator imported successfully")
```

## Building Connectivity Models

### Load Default Connectivity

```python
from tvb.simulator.lab import connectivity

# Load default connectivity (76 regions)
conn = connectivity.Connectivity.from_file()

print(f"Regions: {conn.number_of_regions}")
print(f"Region labels: {conn.region_labels[:5]}")
print(f"Weights shape: {conn.weights.shape}")
print(f"Tract lengths shape: {conn.tract_lengths.shape}")

# Connectivity properties
print(f"Average degree: {conn.weights.sum(axis=1).mean():.2f}")
```

### Load Different Atlases

```python
from tvb.datatypes import connectivity

# Available connectivity datasets
from tvb.simulator.lab import connectivity

# Load 76-region connectivity (default)
conn_76 = connectivity.Connectivity.from_file()

# Load 192-region connectivity
conn_192 = connectivity.Connectivity.from_file(
    source_file="connectivity_192.zip"
)

# Load 998-region connectivity (high resolution)
conn_998 = connectivity.Connectivity.from_file(
    source_file="connectivity_998.zip"
)

print(f"76 regions: {conn_76.number_of_regions}")
print(f"192 regions: {conn_192.number_of_regions}")
print(f"998 regions: {conn_998.number_of_regions}")
```

### Import Custom Connectivity

```python
import numpy as np
from tvb.datatypes import connectivity

# Create custom connectivity from your tractography
# Weights: structural connectivity matrix (NxN)
# Tract lengths: fiber lengths in mm (NxN)
# Region labels: list of region names
# Centers: region coordinates in mm (Nx3)

weights = np.load('my_sc_matrix.npy')
tract_lengths = np.load('my_tract_lengths.npy')
region_labels = np.loadtxt('region_labels.txt', dtype=str)
centers = np.load('region_centers.npy')

# Create connectivity object
conn = connectivity.Connectivity(
    weights=weights,
    tract_lengths=tract_lengths,
    region_labels=region_labels,
    centres=centers
)

# Configure connectivity
conn.configure()

# Save for future use
conn.to_file('my_connectivity.zip')
```

### Visualize Connectivity

```python
import matplotlib.pyplot as plt
import numpy as np

# Plot connectivity matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Structural connectivity weights
im1 = axes[0].imshow(conn.weights, cmap='hot', interpolation='nearest')
axes[0].set_title('Structural Connectivity')
axes[0].set_xlabel('Region')
axes[0].set_ylabel('Region')
plt.colorbar(im1, ax=axes[0], label='Weight')

# Tract lengths
im2 = axes[1].imshow(conn.tract_lengths, cmap='viridis', interpolation='nearest')
axes[1].set_title('Tract Lengths (mm)')
axes[1].set_xlabel('Region')
axes[1].set_ylabel('Region')
plt.colorbar(im2, ax=axes[1], label='Length (mm)')

plt.tight_layout()
plt.savefig('connectivity_visualization.png', dpi=300)
```

## Neural Mass Models

### Wilson-Cowan Oscillator

```python
from tvb.simulator.lab import models

# Wilson-Cowan model (excitatory-inhibitory)
wc = models.WilsonCowan()

# Model parameters
print("Default parameters:")
print(f"c_ee (E-E coupling): {wc.c_ee}")
print(f"c_ei (E-I coupling): {wc.c_ei}")
print(f"c_ie (I-E coupling): {wc.c_ie}")
print(f"c_ii (I-I coupling): {wc.c_ii}")
print(f"tau_e (E time constant): {wc.tau_e}")
print(f"tau_i (I time constant): {wc.tau_i}")

# Customize parameters
wc.c_ee = 12.0
wc.c_ei = 4.0
wc.c_ie = 13.0
wc.c_ii = 11.0
wc.a_e = 1.2
wc.a_i = 2.0
```

### Generic 2D Oscillator

```python
from tvb.simulator.lab import models

# Generic 2D oscillator (flexible dynamics)
g2d = models.Generic2dOscillator()

# Parameters for limit cycle
g2d.a = -0.5  # Bifurcation parameter
g2d.b = -10.0
g2d.c = 0.0
g2d.d = 0.02
g2d.e = 3.0
g2d.f = 1.0
g2d.g = 0.0
g2d.alpha = 1.0
g2d.beta = 1.0
g2d.tau = 1.0

print(f"State variables: {g2d.state_variables}")
print(f"Number of state variables: {g2d.nvar}")
```

### Kuramoto Oscillator

```python
from tvb.simulator.lab import models

# Kuramoto phase oscillator
kuramoto = models.Kuramoto()

# Natural frequency
kuramoto.omega = 1.0  # radians/ms

print("Kuramoto oscillator configured")
print(f"State variables: {kuramoto.state_variables}")
```

### Epileptor Model

```python
from tvb.simulator.lab import models

# Epileptor for seizure modeling
epileptor = models.Epileptor()

# Seizure parameters
epileptor.x0 = -1.6  # Excitability (< -2: healthy, > -1.6: seizure-prone)
epileptor.Iext = 3.1  # External input
epileptor.Iext2 = 0.45
epileptor.r = 0.00035  # Permittivity coupling
epileptor.slope = 0.0

print("Epileptor model for seizure simulation")
print(f"Excitability (x0): {epileptor.x0}")
print(f"Number of state variables: {epileptor.nvar}")
```

### Jansen-Rit Model

```python
from tvb.simulator.lab import models

# Jansen-Rit neural mass model
jr = models.JansenRit()

# Pyramidal cell parameters
jr.A = 3.25  # Maximum amplitude of EPSP
jr.B = 22.0  # Maximum amplitude of IPSP
jr.a = 0.1   # Inverse time constant of EPSP
jr.b = 0.05  # Inverse time constant of IPSP

# Connectivity parameters
jr.v0 = 5.52  # Firing threshold
jr.nu_max = 0.0025  # Maximum firing rate
jr.r = 0.56  # Sigmoid steepness

print("Jansen-Rit model configured")
```

## Simulation Configuration

### Basic Simulation Setup

```python
from tvb.simulator.lab import *
import numpy as np

# Load connectivity
conn = connectivity.Connectivity.from_file()
conn.speed = 4.0  # Conduction velocity (mm/ms)

# Choose model
model = models.Generic2dOscillator()

# Coupling function
coupling = coupling.Linear(a=0.0152)

# Integration scheme
heunint = integrators.HeunDeterministic(dt=2**-4)

# Monitors
mon_tavg = monitors.TemporalAverage(period=2**-2)
mon_raw = monitors.Raw()

# Create simulator
sim = simulator.Simulator(
    model=model,
    connectivity=conn,
    coupling=coupling,
    integrator=heunint,
    monitors=[mon_tavg, mon_raw]
)

# Configure simulator
sim.configure()

print("Simulator configured successfully")
print(f"Simulation timestep: {sim.integrator.dt} ms")
print(f"Number of regions: {sim.connectivity.number_of_regions}")
```

### Add Noise

```python
from tvb.simulator.lab import noise

# Additive white noise
white_noise = noise.Additive(nsig=np.array([0.01]))

# Configure simulator with noise
sim = simulator.Simulator(
    model=model,
    connectivity=conn,
    coupling=coupling,
    integrator=heunint,
    monitors=[mon_tavg],
    noise=white_noise
)

sim.configure()
```

### Configure BOLD Monitor

```python
from tvb.simulator.lab import monitors

# BOLD monitor for fMRI simulation
bold_monitor = monitors.Bold(period=2000.0)  # 2 seconds TR

# Balloon-Windkessel parameters
bold_monitor.hrf_kernel = monitors.Bold.compute_hrf()

# Multiple monitors
mon_bold = monitors.Bold(period=2000.0)
mon_eeg = monitors.EEG()
mon_tavg = monitors.TemporalAverage(period=1.0)

sim = simulator.Simulator(
    model=model,
    connectivity=conn,
    coupling=coupling,
    integrator=heunint,
    monitors=[mon_bold, mon_eeg, mon_tavg]
)

sim.configure()
```

## Running Simulations

### Basic Simulation

```python
# Run simulation
simulation_length = 1000.0  # ms

# Execute simulation
(tavg_time, tavg_data), = sim.run(simulation_length=simulation_length)

print(f"Simulation complete")
print(f"Data shape: {tavg_data.shape}")  # (time, state_var, nodes, modes)
print(f"Time points: {len(tavg_time)}")
```

### Long Simulation with Progress

```python
import numpy as np

# Longer simulation with chunked execution
simulation_length = 10000.0  # 10 seconds
chunk_length = 1000.0

all_time = []
all_data = []

for i in range(int(simulation_length / chunk_length)):
    (tavg_time, tavg_data), = sim.run(simulation_length=chunk_length)

    all_time.append(tavg_time)
    all_data.append(tavg_data)

    print(f"Completed {(i+1)*chunk_length} ms")

# Concatenate results
full_time = np.concatenate(all_time)
full_data = np.concatenate(all_data, axis=0)

print(f"Total simulation: {full_time[-1]:.2f} ms")
print(f"Final data shape: {full_data.shape}")
```

### BOLD Simulation

```python
from tvb.simulator.lab import *

# Setup for BOLD simulation
conn = connectivity.Connectivity.from_file()
conn.speed = 4.0

model = models.Generic2dOscillator()
coupling = coupling.Linear(a=0.0152)
heunint = integrators.HeunDeterministic(dt=2**-4)

# BOLD monitor with 2-second TR
mon_bold = monitors.Bold(period=2000.0)

sim = simulator.Simulator(
    model=model,
    connectivity=conn,
    coupling=coupling,
    integrator=heunint,
    monitors=[mon_bold]
)

sim.configure()

# Simulate 5 minutes of resting-state
(bold_time, bold_data), = sim.run(simulation_length=5*60*1000.0)

print(f"BOLD time points: {len(bold_time)}")
print(f"BOLD data shape: {bold_data.shape}")  # (timepoints, 1, regions, 1)

# Extract BOLD signal per region
bold_signal = bold_data[:, 0, :, 0]  # (time, regions)
print(f"BOLD signal shape: {bold_signal.shape}")
```

### Parameter Sweep

```python
import numpy as np

# Sweep coupling strength
coupling_values = np.linspace(0.005, 0.02, 5)
results = {}

for c_val in coupling_values:
    # Configure coupling
    coupling = coupling.Linear(a=c_val)

    # Create simulator
    sim = simulator.Simulator(
        model=model,
        connectivity=conn,
        coupling=coupling,
        integrator=heunint,
        monitors=[mon_tavg]
    )
    sim.configure()

    # Run simulation
    (tavg_time, tavg_data), = sim.run(simulation_length=1000.0)

    # Store results
    results[c_val] = tavg_data
    print(f"Completed coupling = {c_val:.4f}")

print(f"Parameter sweep complete: {len(results)} conditions")
```

## Analysis and Visualization

### Time Series Visualization

```python
import matplotlib.pyplot as plt

# Plot time series for first 5 regions
fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

for i in range(5):
    axes[i].plot(tavg_time, tavg_data[:, 0, i, 0], linewidth=0.5)
    axes[i].set_ylabel(f'{conn.region_labels[i]}')
    axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (ms)')
axes[0].set_title('Regional Time Series')
plt.tight_layout()
plt.savefig('time_series.png', dpi=300)
```

### Functional Connectivity from Simulation

```python
import numpy as np

# Extract time series (time x regions)
time_series = tavg_data[:, 0, :, 0]

# Compute functional connectivity (Pearson correlation)
fc_matrix = np.corrcoef(time_series.T)

# Visualize FC
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Structural connectivity
im1 = axes[0].imshow(conn.weights, cmap='hot', interpolation='nearest')
axes[0].set_title('Structural Connectivity')
plt.colorbar(im1, ax=axes[0])

# Functional connectivity
im2 = axes[1].imshow(fc_matrix, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
axes[1].set_title('Simulated Functional Connectivity')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig('sc_vs_fc.png', dpi=300)
```

### Power Spectrum Analysis

```python
from scipy import signal

# Compute power spectral density
fs = 1000.0 / (tavg_time[1] - tavg_time[0])  # Sampling frequency

# Average across regions
mean_signal = time_series.mean(axis=1)

# Compute PSD
freqs, psd = signal.welch(mean_signal, fs=fs, nperseg=256)

# Plot
plt.figure(figsize=(10, 4))
plt.semilogy(freqs, psd)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.title('Power Spectral Density')
plt.grid(True, alpha=0.3)
plt.xlim([0, 50])
plt.savefig('power_spectrum.png', dpi=300)
```

### Compare Simulated vs Empirical FC

```python
import numpy as np
from scipy.stats import pearsonr

# Load empirical FC
empirical_fc = np.load('empirical_fc.npy')

# Compute simulated FC
simulated_fc = np.corrcoef(time_series.T)

# Extract upper triangle (excluding diagonal)
mask = np.triu(np.ones_like(empirical_fc), k=1).astype(bool)
emp_fc_vec = empirical_fc[mask]
sim_fc_vec = simulated_fc[mask]

# Correlation between empirical and simulated FC
r, p = pearsonr(emp_fc_vec, sim_fc_vec)

print(f"Correlation: r = {r:.3f}, p = {p:.2e}")

# Scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(emp_fc_vec, sim_fc_vec, alpha=0.3, s=5)
plt.xlabel('Empirical FC')
plt.ylabel('Simulated FC')
plt.title(f'FC Comparison (r = {r:.3f})')
plt.plot([-1, 1], [-1, 1], 'r--', label='Identity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('fc_comparison.png', dpi=300)
```

## Personalized Brain Models

### Load Subject-Specific Connectivity

```python
import numpy as np
from tvb.datatypes import connectivity

# Load structural connectivity from tractography
sc_matrix = np.load('subject_01_sc.npy')
tract_lengths = np.load('subject_01_lengths.npy')

# Load parcellation info
region_labels = np.loadtxt('subject_01_labels.txt', dtype=str)
region_centers = np.load('subject_01_centers.npy')

# Create connectivity
conn = connectivity.Connectivity(
    weights=sc_matrix,
    tract_lengths=tract_lengths,
    region_labels=region_labels,
    centres=region_centers,
    speed=4.0
)

conn.configure()

print(f"Subject-specific connectivity: {conn.number_of_regions} regions")
print(f"Mean connectivity strength: {conn.weights.mean():.3f}")
```

### Virtual Lesion Study

```python
import numpy as np

# Original connectivity
conn_healthy = connectivity.Connectivity.from_file()

# Create lesioned connectivity (remove region)
lesion_idx = 10  # Index of region to lesion

conn_lesion = connectivity.Connectivity.from_file()
conn_lesion.weights[lesion_idx, :] = 0
conn_lesion.weights[:, lesion_idx] = 0
conn_lesion.tract_lengths[lesion_idx, :] = 0
conn_lesion.tract_lengths[:, lesion_idx] = 0

conn_lesion.configure()

# Simulate healthy brain
sim_healthy = simulator.Simulator(
    model=models.Generic2dOscillator(),
    connectivity=conn_healthy,
    coupling=coupling.Linear(a=0.0152),
    integrator=integrators.HeunDeterministic(dt=2**-4),
    monitors=[monitors.TemporalAverage(period=1.0)]
)
sim_healthy.configure()

# Simulate lesioned brain
sim_lesion = simulator.Simulator(
    model=models.Generic2dOscillator(),
    connectivity=conn_lesion,
    coupling=coupling.Linear(a=0.0152),
    integrator=integrators.HeunDeterministic(dt=2**-4),
    monitors=[monitors.TemporalAverage(period=1.0)]
)
sim_lesion.configure()

# Run both simulations
(_, data_healthy), = sim_healthy.run(simulation_length=5000.0)
(_, data_lesion), = sim_lesion.run(simulation_length=5000.0)

print("Virtual lesion simulation complete")
```

### Patient-Specific Epilepsy Model

```python
from tvb.simulator.lab import *

# Patient connectivity
conn = connectivity.Connectivity.from_file()

# Epileptor model
epileptor = models.Epileptor()

# Define epileptogenic zone (region indices)
ez_regions = [15, 16, 17]  # Example: temporal lobe

# Set heterogeneous excitability
x0_values = np.ones(conn.number_of_regions) * -2.2  # Healthy
x0_values[ez_regions] = -1.5  # Epileptogenic

# Configure model with heterogeneous parameters
epileptor.x0 = x0_values

# Setup simulation
coupling = coupling.Difference(a=1.0)
heunint = integrators.HeunStochastic(dt=0.05)

mon = monitors.TemporalAverage(period=1.0)

sim = simulator.Simulator(
    model=epileptor,
    connectivity=conn,
    coupling=coupling,
    integrator=heunint,
    monitors=[mon]
)

sim.configure()

# Simulate seizure propagation
(time, data), = sim.run(simulation_length=10000.0)

print("Seizure simulation complete")
print(f"Epileptogenic zone: {[conn.region_labels[i] for i in ez_regions]}")
```

## Integration with Neuroimaging Pipelines

### Load Connectivity from MRtrix3

```python
import numpy as np
from tvb.datatypes import connectivity

# Load MRtrix3 outputs
# Structural connectivity matrix from tck2connectome
sc_matrix = np.loadtxt('connectome.csv', delimiter=',')

# Tract lengths (mean length per connection)
lengths = np.loadtxt('mean_lengths.csv', delimiter=',')

# Load parcellation labels
with open('parcellation_labels.txt', 'r') as f:
    labels = [line.strip() for line in f]

# Load region centers (from label centroids)
centers = np.loadtxt('region_centers.txt')

# Create TVB connectivity
conn = connectivity.Connectivity(
    weights=sc_matrix,
    tract_lengths=lengths,
    region_labels=np.array(labels),
    centres=centers,
    speed=4.0
)

conn.configure()
print(f"Connectivity from MRtrix3: {conn.number_of_regions} regions")
```

### Validate Against Empirical fMRI

```python
import numpy as np
from scipy.stats import pearsonr

# Load empirical resting-state fMRI data
empirical_bold = np.load('empirical_bold.npy')  # (time, regions)

# Simulate BOLD
from tvb.simulator.lab import *

conn = connectivity.Connectivity.from_file()
model = models.Generic2dOscillator()
coupling = coupling.Linear(a=0.0152)
heunint = integrators.HeunDeterministic(dt=2**-4)
mon_bold = monitors.Bold(period=2000.0)

sim = simulator.Simulator(
    model=model,
    connectivity=conn,
    coupling=coupling,
    integrator=heunint,
    monitors=[mon_bold]
)
sim.configure()

# Match empirical duration
duration = empirical_bold.shape[0] * 2000.0  # TR = 2s
(bold_time, bold_data), = sim.run(simulation_length=duration)

simulated_bold = bold_data[:, 0, :, 0]

# Compute functional connectivity
emp_fc = np.corrcoef(empirical_bold.T)
sim_fc = np.corrcoef(simulated_bold.T)

# Compare
mask = np.triu(np.ones_like(emp_fc), k=1).astype(bool)
r, p = pearsonr(emp_fc[mask], sim_fc[mask])

print(f"FC correlation with empirical data: r = {r:.3f}, p = {p:.2e}")
```

## Integration with Claude Code

When helping users with TVB:

1. **Environment Check:**
   ```python
   import tvb
   print(f"TVB version: {tvb.__version__}")
   ```

2. **Data Preparation:** Ensure structural connectivity matrices are symmetric and properly normalized

3. **Model Selection:** Choose appropriate neural mass model based on research question

4. **Common Issues:**
   - Memory errors with high-resolution connectivity (use coarser parcellation)
   - Long simulation times (reduce simulation length or increase dt)
   - Numerical instability (decrease integration timestep)
   - Connectivity import errors (check matrix dimensions and format)

5. **Performance:** Use appropriate integration timestep and monitor periods

## Best Practices

- Start with default connectivity and models before using custom data
- Validate structural connectivity (check symmetry, remove self-connections)
- Use appropriate conduction velocity (typically 3-6 mm/ms)
- Monitor simulation for numerical stability
- Compare simulated FC with empirical data to validate parameters
- Use parameter sweeps to explore model behavior
- Save configured simulators for reproducibility
- Document all model parameters in methods
- Use appropriate integration timestep for model dynamics
- Validate results against empirical data

## Troubleshooting

**Problem:** "Simulation produces NaN values"
**Solution:** Decrease integration timestep (dt), check connectivity for invalid values, reduce coupling strength

**Problem:** "Memory error with large connectivity"
**Solution:** Use coarser parcellation, reduce simulation length, process in chunks

**Problem:** "Simulated FC doesn't match empirical data"
**Solution:** Optimize coupling strength, adjust conduction velocity, try different neural mass models, check connectivity normalization

**Problem:** "Web interface not loading"
**Solution:** Check Docker container status, verify port mapping, check firewall settings

## Resources

- TVB Documentation: https://docs.thevirtualbrain.org/
- Tutorials: https://www.thevirtualbrain.org/tvb/zwei/tutorials
- GitHub: https://github.com/the-virtual-brain/tvb-root
- Forum: https://groups.google.com/g/tvb-users
- Scientific Portal: https://www.thevirtualbrain.org/tvb/zwei/brainsimulator-project

## Related Tools

- **MRtrix3:** Tractography for structural connectivity (see `mrtrix3.md`)
- **DIPY:** Diffusion processing (see `dipy.md`)
- **fMRIPrep:** Functional MRI preprocessing (see `fmriprep.md`)
- **Nilearn:** fMRI analysis (see `nilearn.md`)
- **BrainIAK:** Advanced fMRI analysis (see `brainiak.md`)
- **FreeSurfer:** Parcellation (see `freesurfer.md`)

## Citation

```bibtex
@article{sanzleon2013tvb,
  title={The Virtual Brain: a simulator of primate brain network dynamics},
  author={Sanz Leon, Paula and Knock, Stuart A. and Spiegler, Andreas and others},
  journal={Frontiers in Neuroinformatics},
  volume={7},
  pages={10},
  year={2013},
  doi={10.3389/fninf.2013.00010}
}
```

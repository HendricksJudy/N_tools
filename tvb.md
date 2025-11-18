# The Virtual Brain (TVB): Whole-Brain Network Simulation Platform

## Overview

The Virtual Brain (TVB) is a comprehensive neuroinformatics platform for simulating large-scale brain network dynamics using biophysically realistic neural mass models. TVB integrates structural connectivity derived from diffusion MRI tractography with mathematical models of neural activity to simulate whole-brain dynamics across multiple spatial and temporal scales. This enables researchers to explore how brain structure constrains function, test mechanistic hypotheses about brain dynamics, create personalized brain models for clinical applications, and generate testable predictions that can be validated with empirical neuroimaging data.

TVB represents a paradigm shift from purely statistical analysis of neuroimaging data to mechanistic, theory-driven computational modeling. By combining anatomical connectivity with biophysical models of neural dynamics, TVB can simulate various neuroimaging modalities (fMRI BOLD, EEG, MEG, local field potentials) and explore how local neural activity gives rise to large-scale brain networks. This approach is particularly valuable for understanding brain diseases, planning neurosurgical interventions, and testing pharmaceutical compounds in silico before clinical trials.

**Key Features:**
- Large-scale brain network simulation (whole-brain models with 50-10,000+ regions)
- Multiple neural mass models (Wilson-Cowan, Kuramoto, Jansen-Rit, Wong-Wang, and more)
- Integration of structural connectivity from DTI/DSI tractography
- Multi-modal simulation (fMRI BOLD, EEG, MEG, LFP, spike trains)
- Parameter space exploration and sensitivity analysis
- Individual-specific brain models from patient neuroimaging data
- Interactive web-based GUI and Python scripting API
- GPU acceleration for computationally intensive simulations
- Virtual lesions and intervention modeling (DBS, TMS, pharmaceuticals)
- Time-varying dynamics and stimulus-driven simulations

**Primary Use Cases:**
- Understanding structure-function relationships in healthy and diseased brains
- Predicting effects of lesions, tumors, or neurosurgical interventions
- Testing pharmaceutical compounds on neural network dynamics
- Personalized medicine with individual-specific brain models
- Exploring criticality and phase transitions in brain dynamics
- Generating hypotheses for empirical validation
- Education in computational neuroscience and dynamical systems

**Citation:**
```
Sanz Leon, P., Knock, S. A., Woodman, M. M., Domide, L., Mersmann, J., McIntosh, A. R.,
& Jirsa, V. (2013). The Virtual Brain: a simulator of primate brain network dynamics.
Frontiers in Neuroinformatics, 7, 10.
```

## Installation

### Docker Installation (Recommended)

Docker provides the easiest installation with all dependencies pre-configured:

```bash
# Pull TVB Docker image
docker pull thevirtualbrain/tvb-run

# Run TVB web interface
docker run -d -p 8080:8080 thevirtualbrain/tvb-run

# Access web interface at http://localhost:8080
# Default credentials: admin / pass

# For Python API usage
docker run -it --rm \
  -v $(pwd):/work \
  thevirtualbrain/tvb-run \
  python /work/my_simulation.py
```

### Python Package Installation

```bash
# Create dedicated environment
conda create -n tvb python=3.9
conda activate tvb

# Install TVB framework
pip install tvb-library tvb-data tvb-gdist

# For GUI/web interface
pip install tvb-framework

# Optional: GPU support (CUDA required)
pip install cupy

# Verify installation
python -c "import tvb.simulator; print(tvb.simulator.__version__)"
```

### Web Interface Setup

```bash
# Initialize TVB database
tvb-start --profile

# Start web server
tvb-start web

# Access at http://localhost:8080
# First launch creates configuration

# Stop server
tvb-stop
```

### Download Example Data

```bash
# TVB includes sample connectivity matrices
python << EOF
from tvb.datatypes.connectivity import Connectivity
conn = Connectivity.from_file()
print(f"Default connectivity: {conn.number_of_regions} regions")
EOF

# Download additional datasets
# Available at: https://github.com/the-virtual-brain/tvb-data
```

## Building Structural Connectomes

**Example 1: Load Default Connectivity**

```python
from tvb.simulator.lab import *
import numpy as np
import matplotlib.pyplot as plt

# Load built-in connectivity (76 regions, AAL atlas)
conn = connectivity.Connectivity.from_file()

print(f"Number of regions: {conn.number_of_regions}")
print(f"Connectivity shape: {conn.weights.shape}")
print(f"Region labels: {conn.region_labels[:5]}")
# Output: ['rANG', 'lANG', 'rCAL', 'lCAL', 'rCUN']

# Visualize connectivity matrix
plt.figure(figsize=(10, 8))
plt.imshow(conn.weights, cmap='hot', interpolation='nearest')
plt.colorbar(label='Connection Strength')
plt.title('Structural Connectivity Matrix')
plt.xlabel('Region')
plt.ylabel('Region')
plt.savefig('connectivity_matrix.png', dpi=300)
```

**Example 2: Load Custom Connectivity from FreeSurfer/Tractography**

```python
# Assume you have:
# - weights.txt: NxN connectivity matrix from tractography
# - distances.txt: NxN fiber length matrix
# - labels.txt: region names from parcellation
# - centers.txt: region centroids (x, y, z)

import numpy as np
from tvb.datatypes import connectivity

# Load data
weights = np.loadtxt('weights.txt')
distances = np.loadtxt('distances.txt')
labels = np.loadtxt('labels.txt', dtype=str)
centers = np.loadtxt('centers.txt')

# Create TVB connectivity object
custom_conn = connectivity.Connectivity(
    weights=weights,
    tract_lengths=distances,
    region_labels=labels,
    centres=centers
)

# Normalize weights
custom_conn.weights /= custom_conn.weights.max()

# Configure TVB-required attributes
custom_conn.configure()

# Save for future use
custom_conn.save('my_custom_connectivity.zip')

# Load later
loaded_conn = connectivity.Connectivity.from_file('my_custom_connectivity.zip')
```

**Example 3: Processing HCP Structural Connectivity**

```python
# Convert HCP connectome to TVB format
# Assumes DSI Studio or MRtrix3 connectivity output

import pandas as pd

# Load HCP-style connectivity (e.g., from Schaefer 400 parcellation)
connectivity_file = 'sub-001_Schaefer400_connectivity.csv'
conn_df = pd.read_csv(connectivity_file, index_col=0)

# Extract components
weights = conn_df.values
labels = np.array(conn_df.columns)

# Estimate distances (if not available, use Euclidean)
# Load region coordinates
coords_file = 'Schaefer400_centers.txt'
centers = np.loadtxt(coords_file)

# Compute pairwise distances
from scipy.spatial.distance import pdist, squareform
distances = squareform(pdist(centers))

# Create connectivity
hcp_conn = connectivity.Connectivity(
    weights=weights,
    tract_lengths=distances,
    region_labels=labels,
    centres=centers
)
hcp_conn.configure()

print(f"HCP connectivity: {hcp_conn.number_of_regions} regions")
```

## Neural Mass Models

**Example 4: Wilson-Cowan Model**

```python
from tvb.simulator.models import WilsonCowan
import numpy as np

# Initialize Wilson-Cowan model
wc = WilsonCowan()

# Model parameters
wc.c_ee = np.array([12.0])  # Excitatory-excitatory coupling
wc.c_ei = np.array([4.0])   # Inhibitory-excitatory coupling
wc.c_ie = np.array([13.0])  # Excitatory-inhibitory coupling
wc.c_ii = np.array([11.0])  # Inhibitory-inhibitory coupling
wc.tau_e = np.array([10.0]) # Excitatory time constant
wc.tau_i = np.array([10.0]) # Inhibitory time constant
wc.a_e = np.array([1.2])    # Excitatory threshold
wc.a_i = np.array([2.0])    # Inhibitory threshold

# State variables: E (excitatory), I (inhibitory)
print(f"State variables: {wc.state_variables}")
print(f"Excitatory population activity: E")
print(f"Inhibitory population activity: I")

# Examine phase space
e_range = np.linspace(0, 1, 100)
i_range = np.linspace(0, 1, 100)

# Nullclines can be computed for stability analysis
```

**Example 5: Reduced Wong-Wang Model for fMRI**

```python
from tvb.simulator.models import ReducedWongWang

# Optimized for fMRI BOLD simulation
rww = ReducedWongWang()

# Key parameters
rww.a = np.array([0.270])    # Gain parameter
rww.b = np.array([0.108])    # Input-output function parameter
rww.d = np.array([154.0])    # Time scale parameter
rww.gamma = np.array([0.641]) # Kinetic parameter
rww.tau_s = np.array([100.0]) # Synaptic time constant (ms)
rww.w = np.array([0.6])      # Recurrent connection weight
rww.J_N = np.array([0.2609]) # Synaptic coupling

# This model produces realistic BOLD fluctuations
# State variable: S (synaptic gating variable)
print(f"State variables: {rww.state_variables}")
# Output: ['S']
```

**Example 6: Jansen-Rit Model for EEG**

```python
from tvb.simulator.models import JansenRit

# Neural mass model that generates EEG-like signals
jr = JansenRit()

# Parameters
jr.A = np.array([3.25])  # Maximum amplitude of excitatory PSP
jr.B = np.array([22.0])  # Maximum amplitude of inhibitory PSP
jr.a = np.array([100.0]) # Inverse of excitatory time constant
jr.b = np.array([50.0])  # Inverse of inhibitory time constant
jr.v0 = np.array([6.0])  # Firing threshold
jr.nu_max = np.array([0.0025])  # Maximum firing rate
jr.r = np.array([0.56]) # Steepness of sigmoid

# Six state variables representing pyramidal and interneuron populations
print(f"State variables: {jr.state_variables}")
# Output: ['y0', 'y1', 'y2', 'y3', 'y4', 'y5']

# Can generate alpha rhythms (8-12 Hz)
```

## Simulating Brain Dynamics

**Example 7: Basic Resting-State Simulation**

```python
from tvb.simulator.lab import *
import numpy as np

# 1. Set up connectivity
conn = connectivity.Connectivity.from_file()

# 2. Choose neural mass model
model = models.Generic2dOscillator()

# 3. Configure coupling (how regions interact)
coupling_strength = 0.042
coupl = coupling.Linear(a=np.array([coupling_strength]))

# 4. Set up integration scheme
heunint = integrators.HeunDeterministic(dt=2**-4)  # 0.0625 ms timestep

# 5. Add noise
noise = noise.Additive(nsig=np.array([2**-10]))

# 6. Create monitors (what to record)
mon_tavg = monitors.TemporalAverage(period=2.0)  # Sample every 2 ms
mon_raw = monitors.Raw()

# 7. Initialize simulator
sim = simulator.Simulator(
    model=model,
    connectivity=conn,
    coupling=coupl,
    integrator=heunint,
    monitors=[mon_tavg, mon_raw]
)
sim.configure()

# 8. Run simulation
simulation_length = 10000.0  # milliseconds (10 seconds)
(tavg_time, tavg_data), (raw_time, raw_data) = sim.run(
    simulation_length=simulation_length
)

print(f"Time series shape: {tavg_data.shape}")
# Shape: (timepoints, state_variables, nodes, modes)
# Example: (5000, 2, 76, 1) = 5000 timepoints, 2 state vars, 76 regions

# Extract data for analysis
timeseries = tavg_data[:, 0, :, 0]  # First state variable, all regions
print(f"Timeseries shape: {timeseries.shape}")  # (5000, 76)
```

**Example 8: Simulate BOLD fMRI Signals**

```python
from tvb.simulator.lab import *
import numpy as np
import matplotlib.pyplot as plt

# Set up simulation with BOLD monitor
conn = connectivity.Connectivity.from_file()
model = models.ReducedWongWang()
coupl = coupling.Linear(a=np.array([0.014]))
heunint = integrators.HeunStochastic(dt=0.1, noise=noise.Additive(nsig=np.array([0.001])))

# BOLD monitor with hemodynamic model
bold_monitor = monitors.Bold(period=2000.0)  # TR = 2000 ms = 2 sec

sim = simulator.Simulator(
    model=model,
    connectivity=conn,
    coupling=coupl,
    integrator=heunint,
    monitors=[bold_monitor]
)
sim.configure()

# Run simulation (need ~30 seconds to stabilize BOLD)
(bold_time, bold_data), = sim.run(simulation_length=60000.0)

# BOLD data shape: (timepoints, 1, regions, 1)
bold_ts = bold_data[:, 0, :, 0]  # Shape: (30, 76) for 60s at TR=2s

# Plot BOLD timeseries for a few regions
plt.figure(figsize=(12, 6))
for i in range(5):
    plt.plot(bold_time, bold_ts[:, i], label=f'Region {conn.region_labels[i]}')
plt.xlabel('Time (ms)')
plt.ylabel('BOLD Signal')
plt.legend()
plt.title('Simulated fMRI BOLD Signals')
plt.savefig('bold_timeseries.png', dpi=300)

# Compute functional connectivity
from scipy.stats import pearsonr
n_regions = bold_ts.shape[1]
fc_simulated = np.zeros((n_regions, n_regions))
for i in range(n_regions):
    for j in range(n_regions):
        fc_simulated[i, j], _ = pearsonr(bold_ts[:, i], bold_ts[:, j])

# Compare to empirical FC
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(fc_simulated, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar()
plt.title('Simulated FC')
plt.subplot(1, 2, 2)
# Load empirical FC for comparison (if available)
# plt.imshow(fc_empirical, cmap='RdBu_r', vmin=-1, vmax=1)
plt.title('Empirical FC')
plt.tight_layout()
plt.savefig('fc_comparison.png', dpi=300)
```

**Example 9: Simulate EEG with Forward Model**

```python
from tvb.simulator.lab import *
import numpy as np

# Use Jansen-Rit model for EEG
conn = connectivity.Connectivity.from_file()
model = models.JansenRit()
coupl = coupling.SigmoidalJansenRit(a=np.array([0.1]))
heunint = integrators.HeunStochastic(dt=2**-4, noise=noise.Additive(nsig=np.array([0.001])))

# EEG monitor with forward projection
eeg_monitor = monitors.EEG(
    period=4.0,  # 250 Hz sampling (4 ms)
    projection=None  # Use default projection matrix
)

sim = simulator.Simulator(
    model=model,
    connectivity=conn,
    coupling=coupl,
    integrator=heunint,
    monitors=[eeg_monitor]
)
sim.configure()

# Run simulation
(eeg_time, eeg_data), = sim.run(simulation_length=10000.0)

# EEG data: (timepoints, electrodes, 1, 1)
eeg_signals = eeg_data[:, :, 0, 0]
print(f"EEG signals shape: {eeg_signals.shape}")
# Shape: (2500, 62) for 10s at 250 Hz, 62 electrodes

# Plot EEG from a few electrodes
plt.figure(figsize=(14, 8))
for i in range(min(5, eeg_signals.shape[1])):
    plt.plot(eeg_time, eeg_signals[:, i] + i*50, label=f'Electrode {i+1}')
plt.xlabel('Time (ms)')
plt.ylabel('EEG Amplitude (offset for visualization)')
plt.title('Simulated EEG Signals')
plt.legend()
plt.savefig('eeg_signals.png', dpi=300)

# Spectral analysis
from scipy.signal import welch
freqs, psd = welch(eeg_signals[:, 0], fs=250, nperseg=256)
plt.figure()
plt.semilogy(freqs, psd)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('EEG Power Spectrum')
plt.xlim([0, 40])
plt.savefig('eeg_spectrum.png', dpi=300)
```

## Parameter Space Exploration

**Example 10: Global Coupling Sweep**

```python
from tvb.simulator.lab import *
import numpy as np
import matplotlib.pyplot as plt

conn = connectivity.Connectivity.from_file()
model = models.ReducedWongWang()

# Range of coupling strengths to test
coupling_values = np.linspace(0.001, 0.05, 20)

# Storage for results
fc_simulated_all = []

for coupling_strength in coupling_values:
    print(f"Simulating coupling = {coupling_strength:.4f}")

    coupl = coupling.Linear(a=np.array([coupling_strength]))
    heunint = integrators.HeunStochastic(dt=0.1, noise=noise.Additive(nsig=np.array([0.001])))
    bold_monitor = monitors.Bold(period=2000.0)

    sim = simulator.Simulator(
        model=model,
        connectivity=conn,
        coupling=coupl,
        integrator=heunint,
        monitors=[bold_monitor]
    )
    sim.configure()

    # Run simulation
    (bold_time, bold_data), = sim.run(simulation_length=60000.0)
    bold_ts = bold_data[:, 0, :, 0]

    # Compute FC
    fc = np.corrcoef(bold_ts.T)
    fc_simulated_all.append(fc)

fc_simulated_all = np.array(fc_simulated_all)

# Compare to empirical FC (load your own)
# empirical_fc = np.load('empirical_fc.npy')

# Compute correlation between simulated and empirical FC
fc_correlations = []
for fc_sim in fc_simulated_all:
    # Flatten upper triangle
    # triu_indices = np.triu_indices(fc_sim.shape[0], k=1)
    # corr = np.corrcoef(fc_sim[triu_indices], empirical_fc[triu_indices])[0, 1]
    # fc_correlations.append(corr)
    pass

# Plot coupling vs FC fit
# plt.plot(coupling_values, fc_correlations)
# plt.xlabel('Global Coupling Strength')
# plt.ylabel('FC Correlation (Simulated vs Empirical)')
# plt.title('Parameter Optimization')
```

**Example 11: Bifurcation Analysis**

```python
from tvb.simulator.lab import *
import numpy as np

# Explore how changing a parameter affects dynamics
conn = connectivity.Connectivity.from_file()
model = models.Generic2dOscillator()

# Vary parameter (e.g., global inhibition)
inhibition_values = np.linspace(-2.0, 0.0, 50)
mean_activity = []
oscillation_amplitude = []

for tau in inhibition_values:
    model.tau = np.array([tau])

    sim = simulator.Simulator(
        model=model,
        connectivity=conn,
        coupling=coupling.Linear(a=np.array([0.042])),
        integrator=integrators.HeunDeterministic(dt=2**-4),
        monitors=[monitors.TemporalAverage(period=1.0)]
    )
    sim.configure()

    (time, data), = sim.run(simulation_length=10000.0)

    # Analyze activity
    ts = data[:, 0, :, 0]
    mean_activity.append(ts.mean())
    oscillation_amplitude.append(ts.std())

# Plot bifurcation diagram
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(inhibition_values, mean_activity)
plt.xlabel('Inhibition Parameter')
plt.ylabel('Mean Activity')
plt.subplot(1, 2, 2)
plt.plot(inhibition_values, oscillation_amplitude)
plt.xlabel('Inhibition Parameter')
plt.ylabel('Oscillation Amplitude (std)')
plt.tight_layout()
plt.savefig('bifurcation_analysis.png', dpi=300)
```

## Disease Modeling and Interventions

**Example 12: Virtual Lesion**

```python
from tvb.simulator.lab import *
import numpy as np

conn = connectivity.Connectivity.from_file()

# Simulate lesion by removing connections to/from specific regions
# Example: Lesion in region 10 (simulating stroke)
lesion_region = 10

# Create lesioned connectivity
conn_lesioned = conn.copy()
conn_lesioned.weights[lesion_region, :] = 0
conn_lesioned.weights[:, lesion_region] = 0
conn_lesioned.configure()

# Simulate with intact connectivity
model = models.ReducedWongWang()
coupl = coupling.Linear(a=np.array([0.014]))
heunint = integrators.HeunStochastic(dt=0.1, noise=noise.Additive(nsig=np.array([0.001])))
bold_monitor = monitors.Bold(period=2000.0)

sim_intact = simulator.Simulator(
    model=model,
    connectivity=conn,
    coupling=coupl,
    integrator=heunint,
    monitors=[bold_monitor]
)
sim_intact.configure()

(time_intact, data_intact), = sim_intact.run(simulation_length=60000.0)
bold_intact = data_intact[:, 0, :, 0]
fc_intact = np.corrcoef(bold_intact.T)

# Simulate with lesion
sim_lesioned = simulator.Simulator(
    model=model,
    connectivity=conn_lesioned,
    coupling=coupl,
    integrator=heunint,
    monitors=[bold_monitor]
)
sim_lesioned.configure()

(time_lesioned, data_lesioned), = sim_lesioned.run(simulation_length=60000.0)
bold_lesioned = data_lesioned[:, 0, :, 0]
fc_lesioned = np.corrcoef(bold_lesioned.T)

# Compare FC before and after lesion
plt.figure(figsize=(14, 5))
plt.subplot(1, 3, 1)
plt.imshow(fc_intact, cmap='RdBu_r', vmin=-1, vmax=1)
plt.title('FC Intact')
plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(fc_lesioned, cmap='RdBu_r', vmin=-1, vmax=1)
plt.title('FC After Lesion')
plt.colorbar()
plt.subplot(1, 3, 3)
plt.imshow(fc_intact - fc_lesioned, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
plt.title('FC Difference')
plt.colorbar()
plt.tight_layout()
plt.savefig('lesion_effects.png', dpi=300)
```

**Example 13: Simulating Alzheimer's Disease**

```python
# Alzheimer's disease affects specific connectivity patterns
# Model by reducing long-range connections

conn = connectivity.Connectivity.from_file()
distances = conn.tract_lengths

# Reduce long-distance connections (> 50mm)
conn_ad = conn.copy()
long_connections = distances > 50
conn_ad.weights[long_connections] *= 0.5  # 50% reduction

# Alternative: Target specific networks (e.g., default mode)
# dmn_regions = [0, 1, 5, 6, ...]  # DMN region indices
# for i in dmn_regions:
#     for j in dmn_regions:
#         conn_ad.weights[i, j] *= 0.3  # 70% reduction in DMN

conn_ad.configure()

# Run simulations and compare
# ... (similar to lesion example)
```

**Example 14: Deep Brain Stimulation (DBS) Simulation**

```python
from tvb.simulator.lab import *
import numpy as np

# Simulate DBS by adding external input to specific region(s)
conn = connectivity.Connectivity.from_file()
model = models.ReducedWongWang()

# DBS target (e.g., subthalamic nucleus index)
dbs_target = 25
stimulus_strength = 0.5  # mV

# Create stimulus pattern (high-frequency pulses, 130 Hz)
stimulus = patterns.StimuliRegion(
    temporal=equations.PulseTrain(
        onset=5000.0,  # Start at 5 seconds
        period=7.69,   # 130 Hz
        pulse_width=0.5  # 0.5 ms pulses
    ),
    spatial=None,
    weight=np.array([stimulus_strength])
)

# Specify which regions receive stimulus
stimulus.connectivity = conn
stimulus.spatial = np.zeros((conn.number_of_regions,))
stimulus.spatial[dbs_target] = 1.0  # Only stimulate DBS target

# Run simulation with DBS
sim = simulator.Simulator(
    model=model,
    connectivity=conn,
    coupling=coupling.Linear(a=np.array([0.014])),
    integrator=integrators.HeunStochastic(dt=0.1, noise=noise.Additive(nsig=np.array([0.001]))),
    monitors=[monitors.Bold(period=2000.0)],
    stimulus=stimulus
)
sim.configure()

(bold_time, bold_data), = sim.run(simulation_length=60000.0)

# Compare activity before and during DBS
bold_ts = bold_data[:, 0, :, 0]
before_dbs = bold_ts[:12, :]  # First 24 seconds (12 TRs)
during_dbs = bold_ts[12:, :]  # After DBS onset

print(f"Mean activity before DBS: {before_dbs.mean():.3f}")
print(f"Mean activity during DBS: {during_dbs.mean():.3f}")
```

## Advanced Features

**Example 15: Time-Varying Connectivity**

```python
# Simulate dynamic functional connectivity changes

from tvb.simulator.lab import *
import numpy as np

conn = connectivity.Connectivity.from_file()

# Define time-varying coupling function
class TimeVaryingCoupling(coupling.Linear):
    def __call__(self, g_ij, x_i, x_j):
        # Sinusoidally modulate coupling strength
        t = self.time_step * self.dt
        modulation = 1.0 + 0.3 * np.sin(2 * np.pi * t / 10000.0)  # 10s period
        return modulation * super().__call__(g_ij, x_i, x_j)

# Use in simulation
# (Requires custom implementation)
```

**Example 16: GPU-Accelerated Simulation**

```python
# For very large networks (>1000 regions) or long simulations

# Install CuPy for GPU support
# pip install cupy-cuda11x  # Match your CUDA version

import cupy as cp
from tvb.simulator.lab import *

# TVB can leverage GPU for certain operations
# Particularly useful for large parameter sweeps

# Enable GPU in configuration
# This is experimental and depends on TVB version
```

## Integration with Neuroimaging

**Example 17: Fitting Model to Empirical FC**

```python
from tvb.simulator.lab import *
import numpy as np
from scipy.optimize import minimize

# Load empirical functional connectivity
empirical_fc = np.load('empirical_fc.npy')  # From resting-state fMRI

conn = connectivity.Connectivity.from_file()
model = models.ReducedWongWang()

def objective_function(params):
    """Optimize coupling strength and other parameters."""
    coupling_strength, noise_level = params

    coupl = coupling.Linear(a=np.array([coupling_strength]))
    heunint = integrators.HeunStochastic(
        dt=0.1,
        noise=noise.Additive(nsig=np.array([noise_level]))
    )
    bold_monitor = monitors.Bold(period=2000.0)

    sim = simulator.Simulator(
        model=model,
        connectivity=conn,
        coupling=coupl,
        integrator=heunint,
        monitors=[bold_monitor]
    )
    sim.configure()

    (bold_time, bold_data), = sim.run(simulation_length=60000.0)
    bold_ts = bold_data[:, 0, :, 0]

    # Compute simulated FC
    fc_simulated = np.corrcoef(bold_ts.T)

    # Correlation between empirical and simulated FC
    triu_indices = np.triu_indices(fc_simulated.shape[0], k=1)
    corr = np.corrcoef(
        fc_simulated[triu_indices],
        empirical_fc[triu_indices]
    )[0, 1]

    # Minimize negative correlation (maximize positive)
    return -corr

# Optimize parameters
initial_params = [0.014, 0.001]
bounds = [(0.001, 0.1), (0.0001, 0.01)]

result = minimize(
    objective_function,
    initial_params,
    bounds=bounds,
    method='L-BFGS-B'
)

print(f"Optimal coupling: {result.x[0]:.4f}")
print(f"Optimal noise: {result.x[1]:.6f}")
print(f"FC correlation: {-result.fun:.3f}")
```

## Troubleshooting

**Numerical Instabilities:**
```python
# If simulation diverges (NaN or Inf values):

# 1. Reduce integration time step
integrator = integrators.HeunDeterministic(dt=2**-6)  # Smaller dt

# 2. Check coupling strength (too high can cause instability)
coupling_strength = 0.001  # Start low

# 3. Verify connectivity normalization
conn.weights /= conn.weights.max()

# 4. Add damping to model parameters
# 5. Check initial conditions
```

**Memory Issues:**
```python
# For large networks or long simulations:

# 1. Reduce monitor sampling frequency
bold_monitor = monitors.Bold(period=3000.0)  # Lower sampling

# 2. Use chunked simulation
chunk_length = 10000.0  # 10 seconds
total_length = 100000.0
n_chunks = int(total_length / chunk_length)

for i in range(n_chunks):
    (time, data), = sim.run(simulation_length=chunk_length)
    # Process and save each chunk
    np.save(f'chunk_{i}.npy', data)
    # Clear memory
    del data

# 3. Reduce network size (coarser parcellation)
```

## Best Practices

**Model Selection:**
- Wilson-Cowan: General neural dynamics, simple and interpretable
- Generic2dOscillator: Oscillatory dynamics, for rhythms
- ReducedWongWang: fMRI BOLD simulation, slow dynamics
- Jansen-Rit: EEG/MEG, fast oscillations (alpha, beta)

**Parameter Estimation:**
- Start with literature values
- Fit one parameter at a time
- Use empirical FC as fitting target
- Validate with independent data

**Simulation Setup:**
- Warm up simulations (discard first 10-20 seconds)
- Verify stable dynamics before long runs
- Check multiple random seeds
- Compare to empirical data at multiple scales

**Reproducibility:**
- Set random seeds
- Document all parameters
- Version control code
- Save connectivity matrices with results

## Integration with Analysis Ecosystem

**FreeSurfer:**
- Use FreeSurfer parcellations (Desikan-Killiany, Destrieux)
- Extract region centers for TVB connectivity

**MRtrix3/DSI Studio:**
- Generate structural connectivity via tractography
- Convert to TVB format

**Nilearn:**
- Load empirical FC for model validation
- Visualize simulation results on brain surfaces

**NetworkX:**
- Graph theoretical analysis of simulated dynamics
- Compare network metrics

## References

**TVB:**
- Sanz Leon et al. (2013). The Virtual Brain: a simulator of primate brain network dynamics. *Frontiers in Neuroinformatics*, 7, 10.
- Ritter et al. (2013). The Virtual Brain integrates computational modeling and multimodal neuroimaging. *Brain Connectivity*, 3(2), 121-145.

**Neural Mass Models:**
- Wilson & Cowan (1972). Excitatory and inhibitory interactions in localized populations of model neurons. *Biophysical Journal*, 12(1), 1-24.
- Jansen & Rit (1995). Electroencephalogram and visual evoked potential generation in a mathematical model. *Biological Cybernetics*, 73(4), 357-366.
- Wong & Wang (2006). A recurrent network mechanism of time integration in perceptual decisions. *Journal of Neuroscience*, 26(4), 1314-1328.

**Applications:**
- Breakspear (2017). Dynamic models of large-scale brain activity. *Nature Neuroscience*, 20(3), 340-352.
- Deco et al. (2019). Awakening: Predicting external stimulation to force transitions. *PNAS*, 116(36), 18088-18097.

**Online Resources:**
- TVB Website: https://www.thevirtualbrain.org
- TVB Documentation: https://docs.thevirtualbrain.org
- TVB GitHub: https://github.com/the-virtual-brain
- TVB Tutorials: https://www.thevirtualbrain.org/tvb/zwei/tutorial-index

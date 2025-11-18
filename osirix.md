# OsiriX - macOS DICOM Viewer and Medical Imaging Platform

## Overview

OsiriX is a macOS-native DICOM image viewer and processing platform widely used in clinical radiology, medical research, and teaching hospitals worldwide. Originally developed as an open-source project at UCLA, OsiriX provides advanced 2D/3D/4D visualization capabilities, multi-planar reconstruction (MPR), maximum intensity projection (MIP), volume rendering, and an extensive plugin architecture. The software excels at handling large DICOM datasets, integrating with hospital PACS (Picture Archiving and Communication System), and offering an intuitive macOS interface optimized for radiologists, clinicians, and medical researchers.

OsiriX transitioned to a commercial model in 2015 (OsiriX MD for FDA-certified clinical use), while the open-source fork Horos continues the original development philosophy. Both versions leverage Apple's Metal graphics framework for high-performance 3D rendering, support all major imaging modalities (CT, MRI, PET, ultrasound, X-ray), and provide DICOM compliance for seamless integration with clinical workflows. OsiriX remains the gold standard for macOS medical imaging, particularly in neurology, radiology, and surgical planning.

**Official Website:** https://www.osirix-viewer.com (OsiriX MD commercial)
**Horos (Open-Source Fork):** https://horosproject.org
**Documentation:** https://osirix-viewer.com/resources/

### Key Features

- **Native macOS Application:** Optimized for macOS with Metal acceleration
- **DICOM Compliance:** Full DICOM 3.0 standard support and PACS integration
- **2D Viewing:** Multi-series display with synchronized scrolling and MPR
- **3D/4D Rendering:** Volume rendering (VR), MIP, surface rendering, 4D time-series
- **ROI Tools:** Comprehensive region-of-interest drawing and quantitative measurements
- **Image Fusion:** Multi-modality overlay (PET/MRI, CT/MRI)
- **PACS Integration:** DICOM send/receive (C-STORE, C-FIND, C-MOVE, WADO)
- **Plugin Architecture:** Extensive plugin ecosystem for custom functionality
- **Database Management:** Efficient DICOM database with search and organization
- **Export Capabilities:** NIfTI, JPEG, TIFF, DICOM, QuickTime movie export
- **Reporting:** Integrated reporting with annotations and measurements
- **Performance:** Hardware acceleration via Metal for real-time 3D interaction

### Applications

- Clinical radiology and diagnostic imaging
- Neurological assessment and surgical planning
- Multi-modality image fusion (PET/MRI, SPECT/CT)
- Research requiring native DICOM viewing without conversion
- Teaching and medical education
- Teleradiology and remote consultation
- Integration with hospital PACS systems

### OsiriX MD vs. Horos

- **OsiriX MD:** Commercial, FDA-certified for clinical diagnosis, professional support
- **Horos:** Free, open-source, community-driven, suitable for research and education
- **Feature Parity:** Core functionality similar, OsiriX MD adds clinical certifications
- **Updates:** OsiriX MD receives regular updates; Horos development slower
- **Recommendation:** Horos for research/education; OsiriX MD for clinical use

### Citation

```bibtex
@article{Rosset2004OsiriX,
  title={OsiriX: an open-source software for navigating in multidimensional DICOM images},
  author={Rosset, Antoine and Spadola, Luca and Ratib, Osman},
  journal={Journal of Digital Imaging},
  volume={17},
  number={3},
  pages={205--216},
  year={2004},
  publisher={Springer}
}

@article{Rosset2006OsiriX,
  title={Osirix: A free software for DICOM images viewing in daily clinical activity},
  author={Rosset, A and Spadola, L and Pysher, L and Ratib, O},
  journal={European Radiology},
  volume={16},
  pages={13--17},
  year={2006}
}
```

---

## Installation

### OsiriX MD (Commercial Version)

```bash
# Download from official website
# https://www.osirix-viewer.com/osirix/osirix-md/

# System Requirements:
# - macOS 10.13 (High Sierra) or later
# - 8GB RAM minimum, 16GB+ recommended
# - Graphics card with Metal support
# - 50GB+ free disk space for DICOM database

# Installation:
# 1. Download OsiriX_MD_X.X.dmg
# 2. Open DMG and drag OsiriX to Applications folder
# 3. Launch OsiriX from Applications
# 4. Enter license key (30-day trial available)

# First Launch:
# - Accept EULA
# - Configure DICOM database location
# - Set up PACS connections (if applicable)
```

### Horos (Open-Source Version)

```bash
# Download from Horos website
# https://horosproject.org/download/

# System Requirements: Same as OsiriX MD

# Installation:
# 1. Download Horos-vX.X.X.dmg
# 2. Open DMG and drag Horos to Applications
# 3. Launch Horos from Applications (no license required)

# Security Note:
# macOS may block unsigned apps. Go to:
# System Preferences > Security & Privacy > General
# Click "Open Anyway" if prompted
```

### Database Setup

```bash
# On first launch, configure database location

# Recommended database location:
# /Users/[username]/Documents/OsiriX Data
# or external SSD for large datasets

# Database structure:
DATABASE/
├── DATABASE.noindex/      # DICOM files
├── INCOMING.noindex/      # Auto-import folder
└── OsiriX.sql            # SQLite database

# Automatic import:
# Drop DICOM files into INCOMING.noindex folder
# OsiriX will automatically import and index them
```

### PACS Configuration

```bash
# Configure PACS server for DICOM query/retrieve

# In OsiriX:
# Preferences > Locations > Add Location

# PACS Server Settings:
# - AE Title: PACS_SERVER (get from IT/admin)
# - Address: pacs.hospital.edu
# - Port: 11112 (default DICOM port)
# - Transfer Syntax: Explicit VR Little Endian

# Test connection with DICOM Echo (C-ECHO)
```

---

## DICOM Management

### Importing DICOM Files

```bash
# Method 1: Drag and drop
# Drag DICOM folder to OsiriX window or database

# Method 2: File menu
# File > Import > Files...
# Select DICOM files or folder

# Method 3: INCOMING folder
# Copy DICOM files to:
# ~/Documents/OsiriX Data/INCOMING.noindex/
# OsiriX auto-imports every 5 seconds

# Method 4: Import from DICOM CD/DVD
# Insert disc, then:
# File > Import > Import a DICOM CD/DVD
```

### Database Organization

```bash
# OsiriX organizes studies by:
# - Patient Name
# - Patient ID
# - Study Date
# - Modality (CT, MRI, PET, etc.)
# - Study Description

# Smart Albums (virtual folders):
# Database > New Smart Album
# Define criteria (e.g., "All Brain MRI from 2023")
# Examples:
# - Modality = MR AND Study Description contains "brain"
# - Study Date > 2023-01-01
# - Patient Name contains "Smith"

# Searching:
# Use search bar (top right) to filter by:
# - Patient name, ID, study description
# - Date range
# - Modality
```

### PACS Query/Retrieve

```bash
# Query remote PACS server for studies

# Step 1: Open PACS Query
# Database window > PACS OnDemand button
# or File > Retrieve Studies...

# Step 2: Enter search criteria
# Patient Name: Smith* (use * as wildcard)
# Patient ID: 123456
# Date Range: Last 30 days
# Modality: MR

# Step 3: Click "Search"
# Results appear in query window

# Step 4: Select studies and click "Retrieve"
# Studies download to local database via C-MOVE

# Tip: Save frequent queries as presets
```

---

## 2D Viewing and MPR

### Basic 2D Navigation

```bash
# Opening a study:
# Double-click study in database

# 2D Viewer controls:
# - Scroll wheel: Navigate through slices
# - Drag: Window/level adjustment
# - Right-click + drag: Zoom
# - Spacebar + drag: Pan

# Multi-series layout:
# View > Tiles > 2x2 (or 3x3, 4x4)
# Each tile shows different series or plane

# Synchronized scrolling:
# Check "Sync" button in toolbar
# All series scroll together through slices
```

### Window/Level Adjustment

```bash
# Adjust brightness/contrast for optimal viewing

# Method 1: Mouse drag
# Click and drag left/right (level), up/down (window)

# Method 2: Presets
# Window > WL/WW > Presets
# Common presets:
# - Brain: W=80, L=40
# - Bone: W=2000, L=300
# - Lung: W=1500, L=-600
# - Soft tissue: W=400, L=40

# Method 3: Manual entry
# Window > WL/WW > Manual...
# Enter exact window and level values

# Tip: Press 'F' for full dynamic range
```

### Multi-Planar Reconstruction (MPR)

```bash
# Generate orthogonal views from volumetric data

# Step 1: Open MPR
# Select series, then:
# 2D Viewer > MPR > Multiplanar Reconstruction

# Step 2: MPR window shows 3 orthogonal planes
# - Axial (top-left)
# - Coronal (top-right)
# - Sagittal (bottom-left)
# - 3D cursor (bottom-right, optional)

# Step 3: Navigate
# Click in any plane to reposition crosshairs
# All planes update simultaneously

# Export MPR:
# File > Export > Current Image to DICOM
```

### Curved MPR

```bash
# Follow curved anatomical structures (e.g., spine, vessels)

# Step 1: Draw curve
# 2D Viewer > ROI > Closed Polygon or Pencil
# Draw along structure (e.g., spinal canal)

# Step 2: Generate Curved MPR
# ROI > Curved MPR
# Displays straightened view along curve

# Use cases:
# - Spinal canal visualization
# - Vessel straightening
# - Nerve tracking
```

---

## 3D/4D Visualization

### Volume Rendering (VR)

```bash
# Create 3D volume from image stack

# Step 1: Open 3D viewer
# 2D Viewer > 3D > Volume Rendering
# or click 3D icon in toolbar

# Step 2: Adjust rendering parameters
# - Transfer function: Bone, Soft Tissue, MIP, etc.
# - Opacity: Adjust transparency curve
# - Color: Change color lookup table (LUT)
# - Shading: Enable/disable lighting effects

# Step 3: Interact with volume
# - Drag: Rotate
# - Right-click drag: Zoom
# - Scroll: Adjust clipping planes
# - Option + drag: Pan

# Presets:
# - Bone (CT): High-density structures visible
# - Brain (MRI): Soft tissue visualization
# - Vessels (MRA/CTA): Vascular structures
```

### Maximum Intensity Projection (MIP)

```bash
# Display brightest voxels along viewing direction

# Step 1: Open MIP viewer
# 2D Viewer > 3D > MIP
# Best for angiography (MRA, CTA)

# Step 2: Set slab thickness
# MIP Settings > Slab Thickness: 10mm - 100mm
# Thicker slab = more vessels visible

# Step 3: Rotate and explore
# Drag to rotate view
# Export rotations as movie:
# Movie > Export as QuickTime

# Use case: Visualize cerebral vasculature from MRA
```

### Surface Rendering

```bash
# Extract and display 3D surfaces

# Step 1: Create ROI
# In 2D viewer, draw ROI around structure
# Copy ROI to all slices (Edit > Propagate ROI)

# Step 2: Generate surface
# ROI > Generate Surface
# Marching cubes algorithm extracts surface

# Step 3: Smooth and refine
# Surface > Smooth (reduce polygonal artifacts)
# Surface > Decimation (reduce polygon count)

# Export surface:
# File > Export > STL (for 3D printing)
# File > Export > VTK (for ParaView)
```

### 4D Time-Series Viewing

```bash
# View dynamic imaging (fMRI, cardiac, DCE-MRI)

# Step 1: Load 4D dataset
# OsiriX automatically detects temporal dimension

# Step 2: Open 4D viewer
# 2D Viewer > 4D > 4D Viewer

# Step 3: Playback controls
# - Play/Pause: Animate through timepoints
# - Speed: Adjust frames per second
# - Slider: Manual timepoint selection

# Movie export:
# Movie > Export 4D Series as QuickTime
# Useful for presentations and publications

# Use cases:
# - fMRI activation over time
# - Cardiac cine sequences
# - Contrast bolus passage (DCE-MRI)
```

---

## ROI and Measurements

### Drawing ROIs

```bash
# Region of Interest tools for quantification

# ROI Types (2D Viewer toolbar):
# - Rectangle: Rectangular ROI
# - Ellipse: Circular/oval ROI
# - Closed Polygon: Manual boundary
# - Pencil: Freehand drawing
# - Brush: Paint-style ROI
# - Point: Single point marker
# - Line: Linear measurement
# - Angle: Angle measurement

# Drawing workflow:
# 1. Select ROI tool from toolbar
# 2. Draw ROI on image
# 3. View measurements in ROI Manager (Window > ROI Manager)

# ROI properties:
# - Mean intensity
# - Standard deviation
# - Min/max intensity
# - Area (2D) or volume (3D)
# - Length (line ROI)
```

### Volume Measurements

```bash
# Quantify 3D structure volumes (e.g., tumor, ventricle)

# Method 1: Manual segmentation on each slice
# Step 1: Draw ROI around structure on slice 1
# Step 2: Advance to next slice, draw again
# Step 3: Repeat for all slices
# Step 4: ROI > Compute Volume
# Result: Total volume in mm³ or cm³

# Method 2: Automated propagation
# Step 1: Draw ROI on one slice
# Step 2: ROI > Propagate ROI (to adjacent slices)
# Step 3: Manually refine propagated ROIs
# Step 4: Compute volume

# Export ROI statistics:
# ROI > Export as CSV
# Columns: Slice, Area, Mean, StdDev, etc.
```

### Intensity Statistics

```bash
# Extract quantitative metrics from ROIs

# Step 1: Draw ROI(s)
# Use any ROI tool to define region(s)

# Step 2: View statistics (ROI Manager)
# - Mean: Average intensity in ROI
# - StdDev: Standard deviation
# - Min/Max: Intensity range
# - Pixels: Number of pixels in ROI
# - Area: ROI area in mm²

# Multi-ROI analysis:
# Draw multiple ROIs (e.g., different brain regions)
# ROI Manager displays statistics for each
# Compare means across ROIs

# Use case: ADC values in stroke lesion vs. normal tissue
```

### Length and Angle Measurements

```bash
# Measure anatomical dimensions

# Linear measurement:
# 1. Select Line tool
# 2. Draw line between two points
# 3. Length displayed in mm (based on DICOM pixel spacing)

# Angle measurement:
# 1. Select Angle tool
# 2. Click three points to define angle
# 3. Angle displayed in degrees

# Calibration check:
# Verify measurements against known dimensions
# Edit > DICOM Tags > Pixel Spacing
# Ensure correct calibration for accurate measurements

# Export measurements:
# ROI > Export Measurements to Excel/CSV
```

---

## Image Processing

### Image Fusion (Multi-Modality Overlay)

```bash
# Overlay PET on MRI, SPECT on CT, etc.

# Step 1: Open both studies
# Load anatomical (e.g., MRI) and functional (e.g., PET)

# Step 2: Initiate fusion
# 2D Viewer > Fusion > Image Fusion
# Select second modality from popup

# Step 3: Adjust fusion parameters
# - Opacity: Blend ratio (slider)
# - Color table: Choose PET color scheme (hot metal, rainbow)
# - Registration: Auto or manual alignment

# Step 4: Navigate fused image
# Scroll through slices with both modalities visible

# Use case: PET/MRI for tumor metabolism + anatomy
```

### Registration Tools

```bash
# Align images from different acquisitions

# Automatic registration:
# Fusion > Register Images
# Algorithm: Mutual information or cross-correlation
# Works best for same-modality images (MRI-MRI)

# Manual registration:
# Fusion > Manual Registration
# Use sliders to adjust:
# - Translation (X, Y, Z)
# - Rotation (pitch, yaw, roll)
# - Scaling
# Verify alignment in fused view

# Save registered image:
# File > Export > Registered Series as New DICOM
```

### Filtering and Enhancement

```bash
# Image processing filters (via plugins or built-in)

# Smoothing (noise reduction):
# Plugins > Filters > Gaussian Smoothing
# Kernel size: 3x3 or 5x5
# Reduces noise but may blur edges

# Sharpening (edge enhancement):
# Plugins > Filters > Sharpen
# Enhances edges and fine details
# Use cautiously to avoid artifacts

# Note: Most analysis should use original data
# Filters for visualization only, not quantification
```

---

## Plugins and Extensions

### Plugin Manager

```bash
# Access OsiriX plugin ecosystem

# Open Plugin Manager:
# Plugins > Plugin Manager
# or Preferences > Plugins

# Browse available plugins:
# - Database plugins (data management)
# - Image filters (processing)
# - ROI tools (advanced segmentation)
# - Export tools (format conversion)

# Install plugin:
# 1. Download .osirixplugin file
# 2. Double-click to install
# 3. Restart OsiriX
# 4. Plugin appears in Plugins menu
```

### Popular Plugins

```bash
# NIfTI Export Plugin:
# Export DICOM to NIfTI format for FSL/SPM analysis
# Plugins > Export to NIfTI

# DICOM Anonymizer:
# Remove patient identifiable information
# Plugins > Anonymize Series

# Morphometry (volume quantification):
# Advanced segmentation and volume calculation
# Plugins > Morphometry

# OsiriX/Horos plugin repository:
# https://www.osirix-viewer.com/resources/plugins/
# https://github.com/horosproject/horos-plugins
```

### Developing Custom Plugins

```bash
# Create plugins using Objective-C or Swift

# Plugin SDK:
# Download from OsiriX website
# Xcode project template included

# Basic plugin structure:
# - PluginFilter.h/.m (main class)
# - Inherits from PlugInFilter base class
# - Implement -long filterImage:(NSString*)menuName
# - Access viewer, images, ROIs via API

# Documentation:
# https://osirix-viewer.com/resources/dicom-image-library/
# API reference and examples available
```

---

## Export and Interoperability

### Export to NIfTI

```bash
# Convert DICOM to NIfTI for neuroimaging analysis

# Method 1: Built-in export (Horos)
# File > Export > NIfTI format
# Select series and output location

# Method 2: Plugin (OsiriX MD)
# Install NIfTI export plugin
# Plugins > Export to NIfTI

# Alternative (command-line):
# Use dcm2niix for batch conversion
# dcm2niix -o /output /path/to/DICOM

# Verify NIfTI orientation:
# Check orientation in FSLeyes or similar
# OsiriX may export in radiological convention
```

### Export Images and Movies

```bash
# Export for presentations and publications

# Still images:
# File > Export > Images
# Formats: JPEG, TIFF, PNG, DICOM
# Options: Current slice, all slices, or range

# Movies (QuickTime):
# Movie > Export Rotation as QuickTime
# Useful for 3D renderings and 4D time-series
# Settings: Frame rate, resolution, codec

# Screenshots:
# Cmd+Shift+4 (macOS screenshot)
# Or Export > Current Image (preserves resolution)
```

### DICOM Anonymization

```bash
# Remove patient identifiable information

# Method 1: Built-in anonymization
# Database > Select studies
# Database > Anonymize...
# Options:
# - Replace patient name with code
# - Shift dates by random offset
# - Remove private tags

# Method 2: Anonymize on export
# File > Export > Anonymized DICOM
# Automatically anonymizes during export

# Verify anonymization:
# Check DICOM tags after anonymization
# Window > DICOM Tags
# Ensure Patient Name, ID, DOB removed
```

---

## Horos Differences

### Feature Comparison

```bash
# Core functionality: Nearly identical
# - 2D/3D/4D viewing: Same
# - MPR, MIP, VR: Same
# - ROI tools: Same
# - DICOM support: Same

# Key differences:
# OsiriX MD:
# - FDA 510(k) cleared for clinical diagnosis
# - Commercial support and updates
# - Advanced plugins (some paid)
# - Cost: ~$700 one-time or subscription

# Horos:
# - Free and open-source
# - Community-driven development
# - Slower update cycle
# - Suitable for research/education (not clinical diagnosis)
```

### Open-Source Advantages

```bash
# Horos benefits:
# - No licensing costs
# - Source code available (GitHub)
# - Community contributions
# - Research-friendly license
# - Ideal for academic institutions

# Limitations:
# - No official technical support
# - FDA approval not applicable
# - Development depends on volunteers
# - Some advanced features lag behind OsiriX MD
```

---

## Troubleshooting

### DICOM Import Issues

**Problem:** Files not importing or appearing corrupted

**Solutions:**
```bash
# Check DICOM validity:
# Open terminal and use DCMTK tools:
dcmdump file.dcm  # Verify DICOM format

# Rebuild database:
# Database > Rebuild Database
# Fixes indexing issues

# Check file permissions:
# Ensure read access to DICOM files
# chmod -R 755 /path/to/DICOM
```

### PACS Connection Problems

**Problem:** Cannot query or retrieve from PACS

**Solutions:**
```bash
# Test network connectivity:
ping pacs.hospital.edu

# Verify PACS settings:
# Preferences > Locations
# Confirm AE Title, address, port

# Test DICOM Echo:
# Locations > Select PACS > Echo
# Should return success if connection OK

# Firewall check:
# Ensure port 11112 (or custom port) not blocked
# Contact IT if hospital firewall blocks DICOM
```

### Performance Optimization

**Problem:** Slow 3D rendering or large dataset loading

**Solutions:**
```bash
# Enable Metal acceleration:
# Preferences > 3D > Use Metal (should be on by default)

# Reduce rendering quality:
# 3D Viewer > Settings > Lower resolution
# Trade quality for speed

# Increase RAM allocation:
# Preferences > Database > Cache size
# Allocate more RAM for database cache

# Use SSD for database:
# Move OsiriX Data folder to SSD
# Significantly improves load times
```

### Metal Rendering Issues

**Problem:** 3D viewer crashes or displays artifacts

**Solutions:**
```bash
# Update macOS:
# Metal requires macOS 10.13+
# Check for system updates

# Update graphics drivers:
# macOS handles drivers automatically
# Ensure latest macOS version installed

# Disable Metal (fallback):
# Preferences > 3D > Uncheck "Use Metal"
# Falls back to OpenGL (slower but more compatible)
```

---

## Best Practices

### Database Management

- **Regular Backups:** Backup ~/Documents/OsiriX Data regularly
- **External Storage:** Use external SSD for large datasets (faster than HDD)
- **Smart Albums:** Organize studies with Smart Albums for quick access
- **Cleanup:** Periodically delete old/unnecessary studies to save space
- **Database Repair:** Rebuild database if experiencing slowdowns

### DICOM Workflow Optimization

- **Batch Import:** Use INCOMING folder for automatic import of multiple studies
- **PACS Integration:** Configure PACS for seamless query/retrieve
- **Templates:** Save common window/level presets for quick access
- **Shortcuts:** Learn keyboard shortcuts for efficiency (see Help menu)

### Clinical vs. Research Usage

- **Clinical (OsiriX MD):** FDA-cleared, use for diagnostic interpretation
- **Research (Horos):** Free, suitable for exploratory analysis and education
- **Anonymization:** Always anonymize data for research publications
- **Validation:** Verify measurements against known standards

### Privacy and Anonymization

- **PHI Protection:** Secure database with strong password (FileVault encryption)
- **Anonymize Before Sharing:** Remove all identifiable information before export
- **HIPAA Compliance:** Ensure database stored on encrypted, secure storage
- **Audit Trails:** Keep records of data access for clinical use

---

## References

1. **OsiriX Publications:**
   - Rosset et al. (2004). OsiriX: An open-source software for navigating in multidimensional DICOM images. *J Digital Imaging*, 17(3):205-216.
   - Rosset et al. (2006). OsiriX: A free software for DICOM images viewing in daily clinical activity. *European Radiology*, 16:13-17.

2. **DICOM Standard:**
   - NEMA PS3 / ISO 12052. Digital Imaging and Communications in Medicine (DICOM) Standard.
   - https://www.dicomstandard.org/

3. **Medical Imaging:**
   - Bushberg et al. (2011). *The Essential Physics of Medical Imaging*. 3rd edition.
   - Sprawls (2000). *Physical Principles of Medical Imaging*. Online textbook.

4. **Horos Project:**
   - https://horosproject.org
   - https://github.com/horosproject/horos

**Official Resources:**
- OsiriX MD: https://www.osirix-viewer.com
- User Manual: https://osirix-viewer.com/resources/
- Plugins: https://www.osirix-viewer.com/resources/plugins/
- Forum: https://groups.google.com/forum/#!forum/osirix-users

# Mango (Multi-image Analysis GUI)

## Overview

**Mango** (Multi-image Analysis GUI) is a versatile medical image viewer developed at the Research Imaging Institute, University of Texas Health Science Center San Antonio. Built in Java for cross-platform compatibility, Mango provides comprehensive viewing capabilities for a wide range of neuroimaging and medical imaging formats. It combines intuitive GUI-based interaction with powerful features for ROI definition, image registration, surface visualization, and quantitative analysis.

Mango excels at manual quality control, ROI drawing, multi-modal image comparison, and DICOM handling. Its extensible plugin architecture and scripting support make it valuable for both clinical research and neuroscience applications where manual interaction with imaging data is required.

**Key Use Cases:**
- Manual ROI definition and segmentation
- Multi-modal image registration and fusion
- Quality control of preprocessing pipelines
- DICOM viewing and conversion to NIFTI
- Clinical image review and annotation
- Teaching and demonstration
- Coordinate-based analysis and reporting
- Quantitative measurements (volumes, distances, intensities)

**Official Website:** http://ric.uthscsa.edu/mango/
**Documentation:** http://ric.uthscsa.edu/mango/mango.html
**Download:** http://ric.uthscsa.edu/mango/downloads.html

---

## Key Features

- **Multi-Format Support:** NIFTI, DICOM, ANALYZE, MINC, PAR/REC, MGH, and more
- **3D Rendering:** Volume and surface visualization
- **Multi-Planar Reformatting:** Axial, sagittal, coronal, and oblique views
- **ROI Drawing:** Freehand, geometric shapes, threshold-based tools
- **Image Registration:** Manual and automated alignment
- **Surface Extraction:** Generate meshes from volumes
- **Coordinate Systems:** MNI, Talairach, native space support
- **Intensity Tools:** Windowing, LUTs, contrast adjustment
- **Measurements:** Distance, angle, volume, intensity statistics
- **DICOM Support:** Browse, view, and convert DICOM series
- **Plugin Architecture:** Extensible with custom plugins
- **Scripting:** JavaScript and Jython for automation
- **Cross-Platform:** Windows, macOS, Linux (Java-based)
- **Export:** Save images, movies, ROIs, surfaces
- **No Installation Required:** Portable, runs from folder
- **Free and Open Source:** No licensing costs

---

## Installation

### Download and Setup

```bash
# Download Mango from official website
# http://ric.uthscsa.edu/mango/downloads.html

# Extract the archive
unzip Mango_unix.zip  # Linux
# or
# Open Mango_macOS.dmg and copy to Applications  # macOS
# or
# Extract Mango_windows.zip  # Windows

# Mango requires Java Runtime Environment (JRE) 8 or higher
java -version  # Check Java installation
```

### Linux

```bash
# Navigate to Mango directory
cd Mango

# Make executable
chmod +x Mango

# Run Mango
./Mango

# Or double-click Mango icon in file browser
```

### macOS

```bash
# Open from Applications folder
open /Applications/Mango.app

# Or from Terminal
/Applications/Mango.app/Contents/MacOS/JavaAppLauncher
```

### Windows

```cmd
rem Double-click Mango.exe in extracted folder
rem Or run from command line
Mango.exe
```

### Java Memory Configuration

```bash
# Edit Mango startup script to increase memory

# Linux/macOS: Edit Mango script
java -Xmx4096m -jar Mango.jar  # 4GB memory

# Windows: Edit Mango.l4j.ini
-Xmx4096m
```

---

## Basic Usage

### Launch and Load Image

```bash
# Launch Mango
./Mango

# Load image from GUI:
# File → Open → Select image file
# Supports: .nii, .nii.gz, .img/.hdr, .dcm, etc.
```

**Command-Line Loading:**

```bash
# Load image from command line
./Mango image.nii.gz

# Load multiple images
./Mango image1.nii.gz image2.nii.gz
```

### Multi-Planar Views

```
Mango displays three orthogonal views:
- Axial (horizontal slices)
- Sagittal (left-right slices)
- Coronal (front-back slices)

Navigation:
- Left-click and drag to change slice
- Mouse wheel to scroll through slices
- Right-click for crosshair positioning
```

### Image Series

```bash
# Load multiple images as series
# File → Open → Select all images → Open as Series

# Navigate series with slider or keyboard
# Useful for:
# - fMRI time series
# - DTI volumes
# - Multi-contrast imaging
```

---

## ROI Drawing

### Basic ROI Tools

```
ROI Drawing Tools (in toolbar):

1. **Freehand:** Draw arbitrary shapes
   - Left-click and drag to draw
   - Double-click to close ROI

2. **Rectangle:** Draw rectangular ROIs
   - Click and drag corners

3. **Ellipse:** Draw elliptical ROIs
   - Click and drag to define

4. **Polygon:** Define multi-sided shapes
   - Click to add vertices
   - Double-click to close

5. **Paint Brush:** Paint voxels directly
   - Adjustable brush size
   - Paint or erase mode
```

### Threshold-Based ROI

```
Create ROI from intensity threshold:

1. Tools → ROI → Threshold
2. Set min and max intensity values
3. Click "Create ROI" to convert to mask
4. Save ROI: File → Save ROI

Example: Isolate brain from background
- Set threshold: 100 to 1000
- Creates binary mask
```

### 3D ROI (Multi-Slice)

```
Draw ROI across multiple slices:

1. Enable "Multiple Slices" mode
2. Draw ROI on current slice
3. Move to next slice
4. Draw ROI again
5. Repeat for all slices

Mango interpolates between slices
to create 3D volume ROI
```

### ROI Operations

```
ROI manipulation:

- **Copy/Paste:** Duplicate ROI across slices
- **Move:** Reposition ROI
- **Resize:** Adjust dimensions
- **Rotate:** Change orientation
- **Boolean Operations:**
  - Union (combine ROIs)
  - Intersection (overlap)
  - Subtraction (remove overlap)

Access via: Tools → ROI
```

### Save and Load ROIs

```bash
# Save ROI
# File → Save ROI → Choose format
# Formats: Mango ROI (.roi), NIFTI (.nii), mask (.img)

# Load ROI
# File → Open ROI → Select ROI file

# Export ROI statistics
# Tools → ROI Statistics → Save to CSV
# Includes: Volume, mean intensity, std, min, max
```

---

## Image Registration

### Manual Registration

```
Manual image alignment:

1. Load reference image (File → Open)
2. Load image to register (File → Add)
3. Tools → Transform → Manual Transform
4. Use translation/rotation sliders:
   - X, Y, Z translation (mm)
   - Roll, pitch, yaw rotation (degrees)
5. Adjust until images aligned
6. File → Save Transformed Image
```

### Automatic Registration

```
Automated image registration:

1. Load reference and moving images
2. Tools → Transform → Auto-Register
3. Select registration method:
   - Rigid (6 DOF)
   - Affine (12 DOF)
4. Set similarity metric:
   - Mutual Information
   - Correlation
   - Sum of Squared Differences
5. Click "Register"
6. Review result and save
```

### Reslicing

```
Reslice image to different space:

1. Load image to reslice
2. Load target image (defines output space)
3. Tools → Transform → Reslice
4. Select interpolation method:
   - Nearest Neighbor (labels)
   - Trilinear (smooth)
   - Cubic (high quality)
5. Save resliced image
```

---

## Surface Visualization

### Extract Surface

```
Generate 3D surface from volume:

1. Load volumetric image
2. Tools → Surface → Extract Surface
3. Set threshold for surface boundary
4. Choose smoothing level
5. Click "Extract"
6. View 3D surface in dedicated window
```

### Surface Display Options

```
3D Surface rendering:

- **Rotation:** Left-click and drag
- **Zoom:** Mouse wheel
- **Transparency:** Adjust alpha slider
- **Color:** Change surface color
- **Lighting:** Modify lighting angle
- **Mesh:** Toggle wireframe display

Tools → Surface → Surface Properties
```

### Overlay on Surface

```
Project statistical maps onto surface:

1. Extract brain surface
2. Load statistical map (overlay)
3. Tools → Surface → Project Overlay
4. Set color map and threshold
5. Render overlay on surface
```

---

## Coordinate Systems

### Navigate to Coordinates

```
Jump to specific coordinates:

1. Tools → Go To Coordinate
2. Enter coordinates:
   - X, Y, Z in mm (world coordinates)
   - or i, j, k (voxel indices)
3. Select coordinate system:
   - Scanner (native)
   - MNI152
   - Talairach
4. Click "Go"

Crosshair moves to specified location
```

### Report Coordinates

```
Get coordinates at cursor:

1. Right-click on image
2. Coordinates displayed in status bar
3. Shows both:
   - Voxel indices (i, j, k)
   - World coordinates (x, y, z, mm)

Copy coordinates:
- Right-click → Copy Coordinates
```

### Coordinate Space Conversion

```
Convert between coordinate systems:

1. Tools → Coordinates → Convert
2. Enter coordinates in source space
3. Select source space (Scanner, MNI, Talairach)
4. Select target space
5. View converted coordinates

Useful for:
- Relating coordinates to atlases
- Cross-study comparisons
```

---

## Intensity Manipulation

### Window and Level

```
Adjust image contrast:

1. Image → Window/Level
2. Adjust sliders:
   - Window: Range of displayed intensities
   - Level: Center of intensity range

Or use mouse:
- Right-click and drag:
  - Horizontal: Adjust window (contrast)
  - Vertical: Adjust level (brightness)
```

### Look-Up Tables (LUTs)

```
Apply color maps:

1. Image → LUT → Select LUT
2. Available LUTs:
   - Gray (default)
   - Hot (activation)
   - Cool (deactivation)
   - Rainbow
   - Custom (create own)

For overlays:
- Different LUT per image
- Adjust transparency for blending
```

### Threshold Display

```
Display only voxels in intensity range:

1. Image → Threshold Display
2. Set min and max intensity
3. Voxels outside range become transparent
4. Useful for isolating tissues or activations
```

---

## Measurements

### Distance Measurement

```
Measure distance between points:

1. Tools → Measurements → Distance
2. Click start point
3. Click end point
4. Distance displayed in mm
5. Measurements saved in log

Multi-point distance:
- Continue clicking to add points
- Cumulative distance calculated
```

### Angle Measurement

```
Measure angle between lines:

1. Tools → Measurements → Angle
2. Click three points:
   - Point 1: First arm
   - Point 2: Vertex (center)
   - Point 3: Second arm
3. Angle displayed in degrees
```

### Volume Calculation

```
Calculate ROI volume:

1. Draw or load ROI
2. Tools → ROI Statistics
3. Volume displayed in mm³ and voxels

Includes:
- Total volume
- Per-slice breakdown
- Statistical measures (if overlay present)
```

### Intensity Profile

```
Extract intensity profile along line:

1. Tools → Measurements → Profile
2. Draw line on image
3. View intensity graph:
   - X-axis: Distance along line
   - Y-axis: Intensity values
4. Export profile data
```

---

## DICOM Handling

### Browse DICOM Directory

```
View DICOM series:

1. File → Browse DICOM
2. Select directory containing DICOM files
3. Mango scans and organizes by series
4. Select series to view
5. Load into viewer
```

### DICOM to NIFTI Conversion

```
Convert DICOM to NIFTI:

1. Load DICOM series
2. File → Save As
3. Select format: NIFTI (.nii or .nii.gz)
4. Choose output location
5. Save

Preserves:
- Orientation
- Voxel dimensions
- Patient information (in header)
```

### DICOM Metadata

```
View DICOM tags:

1. Load DICOM image
2. Image → DICOM Info
3. Browse DICOM tags:
   - Patient info
   - Acquisition parameters
   - Sequence details
   - Scanner information
4. Export metadata to text file
```

---

## Scripting and Automation

### JavaScript Scripting

```javascript
// Example Mango JavaScript
// File → Scripting → JavaScript Console

// Load image
mango.io.open("path/to/image.nii.gz");

// Get image data
var image = mango.series.current;
var dims = image.imageDimensions;

// Print dimensions
print("Dimensions: " + dims[0] + " x " + dims[1] + " x " + dims[2]);

// Get voxel value at (50, 50, 50)
var value = image.getVoxelValue(50, 50, 50);
print("Intensity at (50,50,50): " + value);

// Save screenshot
mango.io.saveScreenshot("screenshot.png");
```

### Batch Processing Script

```javascript
// batch_process.js
// Process multiple subjects

var subjects = ["sub-01", "sub-02", "sub-03"];
var basePath = "/data/subjects/";

for (var i = 0; i < subjects.length; i++) {
    var subject = subjects[i];
    var imagePath = basePath + subject + "/anat/T1w.nii.gz";

    // Load image
    mango.io.open(imagePath);

    // Apply window/level
    mango.display.setWindow(0, 1000);
    mango.display.setLevel(500);

    // Save screenshot
    var outputPath = "screenshots/" + subject + ".png";
    mango.io.saveScreenshot(outputPath);

    // Close image
    mango.io.close();
}

print("Batch processing complete!");
```

### ROI Analysis Script

```javascript
// roi_analysis.js
// Extract statistics from ROI

// Load image and ROI
mango.io.open("image.nii.gz");
mango.roi.load("hippocampus.roi");

// Get ROI statistics
var stats = mango.roi.getStatistics();

print("ROI Statistics:");
print("Volume: " + stats.volume + " mm³");
print("Mean intensity: " + stats.mean);
print("Std Dev: " + stats.stdDev);
print("Min: " + stats.min);
print("Max: " + stats.max);

// Save to file
mango.io.saveText("roi_stats.txt", JSON.stringify(stats, null, 2));
```

---

## Plugin Development

### Basic Plugin Structure

```java
// SimplePlugin.java
// Basic Mango plugin template

import edu.uthscsa.ric.mango.api.*;

public class SimplePlugin implements MangoPlugin {

    @Override
    public String getName() {
        return "Simple Plugin";
    }

    @Override
    public void execute(MangoContext context) {
        // Get current image
        ImageData image = context.getCurrentImage();

        // Get dimensions
        int[] dims = image.getDimensions();

        // Process image
        for (int z = 0; z < dims[2]; z++) {
            for (int y = 0; y < dims[1]; y++) {
                for (int x = 0; x < dims[0]; x++) {
                    float value = image.getVoxel(x, y, z);

                    // Apply processing
                    float newValue = value * 1.5f;  // Example

                    image.setVoxel(x, y, z, newValue);
                }
            }
        }

        // Update display
        context.updateDisplay();
    }
}
```

### Install Plugin

```bash
# Compile plugin
javac -cp Mango.jar SimplePlugin.java

# Create JAR
jar cf SimplePlugin.jar SimplePlugin.class

# Copy to plugins directory
cp SimplePlugin.jar /path/to/Mango/plugins/

# Restart Mango
# Plugin appears in Plugins menu
```

---

## Integration with Claude Code

Mango can be integrated into automated workflows via scripting:

```python
# mango_automation.py
# Automate Mango tasks from Python

import subprocess
import os
from pathlib import Path

class MangoAutomator:
    """Wrapper for Mango in automated pipelines."""

    def __init__(self, mango_path="/path/to/Mango"):
        self.mango_path = Path(mango_path)
        self.mango_exec = self.mango_path / "Mango"

    def view_image(self, image_file):
        """Open image in Mango."""
        cmd = [str(self.mango_exec), str(image_file)]
        subprocess.run(cmd)

    def run_script(self, script_file):
        """Execute Mango JavaScript."""
        cmd = [
            str(self.mango_exec),
            "--script", str(script_file)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout

    def convert_dicom(self, dicom_dir, output_nifti):
        """Convert DICOM to NIFTI using Mango."""
        script = f"""
        mango.io.openDICOM("{dicom_dir}");
        mango.io.save("{output_nifti}", "NIFTI");
        mango.io.close();
        """

        script_file = "temp_convert.js"
        with open(script_file, 'w') as f:
            f.write(script)

        self.run_script(script_file)
        os.remove(script_file)

        print(f"Converted: {output_nifti}")

# Usage
automator = MangoAutomator()

# Convert DICOM series
automator.convert_dicom(
    dicom_dir="/data/dicom/patient001",
    output_nifti="/data/nifti/patient001_T1.nii.gz"
)
```

**Batch QC Workflow:**

```python
# batch_qc.py
# Generate QC screenshots with Mango

from pathlib import Path
import subprocess

def create_qc_script(subjects, output_dir):
    """Generate Mango script for batch QC."""

    script_lines = []

    for subject in subjects:
        image_path = f"/data/derivatives/fmriprep/{subject}/anat/{subject}_T1w.nii.gz"
        output_path = f"{output_dir}/{subject}_qc.png"

        script_lines.extend([
            f'mango.io.open("{image_path}");',
            'mango.display.setWindow(0, 1000);',
            'mango.display.setLevel(500);',
            'mango.display.goToCenter();',
            f'mango.io.saveScreenshot("{output_path}");',
            'mango.io.close();'
        ])

    return "\\n".join(script_lines)

# Create script
subjects = [f"sub-{i:02d}" for i in range(1, 21)]
script_content = create_qc_script(subjects, "qc_screenshots")

# Save and run
with open("batch_qc.js", 'w') as f:
    f.write(script_content)

subprocess.run(["/path/to/Mango/Mango", "--script", "batch_qc.js"])
```

---

## Integration with Other Tools

### FreeSurfer Integration

```bash
# View FreeSurfer outputs in Mango
./Mango /path/to/freesurfer/subject/mri/T1.mgz

# FreeSurfer MGH/MGZ files supported directly
# View aseg (segmentation)
./Mango /path/to/freesurfer/subject/mri/aseg.mgz
```

### FSL Integration

```bash
# View FSL outputs
./Mango /data/fsl_analysis/zstat1.nii.gz

# Load brain mask as overlay
# File → Add → brain_mask.nii.gz
# Adjust opacity for overlay visualization
```

### SPM Integration

```bash
# View SPM statistical maps
./Mango /data/spm_analysis/spmT_0001.nii

# Overlay on anatomical
./Mango structural.nii spmT_0001.nii
```

---

## Troubleshooting

### Problem 1: Mango Won't Launch

**Symptoms:** Application fails to start

**Solution:**
```bash
# Check Java installation
java -version
# Requires Java 8 or higher

# Install Java if missing
# Ubuntu/Debian:
sudo apt-get install openjdk-11-jre

# macOS:
brew install openjdk@11

# Windows: Download from java.com
```

### Problem 2: Out of Memory

**Symptoms:** Crashes with large images

**Solution:**
```bash
# Increase Java heap size
# Edit Mango startup script

# Linux/macOS: Edit Mango
java -Xmx8192m -jar Mango.jar  # 8GB

# Windows: Edit Mango.l4j.ini
-Xmx8192m
```

### Problem 3: Image Won't Load

**Symptoms:** Error loading image file

**Solution:**
```bash
# Check file format
file image.nii.gz

# Ensure file is not corrupted
fslhd image.nii.gz  # Use FSL to check header

# Try converting format
mri_convert input.mgz output.nii.gz  # FreeSurfer
fslchfiletype NIFTI_GZ input.img  # FSL
```

### Problem 4: Registration Fails

**Symptoms:** Auto-registration produces poor results

**Solution:**
```
1. Try different similarity metrics
2. Adjust initialization:
   - Use manual pre-alignment
   - Set initial parameters closer to solution
3. Check image contrast and quality
4. Verify images are in similar spaces
5. Consider using external tools (FSL FLIRT, ANTs)
   then load result in Mango
```

### Problem 5: Display Issues

**Symptoms:** Garbled graphics or slow rendering

**Solution:**
```bash
# Update Java version
java -version  # Check current version

# Try different Java runtime
# OpenJDK vs Oracle JDK

# Update graphics drivers
# Check OpenGL support

# Reduce window size if slow
```

---

## Best Practices

### 1. Quality Control

- **Systematic review:** Check all subjects with same window/level
- **Multiple views:** Inspect axial, sagittal, coronal
- **Zoom in:** Examine details at high magnification
- **Document issues:** Take screenshots of artifacts
- **Standardize:** Use consistent viewing parameters

### 2. ROI Drawing

- **Zoom appropriately:** Draw at high magnification
- **Multiple slices:** Check neighboring slices for consistency
- **Use references:** Consult atlases for anatomical boundaries
- **Save frequently:** Prevent data loss
- **Validate:** Review 3D ROI rendering for errors

### 3. Registration

- **Start with manual:** Pre-align before auto-registration
- **Check result:** Verify alignment visually
- **Use appropriate metric:** Match to image modalities
- **Save transforms:** Keep transformation matrices
- **Document parameters:** Record registration settings

### 4. DICOM Workflow

- **Organize first:** Sort DICOM files into series
- **Check metadata:** Verify patient info before conversion
- **Consistent naming:** Use systematic file naming
- **Preserve orientation:** Verify after conversion
- **Anonymize if needed:** Remove patient identifiers

### 5. Scripting

- **Test interactively first:** Develop workflow in GUI
- **Add error handling:** Check for missing files
- **Log progress:** Print status messages
- **Modularize:** Create reusable functions
- **Comment code:** Document purpose and parameters

---

## Resources

### Official Documentation

- **Website:** http://ric.uthscsa.edu/mango/
- **User Guide:** http://ric.uthscsa.edu/mango/mango.html
- **Download:** http://ric.uthscsa.edu/mango/downloads.html
- **FAQ:** http://ric.uthscsa.edu/mango/faq.html

### Support

- **Email:** mangohelp@uthscsa.edu
- **NITRC Forum:** https://www.nitrc.org/forum/?group_id=91
- **Bug Reports:** Via email to developers

### Publications

- **Mango Paper:** Lancaster et al. (2012) "Mango: Multi Image Analysis GUI"
- **Research Imaging Institute:** http://ric.uthscsa.edu/

---

## Citation

```bibtex
@article{lancaster2012mango,
  title={Mango: multi image analysis GUI},
  author={Lancaster, Jack L and Martinez, Michael J},
  year={2012},
  publisher={Research Imaging Institute, UTHSCSA}
}
```

When using Mango for publications, acknowledge the Research Imaging Institute and cite relevant publications.

---

## Related Tools

- **ITK-SNAP:** Segmentation-focused viewer (see `itksnap.md`)
- **FSLeyes:** FSL viewer (see `fsleyes.md`)
- **MRIcron:** Lightweight viewer (see `mricron.md`)
- **Surfice:** Surface visualization (see `surfice.md`)
- **3D Slicer:** Comprehensive medical imaging platform
- **MIPAV:** NIH medical image processing tool
- **OsiriX:** macOS DICOM viewer (commercial)
- **Horos:** Open-source macOS DICOM viewer

---

**Skill Version:** 1.0
**Last Updated:** 2025-11
**Mango Version Covered:** 4.x
**Maintainer:** Claude Code Neuroimaging Skills

# QSIPrep

**Category:** Neuroimaging - Preprocessing

## Overview

QSIPrep is a neuroimaging tool used for brain imaging analysis and processing.

- **Website:** https://qsiprep.readthedocs.io/
- **Platform:** Linux/Docker
- **Languages:** N/A

## Installation

### Container Installation

```bash
# Docker
docker pull {container_image}

# Singularity
singularity pull {container_image}
```

Refer to the official documentation for the specific container image name.

## Usage Guidance

### Command-line Usage

```bash
# Example command
qsiprep [options] input_file output_file
```

Refer to the tool's documentation for specific command-line options.

## Common Use Cases

- fMRI/MRI data preprocessing
- Motion correction and artifact removal
- Spatial normalization
- Quality control

## Tips

- Check the official documentation at https://qsiprep.readthedocs.io/ for detailed usage
- Ensure all dependencies are properly installed before running
- Consider using containerized versions (Docker/Singularity) for reproducibility


## Integration with Claude Code

When working with QSIPrep:
1. Verify the tool is installed and accessible in your environment
2. Check version compatibility with your data format
3. Use appropriate file paths and ensure proper permissions
4. Monitor memory usage for large datasets
5. Validate outputs before proceeding with analysis

## Resources

- Official documentation: https://qsiprep.readthedocs.io/
- Platform: Linux/Docker

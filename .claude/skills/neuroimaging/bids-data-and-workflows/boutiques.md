# Boutiques - Tool Descriptor Framework

## Overview

**Boutiques** is a framework for describing, validating, and executing command-line tools in a platform-independent and reproducible manner. Developed at McGill University's Neuroinfo Lab, Boutiques uses JSON schema descriptors that capture all aspects of a tool's interface—inputs, outputs, parameters, constraints, and execution requirements—enabling automatic validation, cross-platform execution, and seamless integration with workflow engines and container technologies.

Boutiques ensures tools are correctly installed, properly invoked, and produce expected outputs. Descriptors serve as both machine-readable specifications and human-readable documentation, facilitating tool testing, publishing, sharing via Zenodo, and integration into diverse computational environments (local, HPC, cloud). This standardization eliminates ambiguity in tool usage and enables reproducible science.

**Key Features:**
- JSON-based tool descriptors following JSON Schema
- Automatic input and parameter validation
- Command-line template generation
- Container integration (Docker, Singularity)
- Cross-platform execution (local, HPC, cloud)
- Tool testing and quality assurance frameworks
- Automatic documentation generation
- Publishing and sharing via Zenodo
- Integration with workflow engines (Pydra, Nipype, CBRAIN)
- Error checking and detailed debugging information
- Schema validation and compliance checking
- BIDS Apps integration
- Provenance tracking
- Tool versioning and citation generation

**Primary Use Cases:**
- Standardize neuroimaging tool interfaces
- Validate tool installations and configurations
- Create reproducible tool invocations
- Integrate tools into workflow engines
- Publish and share analysis methods
- Test tools across platforms
- Develop BIDS Apps with validated interfaces
- Document tool usage for reproducibility
- Quality assurance for computational pipelines

**Official Documentation:** https://boutiques.github.io/

---

## Installation

### Install Boutiques

```bash
# Install via pip (recommended)
pip install boutiques

# Or install from GitHub for latest version
pip install git+https://github.com/boutiques/boutiques.git

# Verify installation
bosh --version

# Show available commands
bosh --help

# Note: 'bosh' is the Boutiques command-line tool
```

### Dependencies

```bash
# Core dependencies (auto-installed)
pip install jsonschema simplejson requests

# Optional: For container execution
# Docker
sudo apt-get install docker.io

# Singularity
# See: https://sylabs.io/guides/latest/user-guide/

# For Python API usage
pip install boutiques[all]
```

---

## Tool Descriptors

### Descriptor Structure

A Boutiques descriptor is a JSON file with these sections:

```json
{
  "name": "Tool Name",
  "tool-version": "1.0.0",
  "description": "Brief description",
  "command-line": "tool [INPUT] [OUTPUT] [PARAM]",
  "schema-version": "0.5",
  "inputs": [...],
  "output-files": [...],
  "groups": [...],
  "tags": {...},
  "container-image": {...}
}
```

### Simple Descriptor Example

```json
{
  "name": "FSL BET",
  "tool-version": "6.0.5",
  "description": "Brain Extraction Tool from FSL",
  "command-line": "bet [INPUT_FILE] [OUTPUT_FILE] [OPTIONS]",
  "schema-version": "0.5",
  "inputs": [
    {
      "id": "input_file",
      "name": "Input Image",
      "type": "File",
      "value-key": "[INPUT_FILE]",
      "description": "Input structural image",
      "optional": false
    },
    {
      "id": "output_file",
      "name": "Output Image",
      "type": "String",
      "value-key": "[OUTPUT_FILE]",
      "description": "Output brain-extracted image",
      "optional": false
    },
    {
      "id": "frac_intensity",
      "name": "Fractional Intensity Threshold",
      "type": "Number",
      "value-key": "[OPTIONS]",
      "command-line-flag": "-f",
      "description": "Fractional intensity threshold (0-1)",
      "optional": true,
      "minimum": 0,
      "maximum": 1,
      "default-value": 0.5
    }
  ],
  "output-files": [
    {
      "id": "brain_output",
      "name": "Brain Image",
      "path-template": "[OUTPUT_FILE].nii.gz",
      "description": "Brain-extracted image",
      "optional": false
    }
  ]
}
```

---

## Creating Descriptors

### Describe a Simple Tool

```bash
# Create descriptor for FSL's bet command
cat > fsl-bet.json <<'EOF'
{
  "name": "FSL BET",
  "tool-version": "6.0.5",
  "description": "Brain Extraction Tool",
  "command-line": "bet [INPUT] [OUTPUT] -f [FRAC]",
  "schema-version": "0.5",
  "inputs": [
    {
      "id": "input_image",
      "name": "Input Image",
      "type": "File",
      "value-key": "[INPUT]",
      "optional": false
    },
    {
      "id": "output_basename",
      "name": "Output Basename",
      "type": "String",
      "value-key": "[OUTPUT]",
      "optional": false
    },
    {
      "id": "frac_threshold",
      "name": "Fractional Intensity",
      "type": "Number",
      "value-key": "[FRAC]",
      "default-value": 0.5,
      "minimum": 0,
      "maximum": 1,
      "optional": true
    }
  ],
  "output-files": [
    {
      "id": "brain_image",
      "name": "Brain-extracted Image",
      "path-template": "[OUTPUT]_brain.nii.gz",
      "optional": false
    }
  ]
}
EOF

# Validate descriptor
bosh validate fsl-bet.json
```

### Input Types

```json
{
  "inputs": [
    {
      "id": "file_input",
      "name": "File Input",
      "type": "File",
      "value-key": "[FILE]"
    },
    {
      "id": "string_input",
      "name": "String Parameter",
      "type": "String",
      "value-key": "[STR]"
    },
    {
      "id": "number_input",
      "name": "Numeric Parameter",
      "type": "Number",
      "value-key": "[NUM]",
      "minimum": 0,
      "maximum": 100
    },
    {
      "id": "flag_input",
      "name": "Boolean Flag",
      "type": "Flag",
      "value-key": "[FLAG]",
      "command-line-flag": "--enable-feature"
    },
    {
      "id": "choice_input",
      "name": "Choice Parameter",
      "type": "String",
      "value-key": "[CHOICE]",
      "value-choices": ["option1", "option2", "option3"]
    },
    {
      "id": "list_input",
      "name": "List of Files",
      "type": "File",
      "list": true,
      "value-key": "[FILES]"
    }
  ]
}
```

### Parameter Constraints

```json
{
  "inputs": [
    {
      "id": "threshold",
      "name": "Threshold Value",
      "type": "Number",
      "value-key": "[THRESH]",
      "minimum": 0,
      "maximum": 1,
      "exclusive-minimum": false,
      "exclusive-maximum": false
    },
    {
      "id": "atlas",
      "name": "Atlas Choice",
      "type": "String",
      "value-key": "[ATLAS]",
      "value-choices": ["AAL", "Harvard-Oxford", "Schaefer"],
      "default-value": "AAL"
    },
    {
      "id": "pattern",
      "name": "Subject Pattern",
      "type": "String",
      "value-key": "[PATTERN]",
      "value-pattern": "^sub-[0-9]+$"
    }
  ]
}
```

---

## Validating Descriptors

### Schema Validation

```bash
# Validate descriptor against Boutiques schema
bosh validate fsl-bet.json

# Expected output:
# Descriptor is valid

# If errors exist:
# Error: Missing required field 'name'
# Error: Invalid value for 'schema-version'
```

### Common Validation Errors

```bash
# Missing required field
# Error: 'name' is required

# Invalid schema version
# Error: 'schema-version' must be '0.5'

# Invalid value-key reference
# Error: value-key '[NONEXISTENT]' not found in command-line

# Type mismatch
# Error: 'minimum' requires type 'Number', but input is 'String'

# Fix errors and re-validate
bosh validate fixed-descriptor.json
```

---

## Creating Invocations

### Invocation JSON

An invocation specifies values for a descriptor's inputs:

```json
{
  "input_image": "/data/T1.nii.gz",
  "output_basename": "/output/T1_brain",
  "frac_threshold": 0.6
}
```

### Create and Validate Invocation

```bash
# Create invocation file
cat > bet-invocation.json <<'EOF'
{
  "input_image": "/data/sub-01_T1w.nii.gz",
  "output_basename": "/output/sub-01_brain",
  "frac_threshold": 0.5
}
EOF

# Validate invocation against descriptor
bosh invocation -i bet-invocation.json fsl-bet.json

# Boutiques checks:
# - All required inputs provided
# - Values match type constraints
# - File paths exist (if applicable)
# - Numeric values within min/max bounds
```

### Generate Example Invocation

```bash
# Boutiques can generate example invocation
bosh example fsl-bet.json > example-invocation.json

# Edit example with your values
nano example-invocation.json

# Validate
bosh invocation -i example-invocation.json fsl-bet.json
```

---

## Executing Tools

### Local Execution

```bash
# Execute tool locally with descriptor and invocation
bosh exec launch \
  fsl-bet.json \
  bet-invocation.json

# Boutiques:
# 1. Validates invocation
# 2. Generates command line
# 3. Executes command
# 4. Captures outputs
# 5. Returns results

# View generated command (dry run)
bosh exec simulate \
  fsl-bet.json \
  bet-invocation.json

# Output: bet /data/sub-01_T1w.nii.gz /output/sub-01_brain -f 0.5
```

### Execute with Output Capture

```bash
# Execute and capture outputs
bosh exec launch \
  --stream \
  fsl-bet.json \
  bet-invocation.json \
  > execution-output.json

# execution-output.json contains:
# - Stdout/stderr
# - Exit code
# - Output file paths
# - Execution time
```

---

## Container Integration

### Docker Container Execution

```json
{
  "name": "FSL BET (Docker)",
  "tool-version": "6.0.5",
  "command-line": "bet [INPUT] [OUTPUT] -f [FRAC]",
  "schema-version": "0.5",
  "container-image": {
    "type": "docker",
    "image": "brainlife/fsl:6.0.5"
  },
  "inputs": [
    ...
  ]
}
```

```bash
# Execute in Docker container
bosh exec launch \
  -x \
  --imagepath /tmp/boutiques-cache \
  fsl-bet-docker.json \
  bet-invocation.json

# Boutiques:
# - Pulls Docker image if needed
# - Mounts data directories
# - Executes in container
# - Retrieves outputs
```

### Singularity Container Execution

```json
{
  "container-image": {
    "type": "singularity",
    "image": "docker://brainlife/fsl:6.0.5"
  }
}
```

```bash
# Execute with Singularity
bosh exec launch \
  -x \
  --imagepath /scratch/singularity-cache \
  fsl-bet-singularity.json \
  bet-invocation.json

# Boutiques converts Docker URI to Singularity if needed
```

### Volume Mounts

```json
{
  "custom": {
    "docker-volume-mounts": [
      {
        "host-path": "/data",
        "container-path": "/data"
      },
      {
        "host-path": "/output",
        "container-path": "/output"
      }
    ]
  }
}
```

---

## Output Specification

### Output Files

```json
{
  "output-files": [
    {
      "id": "brain_mask",
      "name": "Brain Mask",
      "path-template": "[OUTPUT]_brain_mask.nii.gz",
      "description": "Binary brain mask",
      "optional": false
    },
    {
      "id": "skull_image",
      "name": "Skull Image",
      "path-template": "[OUTPUT]_skull.nii.gz",
      "description": "Skull-stripped image",
      "optional": true
    }
  ]
}
```

### Path Templates

```json
{
  "output-files": [
    {
      "id": "output",
      "path-template": "[OUTPUT_DIR]/[SUBJECT]_[MODALITY]_processed.nii.gz",
      "path-template-stripped-extensions": [".nii", ".nii.gz"]
    }
  ]
}
```

### Output Validation

```bash
# Execute and verify outputs exist
bosh exec launch fsl-bet.json bet-invocation.json

# Boutiques checks:
# - Required outputs created
# - Output files at expected paths
# - Warns about missing optional outputs
```

---

## Advanced Descriptor Features

### Parameter Groups

```json
{
  "groups": [
    {
      "id": "basic_options",
      "name": "Basic Options",
      "description": "Fundamental processing options",
      "members": ["input_file", "output_file"]
    },
    {
      "id": "advanced_options",
      "name": "Advanced Options",
      "description": "Expert-level parameters",
      "members": ["frac_threshold", "gradient_threshold"],
      "mutually-exclusive": false
    }
  ]
}
```

### Conditional Parameters

```json
{
  "inputs": [
    {
      "id": "enable_registration",
      "name": "Enable Registration",
      "type": "Flag",
      "value-key": "[REGISTER]",
      "command-line-flag": "--register"
    },
    {
      "id": "registration_template",
      "name": "Template",
      "type": "File",
      "value-key": "[TEMPLATE]",
      "optional": true,
      "requires-inputs": ["enable_registration"]
    },
    {
      "id": "fast_mode",
      "name": "Fast Mode",
      "type": "Flag",
      "value-key": "[FAST]",
      "command-line-flag": "--fast",
      "disables-inputs": ["high_quality"]
    }
  ]
}
```

### Environment Variables

```json
{
  "environment-variables": [
    {
      "name": "FSLDIR",
      "value": "/opt/fsl"
    },
    {
      "name": "OMP_NUM_THREADS",
      "value": "8"
    }
  ]
}
```

---

## Tool Testing

### Define Test Invocations

```json
{
  "tests": [
    {
      "name": "Basic test",
      "invocation": {
        "input_image": "/test/data/T1.nii.gz",
        "output_basename": "/test/output/T1_brain"
      },
      "assertions": {
        "exit-code": 0,
        "output-files": [
          {
            "id": "brain_image",
            "exists": true
          }
        ]
      }
    }
  ]
}
```

### Run Tests

```bash
# Execute all tests in descriptor
bosh test fsl-bet.json

# Output:
# Running test: Basic test
# ✓ Exit code: 0
# ✓ Output file exists: /test/output/T1_brain_brain.nii.gz
# All tests passed

# Run specific test
bosh test --name "Basic test" fsl-bet.json
```

### Automated Testing

```bash
# Test tool across platforms
bosh test --executor docker fsl-bet-docker.json
bosh test --executor singularity fsl-bet-singularity.json
bosh test --executor local fsl-bet.json

# Continuous integration
# .github/workflows/test.yml
# - name: Test Boutiques descriptor
#   run: bosh test my-tool.json
```

---

## Publishing and Sharing

### Publish to Zenodo

```bash
# Publish descriptor to Zenodo for permanent DOI
bosh publish fsl-bet.json

# Boutiques will:
# 1. Validate descriptor
# 2. Upload to Zenodo sandbox (test)
# 3. Request confirmation
# 4. Publish with DOI

# You need Zenodo API token
export ZENODO_ACCESS_TOKEN=your_token

# Publish to production
bosh publish --sandbox false fsl-bet.json

# Returns DOI: 10.5281/zenodo.XXXXX
```

### Search Published Tools

```bash
# Search Boutiques tool registry
bosh search "brain extraction"

# Output:
# FSL BET (zenodo.XXXXX)
# ANTs BrainExtraction (zenodo.YYYYY)

# Get descriptor by ID
bosh pull zenodo.XXXXX

# Descriptor downloaded as fsl-bet.json
```

### Import from URL

```bash
# Import descriptor from Zenodo DOI
bosh import 10.5281/zenodo.XXXXX

# Or from direct URL
bosh import https://example.com/tool-descriptor.json

# Imported descriptor is validated and ready to use
```

---

## Integration with Workflows

### Use with Pydra

```python
from pydra import Workflow
from pydra.engine.boutiques import BoutiquesTask

# Create task from Boutiques descriptor
bet_task = BoutiquesTask(
    descriptor='fsl-bet.json',
    name='brain_extraction'
)

# Create workflow
wf = Workflow(name='preprocessing', input_spec=['t1w'])
wf.add(bet_task(
    input_image=wf.lzin.t1w,
    output_basename='brain',
    frac_threshold=0.5
))
wf.set_output([('brain', wf.brain_extraction.lzout.brain_image)])

# Execute
result = wf(t1w='/data/T1.nii.gz')
```

### Use with CBRAIN

```bash
# CBRAIN is a web platform for distributed computing
# Boutiques descriptors can be registered as CBRAIN tools

# Export descriptor for CBRAIN
bosh export cbrain fsl-bet.json > fsl-bet-cbrain.json

# Upload to CBRAIN portal
# Tools are automatically integrated
```

### Integration with Nipype

```python
from nipype.interfaces.base import BoutiquesInterface

# Wrap Boutiques tool as Nipype interface
bet_interface = BoutiquesInterface(
    descriptor='fsl-bet.json',
    container='docker'
)

# Use in Nipype workflow
from nipype import Workflow, Node

wf = Workflow(name='extraction')
bet_node = Node(bet_interface, name='bet')
bet_node.inputs.input_image = '/data/T1.nii.gz'
bet_node.inputs.output_basename = 'brain'

wf.add_nodes([bet_node])
wf.run()
```

---

## BIDS Apps Integration

### BIDS App Descriptor

```json
{
  "name": "My BIDS App",
  "tool-version": "1.0.0",
  "description": "BIDS App for fMRI preprocessing",
  "command-line": "my_app [BIDS_DIR] [OUTPUT_DIR] [ANALYSIS_LEVEL] [OPTIONS]",
  "schema-version": "0.5",
  "inputs": [
    {
      "id": "bids_dir",
      "name": "BIDS Directory",
      "type": "File",
      "value-key": "[BIDS_DIR]",
      "description": "Input BIDS dataset directory",
      "optional": false
    },
    {
      "id": "output_dir",
      "name": "Output Directory",
      "type": "String",
      "value-key": "[OUTPUT_DIR]",
      "description": "Output directory",
      "optional": false
    },
    {
      "id": "analysis_level",
      "name": "Analysis Level",
      "type": "String",
      "value-key": "[ANALYSIS_LEVEL]",
      "value-choices": ["participant", "group"],
      "description": "Analysis level",
      "optional": false
    },
    {
      "id": "participant_label",
      "name": "Participant Labels",
      "type": "String",
      "list": true,
      "value-key": "[OPTIONS]",
      "command-line-flag": "--participant-label",
      "description": "Process specific participants",
      "optional": true
    }
  ],
  "container-image": {
    "type": "docker",
    "image": "mybidsapp/mybidsapp:1.0.0"
  },
  "tags": {
    "domain": "neuroimaging",
    "modality": ["mri"],
    "application-type": "bids"
  }
}
```

### Execute BIDS App

```bash
# Create BIDS App invocation
cat > bidsapp-invocation.json <<'EOF'
{
  "bids_dir": "/data/bids_dataset",
  "output_dir": "/output",
  "analysis_level": "participant",
  "participant_label": ["01", "02", "03"]
}
EOF

# Execute BIDS App via Boutiques
bosh exec launch \
  -x \
  mybidsapp.json \
  bidsapp-invocation.json
```

---

## Best Practices

### Descriptor Design

```bash
# 1. Use descriptive names and IDs
# Good:
"id": "fractional_intensity_threshold"
# Bad:
"id": "param1"

# 2. Provide detailed descriptions
"description": "Fractional intensity threshold (0-1); smaller values give larger brain outline estimates"

# 3. Set sensible defaults
"default-value": 0.5

# 4. Include constraints
"minimum": 0,
"maximum": 1

# 5. Group related parameters
"groups": [{"id": "advanced", "members": [...]}]
```

### Versioning

```json
{
  "tool-version": "6.0.5",
  "descriptor-url": "https://github.com/user/descriptors/v1.0.0/fsl-bet.json",
  "tags": {
    "version-tag": "v1.0.0"
  }
}
```

### Documentation

```bash
# Generate documentation from descriptor
bosh docgen fsl-bet.json > fsl-bet-docs.md

# Creates markdown documentation:
# - Tool overview
# - Input descriptions
# - Output specifications
# - Example usage
```

---

## Error Handling and Debugging

### Validation Errors

```bash
# Detailed error reporting
bosh validate --verbose my-tool.json

# Common errors and fixes:

# Missing command-line value-key
# Error: value-key [MISSING] not in command-line template
# Fix: Add [MISSING] to command-line or fix value-key

# Invalid JSON syntax
# Error: Expecting property name enclosed in double quotes
# Fix: Check JSON syntax (commas, quotes)

# Schema version mismatch
# Error: schema-version must be 0.5
# Fix: Update to current schema version
```

### Execution Debugging

```bash
# Simulate execution (dry run)
bosh exec simulate \
  --verbose \
  my-tool.json \
  invocation.json

# Shows:
# - Generated command line
# - Environment variables
# - Working directory
# - Container configuration (if applicable)

# Debug failed execution
bosh exec launch \
  --stream \
  --debug \
  my-tool.json \
  invocation.json \
  2>&1 | tee debug.log
```

### Container Issues

```bash
# Test container separately
docker run -it brainlife/fsl:6.0.5 bet --help

# Check volume mounts
bosh exec launch \
  --mount /data:/data \
  --mount /output:/output \
  my-tool-docker.json \
  invocation.json

# Use local executor if container fails
bosh exec launch \
  --executor local \
  my-tool.json \
  invocation.json
```

---

## Example: FreeSurfer recon-all Descriptor

```json
{
  "name": "FreeSurfer recon-all",
  "tool-version": "7.2.0",
  "description": "FreeSurfer cortical reconstruction pipeline",
  "command-line": "recon-all -s [SUBJECT_ID] -i [T1_INPUT] [DIRECTIVES]",
  "schema-version": "0.5",
  "container-image": {
    "type": "docker",
    "image": "freesurfer/freesurfer:7.2.0"
  },
  "environment-variables": [
    {
      "name": "SUBJECTS_DIR",
      "value": "/output"
    }
  ],
  "inputs": [
    {
      "id": "subject_id",
      "name": "Subject ID",
      "type": "String",
      "value-key": "[SUBJECT_ID]",
      "description": "Subject identifier",
      "optional": false
    },
    {
      "id": "t1_input",
      "name": "T1 Image",
      "type": "File",
      "value-key": "[T1_INPUT]",
      "description": "T1-weighted anatomical image",
      "optional": false
    },
    {
      "id": "run_all",
      "name": "Run All Stages",
      "type": "Flag",
      "value-key": "[DIRECTIVES]",
      "command-line-flag": "-all",
      "description": "Run all processing stages",
      "optional": true
    },
    {
      "id": "parallel",
      "name": "Parallel Processing",
      "type": "Flag",
      "value-key": "[DIRECTIVES]",
      "command-line-flag": "-parallel",
      "description": "Enable parallel processing",
      "optional": true
    }
  ],
  "output-files": [
    {
      "id": "subject_dir",
      "name": "Subject Directory",
      "path-template": "/output/[SUBJECT_ID]",
      "description": "FreeSurfer subject directory",
      "optional": false
    }
  ],
  "tags": {
    "domain": "neuroimaging",
    "modality": ["mri"],
    "software": "freesurfer"
  }
}
```

---

## Related Tools and Integration

**Container Generation:**
- **NeuroDocker** (Batch 28): Generate containers for Boutiques tools
- **Docker**: Container execution platform
- **Singularity**: HPC container platform

**Workflow Engines:**
- **Pydra** (Batch 28): Execute Boutiques tools in workflows
- **Nipype** (Batch 2): Boutiques interface support
- **CBRAIN**: Web-based distributed computing

**Neuroimaging Tools:**
- **FSL** (Batch 1): Described via Boutiques
- **FreeSurfer** (Batch 1): Boutiques descriptors available
- **AFNI** (Batch 1): Tool standardization

---

## References

- Glatard, T., et al. (2018). Boutiques: a flexible framework for automated application integration in computing platforms. *GigaScience*, 7(5), giy016.
- Sherif, T., et al. (2014). CBRAIN: a web-based, distributed computing platform for collaborative neuroimaging research. *Frontiers in Neuroinformatics*, 8, 54.
- Kiar, G., et al. (2020). Data augmentation through Monte Carlo arithmetic leads to more generalizable classification in connectomics. *bioRxiv*.

**Official Website:** https://boutiques.github.io/
**GitHub Repository:** https://github.com/boutiques/boutiques
**Tool Registry:** https://boutiques.github.io/tools
**Schema Specification:** https://github.com/boutiques/boutiques/blob/master/schema/descriptor.schema.json
**Paper:** https://doi.org/10.1093/gigascience/giy016

## Troubleshooting

- **Descriptor validation fails:** Re-run `boutiques validate <descriptor.json>` and ensure schema version matches your installed Boutiques version.
- **Invocation crashes in container:** Confirm the container image tag matches the descriptor and that mounts/paths are correct.
- **Parameter parsing errors:** Check command-line placeholders in the descriptor for missing `{}` or mismatched types.

## Resources

- Documentation: https://boutiques.github.io/
- Descriptor examples: https://github.com/boutiques/boutiques/tree/master/examples
- Validator: https://github.com/boutiques/boutiques/tree/master/tools/python/boutiques/validator
- Support: https://neurostars.org/ (tag boutiques)

#!/usr/bin/env python3
"""Generate Claude Code skills for neuroimaging tools from CSV data."""

import csv
import os
import re
from pathlib import Path


def sanitize_filename(name):
    """Convert tool name to valid filename."""
    # Remove parentheses and their contents
    name = re.sub(r'\([^)]*\)', '', name)
    # Replace spaces and special chars with hyphens
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[-\s]+', '-', name)
    return name.strip('-').lower()


def get_tool_category(tool_info):
    """Determine tool category based on its purpose."""
    name = tool_info['name'].lower()

    if any(x in name for x in ['preprocessing', 'prep', 'bids']):
        return 'preprocessing'
    elif any(x in name for x in ['viewer', 'visual', 'render', 'display']):
        return 'visualization'
    elif any(x in name for x in ['eeg', 'meg', 'field']):
        return 'electrophysiology'
    elif any(x in name for x in ['dti', 'dwi', 'tract', 'diffusion']):
        return 'diffusion'
    elif any(x in name for x in ['pet', 'spectroscopy', 'mrs']):
        return 'pet-mrs'
    elif any(x in name for x in ['connect', 'network', 'graph']):
        return 'connectivity'
    elif any(x in name for x in ['segment', 'parcell', 'brain-extract']):
        return 'segmentation'
    elif any(x in name for x in ['registration', 'transform', 'warp']):
        return 'registration'
    elif any(x in name for x in ['stat', 'glm', 'analysis']):
        return 'statistics'
    elif any(x in name for x in ['deep', 'neural', 'learn']):
        return 'machine-learning'
    else:
        return 'general'


def generate_skill_content(tool_info):
    """Generate skill markdown content for a tool."""
    name = tool_info['name']
    website = tool_info.get('website', 'N/A')
    platform = tool_info.get('platform', 'N/A')
    languages = tool_info.get('languages', 'N/A')
    category = get_tool_category(tool_info)

    # Build language-specific installation guidance
    install_section = generate_install_section(languages, platform)

    # Build usage guidance
    usage_section = generate_usage_section(name, languages, category)

    content = f"""# {name}

**Category:** Neuroimaging - {category.replace('-', ' ').title()}

## Overview

{name} is a neuroimaging tool used for brain imaging analysis and processing.

- **Website:** {website}
- **Platform:** {platform}
- **Languages:** {languages}

## Installation

{install_section}

## Usage Guidance

{usage_section}

## Common Use Cases

{generate_use_cases(category)}

## Tips

- Check the official documentation at {website} for detailed usage
- Ensure all dependencies are properly installed before running
- Consider using containerized versions (Docker/Singularity) for reproducibility
{generate_language_specific_tips(languages)}

## Integration with Claude Code

When working with {name}:
1. Verify the tool is installed and accessible in your environment
2. Check version compatibility with your data format
3. Use appropriate file paths and ensure proper permissions
4. Monitor memory usage for large datasets
5. Validate outputs before proceeding with analysis

## Resources

- Official documentation: {website}
- Platform: {platform}
"""

    return content


def generate_install_section(languages, platform):
    """Generate installation guidance based on language and platform."""
    if 'Python' in languages:
        return """### Python Installation

```bash
# Using pip
pip install {tool_name}

# Using conda
conda install -c conda-forge {tool_name}
```

Check the official documentation for specific installation instructions."""

    elif 'MATLAB' in languages:
        return """### MATLAB Installation

1. Download the toolbox from the official website
2. Extract to your MATLAB toolbox directory
3. Add to MATLAB path:
   ```matlab
   addpath('/path/to/toolbox');
   savepath;
   ```"""

    elif 'Docker' in platform or 'Singularity' in platform:
        return """### Container Installation

```bash
# Docker
docker pull {container_image}

# Singularity
singularity pull {container_image}
```

Refer to the official documentation for the specific container image name."""

    elif 'Node.js' in languages:
        return """### Node.js Installation

```bash
npm install -g {tool_name}
```"""

    elif 'Java' in languages:
        return """### Java Installation

Download the JAR file or installer from the official website and follow platform-specific installation instructions."""

    else:
        return """### Installation

Download and install from the official website. Follow platform-specific installation instructions provided in the documentation."""


def generate_usage_section(name, languages, category):
    """Generate usage examples based on tool type."""
    if 'Python' in languages:
        return f"""### Python Usage

```python
import {sanitize_filename(name).replace('-', '_')}

# Example workflow
# Load data, process, and save results
# Refer to documentation for specific API usage
```"""

    elif 'MATLAB' in languages:
        return f"""### MATLAB Usage

```matlab
% Add toolbox to path
% Run analysis scripts
% Refer to documentation for specific functions
```"""

    else:
        return f"""### Command-line Usage

```bash
# Example command
{sanitize_filename(name)} [options] input_file output_file
```

Refer to the tool's documentation for specific command-line options."""


def generate_use_cases(category):
    """Generate common use cases based on category."""
    use_cases = {
        'preprocessing': """- fMRI/MRI data preprocessing
- Motion correction and artifact removal
- Spatial normalization
- Quality control""",

        'visualization': """- Brain image visualization
- 3D surface rendering
- Statistical map overlay
- ROI visualization""",

        'electrophysiology': """- EEG/MEG signal processing
- Source localization
- Time-frequency analysis
- Event-related potential analysis""",

        'diffusion': """- DTI/DWI preprocessing
- Tractography
- White matter analysis
- Connectivity mapping""",

        'connectivity': """- Functional connectivity analysis
- Graph theory metrics
- Network visualization
- Hub identification""",

        'segmentation': """- Brain tissue segmentation
- Subcortical structure parcellation
- Cortical surface extraction
- Lesion segmentation""",

        'registration': """- Inter-subject registration
- Template warping
- Spatial normalization
- Atlas alignment""",

        'statistics': """- Statistical parametric mapping
- Group analysis
- Multiple comparison correction
- ROI-based statistics""",

        'machine-learning': """- Deep learning for segmentation
- Automated diagnosis
- Image synthesis
- Quality assessment""",

        'general': """- Neuroimaging data analysis
- Brain imaging research
- Clinical applications
- Research workflows"""
    }

    return use_cases.get(category, use_cases['general'])


def generate_language_specific_tips(languages):
    """Generate language-specific tips."""
    tips = []

    if 'Python' in languages:
        tips.append("- Use virtual environments to manage dependencies")
        tips.append("- Check Python version compatibility")

    if 'MATLAB' in languages:
        tips.append("- Ensure MATLAB license is valid and accessible")
        tips.append("- Check toolbox dependencies")

    if 'Docker' in languages or 'Singularity' in languages:
        tips.append("- Bind mount data directories appropriately")
        tips.append("- Consider using BIDS format for compatibility")

    return '\n'.join(tips) if tips else ''


def read_tool_data():
    """Read and merge tool data from CSV files."""
    tools = {}

    # Read main toolboxes file
    with open('neuroimaging_toolboxes.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['Name'].strip()
            if name:  # Skip empty rows
                tools[name] = {
                    'name': name,
                    'website': row['Website'].strip(),
                    'platform': row['Platform'].strip(),
                    'languages': 'N/A'
                }

    # Add language information
    with open('tool_languages.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['Name'].strip()
            if name in tools and row['Languages'].strip():
                tools[name]['languages'] = row['Languages'].strip()

    return list(tools.values())


def main():
    """Generate all skills."""
    skills_dir = Path('.claude/skills/neuroimaging')
    skills_dir.mkdir(parents=True, exist_ok=True)

    tools = read_tool_data()

    print(f"Generating skills for {len(tools)} tools...")

    for tool_info in tools:
        filename = sanitize_filename(tool_info['name']) + '.md'
        filepath = skills_dir / filename

        content = generate_skill_content(tool_info)

        with open(filepath, 'w') as f:
            f.write(content)

        print(f"Created: {filename}")

    # Create index file
    create_index_file(skills_dir, tools)

    print(f"\nSuccessfully generated {len(tools)} skills in {skills_dir}")


def create_index_file(skills_dir, tools):
    """Create an index of all skills."""
    index_content = """# Neuroimaging Tools Skills

This directory contains Claude Code skills for neuroimaging analysis tools.

## Available Tools

"""

    # Group by category
    by_category = {}
    for tool in tools:
        category = get_tool_category(tool)
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(tool)

    # Generate index by category
    for category in sorted(by_category.keys()):
        index_content += f"\n### {category.replace('-', ' ').title()}\n\n"
        for tool in sorted(by_category[category], key=lambda x: x['name']):
            filename = sanitize_filename(tool['name']) + '.md'
            index_content += f"- [{tool['name']}]({filename})\n"

    index_content += f"\n\n**Total Tools:** {len(tools)}\n"

    with open(skills_dir / 'README.md', 'w') as f:
        f.write(index_content)

    print("Created: README.md (index)")


if __name__ == '__main__':
    main()

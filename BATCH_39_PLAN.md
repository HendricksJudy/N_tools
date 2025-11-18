# Batch 39 Plan: MEG/EEG Forward Modeling

## Overview

**Theme:** MEG/EEG Forward Modeling
**Focus:** Solving the electromagnetic forward problem for source localization
**Target:** 1 skill, 600-650 lines

**Current Progress:** 124/133 skills (93.2%)
**After Batch 38:** 124/133 skills (93.2%)
**After Batch 39:** 125/133 skills (93.9%)

This batch addresses the critical forward problem in MEG/EEG source localization: computing how neural sources generate signals measured at scalp sensors. OpenMEEG provides a comprehensive solution for electromagnetic forward modeling using realistic head geometries derived from MRI.

## Rationale

MEG/EEG source localization requires solving two problems:
1. **Forward Problem:** Given a source location, predict sensor measurements (OpenMEEG)
2. **Inverse Problem:** Given sensor measurements, estimate source locations (MNE-Python, FieldTrip)

OpenMEEG specializes in the forward problem, which is essential for:
- Accurate source localization (EEG/MEG)
- Lead field matrix computation
- Realistic head modeling
- Multi-compartment volume conduction
- Integration with inverse solvers

## Skill to Create

### OpenMEEG (600-650 lines, 20-24 examples)

**Overview:**
OpenMEEG is a C++ software package for solving the electromagnetic forward problem in MEG and EEG. It uses the Boundary Element Method (BEM) to compute how electrical currents in the brain propagate through different tissue compartments (scalp, skull, CSF, brain) to produce signals at MEG/EEG sensors. OpenMEEG integrates with MRI-derived head models and provides accurate lead field matrices essential for source localization.

**Key Features:**
- Boundary Element Method (BEM) for forward modeling
- Realistic head models from MRI segmentation
- Multi-layer geometry (scalp, skull, CSF, brain)
- Supports both MEG and EEG
- Symmetric BEM for efficiency
- Integration with MNE-Python, FieldTrip, Brainstorm
- Lead field matrix computation
- Python and MATLAB interfaces
- GPU acceleration (optional)

**Target Audience:**
- MEG/EEG researchers performing source localization
- Clinical neurophysiologists
- Computational neuroscientists
- Researchers developing inverse methods

**Main Sections:**

1. **Overview** (50 lines)
   - Introduction to electromagnetic forward problem
   - BEM vs. FEM approaches
   - OpenMEEG architecture
   - Applications
   - Citation

2. **Installation** (70 lines)
   - Binary installation (Windows, Mac, Linux)
   - Python package (openmeeg-python)
   - MATLAB interface
   - Building from source
   - Dependencies

3. **Head Model Creation** (100 lines, 3-4 examples)
   - MRI segmentation to surfaces
   - Multi-layer models (3-layer, 4-layer)
   - Surface mesh generation
   - Quality control
   - Example: Create 3-layer BEM

4. **Forward Solution Computation** (110 lines, 4-5 examples)
   - Lead field matrix calculation
   - Source space definition
   - Sensor configuration
   - EEG vs MEG forward models
   - Example: Compute EEG lead fields

5. **Integration with MNE-Python** (100 lines, 3-4 examples)
   - Using OpenMEEG as MNE backend
   - Head model preparation
   - Source localization workflow
   - Example: Complete MEG pipeline

6. **Advanced Features** (80 lines, 2-3 examples)
   - Anisotropic conductivity
   - Dipole fitting
   - Validation against analytical models
   - Example: White matter anisotropy

7. **Quality Control** (60 lines, 2-3 examples)
   - Geometry validation
   - Forward solution verification
   - Topography visualization

8. **Troubleshooting** (50 lines)
   - Common errors
   - Mesh issues
   - Numerical problems

9. **Best Practices** (40 lines)
   - Tissue conductivity values
   - Mesh resolution
   - Source space density

10. **References** (20 lines)
    - OpenMEEG papers
    - BEM methodology
    - Applications

**Code Examples:**
- Create BEM model (Python)
- Compute lead fields (Python)
- MNE integration (Python)
- Validation (Python)
- Advanced modeling (Python)

**Integration Points:**
- MNE-Python for inverse solution
- FreeSurfer for head segmentation
- FieldTrip for MEG/EEG analysis
- Brainstorm for source imaging

## Implementation Checklist

- [ ] 600-650 lines
- [ ] 20-24 code examples
- [ ] Consistent structure
- [ ] Installation instructions
- [ ] Basic and advanced usage
- [ ] Integration examples
- [ ] Troubleshooting
- [ ] Best practices
- [ ] References

## Timeline

**OpenMEEG**: 600-650 lines, 20-24 examples

**Estimated Total:** 600-650 lines, 20-24 examples

## Context & Connections

### Forward Modeling Framework

```
MRI → Segmentation → BEM Surfaces → OpenMEEG → Lead Fields → Inverse Solution
  ↓        ↓             ↓              ↓            ↓            ↓
T1w   FreeSurfer    Mesh Generation  Forward    Source Space  MNE/FieldTrip
```

### Complementary Tools

**Already Covered:**
- **MNE-Python**: Inverse solution (uses OpenMEEG backend)
- **FieldTrip**: Alternative forward/inverse solver
- **FreeSurfer**: MRI segmentation for head model

**New Capability:**
- **OpenMEEG**: Accurate BEM forward solution

## Expected Impact

### Research Community
- Improved source localization accuracy
- Realistic head modeling
- Multi-modal MEG/EEG analysis

### Clinical Applications
- Epilepsy focus localization
- Pre-surgical planning
- Brain-computer interfaces

## Conclusion

Batch 39 provides the essential forward modeling tool OpenMEEG, completing the MEG/EEG analysis pipeline. This brings the collection to **125/133 skills (93.9%)**, with comprehensive coverage of electromagnetic source imaging.

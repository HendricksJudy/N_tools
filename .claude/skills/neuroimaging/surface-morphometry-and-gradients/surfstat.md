# SurfStat

## Overview

SurfStat is a MATLAB toolbox for statistical analysis of surface-based neuroimaging data, particularly cortical surfaces from FreeSurfer and CIVET. Developed by Keith Worsley at McGill University, SurfStat implements linear mixed-effects models and random field theory specifically designed for cortical surface analysis. It excels at analyzing cortical thickness, surface area, curvature, and other vertex-wise measures while properly accounting for the topological structure and spatial smoothness of cortical surfaces.

**Website:** http://www.math.mcgill.ca/keith/surfstat/
**Platform:** MATLAB (Windows/macOS/Linux)
**Language:** MATLAB
**License:** Free for academic use

## Key Features

- Linear models for surface data (fixed and mixed effects)
- Random field theory for multiple comparison correction
- Vertex-wise and cluster-wise inference
- Cortical thickness and surface area analysis
- Support for FreeSurfer and CIVET surfaces
- Longitudinal and repeated-measures designs
- Surface-based smoothing (geodesic and heat kernel)
- Integration with FreeSurfer parcellations
- Visualization on inflated and pial surfaces
- Bilateral hemisphere analysis
- Multivariate analysis across vertices
- Sophisticated contrast specification

## Installation

### Download SurfStat

```bash
# Download from website
# http://www.math.mcgill.ca/keith/surfstat/

# Extract archive
unzip surfstat.zip -d /path/to/surfstat
```

### MATLAB Setup

```matlab
% Add SurfStat to MATLAB path
addpath(genpath('/path/to/surfstat'));
savepath;

% Verify installation
which SurfStatReadSurf
which SurfStatLinMod

% Check version
% SurfStat typically doesn't have version number
% Verify by checking for key functions
```

## Loading Surface Data

### Load FreeSurfer Surfaces

```matlab
% Load template surface mesh (inflated brain)
% FreeSurfer subject: fsaverage
surf_lh = SurfStatReadSurf('/path/to/freesurfer/fsaverage/surf/lh.inflated');
surf_rh = SurfStatReadSurf('/path/to/freesurfer/fsaverage/surf/rh.inflated');

% Combine left and right hemispheres
surf = SurfStatReadSurf({...
    '/path/to/freesurfer/fsaverage/surf/lh.inflated', ...
    '/path/to/freesurfer/fsaverage/surf/rh.inflated'});

% Surface structure contains:
% surf.coord: 3 × n_vertices coordinates
% surf.tri: n_faces × 3 triangle indices
```

### Load Cortical Thickness Data

```matlab
% Load thickness data for multiple subjects
subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05'};
n_subjects = length(subjects);

% Initialize thickness matrix
% Rows = subjects, Columns = vertices
thickness_lh = zeros(n_subjects, size(surf_lh.coord, 2));
thickness_rh = zeros(n_subjects, size(surf_rh.coord, 2));

for s = 1:n_subjects
    % Load left hemisphere thickness
    lh_file = sprintf('/data/%s/surf/lh.thickness.fwhm10.fsaverage.mgh', subjects{s});
    thickness_lh(s, :) = SurfStatReadData(lh_file);

    % Load right hemisphere thickness
    rh_file = sprintf('/data/%s/surf/rh.thickness.fwhm10.fsaverage.mgh', subjects{s});
    thickness_rh(s, :) = SurfStatReadData(rh_file);
end

% Combine hemispheres
thickness = [thickness_lh, thickness_rh];
[n_subjects, n_vertices] = size(thickness);

fprintf('Loaded thickness data: %d subjects, %d vertices\n', n_subjects, n_vertices);
```

### Load Subject Demographics

```matlab
% Read subject information from CSV
demo = readtable('subjects.csv');

% Extract variables
age = demo.age;
sex = demo.sex;  % 1 = Male, 0 = Female
group = demo.group;  % 1 = Patient, 0 = Control
subject_id = demo.subject_id;

% Verify alignment with thickness data
assert(length(age) == n_subjects, 'Demographics mismatch with thickness data');
```

## Basic Statistical Models

### Two-Group Comparison

```matlab
% Compare cortical thickness between patients and controls
% Model: thickness ~ 1 + group

% Create design term
term_group = term(group);

% Fit linear model at each vertex
slm = SurfStatLinMod(thickness, term_group, surf);

% Compute contrast (patients > controls)
contrast = group;  % or equivalently: [0 1] for [intercept, group]
slm = SurfStatT(slm, contrast);

% Multiple comparison correction using random field theory
[pval, peak, clus] = SurfStatP(slm, [], 0.05);

fprintf('Significant clusters: %d\n', length(clus.clusid));
if ~isempty(clus.clusid)
    for c = 1:length(clus.clusid)
        fprintf('  Cluster %d: %d vertices, p = %.4f\n', ...
                c, clus.nverts(c), clus.P(c));
    end
end
```

### One-Sample T-Test

```matlab
% Test if cortical thickness differs from zero (or another value)
% Useful for deviation from atlas or change from baseline

% Subtract baseline or atlas values
baseline_thickness = 2.5;  % mm (typical cortical thickness)
thickness_deviation = thickness - baseline_thickness;

% Model: deviation ~ 1 (intercept only)
slm = SurfStatLinMod(thickness_deviation, 1, surf);

% Test intercept (is mean thickness different from baseline?)
slm = SurfStatT(slm, 1);

% Inference
[pval, peak, clus] = SurfStatP(slm, [], 0.05);
```

### Regression with Continuous Variable

```matlab
% Correlate cortical thickness with age
% Model: thickness ~ 1 + age

term_age = term(age);
slm = SurfStatLinMod(thickness, term_age, surf);

% Test age effect
contrast_age = age;  % or term_age for newer syntax
slm = SurfStatT(slm, contrast_age);

% Multiple comparison correction
[pval, peak, clus] = SurfStatP(slm, [], 0.05);

% Visualize results
figure;
SurfStatViewData(slm.t, surf, 'Age effect on thickness');
SurfStatColLim([-5, 5]);  % T-statistic range
colormap(jet);
colorbar;
```

## ANCOVA with Covariates

### Control for Confounds

```matlab
% Compare groups while controlling for age and sex
% Model: thickness ~ 1 + group + age + sex

% Create design terms
term_group = term(group);
term_age = term(age);
term_sex = term(sex);

% Fit model
slm = SurfStatLinMod(thickness, 1 + term_group + term_age + term_sex, surf);

% Test group effect (controlling for age and sex)
slm = SurfStatT(slm, term_group);

% Inference
[pval, peak, clus] = SurfStatP(slm, [], 0.05);
```

### Interaction Effects

```matlab
% Test Group × Age interaction
% Model: thickness ~ 1 + group + age + group:age

% Create interaction term
term_interaction = term_group * term_age;

% Full model
slm = SurfStatLinMod(thickness, 1 + term_group + term_age + term_interaction, surf);

% Test interaction
slm = SurfStatT(slm, term_interaction);

% Inference
[pval, peak, clus] = SurfStatP(slm, [], 0.05);

% Interpretation: Does age effect differ between groups?
```

## Surface Smoothing

### Geodesic Smoothing

```matlab
% Smooth thickness data along the surface
% FWHM in mm (full-width half-maximum)

FWHM = 10;  % 10mm smoothing kernel

% Smooth each subject's data
thickness_smooth = zeros(size(thickness));

for s = 1:n_subjects
    thickness_smooth(s, :) = SurfStatSmooth(thickness(s, :), surf, FWHM);
end

% Use smoothed data for analysis
slm = SurfStatLinMod(thickness_smooth, term_group, surf);
```

### Heat Kernel Smoothing

```matlab
% Alternative: Heat kernel smoothing
% More isotropic than geodesic on irregular meshes

% Create heat kernel smoothing matrix
% Parameters: surface, FWHM
H = SurfStatHeatKernel(surf, FWHM);

% Apply smoothing
thickness_smooth = thickness * H;  % Matrix multiplication
```

## Longitudinal Analysis

### Repeated Measures

```matlab
% Longitudinal data: multiple timepoints per subject
% Subject IDs for repeated measures
subjects_long = {'sub-01', 'sub-01', 'sub-02', 'sub-02', 'sub-03', 'sub-03'};
timepoint = [1, 2, 1, 2, 1, 2];  % Visit number
subject_id = categorical(subjects_long);

% Load thickness for all visits
% thickness_long: n_visits × n_vertices

% Model with random subject effect
% thickness ~ 1 + timepoint + random(subject)

term_time = term(timepoint);
term_subject = term(subject_id);

% Random effects model
slm = SurfStatLinMod(thickness_long, 1 + term_time + random(term_subject), surf);

% Test time effect (change over visits)
slm = SurfStatT(slm, term_time);

% Inference
[pval, peak, clus] = SurfStatP(slm, [], 0.05);
```

### Growth Curve Analysis

```matlab
% Model cortical development over age
% Multiple scans per subject at different ages

age_visit = [10, 12, 14, 10, 12, 14, 11, 13, 11, 13];  % Age at scan
subject_id = categorical([1, 1, 1, 2, 2, 2, 3, 3, 4, 4]);

% Model: thickness ~ 1 + age + age^2 + random(subject)
term_age_linear = term(age_visit);
term_age_quad = term(age_visit.^2);
term_subject = term(subject_id);

slm = SurfStatLinMod(thickness_long, ...
    1 + term_age_linear + term_age_quad + random(term_subject), surf);

% Test quadratic age effect (non-linear development)
slm = SurfStatT(slm, term_age_quad);

[pval, peak, clus] = SurfStatP(slm, [], 0.05);
```

## Advanced Models

### Multi-Way ANOVA

```matlab
% Two factors: Group (Patient, Control) × Hemisphere (Left, Right)
% Test: Is there a hemisphere-specific group effect?

% Organize data
% Split thickness into separate LH and RH
n_subjects_total = 20;
thickness_lh_only = thickness(:, 1:size(surf_lh.coord, 2));
thickness_rh_only = thickness(:, size(surf_lh.coord, 2)+1:end);

% Stack data
thickness_stacked = [thickness_lh_only; thickness_rh_only];

% Create factors
group_stacked = [group; group];  % Repeat for each hemisphere
hemisphere = [zeros(n_subjects_total, 1); ones(n_subjects_total, 1)];  % LH=0, RH=1

% Model with interaction
term_group = term(group_stacked);
term_hemi = term(hemisphere);
term_interaction = term_group * term_hemi;

slm = SurfStatLinMod(thickness_stacked, 1 + term_group + term_hemi + term_interaction, surf_lh);

% Test interaction
slm = SurfStatT(slm, term_interaction);

[pval, peak, clus] = SurfStatP(slm, [], 0.05);
```

### Vertex-Wise Multivariate Analysis

```matlab
% Multivariate test across multiple measures
% E.g., thickness, curvature, surface area simultaneously

% Load multiple measures
thickness = ...; % n_subjects × n_vertices
curvature = ...; % n_subjects × n_vertices
surf_area = ...; % n_subjects × n_vertices

% Stack measures (3D array: subjects × vertices × measures)
Y = cat(3, thickness, curvature, surf_area);

% Test group effect on multivariate profile
% (Not directly supported in SurfStat, requires custom implementation)
% Alternative: Test each measure separately and combine p-values
```

## Visualization

### Basic Surface Plots

```matlab
% Plot T-statistics on surface
figure('Position', [100, 100, 1200, 600]);

% Left hemisphere
subplot(1, 2, 1);
SurfStatViewData(slm.t(:, 1:end/2), surf_lh, 'Left Hemisphere');
SurfStatColLim([-5, 5]);
colormap(jet);
colorbar;

% Right hemisphere
subplot(1, 2, 2);
SurfStatViewData(slm.t(:, end/2+1:end), surf_rh, 'Right Hemisphere');
SurfStatColLim([-5, 5]);
colormap(jet);
colorbar;
```

### Cluster Visualization

```matlab
% Highlight significant clusters
% Create cluster map (0 = non-significant, 1+ = cluster ID)
cluster_map = zeros(1, n_vertices);

if ~isempty(clus.clusid)
    for c = 1:length(clus.clusid)
        if clus.P(c) < 0.05
            cluster_map(clus.clusid{c}) = c;
        end
    end
end

% Plot clusters
figure;
SurfStatViewData(cluster_map, surf, 'Significant Clusters');
colormap(lines(max(cluster_map)));
colorbar;
```

### Inflated Surface Views

```matlab
% Load inflated surface for better visualization
surf_inflated = SurfStatReadSurf({...
    '/path/to/freesurfer/fsaverage/surf/lh.inflated', ...
    '/path/to/freesurfer/fsaverage/surf/rh.inflated'});

% Plot on inflated surface
figure;
SurfStatViewData(slm.t, surf_inflated, 'T-statistics (Inflated)');
SurfStatColLim([-5, 5]);
colormap(bluewhitered);  % Diverging colormap
colorbar;
```

### Multiple Views

```matlab
% Create publication-quality figure with multiple views
figure('Position', [100, 100, 1600, 1200]);

views = {'lateral', 'medial', 'dorsal', 'ventral'};
for v = 1:4
    subplot(2, 2, v);
    SurfStatViewData(slm.t, surf, sprintf('View: %s', views{v}));

    % Set viewing angle
    switch views{v}
        case 'lateral'
            view([-90, 0]);
        case 'medial'
            view([90, 0]);
        case 'dorsal'
            view([0, 90]);
        case 'ventral'
            view([0, -90]);
    end

    SurfStatColLim([-5, 5]);
    colormap(jet);
end

% Add colorbar to last subplot
colorbar;
```

## Integration with FreeSurfer

### Load FreeSurfer Parcellations

```matlab
% Load Desikan-Killiany parcellation
annot_lh = read_annotation('/path/to/freesurfer/fsaverage/label/lh.aparc.annot');
annot_rh = read_annotation('/path/to/freesurfer/fsaverage/label/rh.aparc.annot');

% Combine hemispheres
parcellation = [annot_lh.vertices; annot_rh.vertices];

% Get parcel labels
parcel_labels = annot_lh.colortable.struct_names;
```

### ROI-Based Analysis

```matlab
% Extract mean thickness per ROI
roi_names = annot_lh.colortable.struct_names;
n_rois = length(roi_names);

roi_thickness = zeros(n_subjects, n_rois);

for r = 1:n_rois
    % Find vertices in this ROI (left hemisphere only for example)
    vertices_in_roi = annot_lh.vertices == r;

    % Mean thickness across ROI vertices
    roi_thickness(:, r) = mean(thickness(:, vertices_in_roi), 2);
end

% Statistical test on ROI means
[h, p, ci, stats] = ttest2(roi_thickness(group == 0, :), ...
                            roi_thickness(group == 1, :));

% Find significant ROIs
sig_rois = find(p < 0.05);
fprintf('Significant ROIs (p < 0.05):\n');
for r = sig_rois
    fprintf('  %s: t = %.2f, p = %.4f\n', roi_names{r}, stats.tstat(r), p(r));
end
```

### Project Volume to Surface

```matlab
% Project volumetric fMRI contrast to surface for visualization
% Requires FreeSurfer's mri_vol2surf

% Run from MATLAB using system command
subject = 'fsaverage';
vol_file = '/data/group_contrast.nii';
surf_file_lh = '/results/lh.group_contrast.mgh';

cmd = sprintf(['mri_vol2surf --src %s --out %s --hemi lh ' ...
               '--projfrac 0.5 --surf white --regheader %s'], ...
               vol_file, surf_file_lh, subject);

system(cmd);

% Load projected data
surf_data_lh = SurfStatReadData(surf_file_lh);

% Visualize
SurfStatViewData(surf_data_lh, surf_lh, 'fMRI Contrast');
```

## Batch Processing

### Multi-Subject Pipeline

```matlab
% Automated analysis pipeline
clear; clc;

% Configuration
subjects_file = 'subject_list.txt';
freesurfer_dir = '/data/freesurfer/';
output_dir = '/results/surfstat/';

% Read subject list
subjects = readtable(subjects_file);
n_subjects = height(subjects);

% Load template surface
surf = SurfStatReadSurf({...
    [freesurfer_dir, 'fsaverage/surf/lh.inflated'], ...
    [freesurfer_dir, 'fsaverage/surf/rh.inflated']});

% Load thickness data
fprintf('Loading thickness data...\n');
thickness = zeros(n_subjects, size([surf{1}.coord, surf{2}.coord], 2));

for s = 1:n_subjects
    lh_file = sprintf('%s%s/surf/lh.thickness.fwhm10.fsaverage.mgh', ...
                      freesurfer_dir, subjects.ID{s});
    rh_file = sprintf('%s%s/surf/rh.thickness.fwhm10.fsaverage.mgh', ...
                      freesurfer_dir, subjects.ID{s});

    thickness(s, :) = [SurfStatReadData(lh_file), SurfStatReadData(rh_file)];
end

% Extract covariates
group = subjects.Group;
age = subjects.Age;
sex = subjects.Sex;

% Run analysis
fprintf('Running statistical analysis...\n');
term_group = term(group);
term_age = term(age);
term_sex = term(sex);

slm = SurfStatLinMod(thickness, 1 + term_group + term_age + term_sex, surf);
slm = SurfStatT(slm, term_group);

% Multiple comparison correction
[pval, peak, clus] = SurfStatP(slm, [], 0.05);

% Save results
fprintf('Saving results...\n');
save(fullfile(output_dir, 'surfstat_results.mat'), 'slm', 'pval', 'peak', 'clus');

% Generate figures
fprintf('Generating figures...\n');
figure;
SurfStatViewData(slm.t, surf, 'Group Effect');
saveas(gcf, fullfile(output_dir, 'group_tstat.png'));

fprintf('Analysis complete!\n');
```

## Multiple Comparison Correction

### Random Field Theory

```matlab
% SurfStat uses random field theory by default
% Corrects for multiple comparisons across vertices and clusters

% Run analysis
slm = SurfStatLinMod(thickness, term_group, surf);
slm = SurfStatT(slm, term_group);

% RFT correction
[pval, peak, clus] = SurfStatP(slm, [], 0.05);

% pval: vertex-wise corrected p-values
% peak: significant peaks with RFT correction
% clus: significant clusters with RFT correction

% Display peaks
if ~isempty(peak.vertid)
    fprintf('Significant peaks (RFT corrected):\n');
    for p = 1:length(peak.vertid)
        fprintf('  Vertex %d: t = %.2f, p = %.4f\n', ...
                peak.vertid(p), peak.t(p), peak.P(p));
    end
end
```

### Cluster-Based Inference

```matlab
% Cluster-forming threshold
cluster_thresh = 0.001;  % Uncorrected p-value for cluster formation

% Find clusters
[pval, peak, clus] = SurfStatP(slm, [], cluster_thresh);

% Significant clusters at alpha = 0.05
alpha = 0.05;
sig_clusters = find([clus.P] < alpha);

fprintf('Significant clusters (p < %.2f):\n', alpha);
for c = sig_clusters
    fprintf('  Cluster %d: %d vertices, peak t = %.2f, p = %.4f\n', ...
            c, clus.nverts(c), max(slm.t(clus.clusid{c})), clus.P(c));
end
```

## Troubleshooting

**Problem:** "Dimensions do not match" error
**Solution:** Ensure thickness data has same number of vertices as surface mesh; check hemisphere alignment

**Problem:** No significant results
**Solution:** Check data quality, verify preprocessing, try different smoothing, increase sample size

**Problem:** Very slow computation
**Solution:** Reduce number of vertices (use lower-resolution mesh), optimize MATLAB settings, use parallel computing

**Problem:** Surface plot looks incorrect
**Solution:** Verify surface file paths, check hemisphere order (LH then RH), ensure correct template (fsaverage)

**Problem:** RFT correction too conservative
**Solution:** Consider cluster-based thresholding, verify smoothness estimation, check for discrete parcellation effects

## Best Practices

1. **Data Quality:**
   - Visually inspect FreeSurfer reconstructions
   - Exclude subjects with poor segmentation
   - Check for outliers in thickness distributions

2. **Smoothing:**
   - Use 10-15mm FWHM for cortical thickness
   - Match smoothing to expected effect size and extent
   - Report smoothing kernel used

3. **Model Specification:**
   - Include relevant covariates (age, sex, scanner)
   - Center continuous predictors
   - Check for multicollinearity

4. **Multiple Comparisons:**
   - Use RFT or cluster-based correction
   - Report both corrected and uncorrected results
   - Consider FDR for exploratory analyses

5. **Reporting:**
   - Report surface template (fsaverage, fsaverage5)
   - Report smoothing FWHM
   - Report correction method and threshold
   - Provide cluster coordinates and sizes
   - Visualize results on surfaces

## Resources

- **Website:** http://www.math.mcgill.ca/keith/surfstat/
- **Tutorial:** http://www.math.mcgill.ca/keith/surfstat/surfstat_tutorial.pdf
- **FreeSurfer:** https://surfer.nmr.mgh.harvard.edu/
- **CIVET:** http://www.bic.mni.mcgill.ca/ServicesSoftware/CIVET
- **Random Field Theory:** Worsley et al. (1992, 1996)

## Citation

```bibtex
@article{worsley1992three,
  title={A three-dimensional statistical analysis for CBF activation studies in human brain},
  author={Worsley, Keith J and Evans, Alan C and Marrett, S and Neelin, P},
  journal={Journal of Cerebral Blood Flow \& Metabolism},
  volume={12},
  number={6},
  pages={900--918},
  year={1992},
  publisher={SAGE Publications Sage UK: London, England}
}

@article{worsley1996unified,
  title={A unified statistical approach for determining significant signals in images of cerebral activation},
  author={Worsley, Keith J and Marrett, Sean and Neelin, Peter and Vandal, Alain C and Friston, Karl J and Evans, Alan C},
  journal={Human brain mapping},
  volume={4},
  number={1},
  pages={58--73},
  year={1996},
  publisher={Wiley Online Library}
}
```

## Related Tools

- **FreeSurfer:** Surface generation and parcellation
- **CIVET:** Alternative surface-based pipeline
- **BrainSpace:** Surface-based gradient analysis
- **BrainStat:** Modern surface statistics framework
- **SPM:** Volumetric statistical analysis
- **CAT12:** Surface-based VBM in SPM

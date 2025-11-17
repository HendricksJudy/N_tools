# QC Automation & Custom Tools

## Overview

This skill covers **custom quality control automation** for neuroimaging data using Python, shell scripting, and integration frameworks. Learn to build custom QC pipelines, automate quality metric extraction, create interactive dashboards, develop quality control databases, and integrate multiple QC tools into cohesive workflows. This is particularly valuable for groups with specific QC needs, large-scale studies, or unique data types not fully covered by existing tools.

**Platform:** Python/Shell
**Language:** Python, Bash
**Dependencies:** pandas, numpy, matplotlib, plotly, dash, sqlite3

## Key Features

- Custom QC metric calculation and extraction
- Automated quality dashboards (Plotly, Dash)
- Integration of multiple QC tools (MRIQC, VisualQC, fMRIPrep)
- Python-based automation scripts
- Shell script batch processing
- Quality control databases (SQLite, PostgreSQL)
- Interactive web dashboards
- Automated flagging and alerting
- Quality report generation
- Multi-modal QC integration
- Custom visualization creation
- Outlier detection algorithms
- Reproducible QC workflows

## QC Automation Basics

### Python Script Template

```python
#!/usr/bin/env python3
"""
Automated QC metric extraction and analysis
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

def extract_mriqc_metrics(mriqc_dir):
    """Extract MRIQC JSON metrics for all subjects"""

    metrics = []
    json_files = Path(mriqc_dir).glob('sub-*_T1w.json')

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        metrics.append({
            'subject_id': json_file.stem.split('_')[0],
            'snr': data.get('snr_total', np.nan),
            'cnr': data.get('cnr', np.nan),
            'fber': data.get('fber', np.nan),
            'efc': data.get('efc', np.nan),
            'fwhm': data.get('fwhm_avg', np.nan)
        })

    return pd.DataFrame(metrics)

def extract_fmriprep_confounds(fmriprep_dir, task='rest'):
    """Extract motion metrics from fMRIPrep confounds"""

    metrics = []
    conf_files = Path(fmriprep_dir).glob(f'sub-*/func/*task-{task}*confounds*.tsv')

    for conf_file in conf_files:
        subject_id = conf_file.parts[-3]
        conf = pd.read_csv(conf_file, sep='\t')

        metrics.append({
            'subject_id': subject_id,
            'mean_fd': np.nanmean(conf['framewise_displacement']),
            'max_fd': np.nanmax(conf['framewise_displacement']),
            'mean_dvars': np.nanmean(conf['std_dvars']),
            'n_volumes': len(conf)
        })

    return pd.DataFrame(metrics)

if __name__ == '__main__':
    # Extract all metrics
    mriqc_metrics = extract_mriqc_metrics('/data/derivatives/mriqc')
    motion_metrics = extract_fmriprep_confounds('/data/derivatives/fmriprep')

    # Merge
    all_metrics = mriqc_metrics.merge(motion_metrics, on='subject_id')

    # Save
    all_metrics.to_csv('qc_metrics_all.csv', index=False)
    print(f"Extracted metrics for {len(all_metrics)} subjects")
```

### Shell Script Automation

```bash
#!/bin/bash
# qc_pipeline.sh - Automated QC workflow

set -e  # Exit on error

BIDS_DIR="/data/bids"
DERIVATIVES="/data/derivatives"
QC_DIR="/data/qc"

# Create output directories
mkdir -p ${QC_DIR}/{mriqc,visualqc,fmriprep_qc,reports}

echo "=== Step 1: Run MRIQC ==="
docker run --rm \
  -v ${BIDS_DIR}:/data:ro \
  -v ${DERIVATIVES}/mriqc:/out \
  nipreps/mriqc:latest \
  /data /out participant

docker run --rm \
  -v ${BIDS_DIR}:/data:ro \
  -v ${DERIVATIVES}/mriqc:/out \
  nipreps/mriqc:latest \
  /data /out group

echo "=== Step 2: Extract MRIQC metrics ==="
python3 extract_mriqc_metrics.py \
  --input ${DERIVATIVES}/mriqc \
  --output ${QC_DIR}/mriqc_metrics.csv

echo "=== Step 3: Extract fMRIPrep motion metrics ==="
python3 extract_fmriprep_metrics.py \
  --input ${DERIVATIVES}/fmriprep \
  --output ${QC_DIR}/motion_metrics.csv

echo "=== Step 4: Generate QC dashboard ==="
python3 generate_dashboard.py \
  --mriqc ${QC_DIR}/mriqc_metrics.csv \
  --motion ${QC_DIR}/motion_metrics.csv \
  --output ${QC_DIR}/reports/dashboard.html

echo "=== Step 5: Flag outliers ==="
python3 flag_outliers.py \
  --input ${QC_DIR}/mriqc_metrics.csv \
  --output ${QC_DIR}/outliers.txt

echo "QC pipeline complete!"
echo "Dashboard: ${QC_DIR}/reports/dashboard.html"
```

## Custom Metric Calculation

### Parse MRIQC JSON Files

```python
import json
import glob
import pandas as pd

def parse_mriqc_json(json_dir):
    """Parse all MRIQC JSON files"""

    all_metrics = []

    # T1w metrics
    for json_file in glob.glob(f'{json_dir}/sub-*_T1w.json'):
        with open(json_file, 'r') as f:
            data = json.load(f)

        subject_id = data['bids_meta']['subject_id']

        metrics = {
            'subject_id': subject_id,
            'modality': 'T1w',
            'snr_total': data.get('snr_total'),
            'cnr': data.get('cnr'),
            'fber': data.get('fber'),
            'efc': data.get('efc'),
            'fwhm_avg': data.get('fwhm_avg'),
            'fwhm_x': data.get('fwhm_x'),
            'fwhm_y': data.get('fwhm_y'),
            'fwhm_z': data.get('fwhm_z'),
            'qi_1': data.get('qi_1'),
            'qi_2': data.get('qi_2'),
            'inu_range': data.get('inu_range'),
            'inu_med': data.get('inu_med'),
            'wm2max': data.get('wm2max')
        }

        all_metrics.append(metrics)

    # BOLD metrics
    for json_file in glob.glob(f'{json_dir}/sub-*_bold.json'):
        with open(json_file, 'r') as f:
            data = json.load(f)

        subject_id = data['bids_meta']['subject_id']
        task = data['bids_meta'].get('task_id', 'unknown')

        metrics = {
            'subject_id': subject_id,
            'modality': 'bold',
            'task': task,
            'fd_mean': data.get('fd_mean'),
            'fd_num': data.get('fd_num'),
            'fd_perc': data.get('fd_perc'),
            'tsnr': data.get('tsnr'),
            'dvars_std': data.get('dvars_std'),
            'dvars_vstd': data.get('dvars_vstd'),
            'gcor': data.get('gcor'),
            'gsr_x': data.get('gsr_x'),
            'gsr_y': data.get('gsr_y'),
            'aor': data.get('aor')
        }

        all_metrics.append(metrics)

    return pd.DataFrame(all_metrics)

# Usage
df = parse_mriqc_json('/data/derivatives/mriqc')
df.to_csv('mriqc_all_metrics.csv', index=False)
```

### Calculate Custom Metrics

```python
import nibabel as nib
import numpy as np

def calculate_tsnr(bold_file, mask_file=None):
    """Calculate temporal SNR from BOLD data"""

    img = nib.load(bold_file)
    data = img.get_fdata()

    if mask_file:
        mask = nib.load(mask_file).get_fdata() > 0
        data = data[mask, :]

    # Temporal mean and std
    mean_signal = np.mean(data, axis=-1)
    std_signal = np.std(data, axis=-1)

    # tSNR = mean / std
    tsnr = mean_signal / (std_signal + 1e-10)

    return np.mean(tsnr)

def calculate_signal_dropout(bold_file, threshold_percentile=10):
    """Detect signal dropout regions"""

    img = nib.load(bold_file)
    data = img.get_fdata()

    # Mean across time
    mean_img = np.mean(data, axis=-1)

    # Threshold (bottom percentile)
    threshold = np.percentile(mean_img, threshold_percentile)

    # Dropout volume
    dropout_voxels = np.sum(mean_img < threshold)
    total_voxels = np.prod(mean_img.shape)

    dropout_pct = dropout_voxels / total_voxels * 100

    return dropout_pct

# Usage
tsnr = calculate_tsnr('sub-01_task-rest_bold.nii.gz',
                       'sub-01_mask.nii.gz')
dropout = calculate_signal_dropout('sub-01_task-rest_bold.nii.gz')

print(f"tSNR: {tsnr:.2f}")
print(f"Signal dropout: {dropout:.2f}%")
```

## Visualization Dashboards

### Matplotlib Dashboard

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_qc_dashboard(metrics_file, output_file='qc_dashboard.png'):
    """Create comprehensive QC dashboard"""

    df = pd.read_csv(metrics_file)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # SNR distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df['snr_total'], bins=30, edgecolor='black')
    ax1.axvline(df['snr_total'].median(), color='r', linestyle='--', label='Median')
    ax1.set_xlabel('SNR')
    ax1.set_title('SNR Distribution')
    ax1.legend()

    # CNR distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df['cnr'], bins=30, edgecolor='black')
    ax2.axvline(df['cnr'].median(), color='r', linestyle='--')
    ax2.set_xlabel('CNR')
    ax2.set_title('CNR Distribution')

    # FBER distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(df['fber'], bins=30, edgecolor='black')
    ax3.axvline(df['fber'].median(), color='r', linestyle='--')
    ax3.set_xlabel('FBER')
    ax3.set_title('FBER Distribution')

    # Motion (Mean FD)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(df['mean_fd'], bins=30, edgecolor='black')
    ax4.axvline(0.5, color='r', linestyle='--', label='Threshold (0.5mm)')
    ax4.set_xlabel('Mean FD (mm)')
    ax4.set_title('Motion Distribution')
    ax4.legend()

    # DVARS
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(df['mean_dvars'], bins=30, edgecolor='black')
    ax5.axvline(1.5, color='r', linestyle='--')
    ax5.set_xlabel('Mean DVARS')
    ax5.set_title('DVARS Distribution')

    # SNR vs Motion
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(df['mean_fd'], df['snr_total'], alpha=0.5)
    ax6.set_xlabel('Mean FD (mm)')
    ax6.set_ylabel('SNR')
    ax6.set_title('SNR vs Motion')

    # Quality matrix
    ax7 = fig.add_subplot(gs[2, :])
    quality_metrics = df[['snr_total', 'cnr', 'fber', 'mean_fd', 'mean_dvars']]
    corr = quality_metrics.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax7)
    ax7.set_title('Quality Metric Correlations')

    plt.suptitle('Quality Control Dashboard', fontsize=16)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Dashboard saved to {output_file}")

# Usage
create_qc_dashboard('qc_metrics_all.csv')
```

### Interactive Plotly Dashboard

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def create_interactive_dashboard(metrics_file, output_html='dashboard.html'):
    """Create interactive Plotly dashboard"""

    df = pd.read_csv(metrics_file)

    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('SNR Distribution', 'CNR Distribution', 'Motion (FD)',
                        'DVARS', 'SNR vs Motion', 'Quality Summary'),
        specs=[[{'type': 'histogram'}, {'type': 'histogram'}, {'type': 'histogram'}],
               [{'type': 'histogram'}, {'type': 'scatter'}, {'type': 'box'}]]
    )

    # SNR histogram
    fig.add_trace(go.Histogram(x=df['snr_total'], name='SNR',
                               marker_color='lightblue'),
                  row=1, col=1)

    # CNR histogram
    fig.add_trace(go.Histogram(x=df['cnr'], name='CNR',
                               marker_color='lightgreen'),
                  row=1, col=2)

    # Motion histogram
    fig.add_trace(go.Histogram(x=df['mean_fd'], name='Mean FD',
                               marker_color='salmon'),
                  row=1, col=3)
    fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                  row=1, col=3)

    # DVARS histogram
    fig.add_trace(go.Histogram(x=df['mean_dvars'], name='DVARS',
                               marker_color='lavender'),
                  row=2, col=1)

    # SNR vs Motion scatter
    fig.add_trace(go.Scatter(x=df['mean_fd'], y=df['snr_total'],
                            mode='markers', name='Subjects',
                            marker=dict(size=8, opacity=0.6)),
                  row=2, col=2)

    # Quality box plots
    for i, metric in enumerate(['snr_total', 'cnr', 'mean_fd']):
        fig.add_trace(go.Box(y=df[metric], name=metric),
                      row=2, col=3)

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Interactive QC Dashboard",
        hovermode='closest'
    )

    fig.write_html(output_html)
    print(f"Interactive dashboard saved to {output_html}")

# Usage
create_interactive_dashboard('qc_metrics_all.csv')
```

### Web Dashboard with Dash

```python
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

# Load data
df = pd.read_csv('qc_metrics_all.csv')

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Neuroimaging QC Dashboard"),

    html.Div([
        html.Label("Select Metric:"),
        dcc.Dropdown(
            id='metric-dropdown',
            options=[
                {'label': 'SNR', 'value': 'snr_total'},
                {'label': 'CNR', 'value': 'cnr'},
                {'label': 'Mean FD', 'value': 'mean_fd'},
                {'label': 'DVARS', 'value': 'mean_dvars'}
            ],
            value='snr_total'
        )
    ]),

    html.Div([
        dcc.Graph(id='histogram'),
        dcc.Graph(id='box-plot')
    ]),

    html.Div([
        html.H3("Subject Table"),
        html.Div(id='subject-table')
    ])
])

@app.callback(
    [Output('histogram', 'figure'),
     Output('box-plot', 'figure')],
    [Input('metric-dropdown', 'value')]
)
def update_graphs(metric):
    # Histogram
    hist = px.histogram(df, x=metric, nbins=30,
                        title=f'{metric} Distribution')

    # Box plot
    box = px.box(df, y=metric, title=f'{metric} Box Plot')

    return hist, box

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
    # Access at http://localhost:8050
```

## Outlier Detection

### Statistical Methods

```python
from scipy.stats import zscore
from scipy.spatial.distance import mahalanobis

def detect_outliers_zscore(df, metrics, threshold=3):
    """Z-score based outlier detection"""

    outliers = pd.DataFrame()

    for metric in metrics:
        z = np.abs(zscore(df[metric].dropna()))
        outliers[f'{metric}_outlier'] = z > threshold

    df['outlier'] = outliers.any(axis=1)

    return df

def detect_outliers_iqr(df, metrics):
    """IQR-based outlier detection"""

    outliers = pd.DataFrame()

    for metric in metrics:
        Q1 = df[metric].quantile(0.25)
        Q3 = df[metric].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers[f'{metric}_outlier'] = (df[metric] < lower) | (df[metric] > upper)

    df['outlier'] = outliers.any(axis=1)

    return df

def detect_outliers_mahalanobis(df, metrics, threshold=3):
    """Multivariate outlier detection using Mahalanobis distance"""

    X = df[metrics].dropna()

    # Covariance matrix
    cov = np.cov(X.T)
    cov_inv = np.linalg.inv(cov)
    mean = X.mean().values

    # Mahalanobis distance
    mahal_dist = X.apply(lambda row: mahalanobis(row, mean, cov_inv), axis=1)

    df.loc[X.index, 'mahal_dist'] = mahal_dist
    df['outlier'] = mahal_dist > threshold

    return df

# Usage
metrics = ['snr_total', 'cnr', 'fber', 'mean_fd', 'mean_dvars']
df = pd.read_csv('qc_metrics_all.csv')

df = detect_outliers_mahalanobis(df, metrics)

outliers = df[df['outlier']]
print(f"Outliers detected: {len(outliers)}/{len(df)}")
outliers.to_csv('qc_outliers.csv', index=False)
```

### Machine Learning Approaches

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def detect_outliers_isolation_forest(df, metrics, contamination=0.1):
    """Isolation Forest for outlier detection"""

    X = df[metrics].dropna()

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Isolation Forest
    clf = IsolationForest(contamination=contamination, random_state=42)
    predictions = clf.fit_predict(X_scaled)

    df.loc[X.index, 'outlier'] = predictions == -1

    return df

# Usage
df = detect_outliers_isolation_forest(df, metrics)
```

## Quality Control Database

### SQLite Database Setup

```python
import sqlite3
import pandas as pd
from datetime import datetime

def create_qc_database(db_file='qc_database.db'):
    """Create QC database schema"""

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Subjects table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS subjects (
            subject_id TEXT PRIMARY KEY,
            scan_date DATE,
            scanner TEXT,
            site TEXT
        )
    ''')

    # MRIQC metrics table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mriqc_t1w (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id TEXT,
            snr_total REAL,
            cnr REAL,
            fber REAL,
            efc REAL,
            fwhm_avg REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
        )
    ''')

    # Motion metrics table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS motion_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id TEXT,
            task TEXT,
            mean_fd REAL,
            max_fd REAL,
            mean_dvars REAL,
            n_volumes INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
        )
    ''')

    # QC decisions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS qc_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id TEXT,
            modality TEXT,
            decision TEXT,  -- pass, fail, questionable
            rater TEXT,
            notes TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
        )
    ''')

    conn.commit()
    conn.close()

def insert_mriqc_metrics(db_file, metrics_df):
    """Insert MRIQC metrics into database"""

    conn = sqlite3.connect(db_file)

    metrics_df.to_sql('mriqc_t1w', conn, if_exists='append', index=False)

    conn.close()

def query_failed_qc(db_file):
    """Query subjects that failed QC"""

    conn = sqlite3.connect(db_file)

    query = """
    SELECT DISTINCT subject_id, modality, decision, notes
    FROM qc_decisions
    WHERE decision = 'fail'
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df

# Usage
create_qc_database()
```

### Database Queries

```python
def get_subject_qc_history(db_file, subject_id):
    """Get complete QC history for a subject"""

    conn = sqlite3.connect(db_file)

    query = f"""
    SELECT
        m.snr_total, m.cnr, m.fber,
        mo.mean_fd, mo.mean_dvars,
        q.decision, q.rater, q.timestamp
    FROM subjects s
    LEFT JOIN mriqc_t1w m ON s.subject_id = m.subject_id
    LEFT JOIN motion_metrics mo ON s.subject_id = mo.subject_id
    LEFT JOIN qc_decisions q ON s.subject_id = q.subject_id
    WHERE s.subject_id = '{subject_id}'
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df
```

## Automated Reporting

### Generate PDF Reports

```python
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def generate_qc_report(metrics_file, output_pdf='qc_report.pdf'):
    """Generate multi-page PDF QC report"""

    df = pd.read_csv(metrics_file)

    with PdfPages(output_pdf) as pdf:
        # Page 1: Summary statistics
        fig, ax = plt.subplots(figsize=(8, 11))
        ax.axis('off')

        summary_text = f"""
        Quality Control Report
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

        Total Subjects: {len(df)}

        SNR Statistics:
          Mean: {df['snr_total'].mean():.2f}
          Median: {df['snr_total'].median():.2f}
          Range: {df['snr_total'].min():.2f} - {df['snr_total'].max():.2f}

        Motion Statistics:
          Mean FD: {df['mean_fd'].mean():.3f} mm
          Subjects with FD > 0.5mm: {(df['mean_fd'] > 0.5).sum()}

        QC Summary:
          Passed: {(df['snr_total'] > 10).sum()}
          Failed: {(df['snr_total'] <= 10).sum()}
        """

        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', family='monospace')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 2: Distributions
        create_qc_dashboard(metrics_file, 'temp_dashboard.png')
        fig = plt.figure(figsize=(11, 8.5))
        img = plt.imread('temp_dashboard.png')
        plt.imshow(img)
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print(f"PDF report saved to {output_pdf}")
```

### Email Notifications

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_qc_alert(subject_id, metrics, recipients):
    """Send email alert for failed QC"""

    sender = 'qc-system@institution.edu'
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)
    msg['Subject'] = f'QC Alert: {subject_id}'

    body = f"""
    Subject {subject_id} failed quality control.

    Metrics:
    - SNR: {metrics['snr_total']:.2f}
    - Mean FD: {metrics['mean_fd']:.3f} mm

    Please review manually.
    """

    msg.attach(MIMEText(body, 'plain'))

    # Send email (configure SMTP server)
    # server = smtplib.SMTP('smtp.institution.edu', 587)
    # server.starttls()
    # server.login(sender, 'password')
    # server.send_message(msg)
    # server.quit()

    print(f"Alert sent for {subject_id}")
```

## Large-Scale Studies

### Parallel Processing

```python
from joblib import Parallel, delayed
import glob

def process_subject_qc(subject_file):
    """Process QC for single subject"""

    # Extract metrics
    metrics = extract_metrics(subject_file)

    # Flag outliers
    is_outlier = check_outlier(metrics)

    return {'subject': subject_file, 'metrics': metrics, 'outlier': is_outlier}

# Parallel execution
subject_files = glob.glob('/data/derivatives/mriqc/sub-*.json')

results = Parallel(n_jobs=8)(
    delayed(process_subject_qc)(f) for f in subject_files
)

# Aggregate results
df = pd.DataFrame(results)
```

### Incremental Updates

```python
def incremental_qc_update(db_file, new_subjects):
    """Update QC database with new subjects only"""

    conn = sqlite3.connect(db_file)

    # Get existing subjects
    existing = pd.read_sql_query("SELECT DISTINCT subject_id FROM subjects", conn)
    existing_ids = set(existing['subject_id'])

    # Filter new subjects
    new_ids = [s for s in new_subjects if s not in existing_ids]

    print(f"New subjects to process: {len(new_ids)}")

    for subject_id in new_ids:
        # Process and insert
        metrics = extract_subject_metrics(subject_id)
        insert_metrics(conn, subject_id, metrics)

    conn.close()
```

## Troubleshooting

**Problem:** Slow metric extraction
**Solution:** Use parallel processing, cache results, optimize I/O

**Problem:** Dashboard won't load
**Solution:** Check file paths, reduce data size, use chunking for large datasets

**Problem:** Database locks
**Solution:** Use WAL mode, reduce concurrent writes, implement retry logic

**Problem:** Memory errors with large datasets
**Solution:** Process in batches, use generators, increase swap space

## Best Practices

1. **Version Control:**
   - Use git for QC scripts
   - Document changes
   - Tag releases

2. **Reproducibility:**
   - Save all parameters
   - Use config files
   - Log all operations

3. **Validation:**
   - Test on known-good data
   - Compare with manual QC
   - Cross-validate tools

4. **Documentation:**
   - Comment code thoroughly
   - Create README files
   - Maintain changelog

5. **Modularity:**
   - Separate extraction, analysis, visualization
   - Reusable functions
   - Clear interfaces

## Resources

- **Pandas Documentation:** https://pandas.pydata.org/docs/
- **Plotly Documentation:** https://plotly.com/python/
- **Dash Documentation:** https://dash.plotly.com/
- **SQLite Tutorial:** https://www.sqlitetutorial.net/
- **Joblib:** https://joblib.readthedocs.io/

## Related Tools

- **MRIQC:** Automated quality metrics
- **VisualQC:** Manual inspection interface
- **fMRIPrep:** Preprocessing QC outputs
- **Pandas:** Data manipulation
- **Plotly/Dash:** Interactive visualization
- **SQLite:** Lightweight database

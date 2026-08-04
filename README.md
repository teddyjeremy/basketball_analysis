# Basketball Analysis

A computer-vision system for analyzing basketball games from video. The project combines object detection, multi-object tracking, court geometry, team identification, possession analysis, movement metrics, event detection, and tactical visualization.

## Features

- Player detection and tracking
- Basketball detection and tracking
- Camera-motion estimation
- Team assignment
- Ball possession estimation
- Pass and interception detection
- Player speed and distance estimation
- Court keypoint detection
- Homography-based court projection
- Tactical-view visualization
- Match-level statistics
- Analysis reports
- Video rendering
- Automated tests

## Architecture

```text
Input Video
    │
    ├── Player Detection ──────┐
    ├── Ball Detection ────────┤
    ├── Court Keypoints ───────┤
    └── Camera Motion ─────────┘
                │
                ▼
        Tracking and Geometry
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
      Teams  Possession  Court Projection
        │       │        │
        └───────┼────────┘
                ▼
       Pass / Interception Events
                │
                ▼
       Speed / Distance Metrics
                │
                ▼
       Analysis + Visualization
                │
                ▼
          Output Video
          + Match Report
```

## Project Structure

```text
basketball_analysis/
├── analysis/
│   ├── __init__.py
│   └── report.py
├── ball_acquisition/
│   └── ball_acquisition.py
├── camera_movement_estimator/
├── court_keypoint_detector/
│   ├── court_keypoint_detector.py
│   └── court_template.py
├── configs/
│   └── config.yaml
├── drawers/
│   ├── ball_drawer.py
│   ├── court_drawer.py
│   ├── player_drawer.py
│   ├── possession_drawer.py
│   └── tactical_view_drawer.py
├── pass_and_interception_detector/
│   └── pass_and_interception_detector.py
├── pipelines/
│   ├── __init__.py
│   ├── match_analysis_pipeline.py
│   └── video_analysis.py
├── scripts/
│   └── run_analysis.py
├── speed_and_distance_calculator/
├── tactical_view_converter/
│   └── tactical_view_converter.py
├── team_assigner/
├── trackers/
├── utils/
│   ├── config_loader.py
│   ├── interpolation.py
│   ├── metrics.py
│   ├── serialization.py
│   └── video_writer.py
├── tests/
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/teddyjeremy/basketball_analysis.git
cd basketball_analysis
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install supervision pyyaml pytest
```

## Models

Place the trained model weights in `models/` and configure their paths in `configs/config.yaml`.

The pipeline is designed around separate models for:

- Player detection
- Basketball detection
- Court keypoint detection

Model weights are intentionally not included in the repository.

## Configuration

Runtime configuration is stored in `configs/config.yaml`.

It controls:

- Input and output videos
- Model paths
- Detection confidence thresholds
- IoU thresholds
- Inference device
- Batch size
- Team assignment
- Possession distance
- Motion calibration
- Court projection dimensions

## Running

Run the analysis pipeline with the default configuration:

```bash
python scripts/run_analysis.py --config configs/config.yaml
```

Specify custom input and output files:

```bash
python scripts/run_analysis.py \
    --config configs/config.yaml \
    --input data/game.mp4 \
    --output outputs/game_analysis.mp4
```

## Testing

Run the test suite with:

```bash
pytest -q
```

The repository also includes a GitHub Actions workflow for automated testing.

## Development Direction

The project is structured as a modular basketball analytics system rather than a single video-processing script. Detection, tracking, team identification, possession, event detection, court geometry, motion analysis, visualization, and reporting are separated so that individual components can be improved or replaced independently.

The long-term goal is to support detailed basketball analysis from broadcast and tactical video, including player movement, team spacing, possession sequences, passing networks, defensive positioning, and court-based tactical analysis.

## License

This repository is intended for educational and research use. Review the licenses and terms of third-party software, models, datasets, and other dependencies before redistribution or commercial use.

# Basketball Analysis

Computer-vision pipeline for basketball video analysis using YOLO, OpenCV, Supervision, NumPy, scikit-learn, and PyTorch.

## Capabilities

- Player detection and tracking
- Basketball detection
- Camera-motion estimation
- Team assignment
- Ball possession estimation
- Pass and interception events
- Player speed and distance estimation
- Court keypoint detection
- Homography-based tactical projection
- Match-level metrics and reports
- Video visualization
- Automated tests

## Project Structure

```text
basketball_analysis/
├── analysis/
├── ball_acquisition/
├── camera_movement_estimator/
├── court_keypoint_detector/
├── configs/
├── drawers/
├── pass_and_interception_detector/
├── pipelines/
├── scripts/
├── speed_and_distance_calculator/
├── tactical_view_converter/
├── team_assigner/
├── trackers/
├── utils/
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

Place trained YOLO weights in `models/` and configure their paths in `configs/config.yaml`.

The pipeline expects separate models for player detection, ball detection, and court keypoints.

## Configuration

Runtime settings are defined in `configs/config.yaml`, including:

- video input and output
- model paths
- detection confidence
- IoU
- batch size
- device
- team count
- possession distance
- motion calibration
- court output dimensions

## Running

```bash
python scripts/run_analysis.py --config configs/config.yaml
```

Override the configured input and output paths:

```bash
python scripts/run_analysis.py --config configs/config.yaml --input data/game.mp4 --output outputs/game_analysis.mp4
```

## Testing

```bash
pytest -q
```

GitHub Actions runs the test suite on pushes and pull requests.

## Architecture

```text
Video
  |
  +--> Player YOLO --> ByteTrack
  |
  +--> Ball YOLO
  |
  +--> Court Keypoints --> Homography
  |
  +--> Camera Motion
          |
          v
Player Tracks + Ball Tracks + Court Geometry
          |
          +--> Team Assignment
          |
          +--> Ball Possession
          |
          +--> Pass / Interception Events
          |
          +--> Speed / Distance
          |
          +--> Tactical Projection
          |
          v
Analysis Report + Visualized Video
```

## Reference

The project architecture and implementation direction are informed by the basketball/football computer-vision analysis work by Abdullah Tarek:

https://github.com/abdullahtarek/basketball_analysis

https://github.com/abdullahtarek/football_analysis

The project is independently extended for basketball analysis rather than being a direct copy of those repositories.

## License

This repository is intended for educational and research use. Verify the licenses and terms of any third-party models, datasets, repositories, and dependencies before redistribution or commercial use.

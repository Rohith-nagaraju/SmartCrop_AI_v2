# SmartCrop AI — Complete Research Rebuild

## Project
**Data-Driven Crop Health Decision Support System Using Machine Learning, Cloud Computing, and Predictive Analytics**

The rebuilt system is designed around a stronger, testable claim: an **explainable multimodal crop-health risk assessment system** combining leaf-image disease recognition, severity estimation, environmental risk, and temporal forecasting.

This is intentionally not framed as a claim that EfficientNet-B0 itself is novel. EfficientNet-B0 is already widely used for crop disease classification; recent work also combines image and environmental information in multimodal systems. The research contribution therefore has to be demonstrated through the integration, evaluation, explainability, uncertainty handling, and ablation experiments rather than by naming a backbone as “new”.

## Architecture

```text
Leaf image ──> Image quality ──> Disease classifier ──> Confidence/unknown handling
                                      │
                                      ├──> Grad-CAM/explanation
                                      └──> Severity estimation / segmentation

Temperature ─┐
Humidity ────┤
Rainfall ────┼──> Transparent DPI ──┐
Soil moisture┘                      │
                                    ├──> Multimodal risk fusion ──> Risk level
Historical environment ──> learned ┘
                                      │
Future weather / time series ──> Forecast ──> 3–7 day risk outlook
                                      │
                                      └──> Recommendation + dashboard/report
```

## Folder structure

```text
smartcrop_ai/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── schemas.py
│   ├── ml/disease_model.py
│   └── services/
│       ├── image_quality.py
│       ├── severity.py
│       ├── risk_engine.py
│       ├── environment_model.py
│       ├── forecast.py
│       ├── recommender.py
│       ├── gradcam.py
│       └── report_generator.py
├── ml/
│   ├── train_disease.py
│   ├── evaluate_disease.py
│   ├── train_baselines.py
│   ├── train_environment.py
│   ├── ablation_study.py
│   ├── train_forecast.py
│   ├── generate_demo_environment.py
│   └── generate_ablation_demo.py
├── frontend/
├── models/
├── data/
├── experiments/
└── tests/
```

## 1. Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Prepare disease dataset

Put images into:

```text
data/disease_dataset/
  train/
    Healthy/
    DiseaseA/
    DiseaseB/
  val/
    Healthy/
    DiseaseA/
    DiseaseB/
  test/
    Healthy/
    DiseaseA/
    DiseaseB/
```

**Use a leakage-safe split.** If multiple images come from the same physical leaf/plant or are generated augmentations of one source image, keep the group in only one split.

## 3. Train disease model

```bash
python ml/train_disease.py --epochs 12
python ml/evaluate_disease.py --split test
```

Outputs:

- `models/disease_model.keras`
- `models/class_names.json`
- `experiments/results/confusion_matrix.csv`

## 4. Baseline comparison — essential for your project defense

Run:

```bash
python ml/train_baselines.py --model mobilenetv2 --epochs 5
python ml/train_baselines.py --model resnet50 --epochs 5
python ml/train_baselines.py --model efficientnetb0 --epochs 5
python ml/train_baselines.py --model customcnn --epochs 5
```

The script records:

- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Parameter count
- Training time
- Inference latency

This gives you evidence for whether the selected vision model is actually appropriate for your dataset.

## 5. Environmental model comparison

Prepare a CSV with:

```text
temperature,humidity,rainfall,soil_moisture,risk_score
```

Then:

```bash
python ml/train_environment.py --csv data/environment.csv
```

It compares Random Forest and Gradient Boosting and saves the selected Random Forest model as `models/environment_model.joblib`.

### Demo pipeline only

```bash
python ml/generate_demo_environment.py
python ml/train_environment.py --csv data/demo_environment.csv
```

**Do not use the demo metrics as final academic results.** They exist only to prove the pipeline runs.

## 6. Ablation study — strongest proof of your multimodal contribution

The study should compare:

| Experiment | Inputs |
|---|---|
| A | Image only |
| B | Environment only |
| C | Image + Environment |
| D | Image + Environment + Severity |
| E | Image + Environment + Severity + Temporal |

For a pipeline test:

```bash
python ml/generate_ablation_demo.py
python ml/ablation_study.py --csv data/demo_ablation.csv
```

For your final paper, replace the demo CSV with **real paired observations** and real model outputs.

## 7. Forecasting

Prepare:

```text
timestamp,field_id,temperature,humidity,rainfall,soil_moisture,risk_score
```

Then:

```bash
python ml/train_forecast.py --csv data/historical_risk.csv
```

The final deployment should use actual future weather forecasts rather than the simple perturbation fallback currently used by the API.

## 8. Run the application

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

Or with Docker:

```bash
docker compose up --build
```

## 9. Important research limitations

### Severity
`app/services/severity.py` currently uses a transparent color heuristic. It is **not a validated lesion segmentation model**. For the final research version, train a crop-specific segmentation model using pixel masks and report segmentation metrics such as Dice and IoU.

### Grad-CAM
The current endpoint includes a UI-safe image fallback. A true Grad-CAM implementation should be connected to the trained Keras model and its final convolutional feature layer before claiming model-localization results.

### Unknown detection
The API marks a prediction as `UNCERTAIN` when the maximum class probability is below `UNKNOWN_THRESHOLD` (default 0.55). This threshold should be calibrated on a validation set and preferably evaluated with reliability/calibration metrics.

### Forecasting
The fallback forecast is not a weather forecast. Replace it with real weather/time-series inputs for final deployment.

### Recommendations
The recommendation engine intentionally gives general integrated-disease-management guidance rather than universal pesticide prescriptions. Crop, region, label requirements, and expert validation must be incorporated before operational chemical recommendations.

## 10. What to show in the final evaluation chapter

### Disease model
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Per-class F1
- Confusion matrix
- Model size / parameter count
- Inference latency

### Severity model
- Dice
- IoU
- MAE for affected-area percentage

### Environmental model
- MAE
- RMSE
- R²

### Forecasting
- MAE
- RMSE
- Temporal hold-out evaluation

### Multimodal fusion
- Risk classification accuracy
- Macro F1
- Ablation table A–E

### Explainability
- Grad-CAM examples
- Unknown/low-confidence examples
- Image-quality examples
- Feature/contribution explanation

## 11. Defense answer: “What is different from existing models?”

Do not answer: “Our EfficientNet model is new.”

Answer instead:

> Existing crop-disease systems commonly focus on image classification. Our system extends the decision process by combining image-based disease recognition with severity estimation, environmental disease pressure, uncertainty handling, explainability, and temporal risk forecasting. We validate the contribution experimentally using model baselines and an ablation study that removes each modality and component. Therefore, the novelty claim is the evaluated multimodal decision-support pipeline rather than the underlying CNN architecture.

This distinction matters because EfficientNet-B0-based plant-disease systems and multimodal crop-health systems already exist in the literature.

## 12. No fabricated results

Never put invented Accuracy, F1, MAE, RMSE, R², latency, or segmentation scores in the report. Run the supplied evaluation scripts on your actual dataset and copy the generated metrics into the report.

from pathlib import Path
from src.inference.predictor import load_trained_model, predict_image

cnn = load_trained_model('custom_cnn')
resnet = load_trained_model('resnet18')

fixtures = sorted(Path('tests/fixtures/ood').glob('*.*'))

print('=== CUSTOM CNN (threshold=0.40) ===')
for f in fixtures:
    res, _ = predict_image(f, cnn, confidence_threshold=0.40)
    print(f"{f.name:30s} -> {res['predicted_class']:20s} conf={res['confidence']:6.2%} status={res['status']}")

print('\n=== RESNET-18 (threshold=0.60) ===')
for f in fixtures:
    res, _ = predict_image(f, resnet, confidence_threshold=0.60)
    print(f"{f.name:30s} -> {res['predicted_class']:20s} conf={res['confidence']:6.2%} status={res['status']}")

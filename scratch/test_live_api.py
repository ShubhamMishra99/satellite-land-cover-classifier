import requests
from pathlib import Path

url = 'http://127.0.0.1:8000/predict'

print('=== 1. TESTING UNSUPPORTED IMAGES (BOTH MODELS) ===')
fixtures = sorted(Path('tests/fixtures/ood').glob('*.*'))
for f in fixtures:
    for model in ['custom_cnn', 'resnet18']:
        with open(f, 'rb') as fp:
            res = requests.post(url, files={'file': (f.name, fp, 'image/jpeg')}, data={'model': model})
            assert res.status_code == 200, f'Error {res.status_code} for {f.name}'
            d = res.json()
            st = d['status']
            assert st == 'unsupported', f'Expected unsupported for {f.name} on {model}, got {st}'
            assert d['is_confident'] is False
            assert d['is_suitable'] is False
            assert 'This image does not appear suitable for satellite land-use classification' in d['uncertainty_reason']
            print(f'PASS: {f.name:25s} [{model:10s}] -> status={st:12s} stage={d.get("suitability_stage_failed")}')

print('\n=== 2. TESTING VALID SATELLITE IMAGES (BOTH MODELS) ===')
val_samples = [
    Path('data/raw/EuroSAT_RGB/Forest/Forest_1.jpg'),
    Path('data/raw/EuroSAT_RGB/Highway/Highway_1.jpg'),
    Path('data/raw/EuroSAT_RGB/Residential/Residential_1.jpg'),
]
for p in val_samples:
    if p.exists():
        for model in ['resnet18', 'custom_cnn']:
            with open(p, 'rb') as fp:
                res = requests.post(url, files={'file': (p.name, fp, 'image/jpeg')}, data={'model': model})
                assert res.status_code == 200
                d = res.json()
                st = d['status']
                assert st == 'accepted', f'Expected accepted for {p.name} on {model}, got {st}'
                assert d['is_confident'] is True
                assert d['is_suitable'] is True
                print(f'PASS: {p.name:25s} [{model:10s}] -> status={st:12s} class={d["predicted_class"]:12s} conf={d["confidence_percentage"]}')

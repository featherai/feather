import sys
import os

def check_imports():
    """Check import status of key dependencies and provide install hints."""
    print(f'Python {sys.version} ({sys.executable})')
    
    DEPS = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('datasets', 'datasets'),
        ('kaggle', 'kaggle'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('streamlit', 'streamlit'),
        ('safetensors', 'safetensors'),
        ('accelerate', 'accelerate'),
        ('bitsandbytes', 'bitsandbytes'),  # optional for 4-bit quantization
    ]
    
    missing = []
    ok = []
    
    for module_name, pip_name in DEPS:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, '__version__', 'unknown')
            ok.append(f'{module_name} {ver}')
        except Exception as e:
            hint = f'pip install {pip_name}'
            missing.append((module_name, type(e).__name__, hint))
    
    if ok:
        print('\nInstalled packages:')
        for pkg in ok:
            print(f'[OK] {pkg}')
    
    if missing:
        print('\nMissing packages:')
        for name, err, hint in missing:
            print(f'[MISSING] {name} ({err})')
            print(f'  Install with: {hint}')

if __name__ == '__main__':
    check_imports()

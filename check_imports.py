import importlib.util
modules = ['torch', 'sklearn', 'matplotlib', 'tqdm', 'pandas', 'numpy']
for m in modules:
    spec = importlib.util.find_spec(m)
    if spec is None:
        print(f"{m}: NOT INSTALLED")
    else:
        try:
            mod = importlib.import_module(m)
            v = getattr(mod, '__version__', 'unknown')
            print(f"{m}: INSTALLED, version={v}")
        except Exception as e:
            print(f"{m}: INSTALLED but import failed: {e}")

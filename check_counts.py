try:
    import main
    print(f"main.ALL_SYMPTOMS: {len(main.ALL_SYMPTOMS)}")
except Exception as e:
    print(f"main import error: {e}")

try:
    import app
    print(f"app.ALL_SYMPTOMS: {len(app.ALL_SYMPTOMS)}")
except Exception as e:
    print(f"app import error: {e}")

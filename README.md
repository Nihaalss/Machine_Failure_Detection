# Machine Failure Monitor Pro

Streamlit app for AI-powered machine failure risk monitoring.

## Run locally

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app2.py
```

## Files

- `app2.py`: Streamlit UI + inference
- `lgbm_machine_model.pkl`: trained model
- `feature_names.pkl`: feature list used for inference

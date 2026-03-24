# AI Worker

This microservice exposes a Flask endpoint for generating interview questions using Groq AI. It is intended to run independently from the Node.js backend.

## Producing and Deploying

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Configuration**
   Copy `.env.example` to `.env` and fill in your `GROQ_API_KEY` and other values.

3. **Local execution**
   ```bash
   python app.py
   ```

   The service listens on port `8000` by default. Use `PORT` environment variable to override.

4. **Production**
   Use `gunicorn` (already declared in `Procfile`):
   ```bash
   gunicorn app:app --bind 0.0.0.0:$PORT
   ```

   Ensure `ALLOWED_ORIGINS` is set and the `GROQ_API_KEY` is provided.

5. **Deployment**
   When deploying to Render, set the root to the `ai_worker` folder, provide a build command of
   `pip install -r requirements.txt && python -m spacy download en_core_web_sm`, and the start command
   `gunicorn app:app --bind 0.0.0.0:$PORT`.

## API Endpoints

- `POST /generate_questions` – expects multipart form-data with `resume` file and form fields `role`, `questionType`, `difficultyLevel`, `company`, `jobTitle`, `numQuestions`.
- `GET /health` – simple health check returning `{status: "healthy"}`.

## Notes

- Uploaded resumes are stored temporarily in `TMP_DIR` and cleaned up automatically.
- Requests are limited by `MAX_CONTENT_LENGTH` (default 10 MB).
- Logging level can be adjusted via the `LOG_LEVEL` environment variable.

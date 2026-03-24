import os
import asyncio
import io
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from docx import Document
import spacy
from groq import Groq, GroqError
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env if present

class Config:
    """Application configuration pulled from environment variables."""
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY must be provided via environment")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
    MAX_CONTENT_LENGTH: int = int(os.getenv("MAX_CONTENT_LENGTH", 10 * 1024 * 1024))  # 10 MB default
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "*")  # comma-separated list
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    TMP_DIR: str = os.getenv("TMP_DIR", "temp_uploads")

# Configure logging with timestamp and level from config
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

# create Flask app and load config
app = Flask(__name__)
app.config.from_object(Config)

# ensure temporary upload directory exists
Path(Config.TMP_DIR).mkdir(exist_ok=True, parents=True)

# setup CORS based on configured origins
origins = [o.strip() for o in Config.ALLOWED_ORIGINS.split(",") if o.strip()]
if origins == ["*"] or not origins:
    CORS(app)
else:
    CORS(app, resources={r"/*": {"origins": origins}})

# Check OCR availability
try:
    OCR_AVAILABLE = pytesseract.get_tesseract_version() is not None
    logger.info("OCR (Tesseract) is available")
except Exception as e:
    OCR_AVAILABLE = False
    logger.warning(f"OCR (Tesseract) not available: {str(e)}")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {str(e)}")
    raise

# initialize Groq client using configuration values
GROQ_API_KEY = Config.GROQ_API_KEY
model_name = Config.MODEL_NAME
try:
    client = Groq(api_key=GROQ_API_KEY)
    logger.info(f"Initialized Groq client with model {model_name}")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {str(e)}")
    raise

# enforce maximum upload size
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

@app.before_request
def log_request_info():
    logger.debug(f"Incoming request {request.method} {request.path}")

@app.errorhandler(Exception)
def handle_exception(e):
    # catch-all to avoid leaking stack traces
    logger.exception("Unhandled exception during request")
    return jsonify({"error": "Internal server error"}), 500


def extract_text_from_pdf(pdf_path):
    try:
        start_time = time.time()
        text = ""
        doc = fitz.open(pdf_path)
        for page in doc:
            page_text = page.get_text("text")
            text += page_text + "\n"
            if OCR_AVAILABLE and len(page_text.strip()) < 50:
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes()))
                ocr_text = pytesseract.image_to_string(img)
                text += ocr_text + "\n"
        doc.close()
        text = re.sub(r'\s+', ' ', text.strip())
        logger.debug(f"Extracted PDF text in {time.time() - start_time:.2f}s: {text[:100]}... (length: {len(text)} chars)")
        return text
    except Exception as e:
        logger.error(f"Failed to extract PDF text {pdf_path}: {str(e)}")
        raise

def extract_text_from_docx(docx_path):
    try:
        start_time = time.time()
        doc = Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        logger.debug(f"Extracted DOCX text in {time.time() - start_time:.2f}s: {text[:100]}... (length: {len(text)} chars)")
        return text
    except Exception as e:
        logger.error(f"Failed to extract DOCX text {docx_path}: {str(e)}")
        raise
    finally:
        if 'doc' in locals():
            del doc

def extract_text(resume_path):
    if resume_path.lower().endswith(".pdf"):
        text = extract_text_from_pdf(resume_path)
        logger.debug(f"Full extracted text (PDF): {text[:500]}... (length: {len(text)} chars)")
        return text
    elif resume_path.lower().endswith(".docx"):
        text = extract_text_from_docx(resume_path)
        logger.debug(f"Full extracted text (DOCX): {text[:500]}... (length: {len(text)} chars)")
        return text
    else:
        logger.error(f"Unsupported file format for {resume_path}. Use PDF or DOCX.")
        raise ValueError("Unsupported file format. Use PDF or DOCX.")

def extract_technical_skills(text):
    try:
        start_time = time.time()
        doc = nlp(text)
        matcher = spacy.matcher.PhraseMatcher(nlp.vocab, attr="LOWER")
        keywords = [
            "Python", "Java", "C++", "JavaScript", "SQL", "AWS", "TensorFlow", "PyTorch",
            "Docker", "Kubernetes", "React", "Django", "Flask", "Machine Learning",
            "Data Science", "MongoDB", "Node.js", "GraphQL", "CI/CD", "Git", "Linux",
            "Hadoop", "Spark", "GCP", "Azure", "REST API", "Microservices",
            "HTML", "CSS", "TypeScript", "Bootstrap", "jQuery", "JSON", "XML",
            "PostgreSQL", "MySQL", "NoSQL", "Pandas", "NumPy", "Matplotlib", "Seaborn",
            "OpenCV", "Scikit-learn", "NLP", "Computer Vision", "Agile", "Scrum"
        ]
        patterns = [nlp.make_doc(keyword.lower()) for keyword in keywords]
        matcher.add("SKILLS", patterns)
        matches = matcher(doc)
        skills = sorted({doc[start:end].text.lower() for _, start, end in matches})
        skills = skills[:5] or ["Python"]
        logger.info(f"Selected top {len(skills)} skills in {time.time() - start_time:.2f}s: {skills}")
        return skills
    except Exception as e:
        logger.error(f"Failed to extract skills: {str(e)}")
        return ["Python"]

def extract_behavioral_traits(text):
    try:
        start_time = time.time()
        traits = []
        patterns = {
            "teamwork": r"(collaborated|worked with|team|coordinated)",
            "leadership": r"(led|managed|supervised|directed)",
            "problem solving": r"(solved|resolved|debugged|fixed)"
        }
        for trait, pattern in patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                traits.append(trait)
        traits = traits[:2] or ["teamwork", "problem solving"]
        logger.info(f"Extracted traits in {time.time() - start_time:.2f}s: {traits}")
        return traits
    except Exception as e:
        logger.error(f"Failed to extract traits: {str(e)}")
        return ["teamwork", "problem solving"]

def validate_skill(skill):
    return len(skill) > 2 and re.match(r'^[a-zA-Z0-9\s+-]+$', skill)

async def async_generate_questions(prompt, resume_text, role, question_type, difficulty_level, skill, count=3):
    start_time = time.time()
    logger.debug(f"Generating questions for skill: {skill}, type: {question_type}")
    difficulty_map = {
        "easy": "basic, beginner-level",
        "medium": "intermediate, practical experience",
        "hard": "advanced, in-depth expertise"
    }
    difficulty_desc = difficulty_map.get(difficulty_level.lower(), "intermediate, practical experience")

    prompt = (
        f"Generate exactly {count} {question_type} interview questions for a {role or 'technical'} position "
        f"at {difficulty_desc} level, focusing on the skill '{skill}'. "
        f"Each question must end with a question mark. Format as:\n1. First question?\n2. Second question?\n3. Third question?"
    )

    logger.debug(f"Prompt for {skill}: {prompt}")
    for attempt in range(3):
        logger.debug(f"Attempt {attempt+1} for {skill}")
        try:
            chat_completion = await asyncio.to_thread(
                client.chat.completions.create,
                messages=[{"role": "user", "content": prompt}],
                model=model_name,
                max_tokens=500,
                temperature=0.7,
                top_p=0.9
            )
            response_text = chat_completion.choices[0].message.content
            logger.debug(f"Attempt {attempt+1} - Groq output for {skill}: {response_text}")

            questions = []
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                match = re.match(r'^(?:\d+\.\s*)?(.+?\?)$', line)
                if match:
                    question = match.group(1).strip()
                    if prompt.lower() not in question.lower():
                        questions.append(question)

            logger.debug(f"Attempt {attempt+1} - Extracted {len(questions)} questions in {time.time() - start_time:.2f}s for {skill}: {questions}")

            if len(questions) >= count:
                return list(dict.fromkeys(questions[:count]))
        except GroqError as e:
            logger.error(f"Attempt {attempt+1} - Groq error for {skill}: {str(e)}, Status: {getattr(e, 'status_code', 'N/A')}")
            if "authentication" in str(e).lower() or getattr(e, 'status_code', None) == 401:
                return []
            elif "rate limit" in str(e).lower() or getattr(e, 'status_code', None) == 429:
                if attempt < 2:
                    await asyncio.sleep(1)
                continue
            else:
                if attempt < 2:
                    continue
                return []
        except Exception as e:
            logger.error(f"Attempt {attempt+1} - Generation failed for {skill}: {str(e)}")
            if attempt < 2:
                continue
            return []

    logger.error(f"Failed to generate {count} questions for {skill} after 3 attempts.")
    if question_type == "technical":
        return [f"What is a basic concept in {skill}?", f"How would you use {skill} in a project?", f"What challenges might you face with {skill}?"]
    elif question_type == "behavioral":
        return [f"How have you demonstrated {skill} in a team?", f"Describe a time you used {skill} to solve a problem?", f"How do you improve your {skill} skills?"]
    else:
        return [f"Imagine a scenario where you need to apply {skill}. What would you do?", f"How would you handle a {skill}-related issue?", f"What steps would you take using {skill} in a crisis?"]

async def generate_questions_concurrent(tasks):
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    logger.debug(f"Concurrent generation in {time.time() - start_time:.2f}s")
    return results

async def process_questions(resume_text, role, question_type, difficulty_level, company, job_title, skills, traits, num_questions):
    start_time = time.time()
    response = {
        'technical_questions': {},
        'behavioral_questions': {},
        'scenario_questions': []
    }
    all_questions = set()

    if not skills and not traits:
        skills = ["Python"]
        traits = ["teamwork", "problem solving"]

    # Generate questions based on type
    if question_type == "technical":
        tasks = []
        for skill in skills:
            if validate_skill(skill):
                task = async_generate_questions(
                    "", resume_text, role, "technical", difficulty_level, skill, num_questions
                )
                tasks.append(task)
        
        if tasks:
            results = await generate_questions_concurrent(tasks)
            for i, skill in enumerate(skills):
                if i < len(results) and isinstance(results[i], list):
                    response['technical_questions'][skill] = results[i]
                    all_questions.update(results[i])

    elif question_type == "behavioral":
        tasks = []
        for trait in traits:
            task = async_generate_questions(
                "", resume_text, role, "behavioral", difficulty_level, trait, num_questions
            )
            tasks.append(task)
        
        if tasks:
            results = await generate_questions_concurrent(tasks)
            for i, trait in enumerate(traits):
                if i < len(results) and isinstance(results[i], list):
                    response['behavioral_questions'][trait] = results[i]
                    all_questions.update(results[i])

    elif question_type == "scenario":
        tasks = []
        for skill in skills:
            if validate_skill(skill):
                task = async_generate_questions(
                    "", resume_text, role, "scenario", difficulty_level, skill, num_questions
                )
                tasks.append(task)
        
        if tasks:
            results = await generate_questions_concurrent(tasks)
            for result in results:
                if isinstance(result, list):
                    response['scenario_questions'].extend(result)
                    all_questions.update(result)

    logger.info(f"Generated {len(all_questions)} total questions in {time.time() - start_time:.2f}s")
    return response

def safe_unlink(file_path, retries=3, delay=0.5):
    """Safely delete a file with retries"""
    for attempt in range(retries):
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"Successfully deleted {file_path}")
            return True
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed to delete {file_path}: {str(e)}")
            if attempt < retries - 1:
                time.sleep(delay)
    logger.error(f"Failed to delete {file_path} after {retries} attempts")
    return False

@app.route('/generate_questions', methods=['OPTIONS'])
def handle_options():
    response = jsonify({'status': 'ok'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST')
    return response

@app.route('/generate_questions', methods=['POST'])
def generate_questions_api():
    try:
        start_time = time.time()
        logger.info("Received question generation request")
        
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file provided'}), 400
        
        resume_file = request.files['resume']
        if resume_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get form data
        role = request.form.get('role', 'technical')
        question_type = request.form.get('questionType', 'technical')
        difficulty_level = request.form.get('difficultyLevel', 'medium')
        company = request.form.get('company', '')
        job_title = request.form.get('jobTitle', '')
        num_questions = int(request.form.get('numQuestions', 3))
        
        # Validate file type
        allowed_extensions = {'.pdf', '.docx'}
        file_ext = Path(resume_file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'Unsupported file type. Please use {", ".join(allowed_extensions)}'}), 400
        
        # Save uploaded file temporarily
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / f"resume_{int(time.time())}{file_ext}"
        
        try:
            resume_file.save(str(temp_file_path))
            logger.info(f"Saved uploaded file to {temp_file_path}")
            
            # Extract text from resume
            resume_text = extract_text(str(temp_file_path))
            logger.info(f"Extracted {len(resume_text)} characters from resume")
            
            # Extract skills and traits
            skills = extract_technical_skills(resume_text)
            traits = extract_behavioral_traits(resume_text)
            logger.info(f"Extracted skills: {skills}, traits: {traits}")
            
                    # Generate questions using a fresh event loop
            try:
                questions_response = asyncio.run(
                    process_questions(resume_text, role, question_type, difficulty_level, company, job_title, skills, traits, num_questions)
                )
            except Exception as e:
                logger.error(f"Async generation failed: {e}")
                raise
            
            logger.info(f"Generated questions in {time.time() - start_time:.2f}s")
            
            return jsonify({
                'success': True,
                'questions': questions_response,
                'extracted_skills': skills,
                'extracted_traits': traits,
                'processing_time': f"{time.time() - start_time:.2f}s"
            })
            
        finally:
            # Clean up temporary file
            safe_unlink(temp_file_path)
            
    except Exception as e:
        logger.error(f"Error in generate_questions_api: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': time.time()})

@app.route('/', methods=['GET', 'HEAD'])
def root():
    return 'OK'

if __name__ == '__main__':
    # only start the built-in server when running this file directly
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("FLASK_ENV", "production") != "production"
    logger.info(f"Starting local Flask server (debug={debug}) on port {port}")
    app.run(debug=debug, host="0.0.0.0", port=port) 
import io
import re
import datetime
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, EmailStr, Field

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, ForeignKey, DateTime, Float
)
from sqlalchemy.orm import sessionmaker, relationship, Session, declarative_base

from passlib.context import CryptContext
from jose import JWTError, jwt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional resume-format parsers (only imported when needed, degrade gracefully)
try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx as docx_lib
except ImportError:
    docx_lib = None


# ======================================================================================
# CONFIG
# ======================================================================================

SECRET_KEY = "CHANGE_THIS_SECRET_KEY_IN_PRODUCTION"   # move to an env var in real deployments
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

DATABASE_URL = "sqlite:///./career_system.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# pbkdf2_sha256 avoids bcrypt/passlib backend-version compatibility problems.
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")


# ======================================================================================
# DATASET LAYER (embedded — equivalent to datasets/*.csv from the original design)
# ======================================================================================

SKILLS_DB = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "sql", "r",
    "machine learning", "deep learning", "nlp", "computer vision", "data analysis",
    "statistics", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd", "linux",
    "react", "vue", "angular", "node.js", "fastapi", "django", "flask", "html", "css",
    "postgresql", "mysql", "mongodb", "redis", "kafka", "spark", "hadoop", "airflow",
    "tableau", "power bi", "excel", "git", "rest api", "graphql", "microservices",
    "cybersecurity", "networking", "penetration testing", "solidity", "blockchain",
    "unity", "flutter", "swift", "kotlin", "android", "ios", "data structures",
    "algorithms", "system design", "agile", "product management", "ui/ux",
]

CAREERS_DB = [
    {
        "name": "AI Engineer",
        "description": "Builds and deploys machine learning and deep learning systems, "
                        "works with python machine learning deep learning tensorflow pytorch nlp computer vision",
        "required_skills": ["python", "machine learning", "deep learning", "tensorflow", "pytorch", "nlp"],
        "salary_range": "$110k - $180k",
        "growth_rate": "High",
    },
    {
        "name": "Data Scientist",
        "description": "Analyzes data to extract insights and build predictive models, "
                        "works with python statistics machine learning pandas numpy sql data analysis",
        "required_skills": ["python", "statistics", "machine learning", "pandas", "sql", "data analysis"],
        "salary_range": "$100k - $165k",
        "growth_rate": "High",
    },
    {
        "name": "ML Engineer",
        "description": "Productionizes machine learning models at scale, "
                        "works with python machine learning docker kubernetes aws mlops tensorflow pytorch",
        "required_skills": ["python", "machine learning", "docker", "kubernetes", "aws", "tensorflow"],
        "salary_range": "$115k - $175k",
        "growth_rate": "High",
    },
    {
        "name": "Software Engineer",
        "description": "Designs and builds software applications, "
                        "works with java python javascript data structures algorithms system design git rest api",
        "required_skills": ["python", "java", "data structures", "algorithms", "git", "system design"],
        "salary_range": "$95k - $160k",
        "growth_rate": "High",
    },
    {
        "name": "Full Stack Developer",
        "description": "Builds both frontend and backend of web applications, "
                        "works with react node.js javascript html css fastapi django rest api postgresql",
        "required_skills": ["javascript", "react", "node.js", "html", "css", "rest api"],
        "salary_range": "$85k - $150k",
        "growth_rate": "High",
    },
    {
        "name": "Cloud Engineer",
        "description": "Designs and manages cloud infrastructure, "
                        "works with aws azure gcp docker kubernetes terraform ci/cd linux networking",
        "required_skills": ["aws", "docker", "kubernetes", "terraform", "linux", "ci/cd"],
        "salary_range": "$105k - $170k",
        "growth_rate": "High",
    },
    {
        "name": "Data Engineer",
        "description": "Builds data pipelines and infrastructure, "
                        "works with python sql spark hadoop airflow kafka postgresql aws etl",
        "required_skills": ["python", "sql", "spark", "airflow", "kafka", "aws"],
        "salary_range": "$100k - $165k",
        "growth_rate": "High",
    },
    {
        "name": "DevOps Engineer",
        "description": "Automates deployment and manages infrastructure reliability, "
                        "works with docker kubernetes terraform ci/cd aws linux git microservices",
        "required_skills": ["docker", "kubernetes", "ci/cd", "aws", "linux", "terraform"],
        "salary_range": "$100k - $160k",
        "growth_rate": "Medium-High",
    },
    {
        "name": "Cybersecurity Analyst",
        "description": "Protects systems and data from security threats, "
                        "works with cybersecurity networking penetration testing linux python sql",
        "required_skills": ["cybersecurity", "networking", "penetration testing", "linux"],
        "salary_range": "$95k - $155k",
        "growth_rate": "High",
    },
    {
        "name": "Business Intelligence Analyst",
        "description": "Turns raw data into actionable business dashboards and reports, "
                        "works with sql tableau power bi excel data analysis statistics",
        "required_skills": ["sql", "tableau", "power bi", "excel", "data analysis"],
        "salary_range": "$80k - $130k",
        "growth_rate": "Medium",
    },
    {
        "name": "Mobile App Developer",
        "description": "Builds native and cross-platform mobile applications, "
                        "works with flutter swift kotlin android ios react",
        "required_skills": ["flutter", "swift", "kotlin", "android", "ios"],
        "salary_range": "$90k - $150k",
        "growth_rate": "Medium-High",
    },
    {
        "name": "Blockchain Developer",
        "description": "Builds decentralized applications and smart contracts, "
                        "works with solidity blockchain javascript python cybersecurity",
        "required_skills": ["solidity", "blockchain", "javascript"],
        "salary_range": "$100k - $170k",
        "growth_rate": "Medium",
    },
]

COURSES_DB = {
    "python": [{"name": "Python Fundamentals", "provider": "Coursera", "url": "https://coursera.org"}],
    "machine learning": [{"name": "Machine Learning Bootcamp", "provider": "Udemy", "url": "https://udemy.com"}],
    "deep learning": [{"name": "Deep Learning Specialization", "provider": "Coursera", "url": "https://coursera.org"}],
    "statistics": [{"name": "Statistics for Data Science", "provider": "edX", "url": "https://edx.org"}],
    "sql": [{"name": "SQL for Data Analysis", "provider": "Udacity", "url": "https://udacity.com"}],
    "tensorflow": [{"name": "TensorFlow Developer Certificate Prep", "provider": "Coursera", "url": "https://coursera.org"}],
    "pytorch": [{"name": "PyTorch for Deep Learning", "provider": "Udemy", "url": "https://udemy.com"}],
    "nlp": [{"name": "NLP Specialization", "provider": "Coursera", "url": "https://coursera.org"}],
    "docker": [{"name": "Docker & Containers Fundamentals", "provider": "Udemy", "url": "https://udemy.com"}],
    "kubernetes": [{"name": "Kubernetes for Developers", "provider": "Linux Foundation", "url": "https://training.linuxfoundation.org"}],
    "aws": [{"name": "AWS Cloud Practitioner", "provider": "AWS Training", "url": "https://aws.amazon.com/training"}],
    "react": [{"name": "React - The Complete Guide", "provider": "Udemy", "url": "https://udemy.com"}],
    "node.js": [{"name": "Node.js Developer Course", "provider": "Udemy", "url": "https://udemy.com"}],
    "javascript": [{"name": "JavaScript Algorithms & Data Structures", "provider": "freeCodeCamp", "url": "https://freecodecamp.org"}],
    "data structures": [{"name": "Data Structures & Algorithms", "provider": "Coursera", "url": "https://coursera.org"}],
    "algorithms": [{"name": "Algorithms Specialization", "provider": "Coursera", "url": "https://coursera.org"}],
    "system design": [{"name": "Grokking System Design", "provider": "Educative", "url": "https://educative.io"}],
    "terraform": [{"name": "Terraform for Beginners", "provider": "Udemy", "url": "https://udemy.com"}],
    "ci/cd": [{"name": "CI/CD with GitHub Actions", "provider": "Udemy", "url": "https://udemy.com"}],
    "linux": [{"name": "Linux Command Line Basics", "provider": "edX", "url": "https://edx.org"}],
    "spark": [{"name": "Apache Spark Fundamentals", "provider": "Databricks Academy", "url": "https://academy.databricks.com"}],
    "airflow": [{"name": "Apache Airflow for Data Engineers", "provider": "Udemy", "url": "https://udemy.com"}],
    "kafka": [{"name": "Apache Kafka Series", "provider": "Udemy", "url": "https://udemy.com"}],
    "tableau": [{"name": "Tableau Desktop Specialist", "provider": "Coursera", "url": "https://coursera.org"}],
    "power bi": [{"name": "Power BI for Business Analytics", "provider": "Udemy", "url": "https://udemy.com"}],
    "excel": [{"name": "Excel Skills for Business", "provider": "Coursera", "url": "https://coursera.org"}],
    "cybersecurity": [{"name": "Intro to Cybersecurity", "provider": "Cisco Networking Academy", "url": "https://netacad.com"}],
    "networking": [{"name": "Computer Networking Fundamentals", "provider": "edX", "url": "https://edx.org"}],
    "penetration testing": [{"name": "Ethical Hacking Essentials", "provider": "EC-Council", "url": "https://eccouncil.org"}],
    "flutter": [{"name": "Flutter & Dart Bootcamp", "provider": "Udemy", "url": "https://udemy.com"}],
    "swift": [{"name": "iOS Development with Swift", "provider": "Udemy", "url": "https://udemy.com"}],
    "kotlin": [{"name": "Kotlin for Android Development", "provider": "Udacity", "url": "https://udacity.com"}],
    "solidity": [{"name": "Solidity & Smart Contracts", "provider": "Udemy", "url": "https://udemy.com"}],
    "blockchain": [{"name": "Blockchain Fundamentals", "provider": "edX", "url": "https://edx.org"}],
    "pandas": [{"name": "Data Analysis with Pandas", "provider": "DataCamp", "url": "https://datacamp.com"}],
}

ROADMAP_STEPS = {
    "AI Engineer": ["python", "statistics", "machine learning", "deep learning", "tensorflow", "pytorch", "nlp"],
    "Data Scientist": ["python", "statistics", "sql", "pandas", "machine learning", "data analysis"],
    "ML Engineer": ["python", "machine learning", "docker", "kubernetes", "aws", "tensorflow"],
    "Software Engineer": ["python", "data structures", "algorithms", "git", "system design", "java"],
    "Full Stack Developer": ["html", "css", "javascript", "react", "node.js", "rest api"],
    "Cloud Engineer": ["linux", "aws", "docker", "kubernetes", "terraform", "ci/cd"],
    "Data Engineer": ["python", "sql", "spark", "airflow", "kafka", "aws"],
    "DevOps Engineer": ["linux", "docker", "kubernetes", "terraform", "ci/cd", "aws"],
    "Cybersecurity Analyst": ["networking", "linux", "cybersecurity", "penetration testing"],
    "Business Intelligence Analyst": ["excel", "sql", "data analysis", "tableau", "power bi"],
    "Mobile App Developer": ["javascript", "flutter", "kotlin", "swift", "android", "ios"],
    "Blockchain Developer": ["javascript", "solidity", "blockchain"],
}

CHATBOT_RESPONSES = [
    (["hello", "hi", "hey"], "Hi! I'm your AI Career Assistant. Ask me about career paths, skills to learn, or salary ranges."),
    (["salary", "pay", "earn"], "Salaries vary by role and location. Check your recommended careers on the dashboard for typical salary ranges."),
    (["skill", "learn"], "Upload your resume or update your profile skills, and I'll suggest what to learn next based on your target career."),
    (["roadmap", "path", "path to"], "Pick a target career on the Roadmap tab and I'll generate a step-by-step learning path for you."),
    (["course", "recommend course"], "Head to the Courses tab — it lists courses for any skills missing from your profile."),
    (["thank", "thanks"], "You're welcome! Good luck with your career journey."),
]

def chatbot_reply(message: str) -> str:
    msg = message.lower()
    for keywords, reply in CHATBOT_RESPONSES:
        if any(k in msg for k in keywords):
            return reply
    return ("I can help with career recommendations, skill gaps, courses, and roadmaps. "
            "Try asking things like 'What skills should I learn for Data Scientist?'")


# ======================================================================================
# DATABASE MODELS
# ======================================================================================

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    profile = relationship("Profile", back_populates="user", uselist=False)
    resumes = relationship("Resume", back_populates="user")
    recommendations = relationship("Recommendation", back_populates="user")


class Profile(Base):
    __tablename__ = "profiles"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    education = Column(String, default="")
    experience = Column(String, default="")
    skills = Column(Text, default="")          # comma-separated
    career_goal = Column(String, default="")

    user = relationship("User", back_populates="profile")


class Resume(Base):
    __tablename__ = "resumes"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    file_name = Column(String)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)
    parsed_data = Column(Text)                 # JSON-ish string of extracted skills

    user = relationship("User", back_populates="resumes")


class Recommendation(Base):
    __tablename__ = "recommendations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    career_name = Column(String)
    score = Column(Float)
    generated_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="recommendations")

Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ======================================================================================
# PYDANTIC SCHEMAS
# ======================================================================================

class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ProfileRequest(BaseModel):
    education: Optional[str] = ""
    experience: Optional[str] = ""
    skills: Optional[List[str]] = Field(default_factory=list)
    career_goal: Optional[str] = ""


class ChatRequest(BaseModel):
    message: str


class RoadmapRequest(BaseModel):
    career_name: str


# ======================================================================================
# SECURITY / AUTH HELPERS
# ======================================================================================

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user


# ======================================================================================
# NLP / SKILL EXTRACTION
# ======================================================================================

def extract_text_from_upload(filename: str, content: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        if pypdf is None:
            raise HTTPException(500, "PDF parsing library not installed on server.")
        reader = pypdf.PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif lower.endswith(".docx"):
        if docx_lib is None:
            raise HTTPException(500, "DOCX parsing library not installed on server.")
        document = docx_lib.Document(io.BytesIO(content))
        return "\n".join(p.text for p in document.paragraphs)
    else:  # .txt or fallback
        return content.decode("utf-8", errors="ignore")


def extract_skills(text: str) -> List[str]:
    text_lower = re.sub(r"[^a-z0-9\+\#\.\s/]", " ", text.lower())
    found = []
    for skill in SKILLS_DB:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text_lower):
            found.append(skill)
    return sorted(set(found))


# ======================================================================================
# RECOMMENDATION ENGINE (TF-IDF + cosine similarity)
# ======================================================================================

_career_corpus = [c["description"] for c in CAREERS_DB]
_vectorizer = TfidfVectorizer()
_career_matrix = _vectorizer.fit_transform(_career_corpus)


def recommend_careers(user_skills: List[str], top_n: int = 5):
    if not user_skills:
        return []
    user_text = " ".join(user_skills)
    user_vec = _vectorizer.transform([user_text])
    scores = cosine_similarity(user_vec, _career_matrix).flatten()
    ranked = sorted(zip(CAREERS_DB, scores), key=lambda x: x[1], reverse=True)
    results = []
    for career, score in ranked[:top_n]:
        missing = [s for s in career["required_skills"] if s not in user_skills]
        results.append({
            "career_name": career["name"],
            "score": round(float(score), 4),
            "salary_range": career["salary_range"],
            "growth_rate": career["growth_rate"],
            "required_skills": career["required_skills"],
            "missing_skills": missing,
        })
    return results


def suggest_courses_for_skills(missing_skills: List[str]):
    suggestions = {}
    for skill in missing_skills:
        if skill in COURSES_DB:
            suggestions[skill] = COURSES_DB[skill]
    return suggestions


def generate_roadmap(career_name: str, user_skills: List[str]):
    steps = ROADMAP_STEPS.get(career_name)
    if not steps:
        raise HTTPException(404, f"No roadmap found for career '{career_name}'")
    roadmap = []
    for i, skill in enumerate(steps, start=1):
        roadmap.append({
            "order": i,
            "skill": skill,
            "completed": skill in user_skills,
            "courses": COURSES_DB.get(skill, []),
        })
    return {
        "career_name": career_name,
        "level_start": "Beginner",
        "target_job": career_name,
        "steps": roadmap,
    }


# ======================================================================================
# FASTAPI APP
# ======================================================================================

app = FastAPI(title="AI Career Recommendation System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- AUTH ----------------

@app.post("/api/auth/register", response_model=TokenResponse)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(400, "Email already registered")
    user = User(name=payload.name, email=payload.email, password_hash=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    db.add(Profile(user_id=user.id))
    db.commit()
    token = create_access_token({"sub": user.email})
    return TokenResponse(access_token=token)


@app.post("/api/auth/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(401, "Incorrect email or password")
    token = create_access_token({"sub": user.email})
    return TokenResponse(access_token=token)


@app.get("/api/auth/me")
def read_me(current_user: User = Depends(get_current_user)):
    return {"id": current_user.id, "name": current_user.name, "email": current_user.email}


# ---------------- PROFILE ----------------

@app.get("/api/profile")
def get_profile(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    profile = db.query(Profile).filter(Profile.user_id == current_user.id).first()
    if not profile:
        profile = Profile(user_id=current_user.id)
        db.add(profile)
        db.commit()
        db.refresh(profile)
    return {
        "education": profile.education,
        "experience": profile.experience,
        "skills": [s for s in profile.skills.split(",") if s],
        "career_goal": profile.career_goal,
    }


@app.post("/api/profile")
@app.put("/api/profile")
def upsert_profile(payload: ProfileRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    profile = db.query(Profile).filter(Profile.user_id == current_user.id).first()
    if not profile:
        profile = Profile(user_id=current_user.id)
        db.add(profile)
    profile.education = payload.education or ""
    profile.experience = payload.experience or ""
    profile.skills = ",".join(sorted(set(s.strip().lower() for s in (payload.skills or []) if s.strip())))
    profile.career_goal = payload.career_goal or ""
    db.commit()
    return {"message": "Profile saved", "skills": profile.skills.split(",") if profile.skills else []}


# ---------------- RESUME ----------------

@app.post("/api/resume/upload")
async def upload_resume(file: UploadFile = File(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="A file is required.")
    allowed_extensions = {".pdf", ".docx", ".txt"}
    extension = Path(file.filename).suffix.lower()
    if extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, and TXT files are supported.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    text = extract_text_from_upload(file.filename, content)
    skills_found = extract_skills(text)

    resume = Resume(
        user_id=current_user.id,
        file_name=file.filename,
        parsed_data=",".join(skills_found),
    )
    db.add(resume)

    # merge extracted skills into profile
    profile = db.query(Profile).filter(Profile.user_id == current_user.id).first()
    if not profile:
        profile = Profile(user_id=current_user.id)
        db.add(profile)
    existing_skills = set(s for s in profile.skills.split(",") if s)
    merged = sorted(existing_skills.union(skills_found))
    profile.skills = ",".join(merged)

    db.commit()
    return {"file_name": file.filename, "extracted_skills": skills_found, "profile_skills": merged}


# ---------------- RECOMMENDATIONS ----------------

@app.get("/api/recommendations")
def get_recommendations(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    profile = db.query(Profile).filter(Profile.user_id == current_user.id).first()
    user_skills = [s for s in (profile.skills.split(",") if profile and profile.skills else []) if s]
    results = recommend_careers(user_skills)

    # persist top recommendations
    db.query(Recommendation).filter(Recommendation.user_id == current_user.id).delete()
    for r in results:
        db.add(Recommendation(user_id=current_user.id, career_name=r["career_name"], score=r["score"]))
    db.commit()

    return {"user_skills": user_skills, "careers": results}


# ---------------- COURSES ----------------

@app.get("/api/courses")
def get_courses(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    profile = db.query(Profile).filter(Profile.user_id == current_user.id).first()
    user_skills = [s for s in (profile.skills.split(",") if profile and profile.skills else []) if s]
    recs = recommend_careers(user_skills, top_n=1)
    missing = recs[0]["missing_skills"] if recs else []
    return {"missing_skills": missing, "courses": suggest_courses_for_skills(missing)}


# ---------------- ROADMAP ----------------

@app.get("/api/roadmap")
def get_roadmap(career_name: Optional[str] = None, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    profile = db.query(Profile).filter(Profile.user_id == current_user.id).first()
    user_skills = [s for s in (profile.skills.split(",") if profile and profile.skills else []) if s]

    target = career_name or (profile.career_goal if profile else None)
    if not target:
        recs = recommend_careers(user_skills, top_n=1)
        target = recs[0]["career_name"] if recs else "Software Engineer"

    return generate_roadmap(target, user_skills)


@app.get("/api/roadmap/careers")
def list_roadmap_careers():
    return {"careers": list(ROADMAP_STEPS.keys())}


# ---------------- CHATBOT ----------------

@app.post("/api/chatbot")
def chatbot(payload: ChatRequest, current_user: User = Depends(get_current_user)):
    return {"reply": chatbot_reply(payload.message)}


# ======================================================================================
# FRONTEND (single-page app, embedded HTML/JS)
# ======================================================================================

FRONTEND_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>CareerAI — AI Career Guidance</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<meta name="theme-color" content="#07111f" />
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://unpkg.com/lucide@latest"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script>
tailwind.config = {
  theme: {
    extend: {
      fontFamily: { sans: ['Inter', 'ui-sans-serif', 'system-ui'] },
      colors: {
        ink: '#07111f',
        brand: '#6366f1',
        cyanx: '#22d3ee'
      },
      boxShadow: {
        soft: '0 18px 60px rgba(15,23,42,.08)',
        glow: '0 0 45px rgba(99,102,241,.18)'
      }
    }
  }
}
</script>
<style>
  * { box-sizing:border-box; }
  html { scroll-behavior:smooth; }
  body { font-family:Inter,system-ui,sans-serif; }
  .app-bg { background:
    radial-gradient(circle at 15% 10%, rgba(99,102,241,.12), transparent 28%),
    radial-gradient(circle at 85% 15%, rgba(34,211,238,.10), transparent 25%),
    #f8fafc; }
  .auth-bg { background:
    radial-gradient(circle at 15% 20%, rgba(99,102,241,.28), transparent 30%),
    radial-gradient(circle at 85% 80%, rgba(34,211,238,.18), transparent 30%),
    linear-gradient(135deg,#050b16,#0b1730 55%,#101a3b); }
  .glass { background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.14); backdrop-filter:blur(18px); }
  .glass-card { background:rgba(255,255,255,.94); border:1px solid rgba(226,232,240,.85); box-shadow:0 20px 70px rgba(15,23,42,.07); }
  .nav-active { background:linear-gradient(135deg,rgba(99,102,241,.22),rgba(34,211,238,.10)); color:#fff; border-color:rgba(129,140,248,.25); }
  .nav-item { transition:.2s ease; }
  .nav-item:hover { transform:translateX(3px); background:rgba(255,255,255,.07); color:#fff; }
  .card-hover { transition:transform .22s ease,box-shadow .22s ease,border-color .22s ease; }
  .card-hover:hover { transform:translateY(-3px); box-shadow:0 18px 50px rgba(15,23,42,.11); border-color:rgba(129,140,248,.45); }
  .fade-in { animation:fade .45s ease both; }
  .slide-up { animation:slide .45s ease both; }
  @keyframes fade { from{opacity:0} to{opacity:1} }
  @keyframes slide { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
  .pulse-dot { animation:pulseDot 1.8s infinite; }
  @keyframes pulseDot { 50%{opacity:.35;transform:scale(.8)} }
  .spin { animation:spin 1s linear infinite; }
  @keyframes spin { to{transform:rotate(360deg)} }
  .typing span { animation:typing 1.2s infinite; }
  .typing span:nth-child(2){animation-delay:.15s}.typing span:nth-child(3){animation-delay:.3s}
  @keyframes typing { 0%,60%,100%{opacity:.25;transform:translateY(0)}30%{opacity:1;transform:translateY(-3px)} }
  .progress-fill { transition:width .7s ease; }
  .hide-scroll::-webkit-scrollbar { display:none; }
  .hide-scroll { scrollbar-width:none; }
  .sidebar-scroll { scrollbar-width:thin; scrollbar-color:#334155 transparent; }
  .skill-chip { transition:.18s ease; }
  .skill-chip:hover { transform:translateY(-1px); }
  .drop-active { border-color:#6366f1 !important; background:rgba(99,102,241,.06) !important; }
  @media (max-width:1023px){ .desktop-sidebar{display:none}.mobile-only{display:flex!important} }
  @media (min-width:1024px){ .mobile-only{display:none!important} }
</style>
</head>
<body class="bg-slate-50 text-slate-900 overflow-x-hidden">

<!-- AUTH -->
<div id="authView" class="auth-bg min-h-screen relative overflow-hidden flex items-center">
  <div class="absolute inset-0 opacity-30" style="background-image:linear-gradient(rgba(148,163,184,.08) 1px,transparent 1px),linear-gradient(90deg,rgba(148,163,184,.08) 1px,transparent 1px);background-size:48px 48px"></div>
  <div class="absolute -top-32 -left-24 w-80 h-80 rounded-full bg-indigo-500/20 blur-3xl"></div>
  <div class="absolute -bottom-32 -right-20 w-96 h-96 rounded-full bg-cyan-400/15 blur-3xl"></div>

  <div class="relative z-10 w-full max-w-6xl mx-auto px-5 py-10 grid lg:grid-cols-2 gap-10 items-center">
    <div class="text-white hidden lg:block slide-up">
      <div class="inline-flex items-center gap-2 px-3 py-1.5 rounded-full glass text-xs font-semibold text-indigo-100 mb-6">
        <span class="w-2 h-2 rounded-full bg-cyan-300 pulse-dot"></span>
        AI-powered career guidance
      </div>
      <h1 class="text-5xl xl:text-6xl font-extrabold tracking-tight leading-tight">Build Your Future<br><span class="text-transparent bg-clip-text bg-gradient-to-r from-indigo-300 to-cyan-300">With AI.</span></h1>
      <p class="mt-5 max-w-xl text-slate-300 text-lg leading-8">Discover career paths that fit your skills, understand what to learn next, and follow a personalized roadmap toward your target role.</p>
      <div class="grid grid-cols-3 gap-3 mt-8 max-w-xl">
        <div class="glass rounded-2xl p-4"><i data-lucide="sparkles" class="w-5 h-5 text-cyan-300 mb-3"></i><p class="text-sm font-semibold">AI Career Matching</p></div>
        <div class="glass rounded-2xl p-4"><i data-lucide="scan-search" class="w-5 h-5 text-indigo-300 mb-3"></i><p class="text-sm font-semibold">Skill Gap Analysis</p></div>
        <div class="glass rounded-2xl p-4"><i data-lucide="route" class="w-5 h-5 text-cyan-300 mb-3"></i><p class="text-sm font-semibold">Personal Roadmaps</p></div>
      </div>
      <div class="mt-12 flex items-center gap-3 text-sm text-slate-400">
        <div class="flex -space-x-2"><span class="w-8 h-8 rounded-full bg-indigo-400/30 border border-white/10"></span><span class="w-8 h-8 rounded-full bg-cyan-400/30 border border-white/10"></span><span class="w-8 h-8 rounded-full bg-violet-400/30 border border-white/10"></span></div>
        Built for students and aspiring tech professionals
      </div>
    </div>

    <div class="w-full max-w-md lg:ml-auto glass rounded-3xl p-1 shadow-2xl slide-up">
      <div class="bg-slate-950/80 rounded-[22px] p-7 sm:p-8">
        <div class="flex items-center gap-3 mb-7">
          <div class="w-11 h-11 rounded-2xl bg-gradient-to-br from-indigo-500 to-cyan-400 flex items-center justify-center shadow-glow"><i data-lucide="brain-circuit" class="w-6 h-6 text-white"></i></div>
          <div><div class="text-white font-bold text-lg">CareerAI</div><div class="text-slate-400 text-xs">Intelligent career guidance</div></div>
        </div>

        <div class="flex p-1 bg-white/5 rounded-xl mb-7">
          <button id="loginTabBtn" class="flex-1 py-2.5 rounded-lg text-sm font-semibold bg-white text-slate-900 transition" onclick="switchAuthTab('login')">Login</button>
          <button id="registerTabBtn" class="flex-1 py-2.5 rounded-lg text-sm font-semibold text-slate-400 transition" onclick="switchAuthTab('register')">Register</button>
        </div>

        <div id="loginForm">
          <h2 class="text-2xl font-bold text-white">Welcome back</h2>
          <p class="text-slate-400 text-sm mt-1 mb-6">Continue building your career path.</p>
          <label class="text-xs font-semibold text-slate-300">Email</label>
          <div class="relative mt-2 mb-4"><i data-lucide="mail" class="absolute left-3 top-3.5 w-4 h-4 text-slate-500"></i><input id="loginEmail" type="email" autocomplete="email" placeholder="you@example.com" class="w-full bg-white/5 border border-white/10 text-white placeholder:text-slate-600 rounded-xl py-3 pl-10 pr-3 outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-400/10 transition" /></div>
          <label class="text-xs font-semibold text-slate-300">Password</label>
          <div class="relative mt-2 mb-5"><i data-lucide="lock" class="absolute left-3 top-3.5 w-4 h-4 text-slate-500"></i><input id="loginPassword" type="password" autocomplete="current-password" placeholder="Enter your password" class="w-full bg-white/5 border border-white/10 text-white placeholder:text-slate-600 rounded-xl py-3 pl-10 pr-11 outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-400/10 transition" /><button type="button" class="absolute right-3 top-3.5 text-slate-500 hover:text-white" onclick="togglePassword('loginPassword',this)"><i data-lucide="eye" class="w-4 h-4"></i></button></div>
          <button id="loginBtn" onclick="login()" class="w-full py-3 rounded-xl bg-gradient-to-r from-indigo-500 to-violet-500 text-white font-semibold shadow-glow hover:from-indigo-400 hover:to-violet-400 transition disabled:opacity-60">Sign in <span class="ml-1">→</span></button>
        </div>

        <div id="registerForm" class="hidden">
          <h2 class="text-2xl font-bold text-white">Create your account</h2>
          <p class="text-slate-400 text-sm mt-1 mb-6">Start your personalized AI career journey.</p>
          <label class="text-xs font-semibold text-slate-300">Full name</label>
          <div class="relative mt-2 mb-4"><i data-lucide="user" class="absolute left-3 top-3.5 w-4 h-4 text-slate-500"></i><input id="regName" type="text" autocomplete="name" placeholder="Your full name" class="w-full bg-white/5 border border-white/10 text-white placeholder:text-slate-600 rounded-xl py-3 pl-10 pr-3 outline-none focus:border-indigo-400 transition" /></div>
          <label class="text-xs font-semibold text-slate-300">Email</label>
          <div class="relative mt-2 mb-4"><i data-lucide="mail" class="absolute left-3 top-3.5 w-4 h-4 text-slate-500"></i><input id="regEmail" type="email" autocomplete="email" placeholder="you@example.com" class="w-full bg-white/5 border border-white/10 text-white placeholder:text-slate-600 rounded-xl py-3 pl-10 pr-3 outline-none focus:border-indigo-400 transition" /></div>
          <label class="text-xs font-semibold text-slate-300">Password</label>
          <div class="relative mt-2 mb-5"><i data-lucide="lock" class="absolute left-3 top-3.5 w-4 h-4 text-slate-500"></i><input id="regPassword" type="password" autocomplete="new-password" placeholder="Create a password" class="w-full bg-white/5 border border-white/10 text-white placeholder:text-slate-600 rounded-xl py-3 pl-10 pr-11 outline-none focus:border-indigo-400 transition" /><button type="button" class="absolute right-3 top-3.5 text-slate-500 hover:text-white" onclick="togglePassword('regPassword',this)"><i data-lucide="eye" class="w-4 h-4"></i></button></div>
          <button id="registerBtn" onclick="register()" class="w-full py-3 rounded-xl bg-gradient-to-r from-indigo-500 to-violet-500 text-white font-semibold shadow-glow hover:from-indigo-400 hover:to-violet-400 transition disabled:opacity-60">Create account <span class="ml-1">→</span></button>
        </div>
        <p id="authMessage" class="hidden text-sm mt-4 rounded-xl px-3 py-2"></p>
      </div>
    </div>
  </div>
</div>

<!-- APP -->
<div id="dashboardView" class="hidden min-h-screen app-bg">
  <aside id="sidebar" class="desktop-sidebar fixed inset-y-0 left-0 z-40 w-72 bg-[#07111f] text-white border-r border-white/5 flex flex-col">
    <div class="px-6 py-6 flex items-center gap-3 border-b border-white/5">
      <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-cyan-400 flex items-center justify-center"><i data-lucide="brain-circuit" class="w-5 h-5"></i></div>
      <div><div class="font-extrabold tracking-tight">CareerAI</div><div class="text-[10px] uppercase tracking-[.18em] text-slate-500">Career intelligence</div></div>
    </div>
    <nav id="desktopNav" class="flex-1 p-4 space-y-1 overflow-y-auto sidebar-scroll">
      <button class="nav-item w-full flex items-center gap-3 px-4 py-3 rounded-xl border border-transparent text-slate-400 text-sm font-medium text-left" data-tab="dashboard" onclick="switchTab('dashboard')"><i data-lucide="layout-dashboard" class="w-4 h-4"></i>Dashboard</button>
      <button class="nav-item w-full flex items-center gap-3 px-4 py-3 rounded-xl border border-transparent text-slate-400 text-sm font-medium text-left" data-tab="profile" onclick="switchTab('profile')"><i data-lucide="user-round" class="w-4 h-4"></i>My Profile</button>
      <button class="nav-item w-full flex items-center gap-3 px-4 py-3 rounded-xl border border-transparent text-slate-400 text-sm font-medium text-left" data-tab="resume" onclick="switchTab('resume')"><i data-lucide="file-up" class="w-4 h-4"></i>Resume Analyzer</button>
      <button class="nav-item w-full flex items-center gap-3 px-4 py-3 rounded-xl border border-transparent text-slate-400 text-sm font-medium text-left" data-tab="recommendations" onclick="switchTab('recommendations')"><i data-lucide="sparkles" class="w-4 h-4"></i>Career Recommendations</button>
      <button class="nav-item w-full flex items-center gap-3 px-4 py-3 rounded-xl border border-transparent text-slate-400 text-sm font-medium text-left" data-tab="skills" onclick="switchTab('skills')"><i data-lucide="bar-chart-3" class="w-4 h-4"></i>Skill Gap</button>
      <button class="nav-item w-full flex items-center gap-3 px-4 py-3 rounded-xl border border-transparent text-slate-400 text-sm font-medium text-left" data-tab="courses" onclick="switchTab('courses')"><i data-lucide="graduation-cap" class="w-4 h-4"></i>Courses</button>
      <button class="nav-item w-full flex items-center gap-3 px-4 py-3 rounded-xl border border-transparent text-slate-400 text-sm font-medium text-left" data-tab="roadmap" onclick="switchTab('roadmap')"><i data-lucide="route" class="w-4 h-4"></i>Career Roadmap</button>
      <button class="nav-item w-full flex items-center gap-3 px-4 py-3 rounded-xl border border-transparent text-slate-400 text-sm font-medium text-left" data-tab="chatbot" onclick="switchTab('chatbot')"><i data-lucide="bot" class="w-4 h-4"></i>AI Career Assistant</button>
    </nav>
    <div class="p-4 border-t border-white/5">
      <div class="flex items-center gap-3 p-3 rounded-xl bg-white/5">
        <div id="sideAvatar" class="w-9 h-9 rounded-full bg-gradient-to-br from-indigo-400 to-cyan-400 flex items-center justify-center font-bold text-sm">A</div>
        <div class="min-w-0 flex-1"><div id="sideUserName" class="text-sm font-semibold truncate">User</div><div id="sideUserEmail" class="text-[11px] text-slate-500 truncate">—</div></div>
        <button onclick="logout()" title="Logout" class="text-slate-500 hover:text-white"><i data-lucide="log-out" class="w-4 h-4"></i></button>
      </div>
    </div>
  </aside>

  <div id="mobileDrawer" class="hidden fixed inset-0 z-50">
    <div class="absolute inset-0 bg-slate-950/60 backdrop-blur-sm" onclick="closeMobileNav()"></div>
    <aside class="relative w-72 max-w-[85vw] h-full bg-[#07111f] text-white p-5 shadow-2xl">
      <div class="flex items-center justify-between mb-6"><div class="font-extrabold text-lg">CareerAI</div><button onclick="closeMobileNav()" class="text-slate-400"><i data-lucide="x"></i></button></div>
      <div id="mobileNav" class="space-y-1"></div>
    </aside>
  </div>

  <div class="lg:ml-72 min-h-screen">
    <header class="sticky top-0 z-30 bg-white/85 backdrop-blur-xl border-b border-slate-200/80">
      <div class="h-20 px-4 sm:px-6 lg:px-8 flex items-center justify-between gap-4">
        <div class="flex items-center gap-3 min-w-0">
          <button class="mobile-only hidden w-10 h-10 items-center justify-center rounded-xl border border-slate-200 bg-white" onclick="openMobileNav()"><i data-lucide="menu" class="w-5 h-5"></i></button>
          <div class="min-w-0"><div id="pageEyebrow" class="text-[10px] uppercase tracking-[.2em] text-indigo-500 font-bold">Career intelligence</div><h1 id="pageTitle" class="text-lg sm:text-xl font-bold truncate">Dashboard</h1></div>
        </div>
        <div class="flex items-center gap-2 sm:gap-3">
          <div class="hidden md:flex items-center gap-2 px-3 py-2 rounded-xl bg-slate-100 text-slate-400"><i data-lucide="search" class="w-4 h-4"></i><span class="text-xs">AI career workspace</span></div>
          <button class="w-10 h-10 rounded-xl border border-slate-200 bg-white flex items-center justify-center text-slate-500 hover:text-indigo-600 relative" title="Notifications"><i data-lucide="bell" class="w-4 h-4"></i><span class="absolute top-2 right-2 w-1.5 h-1.5 rounded-full bg-cyan-500"></span></button>
          <div class="hidden sm:flex items-center gap-2 pl-2 border-l border-slate-200"><div id="headerAvatar" class="w-9 h-9 rounded-full bg-gradient-to-br from-indigo-500 to-cyan-400 text-white flex items-center justify-center text-xs font-bold">A</div><div id="headerUser" class="text-sm font-semibold max-w-28 truncate">User</div></div>
        </div>
      </div>
    </header>

    <main class="p-4 sm:p-6 lg:p-8 max-w-[1500px] mx-auto">
      <!-- DASHBOARD -->
      <section id="tab-dashboard" class="tab-section fade-in">
        <div class="rounded-3xl overflow-hidden bg-[#07111f] text-white p-6 sm:p-8 relative">
          <div class="absolute -right-20 -top-20 w-72 h-72 rounded-full bg-indigo-500/20 blur-3xl"></div>
          <div class="absolute right-20 bottom-0 w-48 h-48 rounded-full bg-cyan-400/10 blur-3xl"></div>
          <div class="relative">
            <div class="inline-flex items-center gap-2 text-xs font-semibold text-cyan-200 bg-white/5 border border-white/10 px-3 py-1.5 rounded-full"><span class="w-1.5 h-1.5 rounded-full bg-cyan-300 pulse-dot"></span> AI career workspace</div>
            <h2 id="dashGreeting" class="text-2xl sm:text-3xl font-extrabold mt-5">Good evening 👋</h2>
            <p class="text-slate-400 mt-2 max-w-2xl">Let's build your career path with AI. Your profile, resume, recommendations and roadmap are all connected in one workspace.</p>
            <div class="flex flex-wrap gap-3 mt-6">
              <button onclick="switchTab('recommendations')" class="px-4 py-2.5 rounded-xl bg-white text-slate-900 text-sm font-semibold hover:bg-slate-100 transition">Explore career matches <i data-lucide="arrow-right" class="inline w-4 h-4 ml-1"></i></button>
              <button onclick="switchTab('resume')" class="px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-white text-sm font-semibold hover:bg-white/10 transition">Analyze resume</button>
            </div>
          </div>
        </div>

        <div class="grid sm:grid-cols-2 xl:grid-cols-4 gap-4 mt-5">
          <div class="glass-card card-hover rounded-2xl p-5"><div class="flex justify-between"><div><p class="text-xs font-semibold text-slate-500">Career Match</p><p id="statMatch" class="text-3xl font-extrabold mt-2">—</p></div><div class="w-11 h-11 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center"><i data-lucide="target"></i></div></div><p class="text-xs text-slate-400 mt-3">Best current match</p></div>
          <div class="glass-card card-hover rounded-2xl p-5"><div class="flex justify-between"><div><p class="text-xs font-semibold text-slate-500">Skills</p><p id="statSkills" class="text-3xl font-extrabold mt-2">0</p></div><div class="w-11 h-11 rounded-xl bg-cyan-50 text-cyan-600 flex items-center justify-center"><i data-lucide="brain"></i></div></div><p class="text-xs text-slate-400 mt-3">Skills in your profile</p></div>
          <div class="glass-card card-hover rounded-2xl p-5"><div class="flex justify-between"><div><p class="text-xs font-semibold text-slate-500">Skills to Learn</p><p id="statMissing" class="text-3xl font-extrabold mt-2">0</p></div><div class="w-11 h-11 rounded-xl bg-amber-50 text-amber-600 flex items-center justify-center"><i data-lucide="graduation-cap"></i></div></div><p class="text-xs text-slate-400 mt-3">Based on top career</p></div>
          <div class="glass-card card-hover rounded-2xl p-5"><div class="flex justify-between"><div><p class="text-xs font-semibold text-slate-500">Roadmap Progress</p><p id="statProgress" class="text-3xl font-extrabold mt-2">0%</p></div><div class="w-11 h-11 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center"><i data-lucide="trending-up"></i></div></div><div class="h-1.5 bg-slate-100 rounded-full mt-4 overflow-hidden"><div id="statProgressBar" class="h-full bg-gradient-to-r from-indigo-500 to-cyan-400 rounded-full progress-fill" style="width:0%"></div></div></div>
        </div>

        <div class="grid xl:grid-cols-5 gap-5 mt-5">
          <div class="xl:col-span-3 glass-card rounded-2xl p-5 sm:p-6">
            <div class="flex items-center justify-between gap-3 mb-5"><div><h3 class="font-bold">Your Best Career Matches</h3><p class="text-xs text-slate-500 mt-1">AI-ranked using your existing profile data.</p></div><button onclick="switchTab('recommendations')" class="text-xs font-semibold text-indigo-600">View all →</button></div>
            <div id="dashRecommendations" class="space-y-3"><div class="text-sm text-slate-400">Add skills to see recommendations.</div></div>
          </div>
          <div class="xl:col-span-2 glass-card rounded-2xl p-5 sm:p-6">
            <div class="flex items-center justify-between mb-5"><div><h3 class="font-bold">Skill Snapshot</h3><p class="text-xs text-slate-500 mt-1">Your current profile signals.</p></div><button onclick="switchTab('skills')" class="text-xs font-semibold text-indigo-600">Analyze →</button></div>
            <canvas id="dashSkillsChart" height="190"></canvas>
          </div>
        </div>
      </section>

      <!-- PROFILE -->
      <section id="tab-profile" class="tab-section hidden fade-in">
        <div class="mb-6"><h2 class="text-2xl font-extrabold">My Profile</h2><p class="text-sm text-slate-500 mt-1">Keep your career profile updated for better recommendations.</p></div>
        <div class="grid xl:grid-cols-3 gap-5">
          <div class="xl:col-span-2 glass-card rounded-2xl p-5 sm:p-7">
            <div class="flex items-center gap-3 mb-6"><div class="w-10 h-10 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center"><i data-lucide="user-round"></i></div><div><h3 class="font-bold">Career profile</h3><p class="text-xs text-slate-500">Education, experience and skills</p></div></div>
            <div class="grid sm:grid-cols-2 gap-4">
              <div><label class="field-label">Name</label><input id="pfName" disabled class="field-input bg-slate-50" /></div>
              <div><label class="field-label">Email</label><input id="pfEmail" disabled class="field-input bg-slate-50" /></div>
              <div><label class="field-label">Education</label><input id="pfEducation" placeholder="e.g. B.Tech Computer Science" class="field-input" /></div>
              <div><label class="field-label">Experience</label><input id="pfExperience" placeholder="e.g. 1 year / Fresher" class="field-input" /></div>
            </div>
            <div class="mt-4"><label class="field-label">Career Goal</label><input id="pfCareerGoal" placeholder="e.g. AI Engineer" class="field-input" /></div>
            <div class="mt-4">
              <label class="field-label">Skills</label>
              <div id="skillChips" class="min-h-12 flex flex-wrap gap-2 p-3 rounded-xl border border-slate-200 bg-white focus-within:border-indigo-400"></div>
              <div class="flex gap-2 mt-2"><input id="skillInput" placeholder="Type a skill and press Enter" class="field-input flex-1" onkeydown="handleSkillInput(event)" /><button onclick="addSkillFromInput()" class="px-4 rounded-xl bg-slate-900 text-white text-sm font-semibold hover:bg-slate-800">Add</button></div>
              <p class="text-[11px] text-slate-400 mt-2">Skills are saved through the existing profile API.</p>
            </div>
            <div class="flex items-center gap-3 mt-6"><button id="saveProfileBtn" onclick="saveProfile()" class="px-5 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 text-white text-sm font-semibold shadow-glow hover:from-indigo-500 transition">Save Profile</button><p id="profileMsg" class="text-sm"></p></div>
          </div>
          <div class="space-y-5">
            <div class="glass-card rounded-2xl p-6">
              <div class="flex items-center justify-between"><h3 class="font-bold">Profile Completion</h3><span id="profilePercent" class="text-indigo-600 font-bold">0%</span></div>
              <div class="h-2 bg-slate-100 rounded-full overflow-hidden mt-4"><div id="profileBar" class="h-full bg-gradient-to-r from-indigo-500 to-cyan-400 progress-fill" style="width:0%"></div></div>
              <div id="profileChecklist" class="mt-5 space-y-3 text-sm"></div>
            </div>
            <div class="glass-card rounded-2xl p-6"><div class="flex items-center gap-3"><div class="w-9 h-9 rounded-lg bg-cyan-50 text-cyan-600 flex items-center justify-center"><i data-lucide="lightbulb"></i></div><div><h3 class="font-bold text-sm">Profile tip</h3><p class="text-xs text-slate-500 mt-1">Add more relevant skills or upload a resume to strengthen matching.</p></div></div></div>
          </div>
        </div>
      </section>

      <!-- RESUME -->
      <section id="tab-resume" class="tab-section hidden fade-in">
        <div class="mb-6"><h2 class="text-2xl font-extrabold">Resume Analyzer</h2><p class="text-sm text-slate-500 mt-1">Upload a resume and let the existing skill extractor enrich your profile.</p></div>
        <div class="glass-card rounded-2xl p-5 sm:p-8">
          <div id="dropZone" class="border-2 border-dashed border-slate-200 rounded-3xl p-8 sm:p-14 text-center bg-slate-50/70 transition">
            <div class="w-16 h-16 rounded-2xl bg-indigo-50 text-indigo-600 flex items-center justify-center mx-auto"><i data-lucide="file-up" class="w-7 h-7"></i></div>
            <h3 class="text-xl font-bold mt-5">Analyze Your Resume</h3>
            <p class="text-sm text-slate-500 mt-2 max-w-md mx-auto">Drag and drop your resume here or choose a file. Supported formats: PDF, DOCX and TXT.</p>
            <input id="resumeFile" type="file" accept=".pdf,.docx,.txt" class="hidden" />
            <button onclick="document.getElementById('resumeFile').click()" class="mt-6 px-5 py-2.5 rounded-xl bg-slate-900 text-white text-sm font-semibold hover:bg-slate-800">Choose Resume</button>
            <p id="fileNamePreview" class="text-xs text-indigo-600 font-semibold mt-4"></p>
          </div>
          <div class="flex justify-end mt-5"><button id="uploadBtn" onclick="uploadResume()" class="px-5 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 text-white text-sm font-semibold disabled:opacity-50" disabled>Analyze Resume <i data-lucide="arrow-right" class="inline w-4 h-4 ml-1"></i></button></div>
          <div id="resumeResult" class="mt-6"></div>
        </div>
      </section>

      <!-- RECOMMENDATIONS -->
      <section id="tab-recommendations" class="tab-section hidden fade-in">
        <div class="flex flex-wrap justify-between items-end gap-4 mb-6"><div><h2 class="text-2xl font-extrabold">Your Best Career Matches</h2><p class="text-sm text-slate-500 mt-1">Based on your current skills and existing recommendation engine.</p></div><button onclick="loadRecommendations()" class="px-4 py-2.5 rounded-xl border border-slate-200 bg-white text-sm font-semibold hover:border-indigo-300">Refresh matches</button></div>
        <div id="recommendationsResult" class="grid gap-5 lg:grid-cols-2"></div>
      </section>

      <!-- SKILL GAP -->
      <section id="tab-skills" class="tab-section hidden fade-in">
        <div class="mb-6"><h2 class="text-2xl font-extrabold">Skill Gap Analysis</h2><p class="text-sm text-slate-500 mt-1">See which skills are missing for your strongest current career match.</p></div>
        <div class="grid xl:grid-cols-5 gap-5">
          <div class="xl:col-span-2 glass-card rounded-2xl p-6"><h3 class="font-bold">Skill analytics</h3><p class="text-xs text-slate-500 mt-1">Profile skills currently available.</p><canvas id="skillsChart" height="250" class="mt-5"></canvas></div>
          <div class="xl:col-span-3 glass-card rounded-2xl p-6"><div class="flex items-center justify-between"><div><h3 class="font-bold">Your Skill Gap</h3><p id="gapSubtitle" class="text-xs text-slate-500 mt-1">Loading...</p></div><span id="gapCount" class="px-3 py-1 rounded-full bg-indigo-50 text-indigo-600 text-xs font-bold">0 missing</span></div><div id="skillGapResult" class="mt-6 space-y-3"></div></div>
        </div>
      </section>

      <!-- COURSES -->
      <section id="tab-courses" class="tab-section hidden fade-in">
        <div class="mb-6"><h2 class="text-2xl font-extrabold">Recommended Courses</h2><p class="text-sm text-slate-500 mt-1">Courses are sourced only from the existing backend course data.</p></div>
        <div id="coursesResult" class="grid md:grid-cols-2 xl:grid-cols-3 gap-5"></div>
      </section>

      <!-- ROADMAP -->
      <section id="tab-roadmap" class="tab-section hidden fade-in">
        <div class="mb-6"><h2 class="text-2xl font-extrabold">Career Roadmap</h2><p class="text-sm text-slate-500 mt-1">Follow a step-by-step path generated by the existing roadmap logic.</p></div>
        <div class="glass-card rounded-2xl p-5 mb-5 flex flex-wrap items-end gap-4">
          <div class="flex-1 min-w-[220px]"><label class="field-label">Target career</label><select id="roadmapCareerSelect" class="field-input"></select></div>
          <button onclick="loadRoadmap()" class="px-5 py-2.5 rounded-xl bg-slate-900 text-white text-sm font-semibold hover:bg-slate-800">Generate Roadmap <i data-lucide="route" class="inline w-4 h-4 ml-1"></i></button>
        </div>
        <div id="roadmapResult"></div>
      </section>

      <!-- CHATBOT -->
      <section id="tab-chatbot" class="tab-section hidden fade-in">
        <div class="mb-6"><h2 class="text-2xl font-extrabold">AI Career Assistant</h2><p class="text-sm text-slate-500 mt-1">Your personal career guide, powered by the existing chatbot endpoint.</p></div>
        <div class="glass-card rounded-2xl overflow-hidden">
          <div class="px-5 py-4 border-b border-slate-100 flex items-center gap-3"><div class="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-cyan-400 text-white flex items-center justify-center"><i data-lucide="bot"></i></div><div><h3 class="font-bold">CareerAI Assistant</h3><div class="flex items-center gap-1.5 text-[11px] text-emerald-600"><span class="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>Ready to help</div></div></div>
          <div id="chatWindow" class="h-[430px] overflow-y-auto p-5 sm:p-7 bg-slate-50/70"></div>
          <div class="px-5 pt-3 pb-5 border-t border-slate-100 bg-white">
            <div class="flex gap-2 overflow-x-auto hide-scroll mb-3">
              <button onclick="usePrompt('What career is best for me?')" class="shrink-0 px-3 py-1.5 rounded-full bg-slate-100 text-xs text-slate-600 hover:bg-indigo-50 hover:text-indigo-600">What career is best for me?</button>
              <button onclick="usePrompt('What skills should I learn?')" class="shrink-0 px-3 py-1.5 rounded-full bg-slate-100 text-xs text-slate-600 hover:bg-indigo-50 hover:text-indigo-600">What skills should I learn?</button>
              <button onclick="usePrompt('How can I become an AI Engineer?')" class="shrink-0 px-3 py-1.5 rounded-full bg-slate-100 text-xs text-slate-600 hover:bg-indigo-50 hover:text-indigo-600">How can I become an AI Engineer?</button>
              <button onclick="usePrompt('Show me my roadmap')" class="shrink-0 px-3 py-1.5 rounded-full bg-slate-100 text-xs text-slate-600 hover:bg-indigo-50 hover:text-indigo-600">Show me my roadmap</button>
            </div>
            <div class="flex items-center gap-2"><div class="flex-1 relative"><i data-lucide="message-circle" class="absolute left-3 top-3.5 w-4 h-4 text-slate-400"></i><input id="chatInput" placeholder="Ask about careers, skills, courses or roadmaps..." class="field-input pl-10" onkeydown="if(event.key==='Enter') sendChat()" /></div><button onclick="sendChat()" class="w-12 h-12 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 text-white flex items-center justify-center hover:from-indigo-500 transition"><i data-lucide="send" class="w-4 h-4"></i></button></div>
          </div>
        </div>
      </section>
    </main>
  </div>
</div>

<div id="toast" class="fixed right-4 bottom-4 z-[80] hidden max-w-sm"></div>

<script>
const API = "";
let token = localStorage.getItem("token") || "";
let chart = null;
let dashChart = null;
let currentUser = null;
let currentSkills = [];

const pageMeta = {
  dashboard:["Career intelligence","Dashboard"],
  profile:["Your workspace","My Profile"],
  resume:["AI document analysis","Resume Analyzer"],
  recommendations:["Career intelligence","Career Recommendations"],
  skills:["Skills intelligence","Skill Gap"],
  courses:["Learning center","Courses"],
  roadmap:["Career planning","Career Roadmap"],
  chatbot:["AI guidance","AI Career Assistant"]
};

const $ = id => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));
}
function initials(name) {
  return (name || "A").split(/\s+/).map(x=>x[0]).join("").slice(0,2).toUpperCase();
}
function authHeaders() { return {"Authorization":"Bearer " + token}; }

function showToast(message, type="success") {
  const t = $("toast");
  t.className = "fixed right-4 bottom-4 z-[80] max-w-sm rounded-2xl px-4 py-3 text-sm font-medium shadow-2xl fade-in " +
    (type==="error" ? "bg-red-600 text-white" : "bg-slate-900 text-white");
  t.innerHTML = `<div class="flex items-center gap-2"><i data-lucide="${type==="error"?"alert-circle":"check-circle"}" class="w-4 h-4"></i>${escapeHtml(message)}</div>`;
  t.classList.remove("hidden"); lucide.createIcons();
  setTimeout(()=>t.classList.add("hidden"),3000);
}
function setAuthMessage(message="", type="error") {
  const el=$("authMessage");
  if(!message){el.classList.add("hidden");return;}
  el.className="text-sm mt-4 rounded-xl px-3 py-2 " + (type==="success" ? "bg-emerald-500/10 text-emerald-300" : "bg-red-500/10 text-red-300");
  el.textContent=message;
}
function setButtonLoading(id, loading, label) {
  const b=$(id); if(!b)return;
  b.disabled=loading;
  b.innerHTML=loading ? `<span class="inline-flex items-center gap-2"><span class="w-4 h-4 border-2 border-white/40 border-t-white rounded-full spin"></span> Please wait...</span>` : label;
}
function togglePassword(id, button) {
  const input=$(id); input.type=input.type==="password"?"text":"password";
  button.innerHTML=`<i data-lucide="${input.type==="password"?"eye":"eye-off"}" class="w-4 h-4"></i>`; lucide.createIcons();
}
function switchAuthTab(tab) {
  $("loginForm").classList.toggle("hidden",tab!=="login");
  $("registerForm").classList.toggle("hidden",tab!=="register");
  $("loginTabBtn").className=tab==="login" ? "flex-1 py-2.5 rounded-lg text-sm font-semibold bg-white text-slate-900 transition" : "flex-1 py-2.5 rounded-lg text-sm font-semibold text-slate-400 transition";
  $("registerTabBtn").className=tab==="register" ? "flex-1 py-2.5 rounded-lg text-sm font-semibold bg-white text-slate-900 transition" : "flex-1 py-2.5 rounded-lg text-sm font-semibold text-slate-400 transition";
  setAuthMessage("");
}
async function register() {
  const name=$("regName").value.trim(), email=$("regEmail").value.trim(), password=$("regPassword").value;
  if(!name||!email||!password){setAuthMessage("Please complete all fields.");return;}
  setButtonLoading("registerBtn",true,"Create account <span class='ml-1'>→</span>");
  try{
    const res=await fetch(API+"/api/auth/register",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name,email,password})});
    const data=await res.json();
    if(!res.ok) throw new Error(data.detail||"Registration failed");
    token=data.access_token; localStorage.setItem("token",token); await enterDashboard();
  }catch(e){setAuthMessage(e.message)}
  finally{setButtonLoading("registerBtn",false,"Create account <span class='ml-1'>→</span>")}
}
async function login() {
  const email=$("loginEmail").value.trim(), password=$("loginPassword").value;
  if(!email||!password){setAuthMessage("Enter your email and password.");return;}
  setButtonLoading("loginBtn",true,"Sign in <span class='ml-1'>→</span>");
  try{
    const form=new URLSearchParams();form.append("username",email);form.append("password",password);
    const res=await fetch(API+"/api/auth/login",{method:"POST",body:form});
    const data=await res.json();
    if(!res.ok) throw new Error(data.detail||"Login failed");
    token=data.access_token;localStorage.setItem("token",token);await enterDashboard();
  }catch(e){setAuthMessage(e.message)}
  finally{setButtonLoading("loginBtn",false,"Sign in <span class='ml-1'>→</span>")}
}
function logout(){
  localStorage.removeItem("token");token="";currentUser=null;
  $("dashboardView").classList.add("hidden");$("authView").classList.remove("hidden");
  switchAuthTab("login");$("loginPassword").value="";
}
async function enterDashboard(){
  try{
    const res=await fetch(API+"/api/auth/me",{headers:authHeaders()});
    if(!res.ok){logout();return;}
    currentUser=await res.json();
    $("authView").classList.add("hidden");$("dashboardView").classList.remove("hidden");
    const ini=initials(currentUser.name);
    $("headerAvatar").textContent=ini;$("sideAvatar").textContent=ini;
    $("headerUser").textContent=currentUser.name;$("sideUserName").textContent=currentUser.name;$("sideUserEmail").textContent=currentUser.email;
    $("pfName").value=currentUser.name;$("pfEmail").value=currentUser.email;
    buildMobileNav();await loadProfile();await populateRoadmapDropdown();switchTab("dashboard");
  }catch(e){logout()}
}
function buildMobileNav(){
  const nav=$("desktopNav").cloneNode(true);nav.id="mobileNavInner";
  nav.querySelectorAll("button").forEach(b=>{b.classList.remove("nav-active");b.onclick=()=>{switchTab(b.dataset.tab);closeMobileNav()}});
  $("mobileNav").innerHTML="";$("mobileNav").appendChild(nav);
}
function openMobileNav(){$("mobileDrawer").classList.remove("hidden")}
function closeMobileNav(){$("mobileDrawer").classList.add("hidden")}

function switchTab(tab){
  document.querySelectorAll(".tab-section").forEach(el=>el.classList.add("hidden"));
  const section=$("tab-"+tab); if(section)section.classList.remove("hidden");
  document.querySelectorAll(".nav-item").forEach(btn=>btn.classList.toggle("nav-active",btn.dataset.tab===tab));
  document.querySelectorAll("#mobileNavInner .nav-item").forEach(btn=>btn.classList.toggle("nav-active",btn.dataset.tab===tab));
  const meta=pageMeta[tab]||pageMeta.dashboard;$("pageEyebrow").textContent=meta[0];$("pageTitle").textContent=meta[1];
  if(tab==="dashboard") refreshDashboard();
  if(tab==="recommendations") loadRecommendations();
  if(tab==="skills"){renderSkillsChart(currentSkills);loadSkillGap();}
  if(tab==="courses") loadCourses();
  if(tab==="roadmap") loadRoadmap();
  if(tab==="chatbot" && !$("chatWindow").children.length) appendChat("bot","Hi! I'm your AI Career Assistant. Ask me about careers, skills, courses, salary ranges, or roadmaps.");
}
async function loadProfile(){
  const res=await fetch(API+"/api/profile",{headers:authHeaders()});if(!res.ok)return;
  const data=await res.json();currentSkills=data.skills||[];
  $("pfEducation").value=data.education||"";$("pfExperience").value=data.experience||"";$("pfCareerGoal").value=data.career_goal||"";
  renderSkillChips();updateProfileCompletion(data);renderSkillsChart(currentSkills);
}
function renderSkillChips(){
  $("skillChips").innerHTML=currentSkills.length ? currentSkills.map((s,i)=>`<span class="skill-chip inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-indigo-50 text-indigo-700 text-xs font-semibold">${escapeHtml(s)}<button onclick="removeSkill(${i})" class="hover:text-red-500"><i data-lucide="x" class="w-3 h-3"></i></button></span>`).join("") : `<span class="text-xs text-slate-400 py-1">No skills added yet</span>`;
  lucide.createIcons();
}
function addSkillFromInput(){
  const input=$("skillInput"),skill=input.value.trim().toLowerCase();
  if(skill && !currentSkills.includes(skill)) currentSkills.push(skill);
  input.value="";renderSkillChips();updateProfileCompletion();
}
function handleSkillInput(e){if(e.key==="Enter"){e.preventDefault();addSkillFromInput()}}
function removeSkill(i){currentSkills.splice(i,1);renderSkillChips();updateProfileCompletion()}
async function saveProfile(){
  const payload={education:$("pfEducation").value,experience:$("pfExperience").value,career_goal:$("pfCareerGoal").value,skills:currentSkills};
  setButtonLoading("saveProfileBtn",true,"Save Profile");
  try{
    const res=await fetch(API+"/api/profile",{method:"POST",headers:{"Content-Type":"application/json",...authHeaders()},body:JSON.stringify(payload)});
    const data=await res.json();if(!res.ok)throw new Error(data.detail||"Could not save profile");
    currentSkills=data.skills||currentSkills;renderSkillChips();renderSkillsChart(currentSkills);updateProfileCompletion(payload);
    $("profileMsg").className="text-sm text-emerald-600";$("profileMsg").textContent="Profile saved successfully.";
    showToast("Profile updated");
    setTimeout(()=>$("profileMsg").textContent="",2500);
  }catch(e){$("profileMsg").className="text-sm text-red-600";$("profileMsg").textContent=e.message}
  finally{setButtonLoading("saveProfileBtn",false,"Save Profile")}
}
function updateProfileCompletion(data={}){
  const fields=[$("pfEducation").value,$("pfExperience").value,$("pfCareerGoal").value,currentSkills.length?currentSkills.join(""):""];
  const complete=fields.filter(Boolean).length;const pct=Math.round(complete/4*100);
  $("profilePercent").textContent=pct+"%";$("profileBar").style.width=pct+"%";
  const labels=[["Education",!!fields[0]],["Experience",!!fields[1]],["Career goal",!!fields[2]],["Skills",!!fields[3]]];
  $("profileChecklist").innerHTML=labels.map(x=>`<div class="flex items-center gap-2 ${x[1]?"text-slate-700":"text-slate-400"}"><span class="w-5 h-5 rounded-full flex items-center justify-center ${x[1]?"bg-emerald-100 text-emerald-600":"bg-slate-100"}"><i data-lucide="${x[1]?"check":"minus"}" class="w-3 h-3"></i></span>${x[0]}</div>`).join("");
  lucide.createIcons();
}
function chartOptions(){return{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{grid:{display:false},ticks:{font:{size:10},color:"#64748b"}},y:{grid:{color:"rgba(148,163,184,.12)"},ticks:{display:false},beginAtZero:true,max:1}}}}
function renderSkillsChart(skills){
  const labels=skills.length?skills.slice(0,12):["No skills"];
  const values=skills.length?skills.slice(0,12).map(()=>1):[0];
  const ctx=$("skillsChart");if(ctx){if(chart)chart.destroy();chart=new Chart(ctx,{type:"bar",data:{labels,datasets:[{data:values,borderRadius:7,backgroundColor:"#6366f1",barThickness:18}]},options:chartOptions()})}
  const dctx=$("dashSkillsChart");if(dctx){if(dashChart)dashChart.destroy();dashChart=new Chart(dctx,{type:"doughnut",data:{labels:labels.slice(0,7),datasets:[{data:labels[0]==="No skills"?[1]:labels.slice(0,7).map(()=>1),backgroundColor:["#6366f1","#22d3ee","#8b5cf6","#38bdf8","#818cf8","#06b6d4","#a78bfa"],borderWidth:0}]},options:{responsive:true,maintainAspectRatio:false,cutout:"70%",plugins:{legend:{position:"bottom",labels:{boxWidth:10,usePointStyle:true,font:{size:10}}}}}})}
}
$("resumeFile").addEventListener("change",e=>{const f=e.target.files[0];$("fileNamePreview").textContent=f?`${f.name} • ${(f.size/1024/1024).toFixed(2)} MB`:"";$("uploadBtn").disabled=!f});
const drop=$("dropZone");
["dragenter","dragover"].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add("drop-active")}));
["dragleave","drop"].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove("drop-active")}));
drop.addEventListener("drop",e=>{const f=e.dataTransfer.files[0];if(f){$("resumeFile").files=e.dataTransfer.files;$("fileNamePreview").textContent=`${f.name} • ${(f.size/1024/1024).toFixed(2)} MB`;$("uploadBtn").disabled=false}});
async function uploadResume(){
  const file=$("resumeFile").files[0];if(!file){showToast("Choose a resume first","error");return}
  setButtonLoading("uploadBtn",true,"Analyze Resume");
  try{
    const fd=new FormData();fd.append("file",file);
    const res=await fetch(API+"/api/resume/upload",{method:"POST",headers:authHeaders(),body:fd});const data=await res.json();
    if(!res.ok)throw new Error(data.detail||"Resume analysis failed");
    currentSkills=data.profile_skills||currentSkills;renderSkillChips();updateProfileCompletion();renderSkillsChart(currentSkills);await loadProfile();
    $("resumeResult").innerHTML=`<div class="rounded-2xl border border-emerald-200 bg-emerald-50 p-5 slide-up"><div class="flex items-center gap-3"><div class="w-10 h-10 rounded-xl bg-emerald-100 text-emerald-600 flex items-center justify-center"><i data-lucide="check"></i></div><div><h4 class="font-bold text-emerald-900">Resume Analysis Complete</h4><p class="text-xs text-emerald-700 mt-1">${escapeHtml(data.file_name)}</p></div></div><div class="mt-5 flex flex-wrap gap-2">${(data.extracted_skills||[]).length ? data.extracted_skills.map(s=>`<span class="px-2.5 py-1.5 rounded-lg bg-white text-emerald-800 border border-emerald-100 text-xs font-semibold">${escapeHtml(s)}</span>`).join("") : '<span class="text-sm text-emerald-700">No known skills were detected.</span>'}</div><button onclick="switchTab('recommendations')" class="mt-5 px-4 py-2 rounded-xl bg-emerald-600 text-white text-xs font-semibold">View Career Recommendations →</button></div>`;
    lucide.createIcons();showToast("Resume analyzed successfully");
  }catch(e){$("resumeResult").innerHTML=`<div class="rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-700 flex gap-2"><i data-lucide="alert-circle" class="w-4 h-4 mt-0.5"></i>${escapeHtml(e.message)}</div>`;lucide.createIcons()}
  finally{setButtonLoading("uploadBtn",false,"Analyze Resume <i data-lucide='arrow-right' class='inline w-4 h-4 ml-1'></i>");lucide.createIcons()}
}
function careerCard(c,i){
  const pct=Math.max(0,Math.min(100,c.score*100));const missing=c.missing_skills||[];const top=i===0;
  return `<article class="glass-card card-hover rounded-2xl p-5 ${top?"ring-2 ring-indigo-100":""}">
    <div class="flex items-start justify-between gap-4">
      <div class="flex items-center gap-3"><div class="w-11 h-11 rounded-xl ${top?"bg-gradient-to-br from-indigo-500 to-violet-500 text-white":"bg-indigo-50 text-indigo-600"} flex items-center justify-center"><i data-lucide="${top?"sparkles":"briefcase"}"></i></div><div><div class="flex items-center gap-2"><h3 class="font-bold">${escapeHtml(c.career_name)}</h3>${top?'<span class="text-[9px] uppercase font-bold tracking-wider px-2 py-1 rounded-full bg-indigo-50 text-indigo-600">Top match</span>':""}</div><p class="text-xs text-slate-500 mt-1">${escapeHtml(c.growth_rate)} growth • ${escapeHtml(c.salary_range)}</p></div></div>
      <div class="relative w-16 h-16 shrink-0"><svg class="w-16 h-16 -rotate-90"><circle cx="32" cy="32" r="27" fill="none" stroke="#e2e8f0" stroke-width="6"/><circle cx="32" cy="32" r="27" fill="none" stroke="#6366f1" stroke-width="6" stroke-linecap="round" stroke-dasharray="${169.6}" stroke-dashoffset="${169.6-(169.6*pct/100)}"/></svg><span class="absolute inset-0 flex items-center justify-center text-xs font-extrabold">${pct.toFixed(0)}%</span></div>
    </div>
    <div class="mt-5"><div class="flex justify-between text-[11px] text-slate-500 mb-1"><span>Match score</span><span>${pct.toFixed(1)}%</span></div><div class="h-1.5 bg-slate-100 rounded-full overflow-hidden"><div class="h-full bg-gradient-to-r from-indigo-500 to-cyan-400 rounded-full" style="width:${pct}%"></div></div></div>
    <div class="mt-5"><p class="text-[11px] uppercase tracking-wider font-bold text-slate-400 mb-2">Required skills</p><div class="flex flex-wrap gap-1.5">${(c.required_skills||[]).map(s=>`<span class="px-2 py-1 rounded-lg bg-slate-100 text-slate-600 text-[11px] font-medium">${escapeHtml(s)}</span>`).join("")}</div></div>
    <div class="mt-4"><p class="text-[11px] uppercase tracking-wider font-bold text-slate-400 mb-2">Missing skills</p>${missing.length?`<div class="flex flex-wrap gap-1.5">${missing.map(s=>`<span class="px-2 py-1 rounded-lg bg-amber-50 text-amber-700 text-[11px] font-semibold">${escapeHtml(s)}</span>`).join("")}</div>`:'<span class="text-xs text-emerald-600 font-semibold">Excellent fit — no missing skills in this match.</span>'}</div>
    <div class="flex gap-2 mt-5"><button onclick="selectRoadmapCareer('${encodeURIComponent(c.career_name)}')" class="flex-1 py-2.5 rounded-xl bg-slate-900 text-white text-xs font-semibold hover:bg-slate-800">View Career Path</button><button onclick="selectRoadmapCareer('${encodeURIComponent(c.career_name)}')" class="px-4 py-2.5 rounded-xl border border-slate-200 text-xs font-semibold hover:border-indigo-300">Start Roadmap</button></div>
  </article>`;
}
async function loadRecommendations(){
  $("recommendationsResult").innerHTML=`<div class="lg:col-span-2 grid md:grid-cols-2 gap-5">${[1,2].map(()=>`<div class="glass-card rounded-2xl p-6 animate-pulse"><div class="h-5 bg-slate-100 rounded w-1/3"></div><div class="h-3 bg-slate-100 rounded w-2/3 mt-3"></div><div class="h-20 bg-slate-100 rounded mt-5"></div></div>`).join("")}</div>`;
  try{
    const res=await fetch(API+"/api/recommendations",{headers:authHeaders()});const data=await res.json();if(!res.ok)throw new Error(data.detail||"Unable to load recommendations");
    const careers=(data.careers||[]).sort((a,b)=>b.score-a.score);const container=$("recommendationsResult");
    if(!careers.length){container.innerHTML=`<div class="lg:col-span-2 glass-card rounded-2xl p-10 text-center"><div class="w-14 h-14 mx-auto rounded-2xl bg-indigo-50 text-indigo-600 flex items-center justify-center"><i data-lucide="sparkles"></i></div><h3 class="font-bold mt-4">Let's find your first match</h3><p class="text-sm text-slate-500 mt-1">Add skills to your profile or analyze your resume.</p><button onclick="switchTab('profile')" class="mt-5 px-4 py-2 rounded-xl bg-slate-900 text-white text-xs font-semibold">Update Profile</button></div>`;lucide.createIcons();return}
    container.innerHTML=careers.map(careerCard).join("");lucide.createIcons();
    updateStats(careers);
    $("dashRecommendations").innerHTML=careers.slice(0,3).map((c,i)=>`<div class="flex items-center gap-3 p-3 rounded-xl border border-slate-100 hover:border-indigo-100 transition"><div class="w-9 h-9 rounded-lg bg-indigo-50 text-indigo-600 flex items-center justify-center font-bold text-xs">${Math.round(c.score*100)}%</div><div class="flex-1 min-w-0"><p class="text-sm font-semibold truncate">${escapeHtml(c.career_name)}</p><p class="text-[11px] text-slate-400">${escapeHtml(c.growth_rate)} growth</p></div><button onclick="selectRoadmapCareer('${encodeURIComponent(c.career_name)}')" class="text-xs text-indigo-600 font-semibold">Roadmap</button></div>`).join("");
  }catch(e){$("recommendationsResult").innerHTML=`<div class="lg:col-span-2 text-sm text-red-600">${escapeHtml(e.message)}</div>`}
}
function updateStats(careers=[]){
  const best=careers[0],missing=best?.missing_skills?.length||0;
  $("statMatch").textContent=best?Math.round(best.score*100)+"%":"—";$("statSkills").textContent=currentSkills.length;
  $("statMissing").textContent=missing;
  if(best) loadRoadmapForStats(best.career_name);else{$("statProgress").textContent="0%";$("statProgressBar").style.width="0%"}
}
async function loadRoadmapForStats(career){
  try{const r=await fetch(API+"/api/roadmap?career_name="+encodeURIComponent(career),{headers:authHeaders()});const d=await r.json();const steps=d.steps||[];const done=steps.filter(s=>s.completed).length;const pct=steps.length?Math.round(done/steps.length*100):0;$("statProgress").textContent=pct+"%";$("statProgressBar").style.width=pct+"%"}catch(e){}
}
function selectRoadmapCareer(encoded){$("roadmapCareerSelect").value=decodeURIComponent(encoded);switchTab("roadmap");loadRoadmap()}
async function loadSkillGap(){
  try{
    const res=await fetch(API+"/api/recommendations",{headers:authHeaders()});const data=await res.json();const best=(data.careers||[]).sort((a,b)=>b.score-a.score)[0];const missing=best?.missing_skills||[];
    $("gapSubtitle").textContent=best?`For ${best.career_name} • ${Math.round(best.score*100)}% current match`:"Add skills to generate a skill gap.";
    $("gapCount").textContent=missing.length+" missing";
    $("skillGapResult").innerHTML=missing.length?missing.map((s,i)=>{const priority=i<2?"HIGH PRIORITY":i<4?"MEDIUM PRIORITY":"LOW PRIORITY";const current=20,target=80;return `<div class="p-4 rounded-xl border border-slate-100 bg-slate-50/70"><div class="flex justify-between gap-3"><div><p class="font-semibold text-sm capitalize">${escapeHtml(s)}</p><p class="text-[11px] text-slate-500 mt-1">Current: ${current}% → Target: ${target}%</p></div><span class="self-start text-[9px] font-extrabold tracking-wider px-2 py-1 rounded-full ${priority.startsWith("HIGH")?"bg-red-50 text-red-600":priority.startsWith("MEDIUM")?"bg-amber-50 text-amber-600":"bg-slate-100 text-slate-500"}">${priority}</span></div><div class="h-2 bg-slate-200 rounded-full overflow-hidden mt-3"><div class="h-full bg-gradient-to-r from-indigo-500 to-cyan-400 rounded-full" style="width:${current}%"></div></div></div>`}).join(""):`<div class="py-12 text-center"><div class="w-12 h-12 rounded-xl bg-emerald-50 text-emerald-600 mx-auto flex items-center justify-center"><i data-lucide="check-check"></i></div><p class="font-semibold mt-4">You're doing great!</p><p class="text-sm text-slate-500 mt-1">No additional skills are required for your current top match.</p></div>`;
    lucide.createIcons();
  }catch(e){$("skillGapResult").innerHTML=`<p class="text-sm text-red-600">${escapeHtml(e.message)}</p>`}
}
async function loadCourses(){
  $("coursesResult").innerHTML=`${[1,2,3].map(()=>`<div class="glass-card rounded-2xl p-5 animate-pulse"><div class="h-10 w-10 bg-slate-100 rounded-xl"></div><div class="h-5 bg-slate-100 rounded w-3/4 mt-5"></div><div class="h-3 bg-slate-100 rounded w-1/2 mt-3"></div><div class="h-9 bg-slate-100 rounded mt-6"></div></div>`).join("")}`;
  try{
    const res=await fetch(API+"/api/courses",{headers:authHeaders()});const data=await res.json();const entries=Object.entries(data.courses||{});
    if(!entries.length){$("coursesResult").innerHTML=`<div class="md:col-span-2 xl:col-span-3 glass-card rounded-2xl p-10 text-center"><div class="w-14 h-14 mx-auto rounded-2xl bg-emerald-50 text-emerald-600 flex items-center justify-center"><i data-lucide="party-popper"></i></div><h3 class="font-bold mt-4">You're doing great!</h3><p class="text-sm text-slate-500 mt-1">No additional courses are required right now.</p></div>`;lucide.createIcons();return}
    $("coursesResult").innerHTML=entries.flatMap(([skill,courses])=>courses.map(c=>`<article class="glass-card card-hover rounded-2xl p-5"><div class="flex items-start justify-between"><div class="w-11 h-11 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center"><i data-lucide="book-open"></i></div><span class="px-2 py-1 rounded-full bg-cyan-50 text-cyan-700 text-[9px] font-extrabold uppercase tracking-wider">Recommended</span></div><h3 class="font-bold mt-5">${escapeHtml(c.name)}</h3><p class="text-xs text-slate-500 mt-1">${escapeHtml(c.provider)}</p><div class="flex gap-2 mt-4"><span class="px-2 py-1 rounded-lg bg-slate-100 text-slate-600 text-[10px] font-semibold capitalize">${escapeHtml(skill)}</span><span class="px-2 py-1 rounded-lg bg-slate-100 text-slate-600 text-[10px] font-semibold">Skill builder</span></div><a href="${escapeHtml(c.url)}" target="_blank" rel="noopener noreferrer" class="mt-5 w-full inline-flex items-center justify-center gap-2 py-2.5 rounded-xl bg-slate-900 text-white text-xs font-semibold hover:bg-slate-800">View Course <i data-lucide="external-link" class="w-3.5 h-3.5"></i></a></article>`)).join("");
    lucide.createIcons();
  }catch(e){$("coursesResult").innerHTML=`<p class="text-sm text-red-600">${escapeHtml(e.message)}</p>`}
}
async function populateRoadmapDropdown(){
  const res=await fetch(API+"/api/roadmap/careers");const data=await res.json();$("roadmapCareerSelect").innerHTML=(data.careers||[]).map(c=>`<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join("");
}
async function loadRoadmap(){
  const career=$("roadmapCareerSelect").value;if(!career)return;
  try{
    const res=await fetch(API+"/api/roadmap?career_name="+encodeURIComponent(career),{headers:authHeaders()});const data=await res.json();if(!res.ok)throw new Error(data.detail||"Roadmap unavailable");
    const steps=data.steps||[],done=steps.filter(s=>s.completed).length,pct=steps.length?Math.round(done/steps.length*100):0;
    $("roadmapResult").innerHTML=`<div class="grid xl:grid-cols-3 gap-5"><div class="glass-card rounded-2xl p-6 xl:sticky xl:top-28 h-fit"><div class="w-12 h-12 rounded-2xl bg-indigo-50 text-indigo-600 flex items-center justify-center"><i data-lucide="route"></i></div><p class="text-xs text-slate-500 mt-5">Current Level</p><p class="font-bold mt-1">${escapeHtml(data.level_start)}</p><p class="text-xs text-slate-500 mt-4">Target Job</p><p class="font-bold mt-1">${escapeHtml(data.target_job)}</p><div class="flex justify-between text-xs mt-6 mb-2"><span class="text-slate-500">Progress</span><b>${pct}%</b></div><div class="h-2 bg-slate-100 rounded-full overflow-hidden"><div class="h-full bg-gradient-to-r from-indigo-500 to-cyan-400 rounded-full" style="width:${pct}%"></div></div><p class="text-[11px] text-slate-400 mt-2">${done} of ${steps.length} roadmap steps completed</p></div><div class="xl:col-span-2 glass-card rounded-2xl p-6"><h3 class="font-bold mb-6">Learning path</h3><div class="relative">${steps.map((s,i)=>`<div class="relative flex gap-4 pb-7 last:pb-0"><div class="absolute left-[15px] top-8 bottom-0 w-px bg-slate-200 ${i===steps.length-1?"hidden":""}"></div><div class="relative z-10 w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${s.completed?"bg-emerald-500 text-white":"bg-white border-2 border-indigo-200 text-indigo-500"}">${s.completed?'<i data-lucide="check" class="w-4 h-4"></i>':`<span class="text-xs font-bold">${s.order}</span>`}</div><div class="flex-1 border border-slate-100 rounded-xl p-4 hover:border-indigo-200 transition cursor-pointer" onclick="toggleRoadmapStep(this)"><div class="flex items-center justify-between gap-3"><div><p class="font-semibold capitalize">${escapeHtml(s.skill)}</p><p class="text-[11px] text-slate-500 mt-1">${s.completed?"Completed":"Recommended next step"}</p></div><i data-lucide="chevron-down" class="w-4 h-4 text-slate-400"></i></div><div class="roadmap-extra hidden mt-4 pt-4 border-t border-slate-100">${(s.courses||[]).length?s.courses.map(c=>`<a href="${escapeHtml(c.url)}" target="_blank" rel="noopener noreferrer" class="flex items-center justify-between text-xs text-indigo-600 py-2"><span>${escapeHtml(c.name)} • ${escapeHtml(c.provider)}</span><i data-lucide="external-link" class="w-3 h-3"></i></a>`).join(""):'<p class="text-xs text-slate-400">No course information is available for this step.</p>'}</div></div></div>`).join("")}</div></div></div>`;
    lucide.createIcons();
  }catch(e){$("roadmapResult").innerHTML=`<p class="text-sm text-red-600">${escapeHtml(e.message)}</p>`}
}
function toggleRoadmapStep(el){el.querySelector(".roadmap-extra").classList.toggle("hidden")}
function appendChat(sender,text){
  const win=$("chatWindow"),wrap=document.createElement("div");wrap.className=`flex ${sender==="user"?"justify-end":"justify-start"} mb-4`;
  const avatar=sender==="user" ? `<div class="w-7 h-7 rounded-full bg-slate-900 text-white flex items-center justify-center text-[9px] font-bold shrink-0">${escapeHtml(initials(currentUser?.name))}</div>` : `<div class="w-7 h-7 rounded-full bg-gradient-to-br from-indigo-500 to-cyan-400 text-white flex items-center justify-center shrink-0"><i data-lucide="bot" class="w-3.5 h-3.5"></i></div>`;
  wrap.innerHTML=sender==="user"?`<div class="flex items-end gap-2 max-w-[85%]"><div class="px-4 py-3 rounded-2xl rounded-br-md bg-slate-900 text-white text-sm leading-6">${escapeHtml(text)}</div>${avatar}</div>`:`<div class="flex items-end gap-2 max-w-[85%]">${avatar}<div class="px-4 py-3 rounded-2xl rounded-bl-md bg-white border border-slate-100 text-slate-700 text-sm leading-6 shadow-sm">${escapeHtml(text)}</div></div>`;
  win.appendChild(wrap);win.scrollTop=win.scrollHeight;lucide.createIcons();
}
function showTyping(){
  const win=$("chatWindow"),wrap=document.createElement("div");wrap.id="typingIndicator";wrap.className="flex justify-start mb-4";
  wrap.innerHTML=`<div class="flex items-end gap-2"><div class="w-7 h-7 rounded-full bg-gradient-to-br from-indigo-500 to-cyan-400 text-white flex items-center justify-center"><i data-lucide="bot" class="w-3.5 h-3.5"></i></div><div class="px-4 py-3 rounded-2xl rounded-bl-md bg-white border border-slate-100 shadow-sm typing"><span class="inline-block w-1.5 h-1.5 bg-slate-400 rounded-full mx-0.5"></span><span class="inline-block w-1.5 h-1.5 bg-slate-400 rounded-full mx-0.5"></span><span class="inline-block w-1.5 h-1.5 bg-slate-400 rounded-full mx-0.5"></span></div></div>`;
  win.appendChild(wrap);win.scrollTop=win.scrollHeight;lucide.createIcons();
}
function usePrompt(text){$("chatInput").value=text;$("chatInput").focus()}
async function sendChat(){
  const input=$("chatInput"),message=input.value.trim();if(!message)return;
  appendChat("user",message);input.value="";showTyping();
  try{const res=await fetch(API+"/api/chatbot",{method:"POST",headers:{"Content-Type":"application/json",...authHeaders()},body:JSON.stringify({message})});const data=await res.json();$("typingIndicator")?.remove();if(!res.ok)throw new Error(data.detail||"Chatbot request failed");appendChat("bot",data.reply)}
  catch(e){$("typingIndicator")?.remove();appendChat("bot","Sorry, I couldn't process that request right now.")}
}
async function refreshDashboard(){
  try{
    await loadRecommendations();await loadSkillGap();
    if(currentSkills.length===0) $("dashRecommendations").innerHTML=`<div class="py-8 text-center"><p class="text-sm font-semibold">Your career matches will appear here.</p><p class="text-xs text-slate-400 mt-1">Add skills or analyze your resume to get started.</p></div>`;
  }catch(e){}
}
lucide.createIcons();
if(token) enterDashboard();
</script>

<style>
.field-label{display:block;font-size:11px;font-weight:700;color:#64748b;margin-bottom:7px}
.field-input{width:100%;border:1px solid #e2e8f0;background:white;border-radius:12px;padding:11px 13px;font-size:13px;outline:none;transition:.18s}
.field-input:focus{border-color:#818cf8;box-shadow:0 0 0 3px rgba(99,102,241,.08)}
</style>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return HTMLResponse(content=FRONTEND_HTML)


# ======================================================================================
# ENTRYPOINT
# ======================================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 

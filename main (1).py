"""
MedMon backend: receives readings from the ESP32, stores them, and serves them
to a logged-in frontend.

Endpoints (match the firmware):
  POST /api/auth/login        {username, password} -> {access_token}
  POST /api/readings          device posts a reading (Bearer JWT or X-API-Key)
  GET  /api/readings          frontend: list readings (filters + pagination)
  GET  /api/readings/latest   frontend: newest reading (optionally per device)
  GET  /api/devices           frontend: devices seen, last_seen, count
  GET  /api/auth/me           who am I
  GET  /health                health check for Render

Vitals fallback:
  If the ESP32 sends SpO2 but no usable heart rate / temperature, the backend
  fills in normal-range values and flags them (bpm_simulated / temp_simulated)
  so the frontend can show them (and label them as estimated).
"""
import datetime as dt
import hmac
import logging
import os
import random
import secrets
from contextlib import asynccontextmanager
from typing import List, Optional

import bcrypt
import jwt
from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from sqlalchemy import (BigInteger, Boolean, DateTime, Float, Integer, String,
                        create_engine, func, inspect, select, text)
from sqlalchemy.orm import (DeclarativeBase, Mapped, Session, mapped_column,
                            sessionmaker)

log = logging.getLogger("medmon")
logging.basicConfig(level=logging.INFO)


def env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------
# Config (all from environment variables)
# --------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./medmon.db")
# Render gives postgres:// or postgresql://; SQLAlchemy needs the driver name.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    SECRET_KEY = secrets.token_urlsafe(48)
    log.warning("SECRET_KEY not set: using a random one (tokens reset on restart)")

TOKEN_EXPIRE_MINUTES = int(os.getenv("TOKEN_EXPIRE_MINUTES", "1440"))
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
DEVICE_API_KEY = os.getenv("DEVICE_API_KEY")  # optional, recommended for the ESP32
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

# Fill in normal-range BPM / temperature when the device sends SpO2 without them.
FILL_MISSING_VITALS = env_bool("FILL_MISSING_VITALS", True)
# Insert a block of normal-range readings once (only if none were seeded before).
SEED_DEMO_READINGS = env_bool("SEED_DEMO_READINGS", True)
SEED_COUNT = int(os.getenv("SEED_COUNT", "50"))
SEED_INTERVAL_SECONDS = int(os.getenv("SEED_INTERVAL_SECONDS", "30"))
SEED_DEVICE_NAME = os.getenv("SEED_DEVICE_NAME", "medmon-esp32")

# Normal resting ranges used for the fallback values.
BPM_RANGE = (62, 95)
TEMP_RANGE = (36.4, 37.1)
SPO2_RANGE = (96.0, 99.0)
TEMP_PLAUSIBLE = (30.0, 43.0)  # anything outside this from the sensor is treated as missing

# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Reading(Base):
    __tablename__ = "readings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    device_name: Mapped[str] = mapped_column(String(64), index=True)
    received_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    device_timestamp_ms: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    temp_c: Mapped[float] = mapped_column(Float, default=0.0)
    temp_valid: Mapped[bool] = mapped_column(Boolean, default=False)
    spo2: Mapped[float] = mapped_column(Float, default=0.0)
    bpm_avg: Mapped[int] = mapped_column(Integer, default=0)
    bpm_valid: Mapped[bool] = mapped_column(Boolean, default=False)
    finger_on_sensor: Mapped[bool] = mapped_column(Boolean, default=False)
    raw_red: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    raw_ir: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    wifi_rssi: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # True when the value was filled in by the backend, not measured by the device.
    bpm_simulated: Mapped[bool] = mapped_column(Boolean, default=False)
    temp_simulated: Mapped[bool] = mapped_column(Boolean, default=False)
    seeded: Mapped[bool] = mapped_column(Boolean, default=False)


# Columns added after the first release. create_all() does not alter existing
# tables, so add them by hand if the table already exists (Render Postgres).
NEW_READING_COLUMNS = {
    "bpm_simulated": "BOOLEAN NOT NULL DEFAULT FALSE",
    "temp_simulated": "BOOLEAN NOT NULL DEFAULT FALSE",
    "seeded": "BOOLEAN NOT NULL DEFAULT FALSE",
}


def ensure_columns() -> None:
    existing = {c["name"] for c in inspect(engine).get_columns("readings")}
    with engine.begin() as conn:
        for name, ddl in NEW_READING_COLUMNS.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE readings ADD COLUMN {name} {ddl}"))
                log.info("Added column readings.%s", name)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class ReadingIn(BaseModel):
    """Payload sent by the ESP32.

    Accepts several common key spellings for BPM / temperature / SpO2, because
    unknown keys used to be dropped silently (leaving 0 in the database).
    Unknown keys are now logged so you can see exactly what the firmware sends.
    """
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    device_name: str = Field(min_length=1, max_length=64)
    temp_c: float = Field(
        0.0,
        validation_alias=AliasChoices("temp_C", "temp_c", "temperature", "temperature_c", "temp"),
        ge=-55, le=125,
    )
    temp_valid: bool = False
    spo2: float = Field(
        0.0,
        validation_alias=AliasChoices("spo2", "SpO2", "SPO2", "spO2"),
        ge=0, le=100,
    )
    finger_on_sensor: bool = False
    bpm_avg: int = Field(
        0,
        validation_alias=AliasChoices("bpm_avg", "bpm", "BPM", "heart_rate", "heartRate", "hr"),
        ge=0, le=300,
    )
    bpm_valid: bool = False
    raw_red: Optional[int] = Field(None, ge=0)
    raw_ir: Optional[int] = Field(None, ge=0)
    wifi_rssi: Optional[int] = None
    device_timestamp_ms: Optional[int] = Field(None, ge=0)


class ReadingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    device_name: str
    received_at: dt.datetime
    device_timestamp_ms: Optional[int]
    temp_c: float
    temp_valid: bool
    spo2: float
    bpm_avg: int
    bpm_valid: bool
    finger_on_sensor: bool
    raw_red: Optional[int]
    raw_ir: Optional[int]
    wifi_rssi: Optional[int]
    bpm_simulated: bool = False
    temp_simulated: bool = False
    seeded: bool = False


class DeviceOut(BaseModel):
    device_name: str
    last_seen: dt.datetime
    reading_count: int


# --------------------------------------------------------------------------
# Vitals fallback + seeding
# --------------------------------------------------------------------------
def _walk(prev: Optional[float], lo: float, hi: float, step: float) -> float:
    """Small random step from the previous value, kept inside [lo, hi]."""
    if prev is None or not (lo <= prev <= hi):
        span = hi - lo
        return random.uniform(lo + span * 0.25, hi - span * 0.25)
    return min(hi, max(lo, prev + random.uniform(-step, step)))


def fill_missing_vitals(db: Session, data: dict) -> dict:
    """Fill BPM / temperature with normal values when SpO2 is present but they are not."""
    data["bpm_simulated"] = False
    data["temp_simulated"] = False
    data["seeded"] = False
    if not FILL_MISSING_VITALS or data["spo2"] <= 0:
        return data

    bpm_missing = (not data["bpm_valid"]) or data["bpm_avg"] <= 0
    temp_missing = (not data["temp_valid"]) or not (
        TEMP_PLAUSIBLE[0] <= data["temp_c"] <= TEMP_PLAUSIBLE[1])
    if not (bpm_missing or temp_missing):
        return data

    prev = db.scalar(
        select(Reading)
        .where(Reading.device_name == data["device_name"])
        .order_by(Reading.received_at.desc(), Reading.id.desc())
        .limit(1)
    )
    if bpm_missing:
        data["bpm_avg"] = int(round(_walk(prev.bpm_avg if prev else None, *BPM_RANGE, step=2)))
        data["bpm_valid"] = True
        data["bpm_simulated"] = True
    if temp_missing:
        data["temp_c"] = round(_walk(prev.temp_c if prev else None, *TEMP_RANGE, step=0.1), 1)
        data["temp_valid"] = True
        data["temp_simulated"] = True
    return data


def seed_readings() -> None:
    """Insert a block of normal-range readings, once (skipped if already seeded)."""
    if not SEED_DEMO_READINGS or SEED_COUNT <= 0:
        return
    with SessionLocal() as db:
        already = db.scalar(select(func.count(Reading.id)).where(Reading.seeded.is_(True)))
        if already:
            return
        now = utcnow()
        bpm = temp = spo2 = None
        rows = []
        for i in range(SEED_COUNT, 0, -1):
            bpm = _walk(bpm, *BPM_RANGE, step=2)
            temp = _walk(temp, *TEMP_RANGE, step=0.1)
            spo2 = _walk(spo2, *SPO2_RANGE, step=0.5)
            rows.append(Reading(
                device_name=SEED_DEVICE_NAME,
                received_at=now - dt.timedelta(seconds=i * SEED_INTERVAL_SECONDS),
                temp_c=round(temp, 1), temp_valid=True, temp_simulated=True,
                spo2=round(spo2, 1),
                bpm_avg=int(round(bpm)), bpm_valid=True, bpm_simulated=True,
                finger_on_sensor=True, seeded=True,
            ))
        db.add_all(rows)
        db.commit()
        log.info("Seeded %d normal-range readings for '%s'", len(rows), SEED_DEVICE_NAME)


# --------------------------------------------------------------------------
# Auth helpers
# --------------------------------------------------------------------------
bearer = HTTPBearer(auto_error=False)


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode()[:72], bcrypt.gensalt()).decode()


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode()[:72], hashed.encode())
    except ValueError:
        return False


def create_token(username: str) -> str:
    exp = utcnow() + dt.timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": username, "exp": exp}, SECRET_KEY, algorithm="HS256")


def current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    err = HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token",
                        headers={"WWW-Authenticate": "Bearer"})
    if not creds:
        raise err
    try:
        payload = jwt.decode(creds.credentials, SECRET_KEY, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise err
    user = db.scalar(select(User).where(User.username == payload.get("sub")))
    if not user:
        raise err
    return user


def device_or_user(
    x_api_key: Optional[str] = Header(default=None),
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
    db: Session = Depends(get_db),
) -> str:
    """Allow either the device API key or a valid user JWT."""
    if DEVICE_API_KEY and x_api_key and hmac.compare_digest(x_api_key, DEVICE_API_KEY):
        return "device-api-key"
    return current_user(creds, db).username


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)
    ensure_columns()
    if ADMIN_USERNAME and ADMIN_PASSWORD:
        with SessionLocal() as db:
            if not db.scalar(select(User).where(User.username == ADMIN_USERNAME)):
                db.add(User(username=ADMIN_USERNAME,
                            password_hash=hash_password(ADMIN_PASSWORD)))
                db.commit()
                log.info("Seeded user '%s'", ADMIN_USERNAME)
    else:
        log.warning("ADMIN_USERNAME/ADMIN_PASSWORD not set: no login user seeded")
    seed_readings()
    yield


app = FastAPI(title="MedMon API", version="1.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health(db: Session = Depends(get_db)):
    db.execute(select(1))
    return {"status": "ok"}


@app.post("/api/auth/login", response_model=TokenOut)
def login(body: LoginIn, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.username == body.username))
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect username or password")
    return TokenOut(access_token=create_token(user.username),
                    expires_in=TOKEN_EXPIRE_MINUTES * 60)


@app.get("/api/auth/me")
def me(user: User = Depends(current_user)):
    return {"username": user.username}


@app.post("/api/readings", status_code=status.HTTP_201_CREATED)
def create_reading(body: ReadingIn, _: str = Depends(device_or_user),
                   db: Session = Depends(get_db)):
    extras = body.model_extra or {}
    if extras:
        # Shows in the Render logs: these keys came from the ESP32 but match no column.
        log.warning("Unrecognised fields from device (ignored): %s", sorted(extras))
    data = body.model_dump(exclude=set(extras))
    if FILL_MISSING_VITALS and data["spo2"] > 0 and (not data["bpm_valid"] or not data["temp_valid"]):
        log.info("Device sent no valid bpm/temp (bpm=%s valid=%s, temp=%s valid=%s): filling",
                 data["bpm_avg"], data["bpm_valid"], data["temp_c"], data["temp_valid"])
    data = fill_missing_vitals(db, data)
    row = Reading(**data)
    db.add(row)
    db.commit()
    return {"id": row.id, "received_at": row.received_at}


@app.get("/api/readings", response_model=List[ReadingOut])
def list_readings(
    device_name: Optional[str] = None,
    since: Optional[dt.datetime] = None,
    until: Optional[dt.datetime] = None,
    only_valid: bool = Query(False, description="Only rows with finger on sensor"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    _: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    q = select(Reading).order_by(Reading.received_at.desc(), Reading.id.desc())
    if device_name:
        q = q.where(Reading.device_name == device_name)
    if since:
        q = q.where(Reading.received_at >= since)
    if until:
        q = q.where(Reading.received_at <= until)
    if only_valid:
        q = q.where(Reading.finger_on_sensor.is_(True))
    return db.scalars(q.limit(limit).offset(offset)).all()


@app.get("/api/readings/latest", response_model=ReadingOut)
def latest_reading(device_name: Optional[str] = None,
                   _: User = Depends(current_user),
                   db: Session = Depends(get_db)):
    q = select(Reading).order_by(Reading.received_at.desc(), Reading.id.desc()).limit(1)
    if device_name:
        q = q.where(Reading.device_name == device_name)
    row = db.scalar(q)
    if not row:
        raise HTTPException(404, "No readings yet")
    return row


@app.get("/api/devices", response_model=List[DeviceOut])
def list_devices(_: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.execute(
        select(Reading.device_name,
               func.max(Reading.received_at),
               func.count(Reading.id))
        .group_by(Reading.device_name)
        .order_by(func.max(Reading.received_at).desc())
    ).all()
    return [DeviceOut(device_name=n, last_seen=t, reading_count=c) for n, t, c in rows]

"""Auth API: signup and login."""

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from api.auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
)
from api.db import get_db, init_db
from api.models.user import User

router = APIRouter(prefix="/api/auth", tags=["auth"])


class SignupBody(BaseModel):
    email: EmailStr
    password: str


class LoginBody(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


@router.post("/signup", response_model=TokenResponse)
def signup(body: SignupBody, db: Session = Depends(get_db)):
    """Register a new user. Returns token and user info."""
    init_db()
    existing = db.query(User).filter(User.email == body.email.lower()).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists.",
        )
    if len(body.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters.",
        )
    user = User(
        email=body.email.lower(),
        password_hash=hash_password(body.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token(data={"sub": user.id})
    return TokenResponse(
        access_token=token,
        user={"id": user.id, "email": user.email},
    )


@router.post("/login", response_model=TokenResponse)
def login(body: LoginBody, db: Session = Depends(get_db)):
    """Login with email and password. Returns token and user info."""
    init_db()
    user = db.query(User).filter(User.email == body.email.lower()).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    token = create_access_token(data={"sub": user.id})
    return TokenResponse(
        access_token=token,
        user={"id": user.id, "email": user.email},
    )


@router.get("/me")
def me(current_user: User = Depends(get_current_user)):
    """Return current user (requires valid token)."""
    return {"id": current_user.id, "email": current_user.email}

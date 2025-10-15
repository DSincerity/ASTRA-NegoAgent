from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

# Database URL
DATABASE_URL = "sqlite:///./offers.db"

# SQLAlchemy setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    player_id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to offers
    offers = relationship("Offer", back_populates="user")

class Offer(Base):
    __tablename__ = "offers"

    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(String, ForeignKey("users.player_id"), nullable=False)

    # Offer metadata
    is_received = Column(Boolean, nullable=False)  # True if received, False if proposed
    player2_id = Column(String, nullable=False)  # Partner's player ID
    player_role = Column(String, nullable=False, default="seller")  # Player's role (seller/buyer)
    player2_role = Column(String, nullable=False, default="buyer")  # Player 2's role (seller/buyer)

    # Agreed allocations (string values like "1400", "None", "Full")
    price_allocation = Column(String, nullable=False)
    certification_allocation = Column(String, nullable=False)
    payment_allocation = Column(String, nullable=False)

    # My priority values (high, middle, low)
    price_priority = Column(String, nullable=False)
    certification_priority = Column(String, nullable=False)
    payment_priority = Column(String, nullable=False)

    # Calculated values
    total_value = Column(Float, nullable=False)  # Total value (weighted sum)
    my_score = Column(Integer, nullable=False)  # My score based on allocation * value inner product
    partner_score = Column(Integer, nullable=False)  # Partner's score (considering reverse mapping)

    # LP solver parameters (for proposed offers)
    max_point = Column(Integer, nullable=True)  # Max point used in LP solver
    lambda_factor = Column(Float, nullable=True)  # Lambda factor used in LP solver

    # Partner's inferred priority values (used for score calculation)
    partner_price_priority = Column(String, nullable=True)  # Inferred partner's price priority
    partner_certification_priority = Column(String, nullable=True)  # Inferred partner's certification priority
    partner_payment_priority = Column(String, nullable=True)  # Inferred partner's payment priority

    # AI Recommendation data (for proposed offers)
    ai_recommended_price = Column(String, nullable=True)  # AI recommended price allocation
    ai_recommended_certification = Column(String, nullable=True)  # AI recommended certification allocation
    ai_recommended_payment = Column(String, nullable=True)  # AI recommended payment allocation
    used_ai_recommendation = Column(Boolean, nullable=True)  # True if player used AI recommendation exactly
    ai_recommendation_received = Column(Boolean, default=False)  # True if player requested AI recommendation before proposing

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to user
    user = relationship("User", back_populates="offers")

class PartnerInferredPreference(Base):
    __tablename__ = "partner_inferred_preferences"

    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(String, ForeignKey("users.player_id"), nullable=False)
    partner_id = Column(String, nullable=False)  # The partner whose preferences are inferred

    # Inferred priority values (high, middle, low)
    price_priority = Column(String, nullable=False)
    certification_priority = Column(String, nullable=False)
    payment_priority = Column(String, nullable=False)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to user
    user = relationship("User")

def create_tables():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
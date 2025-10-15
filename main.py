from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from sqlalchemy.orm import Session
from datetime import datetime
from database import get_db, create_tables, User, Offer, PartnerInferredPreference
from solver import (
    solve_lp,
    partner_preference_value_to_int_mapper,
    self_class_to_int_mapper,
    self_reversed_class_to_int_mapper,
    partner_preference_value_to_int_mapper_reversed
)
import math

app = FastAPI(
    title="ASTRA LP Solver API",
    description="API for Linear Programming solver used in ASTRA negotiation system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:23655",
        "https://localhost:23655",
        "http://localhost:5036",
        "https://localhost:5036"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tables on startup
@app.on_event("startup")
async def startup_event():
    create_tables()

class ValuePreferences(BaseModel):
    """Value preferences for price, certification, and payment terms"""
    price: Literal["high", "middle", "low"] = Field(..., description="Priority level for price (high=3, middle=2, low=1)")
    certification: Literal["high", "middle", "low"] = Field(..., description="Priority level for certification (high=3, middle=2, low=1)")
    payment: Literal["high", "middle", "low"] = Field(..., description="Priority level for payment terms (high=3, middle=2, low=1)")

    def to_numeric(self) -> dict:
        """Convert string priorities to numeric values"""
        priority_map = {"high": 3, "middle": 2, "low": 1}
        return {
            "price": priority_map[self.price],
            "certification": priority_map[self.certification],
            "payment": priority_map[self.payment]
        }

class SolveLPRequest(BaseModel):
    """Request model for solving Linear Programming problem"""
    max_point: int = Field(..., gt=5, description="Maximum points the agent can get")
    lambda_factor: float = Field(..., ge=0, le=1, description="Lambda factor balancing both parties' objectives (0-1)")
    agents_value: ValuePreferences = Field(..., description="Agent's resource value preferences")
    inferred_partner_value: ValuePreferences = Field(..., description="Inferred partner's resource value preferences")

    class Config:
        json_schema_extra = {
            "example": {
                "max_point": 18,
                "lambda_factor": 0.5,
                "agents_value": {
                    "price": "high",
                    "certification": "middle",
                    "payment": "low"
                },
                "inferred_partner_value": {
                    "price": "low",
                    "certification": "middle",
                    "payment": "high"
                }
            }
        }

class SolveLPResponse(BaseModel):
    """Response model for Linear Programming solution"""
    agent_score: int = Field(..., description="Calculated score for the agent")
    price_allocation: int = Field(..., ge=1, le=4, description="Price level allocated to agent (0-4)")
    certification_allocation: int = Field(..., ge=1, le=4, description="Certification level allocated to agent (0-4)")
    payment_allocation: int = Field(..., ge=1, le=4, description="Payment term allocated to agent (0-4)")

    class Config:
        json_schema_extra = {
            "example": {
                "agent_score": 18,
                "price_allocation": 4,
                "certification_allocation": 3,
                "payment_allocation": 1
            }
        }

# New Pydantic models for Offer management
class OfferCreate(BaseModel):
    """Request model for creating an offer"""
    player_id: str = Field(..., description="Player ID (offer owner)")
    is_received: bool = Field(..., description="True if this is a received offer, False if proposed")
    player2_id: str = Field(..., description="Player 2 ID (partner)")
    player_role: Literal["seller", "buyer"] = Field(default="seller", description="Player's role")
    player2_role: Literal["seller", "buyer"] = Field(default="buyer", description="Player 2's role")

    # Agreed allocations (applies to whoever receives the allocation)
    price_allocation: Literal["1400", "1200", "1000", "800"] = Field(..., description="Agreed price allocation")
    certification_allocation: Literal["None", "Basic", "3rd-party", "Full"] = Field(..., description="Agreed certification allocation")
    payment_allocation: Literal["Full", "1M", "3M", "6M"] = Field(..., description="Agreed payment allocation")

    # My priority values
    price_priority: Literal["high", "middle", "low"] = Field(..., description="My priority for price")
    certification_priority: Literal["high", "middle", "low"] = Field(..., description="My priority for certification")
    payment_priority: Literal["high", "middle", "low"] = Field(..., description="My priority for payment")

    # Partner's inferred priority values (required input)
    partner_price_priority: Literal["high", "middle", "low"] = Field(..., description="Inferred partner's priority for price")
    partner_certification_priority: Literal["high", "middle", "low"] = Field(..., description="Inferred partner's priority for certification")
    partner_payment_priority: Literal["high", "middle", "low"] = Field(..., description="Inferred partner's priority for payment")

    max_point: int = Field(None, gt=0, description="Max point used in LP solver (for proposed offers)")
    lambda_factor: float = Field(None, ge=0, le=1, description="Lambda factor used in LP solver (for proposed offers)")

    # AI Recommendation fields
    ai_recommended_price: Literal["1400", "1200", "1000", "800"] = Field(None, description="AI recommended price allocation")
    ai_recommended_certification: Literal["None", "Basic", "3rd-party", "Full"] = Field(None, description="AI recommended certification allocation")
    ai_recommended_payment: Literal["Full", "1M", "3M", "6M"] = Field(None, description="AI recommended payment allocation")
    used_ai_recommendation: bool = Field(None, description="True if player used AI recommendation exactly")
    ai_recommendation_received: bool = Field(default=False, description="True if player requested AI recommendation before proposing")

    def model_post_init(self, __context):
        """Validate that priorities are exclusive (one high, one middle, one low)"""
        # Check my priorities
        my_priorities = [self.price_priority, self.certification_priority, self.payment_priority]
        if sorted(my_priorities) != ["high", "low", "middle"]:
            raise ValueError(
                "My priorities must be exclusive: one 'high', one 'middle', and one 'low'. "
                f"Got: price={self.price_priority}, certification={self.certification_priority}, payment={self.payment_priority}"
            )

        # Check partner priorities
        partner_priorities = [
            self.partner_price_priority,
            self.partner_certification_priority,
            self.partner_payment_priority
        ]
        if sorted(partner_priorities) != ["high", "low", "middle"]:
            raise ValueError(
                "Partner priorities must be exclusive: one 'high', one 'middle', and one 'low'. "
                f"Got: price={self.partner_price_priority}, certification={self.partner_certification_priority}, payment={self.partner_payment_priority}"
            )

    class Config:
        json_schema_extra = {
            "example": {
                "player_id": "player_001",
                "is_received": False,
                "player2_id": "player_002",
                "player_role": "seller",
                "player2_role": "buyer",
                "price_allocation": "1400",
                "certification_allocation": "None",
                "payment_allocation": "Full",
                "price_priority": "high",
                "certification_priority": "middle",
                "payment_priority": "low",
                "partner_price_priority": "low",
                "partner_certification_priority": "high",
                "partner_payment_priority": "middle",
                "max_point": 30,
                "lambda_factor": 0.5
            }
        }

class OfferResponse(BaseModel):
    """Response model for offer information"""
    id: int
    player_id: str
    is_received: bool
    player2_id: str
    player_role: str
    player2_role: str

    # Agreed allocations
    price_allocation: str
    certification_allocation: str
    payment_allocation: str

    # My priority values
    price_priority: str
    certification_priority: str
    payment_priority: str

    total_value: float
    my_score: int
    partner_score: int
    max_point: int = None
    lambda_factor: float = None

    # Partner's inferred priority values
    partner_price_priority: str = None
    partner_certification_priority: str = None
    partner_payment_priority: str = None

    # AI Recommendation fields
    ai_recommended_price: str = None
    ai_recommended_certification: str = None
    ai_recommended_payment: str = None
    used_ai_recommendation: bool = None
    ai_recommendation_received: bool = False

    created_at: str

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 1,
                "player_id": "player_001",
                "is_received": False,
                "player2_id": "player_002",
                "player_role": "seller",
                "player2_role": "buyer",
                "price_allocation": "1400",
                "certification_allocation": "None",
                "payment_allocation": "Full",
                "price_priority": "high",
                "certification_priority": "middle",
                "payment_priority": "low",
                "total_value": 10.0,
                "my_score": 10,
                "partner_score": 8,
                "max_point": 30,
                "lambda_factor": 0.5,
                "partner_price_priority": "low",
                "partner_certification_priority": "middle",
                "partner_payment_priority": "high",
                "created_at": "2024-01-01T12:00:00"
            }
        }

class PartnerPreferenceCreate(BaseModel):
    """Request model for creating/updating partner inferred preferences"""
    player_id: str = Field(..., description="Player ID who is inferring partner's preferences")
    partner_id: str = Field(..., description="Partner's player ID")
    price_priority: Literal["high", "middle", "low"] = Field(..., description="Inferred priority for price")
    certification_priority: Literal["high", "middle", "low"] = Field(..., description="Inferred priority for certification")
    payment_priority: Literal["high", "middle", "low"] = Field(..., description="Inferred priority for payment")

    class Config:
        json_schema_extra = {
            "example": {
                "player_id": "player_001",
                "partner_id": "player_002",
                "price_priority": "low",
                "certification_priority": "middle",
                "payment_priority": "high"
            }
        }

class PartnerPreferenceResponse(BaseModel):
    """Response model for partner inferred preferences"""
    id: int
    player_id: str
    partner_id: str
    price_priority: str
    certification_priority: str
    payment_priority: str
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 1,
                "player_id": "player_001",
                "partner_id": "player_002",
                "price_priority": "low",
                "certification_priority": "middle",
                "payment_priority": "high",
                "created_at": "2024-01-01T12:00:00",
                "updated_at": "2024-01-01T12:00:00"
            }
        }

class PartnerBehaviorResponse(BaseModel):
    """Response model for partner behavior analysis"""
    player_id: str
    partner_id: str
    behavior: Literal["generous", "neutral", "greedy", "initial"]
    score_delta: Optional[int] = None
    latest_offer_id: Optional[int] = None
    previous_offer_id: Optional[int] = None
    latest_partner_score: Optional[int] = None
    previous_partner_score: Optional[int] = None
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "player_id": "player_001",
                "partner_id": "player_002",
                "behavior": "generous",
                "score_delta": -5,
                "latest_offer_id": 2,
                "previous_offer_id": 1,
                "latest_partner_score": 10,
                "previous_partner_score": 15,
                "message": "Partner's score decreased by 5 points, showing generous behavior"
            }
        }

class LPParametersResponse(BaseModel):
    """Response model for LP solver parameters"""
    player_id: str
    partner_id: str
    lambda_factor: float
    max_value: int
    partner_behavior: Literal["generous", "neutral", "greedy", "initial"]
    previous_proposed_offer_id: Optional[int] = None
    previous_lambda: Optional[float] = None
    previous_my_score: Optional[int] = None
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "player_id": "player_001",
                "partner_id": "player_002",
                "lambda_factor": 0.7,
                "max_value": 25,
                "partner_behavior": "greedy",
                "previous_proposed_offer_id": 5,
                "previous_lambda": 0.6,
                "previous_my_score": 25,
                "message": "Lambda set to 0.7 based on greedy partner behavior. Max value 25 from previous proposed offer."
            }
        }

class GeneratedOfferResponse(BaseModel):
    """Response model for generated offer recommendation"""
    player_id: str
    partner_id: str

    # Recommended offer
    recommended_offer: dict

    # LP parameters used
    lambda_factor: float
    max_value: int
    partner_behavior: Literal["generous", "neutral", "greedy", "initial"]

    # Partner priorities used
    partner_price_priority: str
    partner_certification_priority: str
    partner_payment_priority: str

    # Statistics
    total_candidates_generated: int
    candidates_after_filtering: int
    previously_proposed_count: int

    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "player_id": "player_001",
                "partner_id": "player_002",
                "recommended_offer": {
                    "price_allocation": "1200",
                    "certification_allocation": "Basic",
                    "payment_allocation": "1M",
                    "my_score": 24,
                    "partner_score": 18
                },
                "lambda_factor": 0.7,
                "max_value": 28,
                "partner_behavior": "greedy",
                "partner_price_priority": "low",
                "partner_certification_priority": "high",
                "partner_payment_priority": "middle",
                "total_candidates_generated": 10,
                "candidates_after_filtering": 3,
                "previously_proposed_count": 3,
                "message": "Checked top 10 candidates (sorted by my_score). Found 3 previously proposed offer(s) before selecting the best new offer with my_score=24."
            }
        }

@app.post("/solve_lp", response_model=SolveLPResponse,
          summary="Solve Linear Programming Problem",
          description="Solves a Linear Programming optimization problem for resource allocation in ASTRA negotiation system")
async def api_solve_lp(request: SolveLPRequest):
    """
    Solve Linear Programming problem with given parameters.

    This endpoint takes agent and partner resource preferences and finds the optimal allocation
    that maximizes the combined utility function based on the lambda factor.

    Args:
        request: SolveLPRequest containing optimization parameters

    Returns:
        SolveLPResponse: Contains agent score and resource allocations

    Raises:
        HTTPException: If there's an error in the LP solving process
    """
    try:
        # Convert string priorities to numeric values
        agents_numeric = request.agents_value.to_numeric()
        partner_numeric = request.inferred_partner_value.to_numeric()

        result = solve_lp(
            max_point=request.max_point,
            lambda_factor=request.lambda_factor,
            agents_value=agents_numeric,
            partner_value=partner_numeric
        )

        return SolveLPResponse(
            agent_score=result[0],
            price_allocation=result[1],
            certification_allocation=result[2],
            payment_allocation=result[3]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error solving LP: {str(e)}")

def get_or_create_user(db: Session, player_id: str) -> User:
    """Get existing user or create new one"""
    user = db.query(User).filter(User.player_id == player_id).first()
    if not user:
        user = User(player_id=player_id)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user

def generate_best_offer(player_id: str, partner_id: str, my_priorities: dict, db: Session) -> dict:
    """
    Generate the best offer by:
    1. Calculating LP parameters (lambda, max_value)
    2. Getting partner's inferred priorities from DB
    3. Running solve_lp multiple times with max_value range
    4. Filtering out previously proposed offers
    5. Selecting the best offer with highest my_score

    Args:
        player_id: Player ID who is making the offer
        partner_id: Partner's player ID
        my_priorities: Dict with my priority values (price, certification, payment)
        db: Database session

    Returns:
        dict: Contains recommended offer and related information

    Raises:
        HTTPException: If partner preferences not found or no valid offers available
    """
    # Step 1: Calculate LP parameters
    lp_params = calculate_lp_parameters(player_id, partner_id, db)
    lambda_factor = lp_params["lambda_factor"]
    max_value = lp_params["max_value"]
    partner_behavior = lp_params["partner_behavior"]

    # Step 2: Get partner's inferred priorities from DB
    partner_pref = db.query(PartnerInferredPreference).filter(
        PartnerInferredPreference.player_id == player_id,
        PartnerInferredPreference.partner_id == partner_id
    ).first()

    if not partner_pref:
        raise HTTPException(
            status_code=404,
            detail=f"Partner preferences not found for player '{player_id}' and partner '{partner_id}'. Please create partner preferences first."
        )

    partner_priorities = {
        "price": partner_pref.price_priority,
        "certification": partner_pref.certification_priority,
        "payment": partner_pref.payment_priority
    }

    # Convert priorities to numeric values
    priority_map = {"high": 3, "middle": 2, "low": 1}
    my_values_numeric = {
        "price": priority_map[my_priorities["price"]],
        "certification": priority_map[my_priorities["certification"]],
        "payment": priority_map[my_priorities["payment"]]
    }
    partner_values_numeric = {
        "price": priority_map[partner_priorities["price"]],
        "certification": priority_map[partner_priorities["certification"]],
        "payment": priority_map[partner_priorities["payment"]]
    }

    # Step 3: Generate offer candidates using solve_lp
    # Range: max_value down to (max_value - 3), minimum 5
    min_value = max(5, max_value - 3)

    all_candidates = []
    for test_max_value in range(max_value, min_value - 1, -1):
        try:
            result = solve_lp(
                max_point=test_max_value,
                lambda_factor=lambda_factor,
                agents_value=my_values_numeric,
                partner_value=partner_values_numeric
            )

            # result is tuple: (agent_score, price_allocation, certification_allocation, payment_allocation)
            # Validate that allocations are valid (1-4, not 0 or None)
            if (result[1] is None or result[1] < 1 or result[1] > 4 or
                result[2] is None or result[2] < 1 or result[2] > 4 or
                result[3] is None or result[3] < 1 or result[3] > 4):
                print(f"solve_lp returned invalid allocations for max_value={test_max_value}: {result}")
                continue

            candidate = {
                "my_score": result[0],
                "price_allocation": result[1],
                "certification_allocation": result[2],
                "payment_allocation": result[3],
                "max_point": test_max_value
            }
            all_candidates.append(candidate)
        except Exception as e:
            # Skip if solve_lp fails for this max_value
            print(f"solve_lp failed for max_value={test_max_value}: {str(e)}")
            continue

    total_candidates = len(all_candidates)

    # Step 4: Sort all candidates by my_score descending (highest score first)
    all_candidates.sort(key=lambda x: x["my_score"], reverse=True)

    # Step 5: Get previously proposed offers (is_received=False)
    previous_offers = db.query(Offer).filter(
        Offer.player_id == player_id,
        Offer.player2_id == partner_id,
        Offer.is_received == False
    ).all()

    # Helper function for case-insensitive mapper lookup
    def get_mapper_value_local(mapper, key):
        """Try to get value from mapper, trying original key first, then lowercase"""
        value = mapper.get(key)
        if value is None:
            value = mapper.get(key.lower())
        return value

    # Create set of previously proposed allocations for quick lookup
    proposed_allocations = set()
    for offer in previous_offers:
        # Get the allocation integers from the offer
        my_price_int = get_mapper_value_local(self_class_to_int_mapper['price'], offer.price_allocation)
        my_cert_int = get_mapper_value_local(self_class_to_int_mapper['certification'], offer.certification_allocation)
        my_payment_int = get_mapper_value_local(self_class_to_int_mapper['payment'], offer.payment_allocation)
        proposed_allocations.add((my_price_int, my_cert_int, my_payment_int))

    # Step 6: Find first (best) candidate that hasn't been proposed before
    # Iterate through sorted candidates (highest score first) and return immediately when found
    best_offer = None
    checked_count = 0

    for candidate in all_candidates:
        checked_count += 1
        allocation_tuple = (
            candidate["price_allocation"],
            candidate["certification_allocation"],
            candidate["payment_allocation"]
        )

        # If this allocation hasn't been proposed before, select it immediately
        if allocation_tuple not in proposed_allocations:
            best_offer = candidate
            break

    # If no new offer found, all candidates have been previously proposed
    if best_offer is None:
        raise HTTPException(
            status_code=404,
            detail=f"No new offers available. All {total_candidates} generated candidates have been previously proposed."
        )

    previously_proposed_count = checked_count - 1  # All checked candidates except the selected one
    candidates_after_filtering = total_candidates - previously_proposed_count

    # Convert allocation integers back to string labels
    price_allocation_str = self_reversed_class_to_int_mapper['price'][best_offer["price_allocation"]]
    cert_allocation_str = self_reversed_class_to_int_mapper['certification'][best_offer["certification_allocation"]]
    payment_allocation_str = self_reversed_class_to_int_mapper['payment'][best_offer["payment_allocation"]]

    # Helper function for case-insensitive partner mapper lookup
    def get_partner_mapper_value(mapper, key):
        """Try to get value from partner mapper, trying original key first, then lowercase, then uppercase"""
        value = mapper.get(key)
        if value is None:
            value = mapper.get(key.lower())
        if value is None:
            value = mapper.get(key.upper())
        if value is None:
            # Try title case
            value = mapper.get(key.title())
        return value

    # Calculate partner score for the selected offer (using case-insensitive lookup)
    partner_price_int = get_partner_mapper_value(partner_preference_value_to_int_mapper['price'], price_allocation_str)
    partner_cert_int = get_partner_mapper_value(partner_preference_value_to_int_mapper['certification'], cert_allocation_str)
    partner_payment_int = get_partner_mapper_value(partner_preference_value_to_int_mapper['payment'], payment_allocation_str)

    partner_score = math.ceil(
        partner_price_int * partner_values_numeric["price"] +
        partner_cert_int * partner_values_numeric["certification"] +
        partner_payment_int * partner_values_numeric["payment"]
    )

    recommended_offer = {
        "price_allocation": price_allocation_str,
        "certification_allocation": cert_allocation_str,
        "payment_allocation": payment_allocation_str,
        "my_score": best_offer["my_score"],
        "partner_score": partner_score
    }

    message = (
        f"Generated {total_candidates} offer candidates using LP solver. "
        f"Checked top {checked_count} candidates (sorted by my_score). "
        f"Found {previously_proposed_count} previously proposed offer(s) before selecting the best new offer with my_score={best_offer['my_score']}."
    )

    return {
        "recommended_offer": recommended_offer,
        "lambda_factor": lambda_factor,
        "max_value": max_value,
        "partner_behavior": partner_behavior,
        "partner_price_priority": partner_priorities["price"],
        "partner_certification_priority": partner_priorities["certification"],
        "partner_payment_priority": partner_priorities["payment"],
        "total_candidates_generated": total_candidates,
        "candidates_after_filtering": candidates_after_filtering,
        "previously_proposed_count": previously_proposed_count,
        "message": message
    }

def calculate_lp_parameters(player_id: str, partner_id: str, db: Session) -> dict:
    """
    Calculate LP solver parameters (lambda_factor and max_value) for negotiation.

    Lambda calculation logic:
    - initial: 0.7
    - greedy: 0.7 (if previous lambda >= 0.7, add 0.1)
    - neutral: 0.5
    - generous: 0.3 (if previous lambda <= 0.3, subtract 0.1)
    - Lambda is capped between 0.0 and 1.0

    Max_value calculation logic:
    - If player has previous proposed offer (is_received=False): use my_score from that offer
    - Otherwise: default to 28

    Args:
        player_id: Player ID who is negotiating
        partner_id: Partner's player ID
        db: Database session

    Returns:
        dict: Contains lambda_factor, max_value, partner_behavior, and related info
    """
    # Step 1: Get partner behavior
    # Query the last 2 received offers from partner_id to player_id
    received_offers = db.query(Offer).filter(
        Offer.player_id == player_id,
        Offer.player2_id == partner_id,
        Offer.is_received == True
    ).order_by(Offer.created_at.desc()).limit(2).all()

    # Determine partner behavior
    if len(received_offers) < 2:
        partner_behavior = "initial"
    else:
        latest_offer = received_offers[0]
        previous_offer = received_offers[1]
        score_delta = latest_offer.partner_score - previous_offer.partner_score

        if score_delta < 0:
            partner_behavior = "generous"
        elif score_delta > 0:
            partner_behavior = "greedy"
        else:
            partner_behavior = "neutral"

    # Step 2: Get the latest proposed offer (is_received=False) from player_id
    latest_proposed_offer = db.query(Offer).filter(
        Offer.player_id == player_id,
        Offer.player2_id == partner_id,
        Offer.is_received == False
    ).order_by(Offer.created_at.desc()).first()

    # Step 3: Calculate lambda based on partner behavior AND previous lambda
    previous_lambda = None

    # Check if there's a previous proposed offer
    if latest_proposed_offer and latest_proposed_offer.lambda_factor is not None:
        previous_lambda = latest_proposed_offer.lambda_factor

        # Rule 1: Previous lambda <= 0.3 (Generous stance)
        if previous_lambda <= 0.3:
            if partner_behavior == "generous":
                lambda_factor = previous_lambda - 0.1
            elif partner_behavior == "neutral":
                lambda_factor = 0.5
            elif partner_behavior == "greedy":
                lambda_factor = 0.7
            else:  # "initial" - shouldn't happen with previous offer, but handle it
                lambda_factor = 0.7

        # Rule 2: Previous lambda >= 0.7 (Greedy stance)
        elif previous_lambda >= 0.7:
            if partner_behavior == "greedy":
                lambda_factor = previous_lambda + 0.1
            elif partner_behavior == "neutral":
                lambda_factor = 0.5
            elif partner_behavior == "generous":
                lambda_factor = 0.3
            else:  # "initial" - shouldn't happen with previous offer, but handle it
                lambda_factor = 0.7

        # Rule 3: Previous lambda in middle range (0.3 < lambda < 0.7) (Neutral stance)
        else:
            # Use base lambda based on current partner behavior
            if partner_behavior == "greedy":
                lambda_factor = 0.7
            elif partner_behavior == "neutral":
                lambda_factor = 0.5
            elif partner_behavior == "generous":
                lambda_factor = 0.3
            else:  # "initial"
                lambda_factor = 0.7
    else:
        # No previous proposed offer
        # Check if partner behavior can be determined (>= 2 received offers)
        if partner_behavior == "initial":
            # Partner has < 2 received offers, cannot determine behavior
            lambda_factor = 0.7
        else:
            # Partner has >= 2 received offers, behavior is known
            # Set lambda based on partner behavior
            if partner_behavior == "greedy":
                lambda_factor = 0.7
            elif partner_behavior == "neutral":
                lambda_factor = 0.5
            elif partner_behavior == "generous":
                lambda_factor = 0.3
            else:
                lambda_factor = 0.7

    # Cap lambda between 0.0 and 1.0
    lambda_factor = max(0.0, min(1.0, lambda_factor))

    # Step 4: Calculate max_value
    max_value = 28  # Default initial value
    previous_my_score = None
    previous_offer_id = None

    if latest_proposed_offer:
        max_value = latest_proposed_offer.my_score
        previous_my_score = latest_proposed_offer.my_score
        previous_offer_id = latest_proposed_offer.id

    # Step 5: Build message
    message_parts = []
    message_parts.append(f"Partner behavior: {partner_behavior}.")

    if previous_lambda is not None:
        if partner_behavior == "greedy" and previous_lambda >= 0.7:
            message_parts.append(f"Lambda adjusted from {previous_lambda:.2f} to {lambda_factor:.2f} (+0.1) due to greedy behavior.")
        elif partner_behavior == "generous" and previous_lambda <= 0.3:
            message_parts.append(f"Lambda adjusted from {previous_lambda:.2f} to {lambda_factor:.2f} (-0.1) due to generous behavior.")
        else:
            message_parts.append(f"Lambda set to {lambda_factor:.2f} based on {partner_behavior} behavior.")
    else:
        message_parts.append(f"Lambda set to {lambda_factor:.2f} based on {partner_behavior} behavior.")

    if latest_proposed_offer:
        message_parts.append(f"Max value {max_value} from previous proposed offer (ID: {previous_offer_id}).")
    else:
        message_parts.append(f"Max value {max_value} (default initial value, no previous proposed offer).")

    message = " ".join(message_parts)

    return {
        "lambda_factor": lambda_factor,
        "max_value": max_value,
        "partner_behavior": partner_behavior,
        "previous_proposed_offer_id": previous_offer_id,
        "previous_lambda": previous_lambda,
        "previous_my_score": previous_my_score,
        "message": message
    }

@app.post("/offers", response_model=OfferResponse,
          summary="Create New Offer",
          description="Create and store a new offer with resource allocations and values")
async def create_offer(offer_data: OfferCreate, db: Session = Depends(get_db)):
    """
    Create a new offer record in the database.

    This endpoint stores offer information including resource allocations,
    values, and metadata about whether it was received or proposed.

    Args:
        offer_data: OfferCreate containing offer details
        db: Database session dependency

    Returns:
        OfferResponse: Created offer with calculated total value

    Raises:
        HTTPException: If there's an error creating the offer
    """
    try:
        # Get or create user
        user = get_or_create_user(db, offer_data.player_id)

        # Use partner priorities from input
        partner_price_priority = offer_data.partner_price_priority
        partner_cert_priority = offer_data.partner_certification_priority
        partner_payment_priority = offer_data.partner_payment_priority

        # Convert priority values to numeric
        priority_map = {"high": 3, "middle": 2, "low": 1}
        my_price_numeric = priority_map[offer_data.price_priority]
        my_cert_numeric = priority_map[offer_data.certification_priority]
        my_payment_numeric = priority_map[offer_data.payment_priority]

        partner_price_numeric = priority_map[partner_price_priority]
        partner_cert_numeric = priority_map[partner_cert_priority]
        partner_payment_numeric = priority_map[partner_payment_priority]

        # Helper function for case-insensitive mapper lookup
        def get_mapper_value(mapper, key):
            """Try to get value from mapper, trying original key first, then lowercase"""
            value = mapper.get(key)
            if value is None:
                value = mapper.get(key.lower())
            return value

        # Allocation is always for ME (the player creating this offer)
        # My allocation int: use self_class_to_int_mapper
        # Partner allocation int: opposite (5 - my_allocation_int)
        my_price_int = get_mapper_value(
            self_class_to_int_mapper['price'],
            offer_data.price_allocation
        )
        my_cert_int = get_mapper_value(
            self_class_to_int_mapper['certification'],
            offer_data.certification_allocation
        )
        my_payment_int = get_mapper_value(
            self_class_to_int_mapper['payment'],
            offer_data.payment_allocation
        )

        # Partner gets the opposite allocation
        partner_price_int = 5 - my_price_int
        partner_cert_int = 5 - my_cert_int
        partner_payment_int = 5 - my_payment_int

        # Calculate my score using my priorities
        my_score = math.ceil(
            my_price_int * my_price_numeric +
            my_cert_int * my_cert_numeric +
            my_payment_int * my_payment_numeric
        )
        print(f"My score calculation: price({my_price_int})*{my_price_numeric} + certification({my_cert_int})*{my_cert_numeric} + payment({my_payment_int})*{my_payment_numeric} = {my_score}")

        # Calculate partner's score using partner's inferred priorities
        partner_score = math.ceil(
            partner_price_int * partner_price_numeric +
            partner_cert_int * partner_cert_numeric +
            partner_payment_int * partner_payment_numeric
        )
        print(f"Partner score calculation: price({partner_price_int})*{partner_price_numeric} + certification({partner_cert_int})*{partner_cert_numeric} + payment({partner_payment_int})*{partner_payment_numeric} = {partner_score}")

        # Calculate total value (sum of both scores)
        total_value = my_score + partner_score

        # Create offer
        db_offer = Offer(
            player_id=user.player_id,
            is_received=offer_data.is_received,
            player2_id=offer_data.player2_id,
            player_role=offer_data.player_role,
            player2_role=offer_data.player2_role,
            price_allocation=offer_data.price_allocation,
            certification_allocation=offer_data.certification_allocation,
            payment_allocation=offer_data.payment_allocation,
            price_priority=offer_data.price_priority,
            certification_priority=offer_data.certification_priority,
            payment_priority=offer_data.payment_priority,
            total_value=total_value,
            my_score=my_score,
            partner_score=partner_score,
            max_point=offer_data.max_point,
            lambda_factor=offer_data.lambda_factor,
            partner_price_priority=partner_price_priority,
            partner_certification_priority=partner_cert_priority,
            partner_payment_priority=partner_payment_priority,
            ai_recommended_price=offer_data.ai_recommended_price,
            ai_recommended_certification=offer_data.ai_recommended_certification,
            ai_recommended_payment=offer_data.ai_recommended_payment,
            used_ai_recommendation=offer_data.used_ai_recommendation,
            ai_recommendation_received=offer_data.ai_recommendation_received
        )

        db.add(db_offer)
        db.commit()
        db.refresh(db_offer)

        # Convert to response model
        response = OfferResponse(
            id=db_offer.id,
            player_id=user.player_id,
            is_received=db_offer.is_received,
            player2_id=db_offer.player2_id,
            player_role=db_offer.player_role,
            player2_role=db_offer.player2_role,
            price_allocation=db_offer.price_allocation,
            certification_allocation=db_offer.certification_allocation,
            payment_allocation=db_offer.payment_allocation,
            price_priority=db_offer.price_priority,
            certification_priority=db_offer.certification_priority,
            payment_priority=db_offer.payment_priority,
            total_value=db_offer.total_value,
            my_score=db_offer.my_score,
            partner_score=db_offer.partner_score,
            max_point=db_offer.max_point,
            lambda_factor=db_offer.lambda_factor,
            partner_price_priority=db_offer.partner_price_priority,
            partner_certification_priority=db_offer.partner_certification_priority,
            partner_payment_priority=db_offer.partner_payment_priority,
            ai_recommended_price=db_offer.ai_recommended_price,
            ai_recommended_certification=db_offer.ai_recommended_certification,
            ai_recommended_payment=db_offer.ai_recommended_payment,
            used_ai_recommendation=db_offer.used_ai_recommendation,
            ai_recommendation_received=db_offer.ai_recommendation_received,
            created_at=db_offer.created_at.isoformat()
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error creating offer: {str(e)}")

@app.get("/offers/{player_id}", response_model=List[OfferResponse],
         summary="Get User Offers",
         description="Retrieve all offers for a specific player")
async def get_user_offers(player_id: str, db: Session = Depends(get_db)):
    """
    Get all offers for a specific player.

    This endpoint retrieves all stored offers (both received and proposed)
    for the specified player ID.

    Args:
        player_id: Player ID to get offers for
        db: Database session dependency

    Returns:
        List[OfferResponse]: List of all offers for the player

    Raises:
        HTTPException: If player is not found or there's a database error
    """
    try:
        # Check if user exists
        user = db.query(User).filter(User.player_id == player_id).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"Player '{player_id}' not found")

        # Get all offers for the user
        offers = db.query(Offer).filter(Offer.player_id == user.player_id).order_by(Offer.created_at.desc()).all()

        # Convert to response models
        response_offers = []
        for offer in offers:
            response_offers.append(OfferResponse(
                id=offer.id,
                player_id=user.player_id,
                is_received=offer.is_received,
                player2_id=offer.player2_id,
                player_role=offer.player_role,
                player2_role=offer.player2_role,
                price_allocation=offer.price_allocation,
                certification_allocation=offer.certification_allocation,
                payment_allocation=offer.payment_allocation,
                price_priority=offer.price_priority,
                certification_priority=offer.certification_priority,
                payment_priority=offer.payment_priority,
                total_value=offer.total_value,
                my_score=offer.my_score,
                partner_score=offer.partner_score,
                max_point=offer.max_point,
                lambda_factor=offer.lambda_factor,
                partner_price_priority=offer.partner_price_priority,
                partner_certification_priority=offer.partner_certification_priority,
                partner_payment_priority=offer.partner_payment_priority,
                ai_recommended_price=offer.ai_recommended_price,
                ai_recommended_certification=offer.ai_recommended_certification,
                ai_recommended_payment=offer.ai_recommended_payment,
                used_ai_recommendation=offer.used_ai_recommendation,
                ai_recommendation_received=offer.ai_recommendation_received,
                created_at=offer.created_at.isoformat()
            ))

        return response_offers

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving offers: {str(e)}")

@app.get("/offers/{player_id}/summary",
         summary="Get User Offer Summary",
         description="Get summarized statistics of player offers")
async def get_user_offer_summary(player_id: str, db: Session = Depends(get_db)):
    """
    Get summary statistics for a player's offers.

    This endpoint provides aggregate information about a player's offers
    including counts of received vs proposed offers and value statistics.

    Args:
        player_id: Player ID to get summary for
        db: Database session dependency

    Returns:
        dict: Summary statistics including counts and averages

    Raises:
        HTTPException: If player is not found or there's a database error
    """
    try:
        # Check if user exists
        user = db.query(User).filter(User.player_id == player_id).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"Player '{player_id}' not found")

        # Get all offers for the user
        offers = db.query(Offer).filter(Offer.player_id == user.player_id).all()

        if not offers:
            return {
                "player_id": player_id,
                "total_offers": 0,
                "received_offers": 0,
                "proposed_offers": 0,
                "average_value": 0.0,
                "max_value": 0.0,
                "min_value": 0.0
            }

        # Calculate statistics
        total_offers = len(offers)
        received_offers = sum(1 for offer in offers if offer.is_received)
        proposed_offers = total_offers - received_offers

        values = [offer.total_value for offer in offers]
        average_value = sum(values) / len(values)
        max_value = max(values)
        min_value = min(values)

        return {
            "player_id": player_id,
            "total_offers": total_offers,
            "received_offers": received_offers,
            "proposed_offers": proposed_offers,
            "average_value": round(average_value, 2),
            "max_value": max_value,
            "min_value": min_value
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving offer summary: {str(e)}")

@app.post("/partner-preferences", response_model=PartnerPreferenceResponse,
          summary="Create Partner Inferred Preferences",
          description="Create or update inferred preferences for a partner")
async def create_partner_preference(pref_data: PartnerPreferenceCreate, db: Session = Depends(get_db)):
    """
    Create or update inferred preferences for a partner.

    If a preference record already exists for the player-partner pair, it will be updated.
    Otherwise, a new record will be created.

    Args:
        pref_data: PartnerPreferenceCreate containing preference details
        db: Database session dependency

    Returns:
        PartnerPreferenceResponse: Created or updated preference record

    Raises:
        HTTPException: If there's an error creating/updating the preference
    """
    try:
        # Get or create user
        user = get_or_create_user(db, pref_data.player_id)

        # Check if preference already exists
        existing_pref = db.query(PartnerInferredPreference).filter(
            PartnerInferredPreference.player_id == pref_data.player_id,
            PartnerInferredPreference.partner_id == pref_data.partner_id
        ).first()

        if existing_pref:
            # Update existing preference
            existing_pref.price_priority = pref_data.price_priority
            existing_pref.certification_priority = pref_data.certification_priority
            existing_pref.payment_priority = pref_data.payment_priority
            existing_pref.updated_at = datetime.utcnow()

            db.commit()
            db.refresh(existing_pref)

            return PartnerPreferenceResponse(
                id=existing_pref.id,
                player_id=existing_pref.player_id,
                partner_id=existing_pref.partner_id,
                price_priority=existing_pref.price_priority,
                certification_priority=existing_pref.certification_priority,
                payment_priority=existing_pref.payment_priority,
                created_at=existing_pref.created_at.isoformat(),
                updated_at=existing_pref.updated_at.isoformat()
            )
        else:
            # Create new preference
            db_pref = PartnerInferredPreference(
                player_id=user.player_id,
                partner_id=pref_data.partner_id,
                price_priority=pref_data.price_priority,
                certification_priority=pref_data.certification_priority,
                payment_priority=pref_data.payment_priority
            )

            db.add(db_pref)
            db.commit()
            db.refresh(db_pref)

            return PartnerPreferenceResponse(
                id=db_pref.id,
                player_id=db_pref.player_id,
                partner_id=db_pref.partner_id,
                price_priority=db_pref.price_priority,
                certification_priority=db_pref.certification_priority,
                payment_priority=db_pref.payment_priority,
                created_at=db_pref.created_at.isoformat(),
                updated_at=db_pref.updated_at.isoformat()
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating/updating partner preference: {str(e)}")

@app.put("/partner-preferences/{player_id}/{partner_id}", response_model=PartnerPreferenceResponse,
         summary="Update Partner Inferred Preferences",
         description="Update existing inferred preferences for a partner")
async def update_partner_preference(
    player_id: str,
    partner_id: str,
    price_priority: Literal["high", "middle", "low"] = None,
    certification_priority: Literal["high", "middle", "low"] = None,
    payment_priority: Literal["high", "middle", "low"] = None,
    db: Session = Depends(get_db)
):
    """
    Update existing inferred preferences for a partner.

    Only provided fields will be updated. If the preference doesn't exist, returns 404.

    Args:
        player_id: Player ID who is inferring partner's preferences
        partner_id: Partner's player ID
        price_priority: Updated priority for price (optional)
        certification_priority: Updated priority for certification (optional)
        payment_priority: Updated priority for payment (optional)
        db: Database session dependency

    Returns:
        PartnerPreferenceResponse: Updated preference record

    Raises:
        HTTPException: If preference not found or there's a database error
    """
    try:
        # Find existing preference
        pref = db.query(PartnerInferredPreference).filter(
            PartnerInferredPreference.player_id == player_id,
            PartnerInferredPreference.partner_id == partner_id
        ).first()

        if not pref:
            raise HTTPException(
                status_code=404,
                detail=f"Partner preference not found for player '{player_id}' and partner '{partner_id}'"
            )

        # Update provided fields
        if price_priority is not None:
            pref.price_priority = price_priority
        if certification_priority is not None:
            pref.certification_priority = certification_priority
        if payment_priority is not None:
            pref.payment_priority = payment_priority

        pref.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(pref)

        return PartnerPreferenceResponse(
            id=pref.id,
            player_id=pref.player_id,
            partner_id=pref.partner_id,
            price_priority=pref.price_priority,
            certification_priority=pref.certification_priority,
            payment_priority=pref.payment_priority,
            created_at=pref.created_at.isoformat(),
            updated_at=pref.updated_at.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating partner preference: {str(e)}")

@app.get("/partner-preferences/{player_id}/{partner_id}", response_model=PartnerPreferenceResponse,
         summary="Get Partner Inferred Preferences",
         description="Retrieve inferred preferences for a specific partner")
async def get_partner_preference(player_id: str, partner_id: str, db: Session = Depends(get_db)):
    """
    Get inferred preferences for a specific partner.

    Args:
        player_id: Player ID who inferred partner's preferences
        partner_id: Partner's player ID
        db: Database session dependency

    Returns:
        PartnerPreferenceResponse: Partner preference record

    Raises:
        HTTPException: If preference not found or there's a database error
    """
    try:
        pref = db.query(PartnerInferredPreference).filter(
            PartnerInferredPreference.player_id == player_id,
            PartnerInferredPreference.partner_id == partner_id
        ).first()

        if not pref:
            raise HTTPException(
                status_code=404,
                detail=f"Partner preference not found for player '{player_id}' and partner '{partner_id}'"
            )

        return PartnerPreferenceResponse(
            id=pref.id,
            player_id=pref.player_id,
            partner_id=pref.partner_id,
            price_priority=pref.price_priority,
            certification_priority=pref.certification_priority,
            payment_priority=pref.payment_priority,
            created_at=pref.created_at.isoformat(),
            updated_at=pref.updated_at.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving partner preference: {str(e)}")

@app.get("/partner-preferences/{player_id}", response_model=List[PartnerPreferenceResponse],
         summary="Get All Partner Inferred Preferences for a Player",
         description="Retrieve all inferred preferences created by a specific player")
async def get_all_partner_preferences(player_id: str, db: Session = Depends(get_db)):
    """
    Get all inferred preferences for partners created by a specific player.

    Args:
        player_id: Player ID who inferred partner preferences
        db: Database session dependency

    Returns:
        List[PartnerPreferenceResponse]: List of all partner preferences

    Raises:
        HTTPException: If there's a database error
    """
    try:
        prefs = db.query(PartnerInferredPreference).filter(
            PartnerInferredPreference.player_id == player_id
        ).order_by(PartnerInferredPreference.updated_at.desc()).all()

        response_prefs = []
        for pref in prefs:
            response_prefs.append(PartnerPreferenceResponse(
                id=pref.id,
                player_id=pref.player_id,
                partner_id=pref.partner_id,
                price_priority=pref.price_priority,
                certification_priority=pref.certification_priority,
                payment_priority=pref.payment_priority,
                created_at=pref.created_at.isoformat(),
                updated_at=pref.updated_at.isoformat()
            ))

        return response_prefs

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving partner preferences: {str(e)}")

@app.get("/offers/{player_id}/partner-behavior/{partner_id}", response_model=PartnerBehaviorResponse,
         summary="Analyze Partner Behavior",
         description="Analyze partner's negotiation behavior by comparing the last 2 received offers")
async def analyze_partner_behavior(player_id: str, partner_id: str, db: Session = Depends(get_db)):
    """
    Analyze partner's negotiation behavior by comparing partner scores from the last 2 received offers.

    This endpoint determines if the partner is being:
    - "generous": partner's score decreased (they're accepting worse deals)
    - "neutral": partner's score unchanged (consistent behavior)
    - "greedy": partner's score increased (they're demanding better deals)

    Args:
        player_id: Player ID who received the offers
        partner_id: Partner's player ID whose behavior to analyze
        db: Database session dependency

    Returns:
        PartnerBehaviorResponse: Analysis of partner's behavior with score delta

    Raises:
        HTTPException: If insufficient data or database error
    """
    try:
        # Query the last 2 received offers from partner_id to player_id
        received_offers = db.query(Offer).filter(
            Offer.player_id == player_id,
            Offer.player2_id == partner_id,
            Offer.is_received == True
        ).order_by(Offer.created_at.desc()).limit(2).all()

        # If fewer than 2 received offers, return 'initial' behavior
        if len(received_offers) < 2:
            offer_count = len(received_offers)
            return PartnerBehaviorResponse(
                player_id=player_id,
                partner_id=partner_id,
                behavior="initial",
                score_delta=None,
                latest_offer_id=received_offers[0].id if offer_count > 0 else None,
                previous_offer_id=None,
                latest_partner_score=received_offers[0].partner_score if offer_count > 0 else None,
                previous_partner_score=None,
                message=f"Initial phase: only {offer_count} received offer(s) from partner. Need at least 2 offers to analyze behavior pattern."
            )

        # Latest offer is first, previous is second (due to desc order)
        latest_offer = received_offers[0]
        previous_offer = received_offers[1]

        # Calculate score delta (latest - previous)
        score_delta = latest_offer.partner_score - previous_offer.partner_score

        # Determine behavior based on delta
        if score_delta < 0:
            behavior = "generous"
            message = f"Partner's score decreased by {abs(score_delta)} points, showing generous behavior"
        elif score_delta > 0:
            behavior = "greedy"
            message = f"Partner's score increased by {score_delta} points, showing greedy behavior"
        else:
            behavior = "neutral"
            message = "Partner's score remained unchanged, showing neutral behavior"

        return PartnerBehaviorResponse(
            player_id=player_id,
            partner_id=partner_id,
            behavior=behavior,
            score_delta=score_delta,
            latest_offer_id=latest_offer.id,
            previous_offer_id=previous_offer.id,
            latest_partner_score=latest_offer.partner_score,
            previous_partner_score=previous_offer.partner_score,  # This is previous
            message=message
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing partner behavior: {str(e)}")

@app.post("/offers/{player_id}/generate-offer/{partner_id}", response_model=GeneratedOfferResponse,
          summary="Generate Best Offer",
          description="Generate the best offer using LP solver, filtering out previously proposed offers")
async def generate_offer(
    player_id: str,
    partner_id: str,
    price_priority: Literal["high", "middle", "low"],
    certification_priority: Literal["high", "middle", "low"],
    payment_priority: Literal["high", "middle", "low"],
    db: Session = Depends(get_db)
):
    """
    Generate the best offer by:
    1. Calculating LP parameters (lambda, max_value) based on partner behavior
    2. Getting partner's inferred priorities from database
    3. Running solve_lp multiple times with max_value range (max_value to max_value-3)
    4. Filtering out previously proposed offers
    5. Selecting the best offer with highest my_score

    Args:
        player_id: Player ID who is making the offer
        partner_id: Partner's player ID
        price_priority: My priority for price (high/middle/low)
        certification_priority: My priority for certification (high/middle/low)
        payment_priority: My priority for payment (high/middle/low)
        db: Database session dependency

    Returns:
        GeneratedOfferResponse: Best offer recommendation with statistics

    Raises:
        HTTPException: If partner preferences not found or no valid offers available
    """
    try:
        # Validate exclusive priorities
        my_priorities_list = [price_priority, certification_priority, payment_priority]
        if sorted(my_priorities_list) != ["high", "low", "middle"]:
            raise HTTPException(
                status_code=422,
                detail="Priorities must be exclusive: one 'high', one 'middle', and one 'low'. "
                       f"Got: price={price_priority}, certification={certification_priority}, payment={payment_priority}"
            )

        my_priorities = {
            "price": price_priority,
            "certification": certification_priority,
            "payment": payment_priority
        }

        # Call the reusable function
        result = generate_best_offer(player_id, partner_id, my_priorities, db)

        return GeneratedOfferResponse(
            player_id=player_id,
            partner_id=partner_id,
            recommended_offer=result["recommended_offer"],
            lambda_factor=result["lambda_factor"],
            max_value=result["max_value"],
            partner_behavior=result["partner_behavior"],
            partner_price_priority=result["partner_price_priority"],
            partner_certification_priority=result["partner_certification_priority"],
            partner_payment_priority=result["partner_payment_priority"],
            total_candidates_generated=result["total_candidates_generated"],
            candidates_after_filtering=result["candidates_after_filtering"],
            previously_proposed_count=result["previously_proposed_count"],
            message=result["message"]
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating offer: {str(e)}")

@app.get("/offers/{player_id}/lp-parameters/{partner_id}", response_model=LPParametersResponse,
         summary="Get LP Solver Parameters",
         description="Calculate optimal lambda and max_value for LP solver based on partner behavior and offer history")
async def get_lp_parameters(player_id: str, partner_id: str, db: Session = Depends(get_db)):
    """
    Calculate LP solver parameters (lambda_factor and max_value) for negotiation.

    This endpoint uses the calculate_lp_parameters function to determine optimal
    negotiation parameters based on partner's historical behavior.

    Args:
        player_id: Player ID who is negotiating
        partner_id: Partner's player ID
        db: Database session dependency

    Returns:
        LPParametersResponse: Calculated lambda_factor and max_value with explanation

    Raises:
        HTTPException: If there's a database error
    """
    try:
        # Call the reusable function
        result = calculate_lp_parameters(player_id, partner_id, db)

        return LPParametersResponse(
            player_id=player_id,
            partner_id=partner_id,
            lambda_factor=result["lambda_factor"],
            max_value=result["max_value"],
            partner_behavior=result["partner_behavior"],
            previous_proposed_offer_id=result["previous_proposed_offer_id"],
            previous_lambda=result["previous_lambda"],
            previous_my_score=result["previous_my_score"],
            message=result["message"]
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error calculating LP parameters: {str(e)}")

@app.get("/", summary="API Status", description="Check if the ASTRA LP Solver API is running")
async def root():
    """
    Root endpoint to check API status.

    Returns:
        dict: Status message indicating the API is running
    """
    return {"message": "ASTRA LP Solver API is running", "docs": "/docs", "redoc": "/redoc"}

@app.get("/db-viewer", response_class=HTMLResponse,
         summary="Database Viewer",
         description="View all users and offers in the database")
async def db_viewer(db: Session = Depends(get_db)):
    """
    Simple HTML viewer for database contents.

    Displays all users and their offers in a readable format.
    """
    users = db.query(User).all()

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ASTRA Database Viewer</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
            }
            .user-card {
                background: white;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .user-header {
                border-bottom: 2px solid #4CAF50;
                padding-bottom: 10px;
                margin-bottom: 15px;
            }
            .user-name {
                color: #4CAF50;
                font-size: 24px;
                font-weight: bold;
            }
            .user-meta {
                color: #666;
                font-size: 14px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }
            th {
                background-color: #4CAF50;
                color: white;
                padding: 12px;
                text-align: left;
            }
            td {
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .received {
                background-color: #e3f2fd;
            }
            .proposed {
                background-color: #fff3e0;
            }
            .no-offers {
                color: #999;
                font-style: italic;
                padding: 20px;
                text-align: center;
            }
            .stats {
                background-color: #f0f0f0;
                padding: 10px;
                border-radius: 4px;
                margin: 10px 0;
            }
            .refresh-btn {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                margin-bottom: 20px;
            }
            .refresh-btn:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <h1>🗄️ ASTRA Database Viewer</h1>
        <button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>
    """

    if not users:
        html_content += '<div class="no-offers">No users found in database</div>'
    else:
        for user in users:
            offers = db.query(Offer).filter(Offer.user_id == user.id).order_by(Offer.created_at.desc()).all()

            received_count = sum(1 for o in offers if o.is_received)
            proposed_count = len(offers) - received_count

            html_content += f'''
            <div class="user-card">
                <div class="user-header">
                    <div class="user-name">👤 {user.player_id}</div>
                    <div class="user-meta">Created: {user.created_at.strftime("%Y-%m-%d %H:%M:%S")}</div>
                </div>

                <div class="stats">
                    <strong>Total Offers:</strong> {len(offers)}
                    | <strong>Received:</strong> {received_count}
                    | <strong>Proposed:</strong> {proposed_count}
                </div>
            '''

            if offers:
                html_content += '''
                <table>
                    <tr>
                        <th>ID</th>
                        <th>Type</th>
                        <th>Partner</th>
                        <th>Roles</th>
                        <th>Allocation</th>
                        <th>My Priority</th>
                        <th>My Score</th>
                        <th>Partner Score</th>
                        <th>Total</th>
                        <th>Max Point</th>
                        <th>Lambda</th>
                        <th>Created At</th>
                    </tr>
                '''

                for offer in offers:
                    row_class = "received" if offer.is_received else "proposed"
                    offer_type = "📥 Received" if offer.is_received else "📤 Proposed"
                    max_point_display = offer.max_point if offer.max_point else "-"
                    lambda_display = f"{offer.lambda_factor:.2f}" if offer.lambda_factor is not None else "-"

                    html_content += f'''
                    <tr class="{row_class}">
                        <td>{offer.id}</td>
                        <td>{offer_type}</td>
                        <td>{offer.player2_id}</td>
                        <td>
                            P: {offer.player_role}<br/>
                            P2: {offer.player2_role}
                        </td>
                        <td>
                            P: {offer.price_allocation}<br/>
                            C: {offer.certification_allocation}<br/>
                            Pm: {offer.payment_allocation}
                        </td>
                        <td>
                            P: {offer.price_priority}<br/>
                            C: {offer.certification_priority}<br/>
                            Pm: {offer.payment_priority}
                        </td>
                        <td><strong>{offer.my_score}</strong></td>
                        <td><strong>{offer.partner_score}</strong></td>
                        <td>{offer.total_value}</td>
                        <td>{max_point_display}</td>
                        <td>{lambda_display}</td>
                        <td>{offer.created_at.strftime("%Y-%m-%d %H:%M:%S")}</td>
                    </tr>
                    '''

                html_content += '</table>'
            else:
                html_content += '<div class="no-offers">No offers yet</div>'

            html_content += '</div>'

    html_content += '''
    </body>
    </html>
    '''

    return html_content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

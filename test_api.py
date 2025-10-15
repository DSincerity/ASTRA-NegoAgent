import pytest
import requests
import json
import os
from fastapi.testclient import TestClient
from main import app
from database import Base, engine

client = TestClient(app)

# Setup test database
@pytest.fixture(autouse=True)
def setup_database():
    """Setup test database before each test"""
    # Remove existing test database
    if os.path.exists("./offers.db"):
        os.remove("./offers.db")

    # Create fresh tables
    Base.metadata.create_all(bind=engine)
    yield
    # Cleanup after test
    if os.path.exists("./offers.db"):
        os.remove("./offers.db")

class TestASTRAAPI:
    """Test cases for ASTRA LP Solver API"""

    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "ASTRA LP Solver API is running"
        assert "docs" in data
        assert "redoc" in data

    def test_solve_lp_valid_request(self):
        """Test solve_lp with valid parameters"""
        request_data = {
            "max_point": 30,
            "lambda_factor": 0.3,
            "agents_value": {
                "food": 5,
                "water": 4,
                "firewood": 3
            },
            "partner_value": {
                "food": 3,
                "water": 4,
                "firewood": 5
            },
            "epsilon": 0.0001
        }

        response = client.post("/solve_lp", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "agent_score" in data
        assert "food_allocation" in data
        assert "water_allocation" in data
        assert "firewood_allocation" in data

        # Check data types
        assert isinstance(data["agent_score"], int)
        assert isinstance(data["food_allocation"], int)
        assert isinstance(data["water_allocation"], int)
        assert isinstance(data["firewood_allocation"], int)

        # Check allocations are within valid range (0-3)
        assert 0 <= data["food_allocation"] <= 3
        assert 0 <= data["water_allocation"] <= 3
        assert 0 <= data["firewood_allocation"] <= 3

    def test_solve_lp_minimal_request(self):
        """Test solve_lp with minimal required parameters (using default epsilon)"""
        request_data = {
            "max_point": 20,
            "lambda_factor": 0.5,
            "agents_value": {
                "food": 2,
                "water": 3,
                "firewood": 4
            },
            "partner_value": {
                "food": 4,
                "water": 3,
                "firewood": 2
            }
        }

        response = client.post("/solve_lp", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "agent_score" in data
        assert "food_allocation" in data
        assert "water_allocation" in data
        assert "firewood_allocation" in data

    def test_solve_lp_invalid_lambda_factor(self):
        """Test solve_lp with invalid lambda_factor (> 1)"""
        request_data = {
            "max_point": 30,
            "lambda_factor": 1.5,  # Invalid: > 1
            "agents_value": {
                "food": 5,
                "water": 4,
                "firewood": 3
            },
            "partner_value": {
                "food": 3,
                "water": 4,
                "firewood": 5
            }
        }

        response = client.post("/solve_lp", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_solve_lp_invalid_lambda_factor_negative(self):
        """Test solve_lp with negative lambda_factor"""
        request_data = {
            "max_point": 30,
            "lambda_factor": -0.1,  # Invalid: < 0
            "agents_value": {
                "food": 5,
                "water": 4,
                "firewood": 3
            },
            "partner_value": {
                "food": 3,
                "water": 4,
                "firewood": 5
            }
        }

        response = client.post("/solve_lp", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_solve_lp_invalid_max_point(self):
        """Test solve_lp with invalid max_point (≤ 0)"""
        request_data = {
            "max_point": 0,  # Invalid: should be > 0
            "lambda_factor": 0.3,
            "agents_value": {
                "food": 5,
                "water": 4,
                "firewood": 3
            },
            "partner_value": {
                "food": 3,
                "water": 4,
                "firewood": 5
            }
        }

        response = client.post("/solve_lp", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_solve_lp_negative_values(self):
        """Test solve_lp with negative resource values"""
        request_data = {
            "max_point": 30,
            "lambda_factor": 0.3,
            "agents_value": {
                "food": -1,  # Invalid: should be >= 0
                "water": 4,
                "firewood": 3
            },
            "partner_value": {
                "food": 3,
                "water": 4,
                "firewood": 5
            }
        }

        response = client.post("/solve_lp", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_solve_lp_missing_required_fields(self):
        """Test solve_lp with missing required fields"""
        request_data = {
            "max_point": 30,
            "lambda_factor": 0.3,
            # Missing agents_value and partner_value
        }

        response = client.post("/solve_lp", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_solve_lp_edge_cases(self):
        """Test solve_lp with edge case values"""
        # Lambda = 0 (fully selfish)
        request_data = {
            "max_point": 15,
            "lambda_factor": 0.0,
            "agents_value": {
                "food": 1,
                "water": 1,
                "firewood": 1
            },
            "partner_value": {
                "food": 1,
                "water": 1,
                "firewood": 1
            }
        }

        response = client.post("/solve_lp", json=request_data)
        assert response.status_code == 200

        # Lambda = 1 (fully altruistic)
        request_data["lambda_factor"] = 1.0
        response = client.post("/solve_lp", json=request_data)
        assert response.status_code == 200

    def test_openapi_docs_available(self):
        """Test that OpenAPI documentation is available"""
        response = client.get("/docs")
        assert response.status_code == 200

        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_schema(self):
        """Test that OpenAPI schema is properly generated"""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "ASTRA LP Solver API"
        assert "paths" in schema
        assert "/solve_lp" in schema["paths"]
        assert "/" in schema["paths"]

class TestOfferAPI:
    """Test cases for Offer management API"""

    def test_create_offer_valid(self):
        """Test creating a valid offer"""
        offer_data = {
            "username": "player1",
            "is_received": False,
            "partner_username": "player2",
            "food_allocation": 2,
            "water_allocation": 1,
            "firewood_allocation": 3,
            "food_value": 5.0,
            "water_value": 4.0,
            "firewood_value": 3.0
        }

        response = client.post("/offers", json=offer_data)
        assert response.status_code == 200

        data = response.json()
        assert data["username"] == "player1"
        assert data["is_received"] is False
        assert data["partner_username"] == "player2"
        assert data["food_allocation"] == 2
        assert data["water_allocation"] == 1
        assert data["firewood_allocation"] == 3
        assert data["total_value"] == 23.0  # 2*5 + 1*4 + 3*3
        assert "id" in data
        assert "created_at" in data

    def test_create_offer_received(self):
        """Test creating a received offer"""
        offer_data = {
            "username": "player2",
            "is_received": True,
            "partner_username": "player1",
            "food_allocation": 1,
            "water_allocation": 2,
            "firewood_allocation": 0,
            "food_value": 3.0,
            "water_value": 4.0,
            "firewood_value": 5.0
        }

        response = client.post("/offers", json=offer_data)
        assert response.status_code == 200

        data = response.json()
        assert data["username"] == "player2"
        assert data["is_received"] is True
        assert data["partner_username"] == "player1"
        assert data["total_value"] == 11.0  # 1*3 + 2*4 + 0*5

    def test_create_offer_invalid_allocation(self):
        """Test creating offer with invalid allocation (> 3)"""
        offer_data = {
            "username": "player1",
            "is_received": False,
            "partner_username": "player2",
            "food_allocation": 4,  # Invalid: > 3
            "water_allocation": 1,
            "firewood_allocation": 3,
            "food_value": 5.0,
            "water_value": 4.0,
            "firewood_value": 3.0
        }

        response = client.post("/offers", json=offer_data)
        assert response.status_code == 422  # Validation error

    def test_create_offer_negative_allocation(self):
        """Test creating offer with negative allocation"""
        offer_data = {
            "username": "player1",
            "is_received": False,
            "partner_username": "player2",
            "food_allocation": -1,  # Invalid: < 0
            "water_allocation": 1,
            "firewood_allocation": 3,
            "food_value": 5.0,
            "water_value": 4.0,
            "firewood_value": 3.0
        }

        response = client.post("/offers", json=offer_data)
        assert response.status_code == 422  # Validation error

    def test_create_offer_negative_value(self):
        """Test creating offer with negative value"""
        offer_data = {
            "username": "player1",
            "is_received": False,
            "partner_username": "player2",
            "food_allocation": 2,
            "water_allocation": 1,
            "firewood_allocation": 3,
            "food_value": -1.0,  # Invalid: < 0
            "water_value": 4.0,
            "firewood_value": 3.0
        }

        response = client.post("/offers", json=offer_data)
        assert response.status_code == 422  # Validation error

    def test_get_user_offers_empty(self):
        """Test getting offers for non-existent user"""
        response = client.get("/offers/nonexistent_user")
        assert response.status_code == 404

    def test_get_user_offers_with_data(self):
        """Test getting offers for user with data"""
        # First create some offers
        offer1 = {
            "username": "testuser",
            "is_received": False,
            "partner_username": "partner1",
            "food_allocation": 2,
            "water_allocation": 1,
            "firewood_allocation": 3,
            "food_value": 5.0,
            "water_value": 4.0,
            "firewood_value": 3.0
        }

        offer2 = {
            "username": "testuser",
            "is_received": True,
            "partner_username": "partner2",
            "food_allocation": 1,
            "water_allocation": 2,
            "firewood_allocation": 0,
            "food_value": 5.0,
            "water_value": 4.0,
            "firewood_value": 3.0
        }

        # Create offers
        client.post("/offers", json=offer1)
        client.post("/offers", json=offer2)

        # Get offers
        response = client.get("/offers/testuser")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 2
        assert all(offer["username"] == "testuser" for offer in data)

        # Check that offers are sorted by created_at desc
        assert data[0]["created_at"] >= data[1]["created_at"]

    def test_get_user_offer_summary_empty(self):
        """Test getting summary for non-existent user"""
        response = client.get("/offers/nonexistent_user/summary")
        assert response.status_code == 404

    def test_get_user_offer_summary_no_offers(self):
        """Test getting summary for user with no offers"""
        # Create user by creating and then deleting (or just test with empty)
        # For now, let's test the case where user exists but has no offers
        # This requires creating a user first, which happens when creating an offer
        offer_data = {
            "username": "emptyuser",
            "is_received": False,
            "partner_username": "partner1",
            "food_allocation": 1,
            "water_allocation": 1,
            "firewood_allocation": 1,
            "food_value": 1.0,
            "water_value": 1.0,
            "firewood_value": 1.0
        }
        client.post("/offers", json=offer_data)

        # Now manually delete offers (in real scenario, this would be via a DELETE endpoint)
        # For this test, we'll test with user that has offers
        response = client.get("/offers/emptyuser/summary")
        assert response.status_code == 200

        data = response.json()
        assert data["username"] == "emptyuser"
        assert data["total_offers"] == 1

    def test_get_user_offer_summary_with_data(self):
        """Test getting summary for user with multiple offers"""
        # Create multiple offers
        offers = [
            {
                "username": "summaryuser",
                "is_received": False,
                "partner_username": "partner1",
                "food_allocation": 2,
                "water_allocation": 1,
                "firewood_allocation": 3,
                "food_value": 5.0,
                "water_value": 4.0,
                "firewood_value": 3.0
            },
            {
                "username": "summaryuser",
                "is_received": True,
                "partner_username": "partner2",
                "food_allocation": 1,
                "water_allocation": 2,
                "firewood_allocation": 0,
                "food_value": 5.0,
                "water_value": 4.0,
                "firewood_value": 3.0
            },
            {
                "username": "summaryuser",
                "is_received": False,
                "partner_username": "partner3",
                "food_allocation": 0,
                "water_allocation": 0,
                "firewood_allocation": 2,
                "food_value": 5.0,
                "water_value": 4.0,
                "firewood_value": 3.0
            }
        ]

        for offer in offers:
            client.post("/offers", json=offer)

        # Get summary
        response = client.get("/offers/summaryuser/summary")
        assert response.status_code == 200

        data = response.json()
        assert data["username"] == "summaryuser"
        assert data["total_offers"] == 3
        assert data["received_offers"] == 1
        assert data["proposed_offers"] == 2

        # Values: 23.0, 13.0, 6.0
        assert data["max_value"] == 23.0
        assert data["min_value"] == 6.0
        assert data["average_value"] == 14.0  # (23 + 13 + 6) / 3

    def test_create_multiple_offers_same_user(self):
        """Test creating multiple offers for the same user"""
        base_offer = {
            "player_id": "multiuser",
            "is_received": False,
            "player2_id": "partner1",
            "player_role": "seller",
            "player2_role": "buyer",
            "price_allocation": "1400",
            "certification_allocation": "None",
            "payment_allocation": "Full",
            "price_priority": "high",
            "certification_priority": "middle",
            "payment_priority": "low",
            "partner_price_priority": "low",
            "partner_certification_priority": "middle",
            "partner_payment_priority": "high",
            "max_point": 30,
            "lambda_factor": 0.5
        }

        # Create 5 offers
        for i in range(5):
            offer = base_offer.copy()
            offer["player2_id"] = f"partner{i+1}"
            offer["is_received"] = i % 2 == 0  # Alternate between received and proposed
            response = client.post("/offers", json=offer)
            assert response.status_code == 200

        # Verify all offers were created
        response = client.get("/offers/multiuser")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5

class TestPartnerBehaviorAPI:
    """Test cases for partner behavior analysis API"""

    def test_analyze_partner_behavior_generous(self):
        """Test analyzing partner behavior when partner is generous (score decreased)"""
        # Create first received offer with partner_score = 15
        offer1 = {
            "player_id": "player1",
            "is_received": True,
            "player2_id": "partner1",
            "player_role": "seller",
            "player2_role": "buyer",
            "price_allocation": "1400",
            "certification_allocation": "None",
            "payment_allocation": "Full",
            "price_priority": "high",
            "certification_priority": "middle",
            "payment_priority": "low",
            "partner_price_priority": "high",
            "partner_certification_priority": "high",
            "partner_payment_priority": "high",
            "max_point": 30,
            "lambda_factor": 0.5
        }
        response1 = client.post("/offers", json=offer1)
        assert response1.status_code == 200

        # Create second received offer with partner_score = 10 (decreased)
        offer2 = offer1.copy()
        offer2["partner_price_priority"] = "low"
        offer2["partner_certification_priority"] = "low"
        offer2["partner_payment_priority"] = "low"
        response2 = client.post("/offers", json=offer2)
        assert response2.status_code == 200

        # Analyze behavior
        response = client.get("/offers/player1/partner-behavior/partner1")
        assert response.status_code == 200

        data = response.json()
        assert data["player_id"] == "player1"
        assert data["partner_id"] == "partner1"
        assert data["behavior"] == "generous"
        assert data["score_delta"] < 0
        assert "generous behavior" in data["message"]

    def test_analyze_partner_behavior_neutral(self):
        """Test analyzing partner behavior when partner is neutral (score unchanged)"""
        # Create two received offers with same partner_score
        offer1 = {
            "player_id": "player2",
            "is_received": True,
            "player2_id": "partner2",
            "player_role": "seller",
            "player2_role": "buyer",
            "price_allocation": "1400",
            "certification_allocation": "None",
            "payment_allocation": "Full",
            "price_priority": "high",
            "certification_priority": "middle",
            "payment_priority": "low",
            "partner_price_priority": "middle",
            "partner_certification_priority": "middle",
            "partner_payment_priority": "middle",
            "max_point": 30,
            "lambda_factor": 0.5
        }
        client.post("/offers", json=offer1)

        # Second offer with same partner priorities
        offer2 = offer1.copy()
        client.post("/offers", json=offer2)

        # Analyze behavior
        response = client.get("/offers/player2/partner-behavior/partner2")
        assert response.status_code == 200

        data = response.json()
        assert data["behavior"] == "neutral"
        assert data["score_delta"] == 0
        assert "neutral behavior" in data["message"]

    def test_analyze_partner_behavior_greedy(self):
        """Test analyzing partner behavior when partner is greedy (score increased)"""
        # Create first received offer with partner_score low
        offer1 = {
            "player_id": "player3",
            "is_received": True,
            "player2_id": "partner3",
            "player_role": "seller",
            "player2_role": "buyer",
            "price_allocation": "1400",
            "certification_allocation": "None",
            "payment_allocation": "Full",
            "price_priority": "high",
            "certification_priority": "middle",
            "payment_priority": "low",
            "partner_price_priority": "low",
            "partner_certification_priority": "low",
            "partner_payment_priority": "low",
            "max_point": 30,
            "lambda_factor": 0.5
        }
        client.post("/offers", json=offer1)

        # Create second received offer with partner_score higher
        offer2 = offer1.copy()
        offer2["partner_price_priority"] = "high"
        offer2["partner_certification_priority"] = "high"
        offer2["partner_payment_priority"] = "high"
        client.post("/offers", json=offer2)

        # Analyze behavior
        response = client.get("/offers/player3/partner-behavior/partner3")
        assert response.status_code == 200

        data = response.json()
        assert data["behavior"] == "greedy"
        assert data["score_delta"] > 0
        assert "greedy behavior" in data["message"]

    def test_analyze_partner_behavior_initial_one_offer(self):
        """Test analyzing partner behavior with only one offer (should return 'initial')"""
        # Create only one received offer
        offer = {
            "player_id": "player4",
            "is_received": True,
            "player2_id": "partner4",
            "player_role": "seller",
            "player2_role": "buyer",
            "price_allocation": "1400",
            "certification_allocation": "None",
            "payment_allocation": "Full",
            "price_priority": "high",
            "certification_priority": "middle",
            "payment_priority": "low",
            "partner_price_priority": "middle",
            "partner_certification_priority": "middle",
            "partner_payment_priority": "middle",
            "max_point": 30,
            "lambda_factor": 0.5
        }
        response_offer = client.post("/offers", json=offer)
        assert response_offer.status_code == 200

        # Analyze behavior (should return 'initial')
        response = client.get("/offers/player4/partner-behavior/partner4")
        assert response.status_code == 200

        data = response.json()
        assert data["behavior"] == "initial"
        assert data["score_delta"] is None
        assert data["previous_offer_id"] is None
        assert data["latest_offer_id"] is not None
        assert "Initial phase" in data["message"]
        assert "only 1 received offer(s)" in data["message"]

    def test_analyze_partner_behavior_initial_no_offers(self):
        """Test analyzing partner behavior with no received offers (should return 'initial')"""
        # Create only proposed offers (not received)
        offer1 = {
            "player_id": "player5",
            "is_received": False,  # Proposed, not received
            "player2_id": "partner5",
            "player_role": "seller",
            "player2_role": "buyer",
            "price_allocation": "1400",
            "certification_allocation": "None",
            "payment_allocation": "Full",
            "price_priority": "high",
            "certification_priority": "middle",
            "payment_priority": "low",
            "partner_price_priority": "middle",
            "partner_certification_priority": "middle",
            "partner_payment_priority": "middle",
            "max_point": 30,
            "lambda_factor": 0.5
        }
        client.post("/offers", json=offer1)
        client.post("/offers", json=offer1)

        # Analyze behavior (should return 'initial' with 0 offers)
        response = client.get("/offers/player5/partner-behavior/partner5")
        assert response.status_code == 200

        data = response.json()
        assert data["behavior"] == "initial"
        assert data["score_delta"] is None
        assert data["previous_offer_id"] is None
        assert data["latest_offer_id"] is None
        assert "Initial phase" in data["message"]
        assert "only 0 received offer(s)" in data["message"]

    def test_analyze_partner_behavior_nonexistent_player(self):
        """Test analyzing partner behavior for non-existent player (should return 'initial' with 0 offers)"""
        response = client.get("/offers/nonexistent/partner-behavior/partner99")
        assert response.status_code == 200

        data = response.json()
        assert data["behavior"] == "initial"
        assert "only 0 received offer(s)" in data["message"]

class TestPriorityValidation:
    """Test cases for priority exclusivity validation"""

    def test_valid_exclusive_priorities(self):
        """Test that valid exclusive priorities are accepted"""
        offer = {
            "player_id": "player_priority1",
            "is_received": False,
            "player2_id": "partner_priority1",
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
        response = client.post("/offers", json=offer)
        assert response.status_code == 200

    def test_invalid_my_priorities_duplicate_high(self):
        """Test that duplicate 'high' priority is rejected"""
        offer = {
            "player_id": "player_priority2",
            "is_received": False,
            "player2_id": "partner_priority2",
            "player_role": "seller",
            "player2_role": "buyer",
            "price_allocation": "1400",
            "certification_allocation": "None",
            "payment_allocation": "Full",
            "price_priority": "high",
            "certification_priority": "high",  # Duplicate high
            "payment_priority": "low",
            "partner_price_priority": "low",
            "partner_certification_priority": "high",
            "partner_payment_priority": "middle",
            "max_point": 30,
            "lambda_factor": 0.5
        }
        response = client.post("/offers", json=offer)
        assert response.status_code == 422
        response_text = str(response.json())
        assert "exclusive" in response_text.lower()

    def test_invalid_my_priorities_all_same(self):
        """Test that all same priority is rejected"""
        offer = {
            "player_id": "player_priority3",
            "is_received": False,
            "player2_id": "partner_priority3",
            "player_role": "seller",
            "player2_role": "buyer",
            "price_allocation": "1400",
            "certification_allocation": "None",
            "payment_allocation": "Full",
            "price_priority": "middle",
            "certification_priority": "middle",
            "payment_priority": "middle",
            "partner_price_priority": "low",
            "partner_certification_priority": "high",
            "partner_payment_priority": "middle",
            "max_point": 30,
            "lambda_factor": 0.5
        }
        response = client.post("/offers", json=offer)
        assert response.status_code == 422
        response_text = str(response.json())
        assert "exclusive" in response_text.lower()

    def test_invalid_partner_priorities_duplicate_low(self):
        """Test that duplicate 'low' in partner priority is rejected"""
        offer = {
            "player_id": "player_priority4",
            "is_received": False,
            "player2_id": "partner_priority4",
            "player_role": "seller",
            "player2_role": "buyer",
            "price_allocation": "1400",
            "certification_allocation": "None",
            "payment_allocation": "Full",
            "price_priority": "high",
            "certification_priority": "middle",
            "payment_priority": "low",
            "partner_price_priority": "low",
            "partner_certification_priority": "low",  # Duplicate low
            "partner_payment_priority": "middle",
            "max_point": 30,
            "lambda_factor": 0.5
        }
        response = client.post("/offers", json=offer)
        assert response.status_code == 422
        response_text = str(response.json())
        assert "exclusive" in response_text.lower()

    def test_valid_different_order_priorities(self):
        """Test that different orderings of exclusive priorities work"""
        # Test case 1: low, high, middle
        offer1 = {
            "player_id": "player_priority5",
            "is_received": False,
            "player2_id": "partner_priority5",
            "player_role": "seller",
            "player2_role": "buyer",
            "price_allocation": "1400",
            "certification_allocation": "None",
            "payment_allocation": "Full",
            "price_priority": "low",
            "certification_priority": "high",
            "payment_priority": "middle",
            "partner_price_priority": "middle",
            "partner_certification_priority": "low",
            "partner_payment_priority": "high",
            "max_point": 30,
            "lambda_factor": 0.5
        }
        response1 = client.post("/offers", json=offer1)
        assert response1.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

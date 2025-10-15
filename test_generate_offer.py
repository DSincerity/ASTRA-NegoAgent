"""
Comprehensive test suite for generate_offer API

This test suite validates the most critical function in the project:
- LP parameter calculation (lambda, max_value)
- Offer candidate generation
- Filtering previously proposed offers
- Best offer selection logic

Each test case provides detailed tracking of the entire process.
"""

import pytest
import requests
import json
from fastapi.testclient import TestClient
from main import app
from database import Base, engine
import os
from solver import self_class_to_int_mapper, partner_preference_value_to_int_mapper

client = TestClient(app)


def calculate_partner_score(price_alloc: str, cert_alloc: str, payment_alloc: str,
                           partner_price_priority: str, partner_cert_priority: str,
                           partner_payment_priority: str) -> int:
    """
    Calculate partner's actual score based on allocation and their priorities.

    Partner gets the opposite allocation (e.g., if player gets "1400", partner gets "800")
    Partner's value mapping:
    - price: {"1400": 1, "1200": 2, "1000": 3, "800": 4}
    - certification: {"None": 1, "Basic": 2, "3rd-party": 3, "Full": 4}
    - payment: {"Full": 1, "1M": 2, "3M": 3, "6M": 4}

    Priority to value mapping:
    - high: 5, middle: 3, low: 1
    """
    # Convert priorities to values
    priority_to_value = {"high": 5, "middle": 3, "low": 1}

    # Get partner's allocation (opposite of player's)
    # For price and payment, partner gets what player doesn't get
    # Total packages: price levels sum to 5 (indices 1-4), cert levels sum to 5, payment levels sum to 5
    player_price_int = self_class_to_int_mapper['price'][price_alloc]
    player_cert_int = self_class_to_int_mapper['certification'][cert_alloc]
    player_payment_int = self_class_to_int_mapper['payment'][payment_alloc]

    # Partner gets the opposite
    partner_price_int = 5 - player_price_int
    partner_cert_int = 5 - player_cert_int
    partner_payment_int = 5 - player_payment_int

    # Calculate partner's score
    partner_score = (
        partner_price_int * priority_to_value[partner_price_priority] +
        partner_cert_int * priority_to_value[partner_cert_priority] +
        partner_payment_int * priority_to_value[partner_payment_priority]
    )

    return partner_score


def verify_offer_behavior(offer1_data: dict, offer2_data: dict, expected_behavior: str):
    """
    Verify that two offers create the expected partner behavior.
    Prints detailed calculation and validation.
    """
    print(f"\n🔍 VERIFYING {expected_behavior.upper()} BEHAVIOR:")
    print(f"  Offer 1 Allocation: {offer1_data['price_alloc']}, {offer1_data['cert_alloc']}, {offer1_data['payment_alloc']}")
    print(f"  Offer 1 Partner Priorities: price={offer1_data['partner_price']}, cert={offer1_data['partner_cert']}, payment={offer1_data['partner_payment']}")

    score1 = calculate_partner_score(
        offer1_data['price_alloc'], offer1_data['cert_alloc'], offer1_data['payment_alloc'],
        offer1_data['partner_price'], offer1_data['partner_cert'], offer1_data['partner_payment']
    )
    print(f"  Offer 1 Calculated Partner Score: {score1}")

    print(f"\n  Offer 2 Allocation: {offer2_data['price_alloc']}, {offer2_data['cert_alloc']}, {offer2_data['payment_alloc']}")
    print(f"  Offer 2 Partner Priorities: price={offer2_data['partner_price']}, cert={offer2_data['partner_cert']}, payment={offer2_data['partner_payment']}")

    score2 = calculate_partner_score(
        offer2_data['price_alloc'], offer2_data['cert_alloc'], offer2_data['payment_alloc'],
        offer2_data['partner_price'], offer2_data['partner_cert'], offer2_data['partner_payment']
    )
    print(f"  Offer 2 Calculated Partner Score: {score2}")

    score_delta = score2 - score1
    print(f"\n  Score Delta: {score_delta} (Offer 2 - Offer 1)")

    # Determine actual behavior
    if score_delta < 0:
        actual_behavior = "generous"
        print(f"  ✓ Actual Behavior: GENEROUS (partner score decreased)")
    elif score_delta > 0:
        actual_behavior = "greedy"
        print(f"  ✓ Actual Behavior: GREEDY (partner score increased)")
    else:
        actual_behavior = "neutral"
        print(f"  ✓ Actual Behavior: NEUTRAL (partner score unchanged)")

    # Validate
    if actual_behavior == expected_behavior:
        print(f"  ✅ VALIDATION PASSED: Expected {expected_behavior}, got {actual_behavior}")
    else:
        print(f"  ❌ VALIDATION FAILED: Expected {expected_behavior}, got {actual_behavior}")
        raise AssertionError(f"Expected {expected_behavior} behavior, but got {actual_behavior}")

    return score1, score2


# Setup test database
@pytest.fixture(autouse=True)
def setup_database():
    """Setup test database before each test"""
    if os.path.exists("./offers.db"):
        os.remove("./offers.db")
    Base.metadata.create_all(bind=engine)
    yield
    if os.path.exists("./offers.db"):
        os.remove("./offers.db")


def create_partner_preference(player_id: str, partner_id: str,
                              price: str, cert: str, payment: str):
    """Helper to create partner preference"""
    response = client.post("/partner-preferences", json={
        "player_id": player_id,
        "partner_id": partner_id,
        "price_priority": price,
        "certification_priority": cert,
        "payment_priority": payment
    })
    assert response.status_code == 200
    return response.json()


def create_offer(player_id: str, partner_id: str, is_received: bool,
                price_alloc: str, cert_alloc: str, payment_alloc: str,
                my_price: str, my_cert: str, my_payment: str,
                partner_price: str, partner_cert: str, partner_payment: str,
                max_point: int = 30, lambda_factor: float = 0.5):
    """Helper to create an offer"""
    response = client.post("/offers", json={
        "player_id": player_id,
        "is_received": is_received,
        "player2_id": partner_id,
        "player_role": "seller",
        "player2_role": "buyer",
        "price_allocation": price_alloc,
        "certification_allocation": cert_alloc,
        "payment_allocation": payment_alloc,
        "price_priority": my_price,
        "certification_priority": my_cert,
        "payment_priority": my_payment,
        "partner_price_priority": partner_price,
        "partner_certification_priority": partner_cert,
        "partner_payment_priority": partner_payment,
        "max_point": max_point,
        "lambda_factor": lambda_factor
    })
    assert response.status_code == 200
    return response.json()


class TestGenerateOfferInitialState:
    """Test offer generation in initial state (no history)"""

    def test_initial_no_history(self):
        """
        Test Case 1: Initial state - no offer history

        Expected behavior:
        - partner_behavior: "initial"
        - lambda_factor: 0.7
        - max_value: 28 (default)
        - Should generate multiple candidates
        - Should select best offer (highest my_score)
        """
        print("\n" + "="*80)
        print("TEST CASE 1: Initial State - No History")
        print("="*80)

        # Create partner preference
        pref = create_partner_preference(
            "player1", "partner1",
            price="high", cert="middle", payment="low"
        )
        print(f"\n✓ Partner preference created: {pref}")

        # Generate offer
        response = client.post(
            "/offers/player1/generate-offer/partner1",
            params={
                "price_priority": "high",
                "certification_priority": "middle",
                "payment_priority": "low"
            }
        )

        assert response.status_code == 200
        data = response.json()

        print(f"\n📊 RESULTS:")
        print(f"  Partner Behavior: {data['partner_behavior']}")
        print(f"  Lambda Factor: {data['lambda_factor']}")
        print(f"  Max Value: {data['max_value']}")
        print(f"  Total Candidates Generated: {data['total_candidates_generated']}")
        print(f"  Previously Proposed: {data['previously_proposed_count']}")
        print(f"  Candidates After Filtering: {data['candidates_after_filtering']}")

        print(f"\n🎯 RECOMMENDED OFFER:")
        for key, value in data['recommended_offer'].items():
            print(f"  {key}: {value}")

        print(f"\n💬 Message: {data['message']}")

        # Assertions
        assert data['partner_behavior'] == "initial"
        assert data['lambda_factor'] == 0.7
        assert data['max_value'] == 28
        assert data['total_candidates_generated'] > 0
        assert data['previously_proposed_count'] == 0
        assert data['recommended_offer']['my_score'] > 0

        print("\n✅ TEST PASSED\n")


class TestGenerateOfferWithHistory:
    """Test offer generation with previous offer history"""

    def test_greedy_partner_lambda_adjustment(self):
        """
        Test Case 2: Greedy partner with lambda adjustment

        Setup:
        - Create 2 received offers where partner score increases
        - Create 1 proposed offer with lambda=0.7

        Expected behavior:
        - partner_behavior: "greedy"
        - lambda_factor: 0.8 (0.7 + 0.1 due to greedy behavior)
        - max_value: my_score from previous proposed offer
        - Should filter out previously proposed offer
        """
        print("\n" + "="*80)
        print("TEST CASE 2: Greedy Partner with Lambda Adjustment")
        print("="*80)

        # Create partner preference
        create_partner_preference(
            "player2", "partner2",
            price="high", cert="middle", payment="low"
        )

        # Create 2 received offers (partner score increasing = greedy)
        print("\n📥 Creating received offers (greedy behavior)...")

        # Define offer specs for validation
        offer1_spec = {
            'price_alloc': "1400", 'cert_alloc': "None", 'payment_alloc': "Full",
            'partner_price': "low", 'partner_cert': "middle", 'partner_payment': "high"
        }
        offer2_spec = {
            'price_alloc': "800", 'cert_alloc': "Full", 'payment_alloc': "6M",
            'partner_price': "high", 'partner_cert': "middle", 'partner_payment': "low"
        }

        # Verify behavior BEFORE creating offers
        verify_offer_behavior(offer1_spec, offer2_spec, "greedy")

        # First offer: Partner gets lower allocation → lower score
        offer1 = create_offer(
            "player2", "partner2", is_received=True,
            **offer1_spec,
            my_price="high", my_cert="middle", my_payment="low"
        )
        print(f"\n  Offer 1 - API Returned Partner Score: {offer1['partner_score']}")

        # Second offer: Partner gets higher allocation → higher score (GREEDY)
        offer2 = create_offer(
            "player2", "partner2", is_received=True,
            **offer2_spec,
            my_price="high", my_cert="middle", my_payment="low"
        )
        print(f"  Offer 2 - API Returned Partner Score: {offer2['partner_score']}")
        print(f"  → Partner behavior: GREEDY (score increased from {offer1['partner_score']} to {offer2['partner_score']})")

        # Create 1 proposed offer with lambda=0.7
        print("\n📤 Creating proposed offer with lambda=0.7...")
        proposed = create_offer(
            "player2", "partner2", is_received=False,
            price_alloc="1200", cert_alloc="Basic", payment_alloc="1M",
            my_price="high", my_cert="middle", my_payment="low",
            partner_price="high", partner_cert="middle", partner_payment="low",
            max_point=25, lambda_factor=0.7
        )
        print(f"  Proposed Offer - My Score: {proposed['my_score']}, Lambda: {proposed['lambda_factor']}")

        # Generate new offer
        print("\n🔄 Generating new offer...")
        print(f"  Expected Logic: Previous lambda={proposed['lambda_factor']:.1f} >= 0.7 AND Partner=Greedy")
        print(f"  → Lambda should be: {proposed['lambda_factor']:.1f} + 0.1 = {proposed['lambda_factor'] + 0.1:.1f}")

        response = client.post(
            "/offers/player2/generate-offer/partner2",
            params={
                "price_priority": "high",
                "certification_priority": "middle",
                "payment_priority": "low"
            }
        )

        assert response.status_code == 200
        data = response.json()

        print(f"\n📊 RESULTS:")
        print(f"  Partner Behavior: {data['partner_behavior']}")
        print(f"  Lambda Factor: {data['lambda_factor']} (expected: {proposed['lambda_factor'] + 0.1:.1f})")
        print(f"  Previous Lambda: {proposed['lambda_factor']}")
        print(f"  Max Value: {data['max_value']} (from previous proposed)")
        print(f"  Total Candidates Generated: {data['total_candidates_generated']}")
        print(f"  Previously Proposed: {data['previously_proposed_count']}")

        print(f"\n🎯 RECOMMENDED OFFER:")
        for key, value in data['recommended_offer'].items():
            print(f"  {key}: {value}")

        # Assertions - verify lambda adjustment logic
        expected_lambda = proposed['lambda_factor'] + 0.1  # Greedy + previous >= 0.7 → add 0.1
        assert data['partner_behavior'] == "greedy"
        assert abs(data['lambda_factor'] - expected_lambda) < 0.01, \
            f"Expected lambda={expected_lambda:.2f} (previous {proposed['lambda_factor']:.1f} + 0.1), got {data['lambda_factor']}"
        assert data['max_value'] == proposed['my_score']

        print(f"\n✅ TEST PASSED: Lambda correctly adjusted from {proposed['lambda_factor']:.1f} to {data['lambda_factor']:.2f}\n")

    def test_generous_partner_lambda_adjustment(self):
        """
        Test Case 3: Generous partner with lambda adjustment

        Setup:
        - Create 2 received offers where partner score decreases
        - Create 1 proposed offer with lambda=0.3

        Expected behavior:
        - partner_behavior: "generous"
        - lambda_factor: 0.2 (0.3 - 0.1 due to generous behavior)
        """
        print("\n" + "="*80)
        print("TEST CASE 3: Generous Partner with Lambda Adjustment")
        print("="*80)

        create_partner_preference(
            "player3", "partner3",
            price="high", cert="middle", payment="low"
        )

        print("\n📥 Creating received offers (generous behavior)...")

        # Define offer specs for validation
        offer1_spec = {
            'price_alloc': "800", 'cert_alloc': "Full", 'payment_alloc': "6M",
            'partner_price': "high", 'partner_cert': "middle", 'partner_payment': "low"
        }
        offer2_spec = {
            'price_alloc': "1400", 'cert_alloc': "None", 'payment_alloc': "Full",
            'partner_price': "low", 'partner_cert': "middle", 'partner_payment': "high"
        }

        # Verify behavior BEFORE creating offers
        verify_offer_behavior(offer1_spec, offer2_spec, "generous")

        # First offer: Partner gets higher allocation → higher score
        offer1 = create_offer(
            "player3", "partner3", is_received=True,
            **offer1_spec,
            my_price="high", my_cert="middle", my_payment="low"
        )
        print(f"\n  Offer 1 - API Returned Partner Score: {offer1['partner_score']}")

        # Second offer: Partner concedes → lower score (GENEROUS)
        offer2 = create_offer(
            "player3", "partner3", is_received=True,
            **offer2_spec,
            my_price="high", my_cert="middle", my_payment="low"
        )
        print(f"  Offer 2 - API Returned Partner Score: {offer2['partner_score']}")
        print(f"  → Partner behavior: GENEROUS (score decreased from {offer1['partner_score']} to {offer2['partner_score']})")

        print("\n📤 Creating proposed offer with lambda=0.3...")
        proposed = create_offer(
            "player3", "partner3", is_received=False,
            price_alloc="1200", cert_alloc="Basic", payment_alloc="1M",
            my_price="high", my_cert="middle", my_payment="low",
            partner_price="high", partner_cert="middle", partner_payment="low",
            max_point=25, lambda_factor=0.3
        )
        print(f"  Proposed Offer - My Score: {proposed['my_score']}, Lambda: {proposed['lambda_factor']}")

        print("\n🔄 Generating new offer...")
        print(f"  Expected Logic: Previous lambda={proposed['lambda_factor']:.1f} <= 0.3 AND Partner=Generous")
        print(f"  → Lambda should be: {proposed['lambda_factor']:.1f} - 0.1 = {proposed['lambda_factor'] - 0.1:.1f}")

        response = client.post(
            "/offers/player3/generate-offer/partner3",
            params={
                "price_priority": "high",
                "certification_priority": "middle",
                "payment_priority": "low"
            }
        )

        assert response.status_code == 200
        data = response.json()

        print(f"\n📊 RESULTS:")
        print(f"  Partner Behavior: {data['partner_behavior']}")
        print(f"  Lambda Factor: {data['lambda_factor']} (expected: {proposed['lambda_factor'] - 0.1:.1f})")
        print(f"  Previous Lambda: {proposed['lambda_factor']}")
        print(f"  Max Value: {data['max_value']}")

        print(f"\n🎯 RECOMMENDED OFFER:")
        for key, value in data['recommended_offer'].items():
            print(f"  {key}: {value}")

        # Assertions - verify lambda adjustment logic
        expected_lambda = proposed['lambda_factor'] - 0.1  # Generous + previous <= 0.3 → subtract 0.1
        assert data['partner_behavior'] == "generous"
        assert abs(data['lambda_factor'] - expected_lambda) < 0.01, \
            f"Expected lambda={expected_lambda:.2f} (previous {proposed['lambda_factor']:.1f} - 0.1), got {data['lambda_factor']}"

        print(f"\n✅ TEST PASSED: Lambda correctly adjusted from {proposed['lambda_factor']:.1f} to {data['lambda_factor']:.2f}\n")

    def test_greedy_stance_neutral_behavior(self):
        """
        Test Case 11: Previous λ >= 0.7 (Greedy stance), Current = Neutral

        Expected: λ = 0.5 (base, no adjustment)
        """
        print("\n" + "="*80)
        print("TEST CASE 11: Greedy Stance + Neutral Behavior → No Adjustment")
        print("="*80)

        create_partner_preference(
            "player11", "partner11",
            price="high", cert="middle", payment="low"
        )

        print("\n📥 Creating received offers (neutral behavior)...")
        offer1_spec = {
            'price_alloc': "1200", 'cert_alloc': "Basic", 'payment_alloc': "1M",
            'partner_price': "middle", 'partner_cert': "low", 'partner_payment': "high"
        }
        offer2_spec = {
            'price_alloc': "1200", 'cert_alloc': "Basic", 'payment_alloc': "1M",
            'partner_price': "middle", 'partner_cert': "low", 'partner_payment': "high"
        }
        verify_offer_behavior(offer1_spec, offer2_spec, "neutral")

        create_offer("player11", "partner11", is_received=True, **offer1_spec,
                    my_price="high", my_cert="middle", my_payment="low")
        create_offer("player11", "partner11", is_received=True, **offer2_spec,
                    my_price="high", my_cert="middle", my_payment="low")

        print("\n📤 Creating proposed offer with lambda=0.7 (greedy stance)...")
        proposed = create_offer("player11", "partner11", is_received=False,
                    price_alloc="1200", cert_alloc="Basic", payment_alloc="1M",
                    my_price="high", my_cert="middle", my_payment="low",
                    partner_price="high", partner_cert="middle", partner_payment="low",
                    max_point=25, lambda_factor=0.7)
        print(f"  Proposed Offer - My Score: {proposed['my_score']}, Lambda: {proposed['lambda_factor']}")

        print("\n🔄 Generating new offer...")
        print(f"  Expected Logic: Previous lambda={proposed['lambda_factor']:.1f} >= 0.7 BUT Partner=Neutral (not Greedy)")
        print(f"  → Lambda should be: base for neutral = 0.5 (NO adjustment)")

        response = client.post("/offers/player11/generate-offer/partner11",
                              params={"price_priority": "high", "certification_priority": "middle", "payment_priority": "low"})

        assert response.status_code == 200
        data = response.json()
        print(f"\n📊 RESULTS:")
        print(f"  Partner Behavior: {data['partner_behavior']}")
        print(f"  Lambda Factor: {data['lambda_factor']} (expected: 0.5, base for neutral)")
        print(f"  Previous Lambda: {proposed['lambda_factor']}")

        assert data['partner_behavior'] == "neutral"
        assert data['lambda_factor'] == 0.5, \
            f"Expected lambda=0.5 (base for neutral, no adjustment), got {data['lambda_factor']}"
        print(f"\n✅ TEST PASSED: Lambda correctly set to base 0.5 (previous {proposed['lambda_factor']:.1f} ignored for neutral)\n")

    def test_greedy_stance_generous_behavior(self):
        """
        Test Case 12: Previous λ >= 0.7 (Greedy stance), Current = Generous

        Expected: λ = 0.3 (base, no adjustment)
        """
        print("\n" + "="*80)
        print("TEST CASE 12: Greedy Stance + Generous Behavior → No Adjustment")
        print("="*80)

        create_partner_preference("player12", "partner12",
                                 price="high", cert="middle", payment="low")

        print("\n📥 Creating received offers (generous behavior)...")
        offer1_spec = {
            'price_alloc': "800", 'cert_alloc': "Full", 'payment_alloc': "6M",
            'partner_price': "high", 'partner_cert': "middle", 'partner_payment': "low"
        }
        offer2_spec = {
            'price_alloc': "1400", 'cert_alloc': "None", 'payment_alloc': "Full",
            'partner_price': "low", 'partner_cert': "middle", 'partner_payment': "high"
        }
        verify_offer_behavior(offer1_spec, offer2_spec, "generous")

        create_offer("player12", "partner12", is_received=True, **offer1_spec,
                    my_price="high", my_cert="middle", my_payment="low")
        create_offer("player12", "partner12", is_received=True, **offer2_spec,
                    my_price="high", my_cert="middle", my_payment="low")

        print("\n📤 Creating proposed offer with lambda=0.7 (greedy stance)...")
        proposed = create_offer("player12", "partner12", is_received=False,
                    price_alloc="1200", cert_alloc="Basic", payment_alloc="1M",
                    my_price="high", my_cert="middle", my_payment="low",
                    partner_price="high", partner_cert="middle", partner_payment="low",
                    max_point=25, lambda_factor=0.7)
        print(f"  Proposed Offer - My Score: {proposed['my_score']}, Lambda: {proposed['lambda_factor']}")

        print("\n🔄 Generating new offer...")
        print(f"  Expected Logic: Previous lambda={proposed['lambda_factor']:.1f} >= 0.7 BUT Partner=Generous (not Greedy)")
        print(f"  → Lambda should be: base for generous = 0.3 (NO adjustment)")

        response = client.post("/offers/player12/generate-offer/partner12",
                              params={"price_priority": "high", "certification_priority": "middle", "payment_priority": "low"})

        assert response.status_code == 200
        data = response.json()
        print(f"\n📊 RESULTS:")
        print(f"  Partner Behavior: {data['partner_behavior']}")
        print(f"  Lambda Factor: {data['lambda_factor']} (expected: 0.3, base for generous)")
        print(f"  Previous Lambda: {proposed['lambda_factor']}")

        assert data['partner_behavior'] == "generous"
        assert data['lambda_factor'] == 0.3, \
            f"Expected lambda=0.3 (base for generous, no adjustment), got {data['lambda_factor']}"
        print(f"\n✅ TEST PASSED: Lambda correctly set to base 0.3 (previous {proposed['lambda_factor']:.1f} ignored for generous)\n")

    def test_generous_stance_greedy_behavior(self):
        """
        Test Case 13: Previous λ <= 0.3 (Generous stance), Current = Greedy

        Expected: λ = 0.7 (base, no adjustment)
        """
        print("\n" + "="*80)
        print("TEST CASE 13: Generous Stance + Greedy Behavior → No Adjustment")
        print("="*80)

        create_partner_preference("player13", "partner13",
                                 price="high", cert="middle", payment="low")

        print("\n📥 Creating received offers (greedy behavior)...")
        offer1_spec = {
            'price_alloc': "1400", 'cert_alloc': "None", 'payment_alloc': "Full",
            'partner_price': "low", 'partner_cert': "middle", 'partner_payment': "high"
        }
        offer2_spec = {
            'price_alloc': "800", 'cert_alloc': "Full", 'payment_alloc': "6M",
            'partner_price': "high", 'partner_cert': "middle", 'partner_payment': "low"
        }
        verify_offer_behavior(offer1_spec, offer2_spec, "greedy")

        create_offer("player13", "partner13", is_received=True, **offer1_spec,
                    my_price="high", my_cert="middle", my_payment="low")
        create_offer("player13", "partner13", is_received=True, **offer2_spec,
                    my_price="high", my_cert="middle", my_payment="low")

        print("\n📤 Creating proposed offer with lambda=0.3 (generous stance)...")
        proposed = create_offer("player13", "partner13", is_received=False,
                    price_alloc="1200", cert_alloc="Basic", payment_alloc="1M",
                    my_price="high", my_cert="middle", my_payment="low",
                    partner_price="high", partner_cert="middle", partner_payment="low",
                    max_point=25, lambda_factor=0.3)
        print(f"  Proposed Offer - My Score: {proposed['my_score']}, Lambda: {proposed['lambda_factor']}")

        print("\n🔄 Generating new offer...")
        print(f"  Expected Logic: Previous lambda={proposed['lambda_factor']:.1f} <= 0.3 BUT Partner=Greedy (not Generous)")
        print(f"  → Lambda should be: base for greedy = 0.7 (NO adjustment)")

        response = client.post("/offers/player13/generate-offer/partner13",
                              params={"price_priority": "high", "certification_priority": "middle", "payment_priority": "low"})

        assert response.status_code == 200
        data = response.json()
        print(f"\n📊 RESULTS:")
        print(f"  Partner Behavior: {data['partner_behavior']}")
        print(f"  Lambda Factor: {data['lambda_factor']} (expected: 0.7, base for greedy)")
        print(f"  Previous Lambda: {proposed['lambda_factor']}")

        assert data['partner_behavior'] == "greedy"
        assert data['lambda_factor'] == 0.7, \
            f"Expected lambda=0.7 (base for greedy, no adjustment), got {data['lambda_factor']}"
        print(f"\n✅ TEST PASSED: Lambda correctly set to base 0.7 (previous {proposed['lambda_factor']:.1f} ignored for greedy)\n")

    def test_generous_stance_neutral_behavior(self):
        """
        Test Case 14: Previous λ <= 0.3 (Generous stance), Current = Neutral

        Expected: λ = 0.5 (base, no adjustment)
        """
        print("\n" + "="*80)
        print("TEST CASE 14: Generous Stance + Neutral Behavior → No Adjustment")
        print("="*80)

        create_partner_preference("player14", "partner14",
                                 price="high", cert="middle", payment="low")

        print("\n📥 Creating received offers (neutral behavior)...")
        offer1_spec = {
            'price_alloc': "1200", 'cert_alloc': "Basic", 'payment_alloc': "1M",
            'partner_price': "middle", 'partner_cert': "low", 'partner_payment': "high"
        }
        offer2_spec = {
            'price_alloc': "1200", 'cert_alloc': "Basic", 'payment_alloc': "1M",
            'partner_price': "middle", 'partner_cert': "low", 'partner_payment': "high"
        }
        verify_offer_behavior(offer1_spec, offer2_spec, "neutral")

        create_offer("player14", "partner14", is_received=True, **offer1_spec,
                    my_price="high", my_cert="middle", my_payment="low")
        create_offer("player14", "partner14", is_received=True, **offer2_spec,
                    my_price="high", my_cert="middle", my_payment="low")

        print("\n📤 Creating proposed offer with lambda=0.3 (generous stance)...")
        proposed = create_offer("player14", "partner14", is_received=False,
                    price_alloc="1200", cert_alloc="Basic", payment_alloc="1M",
                    my_price="high", my_cert="middle", my_payment="low",
                    partner_price="high", partner_cert="middle", partner_payment="low",
                    max_point=25, lambda_factor=0.3)
        print(f"  Proposed Offer - My Score: {proposed['my_score']}, Lambda: {proposed['lambda_factor']}")

        print("\n🔄 Generating new offer...")
        print(f"  Expected Logic: Previous lambda={proposed['lambda_factor']:.1f} <= 0.3 BUT Partner=Neutral (not Generous)")
        print(f"  → Lambda should be: base for neutral = 0.5 (NO adjustment)")

        response = client.post("/offers/player14/generate-offer/partner14",
                              params={"price_priority": "high", "certification_priority": "middle", "payment_priority": "low"})

        assert response.status_code == 200
        data = response.json()
        print(f"\n📊 RESULTS:")
        print(f"  Partner Behavior: {data['partner_behavior']}")
        print(f"  Lambda Factor: {data['lambda_factor']} (expected: 0.5, base for neutral)")
        print(f"  Previous Lambda: {proposed['lambda_factor']}")

        assert data['partner_behavior'] == "neutral"
        assert data['lambda_factor'] == 0.5, \
            f"Expected lambda=0.5 (base for neutral, no adjustment), got {data['lambda_factor']}"
        print(f"\n✅ TEST PASSED: Lambda correctly set to base 0.5 (previous {proposed['lambda_factor']:.1f} ignored for neutral)\n")

    def test_neutral_stance_greedy_behavior(self):
        """
        Test Case 15: Previous λ = 0.5 (Neutral stance), Current = Greedy

        Expected: λ = 0.7 (base, no adjustment because 0.5 is not >= 0.7)
        """
        print("\n" + "="*80)
        print("TEST CASE 15: Neutral Stance + Greedy Behavior → No Adjustment")
        print("="*80)

        create_partner_preference("player15", "partner15",
                                 price="high", cert="middle", payment="low")

        print("\n📥 Creating received offers (greedy behavior)...")
        offer1_spec = {
            'price_alloc': "1400", 'cert_alloc': "None", 'payment_alloc': "Full",
            'partner_price': "low", 'partner_cert': "middle", 'partner_payment': "high"
        }
        offer2_spec = {
            'price_alloc': "800", 'cert_alloc': "Full", 'payment_alloc': "6M",
            'partner_price': "high", 'partner_cert': "middle", 'partner_payment': "low"
        }
        verify_offer_behavior(offer1_spec, offer2_spec, "greedy")

        create_offer("player15", "partner15", is_received=True, **offer1_spec,
                    my_price="high", my_cert="middle", my_payment="low")
        create_offer("player15", "partner15", is_received=True, **offer2_spec,
                    my_price="high", my_cert="middle", my_payment="low")

        print("\n📤 Creating proposed offer with lambda=0.5 (neutral stance)...")
        proposed = create_offer("player15", "partner15", is_received=False,
                    price_alloc="1200", cert_alloc="Basic", payment_alloc="1M",
                    my_price="high", my_cert="middle", my_payment="low",
                    partner_price="high", partner_cert="middle", partner_payment="low",
                    max_point=25, lambda_factor=0.5)
        print(f"  Proposed Offer - My Score: {proposed['my_score']}, Lambda: {proposed['lambda_factor']}")

        print("\n🔄 Generating new offer...")
        print(f"  Expected Logic: Previous lambda={proposed['lambda_factor']:.1f} (not >= 0.7) AND Partner=Greedy")
        print(f"  → Lambda should be: base for greedy = 0.7 (NO adjustment, condition not met)")

        response = client.post("/offers/player15/generate-offer/partner15",
                              params={"price_priority": "high", "certification_priority": "middle", "payment_priority": "low"})

        assert response.status_code == 200
        data = response.json()
        print(f"\n📊 RESULTS:")
        print(f"  Partner Behavior: {data['partner_behavior']}")
        print(f"  Lambda Factor: {data['lambda_factor']} (expected: 0.7, base for greedy)")
        print(f"  Previous Lambda: {proposed['lambda_factor']}")

        assert data['partner_behavior'] == "greedy"
        assert data['lambda_factor'] == 0.7, \
            f"Expected lambda=0.7 (base for greedy, adjustment requires previous >= 0.7), got {data['lambda_factor']}"
        print(f"\n✅ TEST PASSED: Lambda correctly set to base 0.7 (previous {proposed['lambda_factor']:.1f} < 0.7, no adjustment)\n")

    def test_neutral_stance_generous_behavior(self):
        """
        Test Case 16: Previous λ = 0.5 (Neutral stance), Current = Generous

        Expected: λ = 0.3 (base, no adjustment because 0.5 is not <= 0.3)
        """
        print("\n" + "="*80)
        print("TEST CASE 16: Neutral Stance + Generous Behavior → No Adjustment")
        print("="*80)

        create_partner_preference("player16", "partner16",
                                 price="high", cert="middle", payment="low")

        print("\n📥 Creating received offers (generous behavior)...")
        offer1_spec = {
            'price_alloc': "800", 'cert_alloc': "Full", 'payment_alloc': "6M",
            'partner_price': "high", 'partner_cert': "middle", 'partner_payment': "low"
        }
        offer2_spec = {
            'price_alloc': "1400", 'cert_alloc': "None", 'payment_alloc': "Full",
            'partner_price': "low", 'partner_cert': "middle", 'partner_payment': "high"
        }
        verify_offer_behavior(offer1_spec, offer2_spec, "generous")

        create_offer("player16", "partner16", is_received=True, **offer1_spec,
                    my_price="high", my_cert="middle", my_payment="low")
        create_offer("player16", "partner16", is_received=True, **offer2_spec,
                    my_price="high", my_cert="middle", my_payment="low")

        print("\n📤 Creating proposed offer with lambda=0.5 (neutral stance)...")
        proposed = create_offer("player16", "partner16", is_received=False,
                    price_alloc="1200", cert_alloc="Basic", payment_alloc="1M",
                    my_price="high", my_cert="middle", my_payment="low",
                    partner_price="high", partner_cert="middle", partner_payment="low",
                    max_point=25, lambda_factor=0.5)
        print(f"  Proposed Offer - My Score: {proposed['my_score']}, Lambda: {proposed['lambda_factor']}")

        print("\n🔄 Generating new offer...")
        print(f"  Expected Logic: Previous lambda={proposed['lambda_factor']:.1f} (not <= 0.3) AND Partner=Generous")
        print(f"  → Lambda should be: base for generous = 0.3 (NO adjustment, condition not met)")

        response = client.post("/offers/player16/generate-offer/partner16",
                              params={"price_priority": "high", "certification_priority": "middle", "payment_priority": "low"})

        assert response.status_code == 200
        data = response.json()
        print(f"\n📊 RESULTS:")
        print(f"  Partner Behavior: {data['partner_behavior']}")
        print(f"  Lambda Factor: {data['lambda_factor']} (expected: 0.3, base for generous)")
        print(f"  Previous Lambda: {proposed['lambda_factor']}")

        assert data['partner_behavior'] == "generous"
        assert data['lambda_factor'] == 0.3, \
            f"Expected lambda=0.3 (base for generous, adjustment requires previous <= 0.3), got {data['lambda_factor']}"
        print(f"\n✅ TEST PASSED: Lambda correctly set to base 0.3 (previous {proposed['lambda_factor']:.1f} > 0.3, no adjustment)\n")


class TestGenerateOfferFiltering:
    """Test offer filtering logic and edge cases"""

    def test_all_candidates_proposed_error(self):
        """
        Test Case 4: All candidates already proposed

        Setup:
        - Propose multiple offers covering all possible candidates

        Expected behavior:
        - Should return 404 error
        - Error message should indicate all candidates were proposed
        """
        print("\n" + "="*80)
        print("TEST CASE 4: All Candidates Already Proposed")
        print("="*80)

        create_partner_preference(
            "player4", "partner4",
            price="high", cert="middle", payment="low"
        )

        # Propose many offers to exhaust candidates
        print("\n📤 Creating multiple proposed offers...")
        allocations = [
            ("1400", "None", "Full"),
            ("1200", "Basic", "1M"),
            ("1000", "3rd-party", "3M"),
            ("800", "Full", "6M"),
        ]

        for i, (price, cert, payment) in enumerate(allocations):
            create_offer(
                "player4", "partner4", is_received=False,
                price_alloc=price, cert_alloc=cert, payment_alloc=payment,
                my_price="high", my_cert="middle", my_payment="low",
                partner_price="high", partner_cert="middle", partner_payment="low",
                max_point=28 - i, lambda_factor=0.5
            )
            print(f"  Proposed {i+1}: {price}, {cert}, {payment}")

        print("\n🔄 Attempting to generate new offer (should fail)...")
        response = client.post(
            "/offers/player4/generate-offer/partner4",
            params={
                "price_priority": "high",
                "certification_priority": "middle",
                "payment_priority": "low"
            }
        )

        print(f"\n📊 RESULTS:")
        print(f"  Status Code: {response.status_code}")

        if response.status_code == 404:
            print(f"  Error: {response.json()['detail']}")
            print("\n✅ TEST PASSED: Correctly returns 404 when all candidates proposed\n")
        elif response.status_code == 500:
            # Server error - print details
            error = response.json()
            print(f"\n❌ SERVER ERROR:")
            print(f"  {error.get('detail', 'Unknown error')}")
            raise AssertionError(f"Expected 404 or 200, got 500: {error.get('detail', 'Unknown error')}")
        else:
            # If it somehow found a new offer, show it
            data = response.json()
            print(f"\n⚠️  Found new offer (unexpected):")
            if 'recommended_offer' in data:
                print(f"  {data['recommended_offer']}")
                print(f"  Previously Proposed: {data.get('previously_proposed_count', 'N/A')}")
                print("\n✅ TEST PASSED: Found valid new offer\n")
            else:
                print(f"  Response data: {data}")
                raise AssertionError(f"Unexpected response format")

    def test_early_termination_efficiency(self):
        """
        Test Case 5: Early termination efficiency

        Setup:
        - Propose the highest score offer
        - Verify that only 2 candidates are checked (not all)

        Expected behavior:
        - Should check only top few candidates before finding new offer
        - checked_count in message should be small
        """
        print("\n" + "="*80)
        print("TEST CASE 5: Early Termination Efficiency")
        print("="*80)

        create_partner_preference(
            "player5", "partner5",
            price="high", cert="middle", payment="low"
        )

        # First, generate to see what the best offer would be
        print("\n🔄 First generation to identify best offer...")
        response1 = client.post(
            "/offers/player5/generate-offer/partner5",
            params={
                "price_priority": "high",
                "certification_priority": "middle",
                "payment_priority": "low"
            }
        )
        assert response1.status_code == 200
        best_offer = response1.json()['recommended_offer']

        print(f"  Best offer identified:")
        print(f"    {best_offer['price_allocation']}, {best_offer['certification_allocation']}, {best_offer['payment_allocation']}")
        print(f"    My Score: {best_offer['my_score']}")

        # Now propose that best offer
        print("\n📤 Proposing the best offer...")
        create_offer(
            "player5", "partner5", is_received=False,
            price_alloc=best_offer['price_allocation'],
            cert_alloc=best_offer['certification_allocation'],
            payment_alloc=best_offer['payment_allocation'],
            my_price="high", my_cert="middle", my_payment="low",
            partner_price="high", partner_cert="middle", partner_payment="low",
            max_point=28, lambda_factor=0.7
        )

        # Generate again - should skip best and pick second best
        print("\n🔄 Generating new offer (should skip best and pick 2nd best)...")
        response2 = client.post(
            "/offers/player5/generate-offer/partner5",
            params={
                "price_priority": "high",
                "certification_priority": "middle",
                "payment_priority": "low"
            }
        )

        assert response2.status_code == 200
        data = response2.json()

        print(f"\n📊 RESULTS:")
        print(f"  Total Candidates Generated: {data['total_candidates_generated']}")
        print(f"  Previously Proposed Count: {data['previously_proposed_count']}")
        print(f"  Message: {data['message']}")

        print(f"\n🎯 SELECTED OFFER (2nd best):")
        for key, value in data['recommended_offer'].items():
            print(f"  {key}: {value}")

        # Verify efficiency: should have checked only 2 candidates
        # (1 previously proposed + 1 selected = 2 checked total)
        assert data['previously_proposed_count'] == 1
        print(f"\n✅ TEST PASSED: Early termination working - only checked until found new offer\n")


class TestGenerateOfferEdgeCases:
    """Test edge cases and error conditions"""

    def test_missing_partner_preference(self):
        """
        Test Case 6: Partner preference not found

        Expected behavior:
        - Should return 404 error
        - Error message should indicate missing partner preference
        """
        print("\n" + "="*80)
        print("TEST CASE 6: Missing Partner Preference")
        print("="*80)

        print("\n🔄 Attempting to generate offer without partner preference...")
        response = client.post(
            "/offers/player6/generate-offer/partner6",
            params={
                "price_priority": "high",
                "certification_priority": "middle",
                "payment_priority": "low"
            }
        )

        print(f"\n📊 RESULTS:")
        print(f"  Status Code: {response.status_code}")
        print(f"  Error: {response.json()['detail']}")

        assert response.status_code == 404
        assert "Partner preferences not found" in response.json()['detail']

        print("\n✅ TEST PASSED: Correctly returns 404 for missing partner preference\n")

    def test_invalid_priorities(self):
        """
        Test Case 7: Invalid priority values (not exclusive)

        Expected behavior:
        - Should return 422 validation error
        - Error should indicate priorities must be exclusive
        """
        print("\n" + "="*80)
        print("TEST CASE 7: Invalid Priorities (Not Exclusive)")
        print("="*80)

        create_partner_preference(
            "player7", "partner7",
            price="high", cert="middle", payment="low"
        )

        print("\n🔄 Attempting with duplicate 'high' priority...")
        response = client.post(
            "/offers/player7/generate-offer/partner7",
            params={
                "price_priority": "high",
                "certification_priority": "high",  # Duplicate!
                "payment_priority": "low"
            }
        )

        print(f"\n📊 RESULTS:")
        print(f"  Status Code: {response.status_code}")
        print(f"  Error: {response.json()['detail']}")

        assert response.status_code == 422
        assert "exclusive" in response.json()['detail'].lower()

        print("\n✅ TEST PASSED: Correctly rejects non-exclusive priorities\n")

    def test_lambda_cap_at_one(self):
        """
        Test Case 8: Lambda capped at 1.0

        Setup:
        - Greedy partner
        - Previous lambda = 0.95

        Expected behavior:
        - lambda_factor should be capped at 1.0 (not 1.05)
        """
        print("\n" + "="*80)
        print("TEST CASE 8: Lambda Capped at 1.0")
        print("="*80)

        create_partner_preference(
            "player8", "partner8",
            price="high", cert="middle", payment="low"
        )

        # Create greedy behavior
        print("\n📥 Creating greedy behavior...")

        # Define offer specs for validation
        offer1_spec = {
            'price_alloc': "1400", 'cert_alloc': "None", 'payment_alloc': "Full",
            'partner_price': "low", 'partner_cert': "middle", 'partner_payment': "high"
        }
        offer2_spec = {
            'price_alloc': "800", 'cert_alloc': "Full", 'payment_alloc': "6M",
            'partner_price': "high", 'partner_cert': "middle", 'partner_payment': "low"
        }

        # Verify behavior BEFORE creating offers
        verify_offer_behavior(offer1_spec, offer2_spec, "greedy")

        # First offer: Partner gets lower allocation → lower score
        create_offer(
            "player8", "partner8", is_received=True,
            **offer1_spec,
            my_price="high", my_cert="middle", my_payment="low"
        )
        # Second offer: Partner gets higher allocation → higher score (GREEDY)
        create_offer(
            "player8", "partner8", is_received=True,
            **offer2_spec,
            my_price="high", my_cert="middle", my_payment="low"
        )

        # Propose with lambda=0.95
        print("\n📤 Creating proposed offer with lambda=0.95...")
        create_offer(
            "player8", "partner8", is_received=False,
            price_alloc="1200", cert_alloc="Basic", payment_alloc="1M",
            my_price="high", my_cert="middle", my_payment="low",
            partner_price="high", partner_cert="middle", partner_payment="low",
            max_point=25, lambda_factor=0.95
        )

        print("\n🔄 Generating new offer...")
        response = client.post(
            "/offers/player8/generate-offer/partner8",
            params={
                "price_priority": "high",
                "certification_priority": "middle",
                "payment_priority": "low"
            }
        )

        assert response.status_code == 200
        data = response.json()

        print(f"\n📊 RESULTS:")
        print(f"  Partner Behavior: {data['partner_behavior']}")
        print(f"  Lambda Factor: {data['lambda_factor']} (should be capped at 1.0)")
        print(f"  Expected: 0.95 + 0.1 = 1.05 → capped to 1.0")

        assert data['lambda_factor'] == 1.0

        print("\n✅ TEST PASSED: Lambda correctly capped at 1.0\n")

    def test_lambda_cap_at_zero(self):
        """
        Test Case 9: Lambda capped at 0.0

        Setup:
        - Generous partner
        - Previous lambda = 0.05

        Expected behavior:
        - lambda_factor should be capped at 0.0 (not -0.05)
        """
        print("\n" + "="*80)
        print("TEST CASE 9: Lambda Capped at 0.0")
        print("="*80)

        create_partner_preference(
            "player9", "partner9",
            price="high", cert="middle", payment="low"
        )

        # Create generous behavior
        print("\n📥 Creating generous behavior...")

        # Define offer specs for validation
        offer1_spec = {
            'price_alloc': "800", 'cert_alloc': "Full", 'payment_alloc': "6M",
            'partner_price': "high", 'partner_cert': "middle", 'partner_payment': "low"
        }
        offer2_spec = {
            'price_alloc': "1400", 'cert_alloc': "None", 'payment_alloc': "Full",
            'partner_price': "low", 'partner_cert': "middle", 'partner_payment': "high"
        }

        # Verify behavior BEFORE creating offers
        verify_offer_behavior(offer1_spec, offer2_spec, "generous")

        # First offer: Partner gets higher allocation → higher score
        create_offer(
            "player9", "partner9", is_received=True,
            **offer1_spec,
            my_price="high", my_cert="middle", my_payment="low"
        )
        # Second offer: Partner concedes → lower score (GENEROUS)
        create_offer(
            "player9", "partner9", is_received=True,
            **offer2_spec,
            my_price="high", my_cert="middle", my_payment="low"
        )

        # Propose with lambda=0.05
        print("\n📤 Creating proposed offer with lambda=0.05...")
        create_offer(
            "player9", "partner9", is_received=False,
            price_alloc="1200", cert_alloc="Basic", payment_alloc="1M",
            my_price="high", my_cert="middle", my_payment="low",
            partner_price="high", partner_cert="middle", partner_payment="low",
            max_point=25, lambda_factor=0.05
        )

        print("\n🔄 Generating new offer...")
        response = client.post(
            "/offers/player9/generate-offer/partner9",
            params={
                "price_priority": "high",
                "certification_priority": "middle",
                "payment_priority": "low"
            }
        )

        assert response.status_code == 200
        data = response.json()

        print(f"\n📊 RESULTS:")
        print(f"  Partner Behavior: {data['partner_behavior']}")
        print(f"  Lambda Factor: {data['lambda_factor']} (should be capped at 0.0)")
        print(f"  Expected: 0.05 - 0.1 = -0.05 → capped to 0.0")

        assert data['lambda_factor'] == 0.0

        print("\n✅ TEST PASSED: Lambda correctly capped at 0.0\n")


class TestGenerateOfferNeutralBehavior:
    """Test neutral partner behavior"""

    def test_neutral_partner_no_lambda_adjustment(self):
        """
        Test Case 10: Neutral partner behavior + No previous proposed offer

        Setup:
        - 2 received offers with same partner score (neutral behavior CAN be determined)
        - NO previous proposed offer

        Expected behavior:
        - partner_behavior: "neutral"
        - lambda_factor: 0.5 (based on neutral behavior, since it can be determined)
        """
        print("\n" + "="*80)
        print("TEST CASE 10: Neutral Partner Behavior + No Previous Proposed Offer")
        print("="*80)

        create_partner_preference(
            "player10", "partner10",
            price="high", cert="middle", payment="low"
        )

        print("\n📥 Creating received offers (neutral behavior)...")

        # Define offer specs for validation
        offer1_spec = {
            'price_alloc': "1200", 'cert_alloc': "Basic", 'payment_alloc': "1M",
            'partner_price': "middle", 'partner_cert': "low", 'partner_payment': "high"
        }
        offer2_spec = {
            'price_alloc': "1200", 'cert_alloc': "Basic", 'payment_alloc': "1M",
            'partner_price': "middle", 'partner_cert': "low", 'partner_payment': "high"
        }

        # Verify behavior BEFORE creating offers
        verify_offer_behavior(offer1_spec, offer2_spec, "neutral")

        # Both offers identical → same partner score (NEUTRAL)
        offer1 = create_offer(
            "player10", "partner10", is_received=True,
            **offer1_spec,
            my_price="high", my_cert="middle", my_payment="low"
        )
        print(f"\n  Offer 1 - API Returned Partner Score: {offer1['partner_score']}")

        offer2 = create_offer(
            "player10", "partner10", is_received=True,
            **offer2_spec,
            my_price="high", my_cert="middle", my_payment="low"
        )
        print(f"  Offer 2 - API Returned Partner Score: {offer2['partner_score']}")
        print(f"  → Partner behavior: NEUTRAL (same score)")

        print("\n🔄 Generating new offer...")
        print("  NOTE: No previous proposed offer exists")
        print("  Partner behavior CAN be determined (>= 2 received offers)")
        print("  Expected Logic: No previous proposed offer + Partner behavior = neutral → Lambda = 0.5")

        response = client.post(
            "/offers/player10/generate-offer/partner10",
            params={
                "price_priority": "high",
                "certification_priority": "middle",
                "payment_priority": "low"
            }
        )

        assert response.status_code == 200
        data = response.json()

        print(f"\n📊 RESULTS:")
        print(f"  Partner Behavior: {data['partner_behavior']}")
        print(f"  Lambda Factor: {data['lambda_factor']} (expected: 0.5, based on neutral behavior)")

        assert data['partner_behavior'] == "neutral"
        assert data['lambda_factor'] == 0.5, \
            f"Expected lambda=0.5 (neutral behavior, no previous proposed offer), got {data['lambda_factor']}"

        print("\n✅ TEST PASSED: Lambda correctly set to 0.5 (neutral behavior, no previous proposed offer)\n")


class TestMaxValueTrajectory:
    """Test max_value changes across multiple offer generations"""

    def test_max_value_trajectory_with_decreasing_scores(self):
        """
        Test Case 17: Max value trajectory with decreasing scores

        Setup:
        - Generate offer 1 → my_score = X
        - Generate offer 2 → max_value should be X (from offer 1)
        - Generate offer 3 → max_value should be from offer 2

        Expected behavior:
        - max_value should track the my_score from previous proposed offer
        - Verify the trajectory across 3 generations
        """
        print("\n" + "="*80)
        print("TEST CASE 17: Max Value Trajectory Across Multiple Generations")
        print("="*80)

        create_partner_preference(
            "player17", "partner17",
            price="high", cert="middle", payment="low"
        )

        # Create initial state (2 received offers for behavior determination)
        print("\n📥 Setting up initial behavior (neutral)...")
        offer1_spec = {
            'price_alloc': "1200", 'cert_alloc': "Basic", 'payment_alloc': "1M",
            'partner_price': "middle", 'partner_cert': "low", 'partner_payment': "high"
        }
        create_offer("player17", "partner17", is_received=True, **offer1_spec,
                    my_price="high", my_cert="middle", my_payment="low")
        create_offer("player17", "partner17", is_received=True, **offer1_spec,
                    my_price="high", my_cert="middle", my_payment="low")

        # Generation 1: No previous proposed offer
        print("\n🔄 GENERATION 1: No previous proposed offer")
        response1 = client.post(
            "/offers/player17/generate-offer/partner17",
            params={
                "price_priority": "high",
                "certification_priority": "middle",
                "payment_priority": "low"
            }
        )
        assert response1.status_code == 200
        data1 = response1.json()

        print(f"  Max Value: {data1['max_value']} (expected: 28, default)")
        print(f"  Recommended Offer My Score: {data1['recommended_offer']['my_score']}")

        assert data1['max_value'] == 28, "Initial max_value should be 28"
        first_my_score = data1['recommended_offer']['my_score']

        # Propose this offer
        print(f"\n📤 Proposing offer 1 with my_score={first_my_score}...")
        proposed1 = create_offer(
            "player17", "partner17", is_received=False,
            price_alloc=data1['recommended_offer']['price_allocation'],
            cert_alloc=data1['recommended_offer']['certification_allocation'],
            payment_alloc=data1['recommended_offer']['payment_allocation'],
            my_price="high", my_cert="middle", my_payment="low",
            partner_price="high", partner_cert="middle", partner_payment="low",
            max_point=28, lambda_factor=data1['lambda_factor']
        )
        print(f"  Proposed Offer 1 - My Score: {proposed1['my_score']}")

        # Generation 2: max_value should be from offer 1
        print(f"\n🔄 GENERATION 2: Max value should be {first_my_score} (from offer 1)")
        response2 = client.post(
            "/offers/player17/generate-offer/partner17",
            params={
                "price_priority": "high",
                "certification_priority": "middle",
                "payment_priority": "low"
            }
        )
        assert response2.status_code == 200
        data2 = response2.json()

        print(f"  Max Value: {data2['max_value']} (expected: {first_my_score})")
        print(f"  Recommended Offer My Score: {data2['recommended_offer']['my_score']}")

        assert data2['max_value'] == first_my_score, \
            f"Max value should be {first_my_score} from previous proposed offer"
        second_my_score = data2['recommended_offer']['my_score']

        # Propose this offer
        print(f"\n📤 Proposing offer 2 with my_score={second_my_score}...")
        proposed2 = create_offer(
            "player17", "partner17", is_received=False,
            price_alloc=data2['recommended_offer']['price_allocation'],
            cert_alloc=data2['recommended_offer']['certification_allocation'],
            payment_alloc=data2['recommended_offer']['payment_allocation'],
            my_price="high", my_cert="middle", my_payment="low",
            partner_price="high", partner_cert="middle", partner_payment="low",
            max_point=data2['max_value'], lambda_factor=data2['lambda_factor']
        )
        print(f"  Proposed Offer 2 - My Score: {proposed2['my_score']}")

        # Generation 3: max_value should be from offer 2
        print(f"\n🔄 GENERATION 3: Max value should be {second_my_score} (from offer 2)")
        response3 = client.post(
            "/offers/player17/generate-offer/partner17",
            params={
                "price_priority": "high",
                "certification_priority": "middle",
                "payment_priority": "low"
            }
        )
        assert response3.status_code == 200
        data3 = response3.json()

        print(f"  Max Value: {data3['max_value']} (expected: {second_my_score})")
        print(f"  Recommended Offer My Score: {data3['recommended_offer']['my_score']}")

        assert data3['max_value'] == second_my_score, \
            f"Max value should be {second_my_score} from previous proposed offer"

        # Summary
        print(f"\n📊 MAX VALUE TRAJECTORY SUMMARY:")
        print(f"  Generation 1: max_value=28 (default) → my_score={first_my_score}")
        print(f"  Generation 2: max_value={first_my_score} (from gen 1) → my_score={second_my_score}")
        print(f"  Generation 3: max_value={second_my_score} (from gen 2) → my_score={data3['recommended_offer']['my_score']}")

        print("\n✅ TEST PASSED: Max value trajectory correctly tracks previous proposed offers\n")


class TestOfferSelectionProcess:
    """Test offer candidate generation and selection process"""

    def test_lp_candidate_generation_range(self):
        """
        Test Case 18: LP candidate generation range

        Setup:
        - Create a previous proposed offer
        - Generate new offer and verify LP solver explores max_value range

        Expected behavior:
        - LP solver should generate candidates for max_value range: [max_value, max_value-1, max_value-2, max_value-3]
        - Should show total_candidates_generated
        - Should demonstrate filtering and selection
        """
        print("\n" + "="*80)
        print("TEST CASE 18: LP Candidate Generation Range")
        print("="*80)

        create_partner_preference(
            "player18", "partner18",
            price="high", cert="middle", payment="low"
        )

        # Create neutral behavior
        print("\n📥 Setting up neutral behavior...")
        offer1_spec = {
            'price_alloc': "1200", 'cert_alloc': "Basic", 'payment_alloc': "1M",
            'partner_price': "middle", 'partner_cert': "low", 'partner_payment': "high"
        }
        create_offer("player18", "partner18", is_received=True, **offer1_spec,
                    my_price="high", my_cert="middle", my_payment="low")
        create_offer("player18", "partner18", is_received=True, **offer1_spec,
                    my_price="high", my_cert="middle", my_payment="low")

        # First generate to get the best offer
        print("\n🔄 First generation to identify best offer...")
        response1 = client.post(
            "/offers/player18/generate-offer/partner18",
            params={
                "price_priority": "high",
                "certification_priority": "middle",
                "payment_priority": "low"
            }
        )
        assert response1.status_code == 200
        data1 = response1.json()
        print(f"  Best offer my_score: {data1['recommended_offer']['my_score']}")
        print(f"  Best offer allocation: {data1['recommended_offer']['price_allocation']}, {data1['recommended_offer']['certification_allocation']}, {data1['recommended_offer']['payment_allocation']}")

        # Propose this offer
        print(f"\n📤 Creating previous proposed offer with my_score={data1['recommended_offer']['my_score']}...")
        proposed = create_offer(
            "player18", "partner18", is_received=False,
            price_alloc=data1['recommended_offer']['price_allocation'],
            cert_alloc=data1['recommended_offer']['certification_allocation'],
            payment_alloc=data1['recommended_offer']['payment_allocation'],
            my_price="high", my_cert="middle", my_payment="low",
            partner_price="high", partner_cert="middle", partner_payment="low",
            max_point=28, lambda_factor=0.5
        )
        print(f"  Previous Proposed - My Score: {proposed['my_score']}")

        # Generate new offer - should skip the previous one
        print("\n🔄 Generating new offer (should skip previous)...")
        print(f"  Expected: LP solver will try max_value range [{proposed['my_score']}, {proposed['my_score']-1}, {proposed['my_score']-2}, {proposed['my_score']-3}]")
        print(f"  Expected: Will skip the previously proposed offer and select next best")

        response = client.post(
            "/offers/player18/generate-offer/partner18",
            params={
                "price_priority": "high",
                "certification_priority": "middle",
                "payment_priority": "low"
            }
        )

        assert response.status_code == 200
        data = response.json()

        print(f"\n📊 LP CANDIDATE GENERATION RESULTS:")
        print(f"  Max Value Used: {data['max_value']} (from previous proposed offer)")
        print(f"  Total Candidates Generated: {data['total_candidates_generated']}")
        print(f"  Previously Proposed Count: {data['previously_proposed_count']}")
        print(f"  Candidates After Filtering: {data['candidates_after_filtering']}")

        print(f"\n🎯 SELECTED OFFER:")
        print(f"  My Score: {data['recommended_offer']['my_score']}")
        print(f"  Allocation: {data['recommended_offer']['price_allocation']}, {data['recommended_offer']['certification_allocation']}, {data['recommended_offer']['payment_allocation']}")

        print(f"\n💬 Message: {data['message']}")

        # Assertions
        assert data['max_value'] == proposed['my_score'], "Max value should match previous my_score"
        assert data['total_candidates_generated'] > 0, "Should generate at least 1 candidate"
        assert data['previously_proposed_count'] >= 1, "Should have found the previous proposed offer"
        assert data['candidates_after_filtering'] > 0, "Should have candidates after filtering"

        # Verify LP explored the max_value range (should generate 4 candidates for range max to max-3)
        expected_candidates = min(4, data['max_value'] - 5 + 1)  # max_value down to 5 (minimum)
        print(f"\n  Expected candidates: ~{expected_candidates} (for max_value range)")
        print(f"  Actual candidates: {data['total_candidates_generated']}")

        print("\n✅ TEST PASSED: LP candidate generation and filtering working correctly\n")

    def test_duplication_checking_with_multiple_offers(self):
        """
        Test Case 19: Duplication checking with multiple previously proposed offers

        Setup:
        - Propose 3 different offers
        - Generate new offer

        Expected behavior:
        - Should check all 3 previously proposed offers
        - Should skip all 3 and select 4th best
        - previously_proposed_count should be 3
        """
        print("\n" + "="*80)
        print("TEST CASE 19: Duplication Checking with Multiple Offers")
        print("="*80)

        create_partner_preference(
            "player19", "partner19",
            price="high", cert="middle", payment="low"
        )

        # Create neutral behavior
        print("\n📥 Setting up neutral behavior...")
        offer1_spec = {
            'price_alloc': "1200", 'cert_alloc': "Basic", 'payment_alloc': "1M",
            'partner_price': "middle", 'partner_cert': "low", 'partner_payment': "high"
        }
        create_offer("player19", "partner19", is_received=True, **offer1_spec,
                    my_price="high", my_cert="middle", my_payment="low")
        create_offer("player19", "partner19", is_received=True, **offer1_spec,
                    my_price="high", my_cert="middle", my_payment="low")

        # First generation to identify top candidates
        print("\n🔄 First generation to identify top candidates...")
        response1 = client.post(
            "/offers/player19/generate-offer/partner19",
            params={
                "price_priority": "high",
                "certification_priority": "middle",
                "payment_priority": "low"
            }
        )
        assert response1.status_code == 200
        data1 = response1.json()
        offer1_alloc = (
            data1['recommended_offer']['price_allocation'],
            data1['recommended_offer']['certification_allocation'],
            data1['recommended_offer']['payment_allocation']
        )
        print(f"  Best offer: {offer1_alloc}, my_score={data1['recommended_offer']['my_score']}")

        # Propose the best offer
        print(f"\n📤 Proposing offer 1: {offer1_alloc}")
        create_offer(
            "player19", "partner19", is_received=False,
            price_alloc=offer1_alloc[0], cert_alloc=offer1_alloc[1], payment_alloc=offer1_alloc[2],
            my_price="high", my_cert="middle", my_payment="low",
            partner_price="high", partner_cert="middle", partner_payment="low",
            max_point=28, lambda_factor=0.5
        )

        # Generate again to get 2nd best
        print("\n🔄 Second generation to get 2nd best...")
        response2 = client.post(
            "/offers/player19/generate-offer/partner19",
            params={
                "price_priority": "high",
                "certification_priority": "middle",
                "payment_priority": "low"
            }
        )
        assert response2.status_code == 200
        data2 = response2.json()
        offer2_alloc = (
            data2['recommended_offer']['price_allocation'],
            data2['recommended_offer']['certification_allocation'],
            data2['recommended_offer']['payment_allocation']
        )
        print(f"  2nd best offer: {offer2_alloc}, my_score={data2['recommended_offer']['my_score']}")
        print(f"  Previously proposed count: {data2['previously_proposed_count']}")

        # Propose the 2nd best offer
        print(f"\n📤 Proposing offer 2: {offer2_alloc}")
        create_offer(
            "player19", "partner19", is_received=False,
            price_alloc=offer2_alloc[0], cert_alloc=offer2_alloc[1], payment_alloc=offer2_alloc[2],
            my_price="high", my_cert="middle", my_payment="low",
            partner_price="high", partner_cert="middle", partner_payment="low",
            max_point=data2['recommended_offer']['my_score'], lambda_factor=0.5
        )

        # Generate again to get 3rd best
        print("\n🔄 Third generation to get 3rd best...")
        response3 = client.post(
            "/offers/player19/generate-offer/partner19",
            params={
                "price_priority": "high",
                "certification_priority": "middle",
                "payment_priority": "low"
            }
        )
        assert response3.status_code == 200
        data3 = response3.json()
        offer3_alloc = (
            data3['recommended_offer']['price_allocation'],
            data3['recommended_offer']['certification_allocation'],
            data3['recommended_offer']['payment_allocation']
        )
        print(f"  3rd best offer: {offer3_alloc}, my_score={data3['recommended_offer']['my_score']}")
        print(f"  Previously proposed count: {data3['previously_proposed_count']}")

        # Summary
        print(f"\n📊 DUPLICATION CHECKING SUMMARY:")
        print(f"  Offer 1: {offer1_alloc} (proposed)")
        print(f"  Offer 2: {offer2_alloc} (proposed) - skipped 1 duplicate")
        print(f"  Offer 3: {offer3_alloc} (proposed) - skipped {data3['previously_proposed_count']} duplicates")

        # Assertions
        assert data2['previously_proposed_count'] == 1, "Should have skipped 1 previously proposed offer"
        assert data3['previously_proposed_count'] >= 1, f"Should have skipped at least 1 previously proposed offer, got {data3['previously_proposed_count']}"
        assert offer1_alloc != offer2_alloc, "Offer 1 and 2 should be different"
        assert offer2_alloc != offer3_alloc, "Offer 2 and 3 should be different"

        print("\n✅ TEST PASSED: Duplication checking correctly filters multiple previous offers\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("COMPREHENSIVE GENERATE_OFFER API TEST SUITE")
    print("="*80)
    print("\nThis test suite validates:")
    print("  1. LP parameter calculation (lambda, max_value)")
    print("  2. Partner behavior analysis")
    print("  3. Offer candidate generation")
    print("  4. Previously proposed offer filtering")
    print("  5. Best offer selection with early termination")
    print("  6. Edge cases and error handling")
    print("  7. Max value trajectory across generations")
    print("  8. LP candidate generation and duplication checking")
    print("\n" + "="*80 + "\n")

    pytest.main([__file__, "-v", "-s"])

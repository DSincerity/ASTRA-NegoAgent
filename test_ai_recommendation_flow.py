#!/usr/bin/env python3
"""
Test script for AI Recommendation Flow
Tests the complete workflow:
1. Create partner preference
2. Generate AI recommendation
3. Create offer with AI recommendation data
4. Verify data is stored correctly
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def test_ai_recommendation_flow():
    """Test the complete AI recommendation flow"""

    # Test data
    player_id = "test_player_001"
    partner_id = "test_partner_002"

    print_section("TEST 1: Create Partner Preference")

    # Step 1: Create partner preference (opponent preference prediction)
    preference_data = {
        "player_id": player_id,
        "partner_id": partner_id,
        "price_priority": "high",
        "certification_priority": "middle",
        "payment_priority": "low"
    }

    response = requests.post(
        f"{BASE_URL}/partner-preferences",
        json=preference_data
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200, f"Failed to create partner preference: {response.text}"
    print("✅ Partner preference created successfully")

    print_section("TEST 2: Generate AI Recommendation")

    # Step 2: Generate AI recommendation
    response = requests.post(
        f"{BASE_URL}/offers/{player_id}/generate-offer/{partner_id}",
        params={
            "price_priority": "low",
            "certification_priority": "high",
            "payment_priority": "middle"
        }
    )

    print(f"Status Code: {response.status_code}")
    ai_response = response.json()
    print(f"Response: {json.dumps(ai_response, indent=2)}")

    assert response.status_code == 200, f"Failed to generate AI recommendation: {response.text}"

    # Extract AI recommendation data
    recommended_offer = ai_response["recommended_offer"]
    max_point = ai_response["max_value"]
    lambda_factor = ai_response["lambda_factor"]

    print("✅ AI recommendation generated successfully")
    print(f"   Recommended: Price={recommended_offer['price_allocation']}, "
          f"Cert={recommended_offer['certification_allocation']}, "
          f"Payment={recommended_offer['payment_allocation']}")
    print(f"   LP Parameters: max_point={max_point}, lambda={lambda_factor}")

    print_section("TEST 3: Create Offer WITH AI Recommendation (Used)")

    # Step 3a: Create offer using EXACT AI recommendation
    offer_with_ai_used = {
        "player_id": player_id,
        "is_received": False,
        "player2_id": partner_id,
        "player_role": "buyer",
        "player2_role": "seller",
        "price_allocation": recommended_offer["price_allocation"],
        "certification_allocation": recommended_offer["certification_allocation"],
        "payment_allocation": recommended_offer["payment_allocation"],
        "price_priority": "low",
        "certification_priority": "high",
        "payment_priority": "middle",
        "partner_price_priority": "high",
        "partner_certification_priority": "middle",
        "partner_payment_priority": "low",
        "max_point": max_point,
        "lambda_factor": lambda_factor,
        # AI recommendation fields
        "ai_recommended_price": recommended_offer["price_allocation"],
        "ai_recommended_certification": recommended_offer["certification_allocation"],
        "ai_recommended_payment": recommended_offer["payment_allocation"],
        "used_ai_recommendation": True,  # Should be True (exact match)
        "ai_recommendation_received": True
    }

    response = requests.post(
        f"{BASE_URL}/offers",
        json=offer_with_ai_used
    )

    print(f"Status Code: {response.status_code}")
    offer_response_1 = response.json()
    print(f"Response: {json.dumps(offer_response_1, indent=2)}")

    assert response.status_code == 200, f"Failed to create offer: {response.text}"
    assert offer_response_1["used_ai_recommendation"] == True, "used_ai_recommendation should be True"
    assert offer_response_1["ai_recommendation_received"] == True, "ai_recommendation_received should be True"
    assert offer_response_1["ai_recommended_price"] == recommended_offer["price_allocation"], "AI price should match"
    assert offer_response_1["ai_recommended_certification"] == recommended_offer["certification_allocation"], "AI cert should match"
    assert offer_response_1["ai_recommended_payment"] == recommended_offer["payment_allocation"], "AI payment should match"

    print("✅ Offer with AI recommendation (USED) created successfully")
    print(f"   Stored: used_ai_recommendation={offer_response_1['used_ai_recommendation']}")
    print(f"   Stored: max_point={offer_response_1['max_point']}, lambda={offer_response_1['lambda_factor']}")

    print_section("TEST 4: Create Offer WITH AI Recommendation (NOT Used)")

    # Step 3b: Create offer that DIFFERS from AI recommendation
    offer_with_ai_not_used = {
        "player_id": player_id,
        "is_received": False,
        "player2_id": partner_id,
        "player_role": "buyer",
        "player2_role": "seller",
        "price_allocation": "800",  # Different from recommendation
        "certification_allocation": "None",  # Different from recommendation
        "payment_allocation": "Full",  # Different from recommendation
        "price_priority": "low",
        "certification_priority": "high",
        "payment_priority": "middle",
        "partner_price_priority": "high",
        "partner_certification_priority": "middle",
        "partner_payment_priority": "low",
        "max_point": max_point,
        "lambda_factor": lambda_factor,
        # AI recommendation fields
        "ai_recommended_price": recommended_offer["price_allocation"],
        "ai_recommended_certification": recommended_offer["certification_allocation"],
        "ai_recommended_payment": recommended_offer["payment_allocation"],
        "used_ai_recommendation": False,  # Should be False (different)
        "ai_recommendation_received": True
    }

    response = requests.post(
        f"{BASE_URL}/offers",
        json=offer_with_ai_not_used
    )

    print(f"Status Code: {response.status_code}")
    offer_response_2 = response.json()
    print(f"Response: {json.dumps(offer_response_2, indent=2)}")

    assert response.status_code == 200, f"Failed to create offer: {response.text}"
    assert offer_response_2["used_ai_recommendation"] == False, "used_ai_recommendation should be False"
    assert offer_response_2["ai_recommendation_received"] == True, "ai_recommendation_received should be True"

    print("✅ Offer with AI recommendation (NOT USED) created successfully")
    print(f"   Stored: used_ai_recommendation={offer_response_2['used_ai_recommendation']}")

    print_section("TEST 5: Verify Offers in Database")

    # Step 4: Get all offers for player
    response = requests.get(f"{BASE_URL}/offers/{player_id}")

    print(f"Status Code: {response.status_code}")
    offers = response.json()
    print(f"Total offers: {len(offers)}")

    for idx, offer in enumerate(offers, 1):
        print(f"\nOffer {idx}:")
        print(f"  Price: {offer['price_allocation']}")
        print(f"  Certification: {offer['certification_allocation']}")
        print(f"  Payment: {offer['payment_allocation']}")
        print(f"  AI Recommended: Price={offer['ai_recommended_price']}, "
              f"Cert={offer['ai_recommended_certification']}, "
              f"Payment={offer['ai_recommended_payment']}")
        print(f"  Used AI Recommendation: {offer['used_ai_recommendation']}")
        print(f"  AI Recommendation Received: {offer['ai_recommendation_received']}")
        print(f"  LP Parameters: max_point={offer['max_point']}, lambda={offer['lambda_factor']}")

    assert response.status_code == 200, f"Failed to get offers: {response.text}"
    assert len(offers) >= 2, "Should have at least 2 offers"

    print("\n✅ All offers verified successfully")

    print_section("TEST SUMMARY")
    print("✅ All tests passed successfully!")
    print("\nVerified functionality:")
    print("  1. Partner preference creation")
    print("  2. AI recommendation generation with LP parameters")
    print("  3. Offer creation with AI recommendation data")
    print("  4. Tracking whether player used AI recommendation exactly")
    print("  5. Database storage of all AI-related fields")
    print("\nThe complete AI recommendation enforcement flow is working correctly!")

if __name__ == "__main__":
    try:
        test_ai_recommendation_flow()
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)
    except requests.exceptions.RequestException as e:
        print(f"\n❌ API request failed: {e}")
        print("\nMake sure ASTRA-API server is running on http://localhost:8000")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

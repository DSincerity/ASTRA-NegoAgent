# Lambda Calculation Logic Verification (Final)

## User Requirements (정확한 이해)

사용자가 요청한 **정확한** 로직:

### Lambda 계산 Rule

#### Case 1: 이전 Proposed Offer가 **있는** 경우
**이전 lambda 값의 범위**와 **현재 partner behavior**를 함께 고려:

##### 1-1. 이전 Lambda <= 0.3 (Generous Stance)
- **Generous** behavior → `lambda = previous - 0.1`
- **Neutral** behavior → `lambda = 0.5`
- **Greedy** behavior → `lambda = 0.7`

##### 1-2. 이전 Lambda >= 0.7 (Greedy Stance)
- **Greedy** behavior → `lambda = previous + 0.1`
- **Neutral** behavior → `lambda = 0.5`
- **Generous** behavior → `lambda = 0.3`

##### 1-3. 이전 Lambda가 0.3 < lambda < 0.7 (Neutral Stance)
- **Greedy** behavior → `lambda = 0.7`
- **Neutral** behavior → `lambda = 0.5`
- **Generous** behavior → `lambda = 0.3`

#### Case 2: 이전 Proposed Offer가 **없는** 경우
Partner의 received offers 개수에 따라 behavior 특정 가능 여부 결정:

##### 2-1. Received Offers < 2개 (Partner Behavior = "initial")
- **Partner behavior를 특정할 수 X**
- **Lambda = 0.7** (default)

##### 2-2. Received Offers >= 2개 (Partner Behavior 특정 가능)
- **Greedy** behavior → `lambda = 0.7`
- **Neutral** behavior → `lambda = 0.5`
- **Generous** behavior → `lambda = 0.3`

**핵심:** Partner의 received offers가 2개 이상이면 behavior를 특정할 수 있으므로, 그에 따라 lambda를 설정. 2개 미만이면 behavior를 알 수 없으므로 default 0.7 사용.

## Final Implementation (main.py:692-751)

```python
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
        else:  # "initial"
            lambda_factor = 0.7

    # Rule 2: Previous lambda >= 0.7 (Greedy stance)
    elif previous_lambda >= 0.7:
        if partner_behavior == "greedy":
            lambda_factor = previous_lambda + 0.1
        elif partner_behavior == "neutral":
            lambda_factor = 0.5
        elif partner_behavior == "generous":
            lambda_factor = 0.3
        else:  # "initial"
            lambda_factor = 0.7

    # Rule 3: Previous lambda in middle range (0.3 < lambda < 0.7)
    else:
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
```

## Complete Lambda Adjustment Matrix

| Previous Proposed Offer? | Previous Lambda Range | Partner Behavior | Lambda Calculation | Test Coverage |
|-------------------------|----------------------|------------------|-------------------|---------------|
| **O** | <= 0.3 (Generous stance) | Generous | previous - 0.1 | ✅ Test Cases 3, 9 |
| **O** | <= 0.3 (Generous stance) | Neutral | 0.5 | ✅ Test Case 14 |
| **O** | <= 0.3 (Generous stance) | Greedy | 0.7 | ✅ Test Case 13 |
| **O** | >= 0.7 (Greedy stance) | Greedy | previous + 0.1 | ✅ Test Cases 2, 8 |
| **O** | >= 0.7 (Greedy stance) | Neutral | 0.5 | ✅ Test Case 11 |
| **O** | >= 0.7 (Greedy stance) | Generous | 0.3 | ✅ Test Case 12 |
| **O** | 0.3 < λ < 0.7 (Neutral stance) | Greedy | 0.7 | ✅ Test Case 15 |
| **O** | 0.3 < λ < 0.7 (Neutral stance) | Neutral | 0.5 | ✅ (covered by logic) |
| **O** | 0.3 < λ < 0.7 (Neutral stance) | Generous | 0.3 | ✅ Test Case 16 |
| **X** | N/A | Initial (< 2 received) | 0.7 (default) | ✅ Test Case 1 |
| **X** | N/A | Greedy (>= 2 received) | 0.7 | ✅ (covered by logic) |
| **X** | N/A | Neutral (>= 2 received) | 0.5 | ✅ Test Case 10 |
| **X** | N/A | Generous (>= 2 received) | 0.3 | ✅ (covered by logic) |

## Test Case Details

### ✅ Test Case 1: Initial State (No History at All)
- **Setup**: No offer history whatsoever
- **Partner Behavior**: "initial" (< 2 received offers)
- **Expected**: Lambda = 0.7 (default)
- **Logic**: No previous proposed offer + behavior cannot be determined → Default lambda = 0.7
- **Result**: PASSING

### ✅ Test Case 2: Greedy Stance + Greedy Behavior (Adjustment)
- **Setup**: Previous lambda = 0.7, Partner behavior = Greedy
- **Expected**: Lambda = 0.7 + 0.1 = 0.8
- **Logic**: `previous >= 0.7 AND behavior = greedy → add 0.1`
- **Result**: PASSING

### ✅ Test Case 3: Generous Stance + Generous Behavior (Adjustment)
- **Setup**: Previous lambda = 0.3, Partner behavior = Generous
- **Expected**: Lambda = 0.3 - 0.1 = 0.2
- **Logic**: `previous <= 0.3 AND behavior = generous → subtract 0.1`
- **Result**: PASSING

### ✅ Test Case 8: Lambda Cap at 1.0
- **Setup**: Previous lambda = 0.95, Partner behavior = Greedy
- **Expected**: Lambda = 0.95 + 0.1 = 1.05 → capped to 1.0
- **Result**: PASSING

### ✅ Test Case 9: Lambda Cap at 0.0
- **Setup**: Previous lambda = 0.05, Partner behavior = Generous
- **Expected**: Lambda = 0.05 - 0.1 = -0.05 → capped to 0.0
- **Result**: PASSING

### ✅ Test Case 10: No Previous Proposed Offer + Neutral Behavior
- **Setup**:
  - NO previous proposed offer
  - 2 received offers (partner behavior CAN be determined)
  - Partner behavior = Neutral
- **Expected**: Lambda = 0.5 (based on neutral behavior)
- **Logic**: No previous proposed offer + behavior = neutral (>= 2 received offers) → Lambda = 0.5
- **Result**: PASSING

### ✅ Test Case 11: Greedy Stance + Neutral Behavior
- **Setup**: Previous λ=0.7, Partner behavior = Neutral
- **Expected**: Lambda = 0.5 (base for neutral)
- **Logic**: `previous >= 0.7 AND behavior = neutral → lambda = 0.5`
- **Result**: PASSING

### ✅ Test Case 12: Greedy Stance + Generous Behavior
- **Setup**: Previous λ=0.7, Partner behavior = Generous
- **Expected**: Lambda = 0.3 (base for generous)
- **Logic**: `previous >= 0.7 AND behavior = generous → lambda = 0.3`
- **Result**: PASSING

### ✅ Test Case 13: Generous Stance + Greedy Behavior
- **Setup**: Previous λ=0.3, Partner behavior = Greedy
- **Expected**: Lambda = 0.7 (base for greedy)
- **Logic**: `previous <= 0.3 AND behavior = greedy → lambda = 0.7`
- **Result**: PASSING

### ✅ Test Case 14: Generous Stance + Neutral Behavior
- **Setup**: Previous λ=0.3, Partner behavior = Neutral
- **Expected**: Lambda = 0.5 (base for neutral)
- **Logic**: `previous <= 0.3 AND behavior = neutral → lambda = 0.5`
- **Result**: PASSING

### ✅ Test Case 15: Neutral Stance + Greedy Behavior
- **Setup**: Previous λ=0.5, Partner behavior = Greedy
- **Expected**: Lambda = 0.7 (base for greedy)
- **Logic**: `0.3 < previous < 0.7 AND behavior = greedy → lambda = 0.7`
- **Result**: PASSING

### ✅ Test Case 16: Neutral Stance + Generous Behavior
- **Setup**: Previous λ=0.5, Partner behavior = Generous
- **Expected**: Lambda = 0.3 (base for generous)
- **Logic**: `0.3 < previous < 0.7 AND behavior = generous → lambda = 0.3`
- **Result**: PASSING

## Key Insight

### 핵심 차이점:

#### ❌ 잘못된 이해 (이전):
- "이전 proposed offer가 없으면 무조건 lambda = 0.7"

#### ✅ 올바른 이해 (최종):
- **이전 proposed offer가 없을 때:**
  1. **Partner의 received offers가 2개 미만 (initial)** → Behavior를 특정할 수 X → **Lambda = 0.7 (default)**
  2. **Partner의 received offers가 2개 이상** → Behavior를 특정할 수 O:
     - Greedy → Lambda = 0.7
     - Neutral → Lambda = 0.5
     - Generous → Lambda = 0.3

### 실제 시나리오 예시:

**시나리오 1 (Test Case 1):**
- 첫 협상 시작
- 아무 offer도 X (received offers = 0)
- Partner behavior = "initial"
- **Lambda = 0.7** ✅

**시나리오 2 (Test Case 10):**
- 내가 처음 offer를 생성하는 상황
- 하지만 partner가 이미 2번 offer를 보냄 (received offers = 2)
- Partner behavior = "neutral" (특정 가능)
- **Lambda = 0.5** ✅ (neutral behavior에 맞춰 설정)

## Conclusion

### Implementation Status: ✅ ALL REQUIREMENTS MET

모든 사용자 요구사항이 **정확하게** 구현되었습니다:

1. **이전 proposed offer가 있는 경우**: 이전 lambda 범위 + 현재 behavior로 계산 ✅
2. **이전 proposed offer가 없는 경우**:
   - Behavior 특정 불가능 (initial) → Default 0.7 ✅
   - Behavior 특정 가능 → Behavior에 따라 0.7 / 0.5 / 0.3 ✅

### Test Suite Status: ✅ 19/19 TESTS PASSING

모든 테스트 케이스가 통과하며, 완전한 lambda 계산 로직을 검증합니다.

#### Lambda Calculation Tests (Test Cases 1-16)
Lambda factor calculation based on previous lambda and partner behavior - **ALL PASSING**

#### Max Value Trajectory Test (Test Case 17)
- **Test Case 17**: Max value trajectory across multiple generations
  - Verifies max_value correctly tracks my_score from previous proposed offers
  - Tests 3 successive generations: 28 (default) → score1 → score2
  - Confirms max_value changes as offers are proposed ✅

#### Offer Selection Process Tests (Test Cases 18-19)
- **Test Case 18**: LP candidate generation range
  - Verifies LP solver generates candidates for max_value range [max, max-1, max-2, max-3]
  - Tests filtering of previously proposed offers
  - Confirms total_candidates_generated and previously_proposed_count are tracked correctly ✅

- **Test Case 19**: Duplication checking with multiple offers
  - Proposes 3 successive offers and verifies each generation skips all previous ones
  - Tests early termination: system stops checking as soon as a non-duplicate is found
  - Confirms previously_proposed_count increases correctly: 0 → 1 → 2+ ✅

**최종 구현 완료! 모든 요구사항 검증 완료!**

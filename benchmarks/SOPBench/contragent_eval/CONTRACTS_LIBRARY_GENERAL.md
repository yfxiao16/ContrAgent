# SOPBench contract library: recurring formula shapes

The 346 per-domain SOP contracts ([CONTRACTS_LIBRARY.md](CONTRACTS_LIBRARY.md))
are **instances of about twelve recurring formula shapes**. The shapes are domain-independent:
each is an assume-guarantee (A/G) formula over an observable predicate; a new
domain is covered by instantiating the relevant shapes with that domain's tool
names, state fields, and thresholds. This is the generalization claim — 70
SOPBench SOPs across 7 service domains reduce to a dozen formula shapes.

Notation: `<G>` = the guarded goal tool; `gate_<g>_active` = per-task scope flag
(this task's SOP includes gate g); `state_*` = a fact ContrAgent observes from the
world state at the action boundary; `param_*` = a policy threshold; `succ` =
`arg_value(<G>, succeeded) >= 1`. Every shape is `G((called(<G>) & succ &
gate_active) -> <condition>)` unless noted.

| # | shape | instances | domains |
|---|---|---:|---|
| 1 | auth-required | 61 | bank, dmv, healthcare, library, university, online_market |
| 2 | status-flag (eligibility boolean) | 71 | all 7 |
| 3 | min-threshold | 46 | dmv, healthcare, hotel, library, university, online_market |
| 4 | entity-exists | 41 | all 7 |
| 5 | check-precedence (ordering) | 35 | bank, dmv, healthcare, library, university |
| 6 | date-window | 34 | dmv, healthcare, hotel, library, university, online_market |
| 7 | max-threshold | 25 | bank, dmv, healthcare, library, university, online_market |
| 8 | sufficient-funds | 10 | bank, hotel, library |
| 9 | count-limit | 8 | dmv, library, university, online_market |
| 10 | or-disjunction | 5 | bank, online_market, university |
| 11 | entity-absent | ~3 | bank, dmv, online_market |
| 12 | unsatisfiable-SOP | 1 | bank (generalizes) |

---

## 1. auth-required
*The action may complete only if the actor has established the required auth state.*
- **Formula:** `... -> arg_value(<G>, prior_logged_in) >= 1`  (admin variant: `prior_authenticated_admin`)
- **Atom:** `prior_logged_in` / `prior_authenticated_admin` — observable session state recovered from the trace (login/auth seen earlier, reset on logout).
- **Covers SOP gates:** `logged_in_user`, `authenticated_admin_password`, `internal_is_admin`.
- Relation to shape 5: the precedence of `login` before `<G>`, strengthened to a state check.

## 2. status-flag (eligibility boolean)
*A required status/eligibility flag on the named entity must hold.*
- **Formula:** `... -> arg_value(<G>, state_<flag>) >= 1`
- **Atom:** a 0/1 fact read off an enum/membership field: `state_policy_active`, `state_valid_membership`, `state_order_delivered`, `state_is_loyalty_member`, `state_vehicle_insurance_valid`, `state_not_on_probation`, `state_credit_excellent`, `state_book_available`, `provider_authorized`, …
- **Covers:** every "X is active / valid / available / member / in good standing" gate.

## 3. min-threshold
*An observed numeric value must meet a policy minimum.*
- **Formula:** `... -> arg_value(<G>, state_<v>) >= arg_value(<G>, param_<min>)`
- **Atom:** `state_credit_score`, `state_gpa`, `state_age`, `state_current_credits`, `state_stock`, … vs `param_*`.
- **Covers:** `minimal_eligible_credit_score`, `above_minimum_age`, `meets_min_gpa_*`, `enough_stock`, `meets_half_time_enrollment`, …

## 4. entity-exists
*A named entity (user / counterparty / product / order / course / room …) must exist in the world state.*
- **Formula:** `... -> arg_value(<G>, state_<entity>_exists) >= 1`
- **Atom:** `state_user_exists`, `state_dest_user_exists`, `state_order_exists`, `state_product_exists`, `state_course_exists`, `state_provider_exists`, `state_coupon_exists`, `state_room_type_valid`, …
- **Covers:** all `internal_check_*_exist(s)` gates (observed by value, not just "check was called").

## 5. check-precedence (ordering)
*A required verification step must occur before the goal completes.*
- **Formula:** `(!(called(<G>) & succ & gate_active) U (called(<check>) | called(login_user))) | G(!(called(<G>) & succ & gate_active))`
- **Atom:** `called(<check>)` — temporal ordering over the trace.
- **Covers:** `internal_check_username_exist must precede <G>` and similar. The value form (shape 4) is stronger where the entity is observable.

## 6. date-window
*The wall clock must fall in an allowed window / before a deadline.*
- **Formula:** `... -> (arg_value(<G>, now_epoch) >= arg_value(<G>, <start>_epoch) & arg_value(<G>, now_epoch) <= arg_value(<G>, <end>_epoch))` (or a single bound).
- **Atom:** `now_epoch` and a grounded `*_epoch` ordinal (renewal/enrollment/registration/exchange/return window, lead time, deadline).
- **Covers:** `within_*_renewal_period`, `within_enrollment_period`, `within_registration_period`, `within_exchange/return_period`, `before_*_deadline`, `is_booking_date_within_lead_time_range`, `coupon_not_expired`, `appointment_date_valid`, `before_test_date`.

## 7. max-threshold
*An observed value must stay under a policy maximum.*
- **Formula:** `... -> arg_value(<G>, state_<v>) <= arg_value(<G>, param_<max>)` (or `<`)
- **Covers:** `maximum_owed_balance`, `maximum_deposit/exchange`, `meets_income_requirements`, `claim_within_limits`, `credits_within_limit`, … (amounts are unit-normalised, e.g. `amount_dollars`).

## 8. sufficient-funds
*The actor's balance must cover a cost computed from observable facts.*
- **Formula:** `... -> arg_value(<G>, state_balance) >= arg_value(<G>, <cost>)`
- **Atom:** `state_balance` vs a grounded cost (`amount`, `booking_fee = nights*rate`, `late_fee_total = count*fee`, …). Products are materialised in grounding (the LTL DSL has no binary multiply).
- **Covers:** `sufficient_account_balance*`, `sufficient_amount_for_booking`, late-fee / membership-fee gates.

## 9. count-limit
*A count of prior items must stay under a configured cap.*
- **Formula:** `... -> arg_value(<G>, <count>) < arg_value(<G>, param_<max>)`
- **Atom:** `state_test_attempts`, `minor_count`, `state_order_exchanges`, `borrowed_count`, `state_major_changes`, …
- **Covers:** `within_attempt_limit`, `under_max_minors/major_changes`, `less_than_max_exchanges`, `within_borrow_limit`, `has_exceeded_maximum_stays`.

## 10. or-disjunction
*An OR-structured SOP: at least one alternative precondition must hold.* Scoped by
`_present` flags (an OR branch is not unconditionally required, so each disjunct
can't be enforced alone).
- **Formula:** `... & gate_<a>_present & gate_<b>_present) -> (<cond_a> | <cond_b>)`
- **Covers:** `pay_loan` (balance ≥ owed OR ≥ request), online_market credit-OR-window, university requirement disjunctions, `meets_division_requirements`.

## 11. entity-absent
*A named entity must NOT already exist (creation / first-time actions).*
- **Formula:** `... -> arg_value(<G>, state_<entity>_exists) <= 0`
- **Covers:** `open_account` (username free), `register_vehicle` (plate unregistered), `unique_review`, `provider_not_already_authorized`, `coupon_not_already_used`, `not_already_added_shipping_address`.

## 12. unsatisfiable-SOP
*If the declared SOP is self-contradictory (e.g. requires an entity to both exist
and not exist), no execution can satisfy it → never permit.*
- **Formula:** `... & gate_existence_contradiction) -> arg_value(<G>, gate_existence_contradiction) <= 0`
- **Covers:** degenerate contradictory SOPs (observed in `open_account`).

---

## What does not reduce to a general shape (honest scope)
A residue of gates is genuinely domain-specific *and* not observable as a flat
fact, so it has no reusable shape and bounds recall in hotel/university:
- **cross-product over a ledger** — e.g. hotel reservation-modification fee =
  (new nights × new rate) − (old nights × old rate) over the booking record.
- **timetable conflict** — university `no_schedule_conflict` / `no_exam_conflict`
  (pairwise cross-product over the full enrolled timetable).
- **requirement-set logic** — `major_requirements_met`, `meets_minor_prerequisites`,
  `minor_compatible_with_major` (set algebra over curriculum tables).
These are the boundary of a deterministic, observable-state enforcement layer.

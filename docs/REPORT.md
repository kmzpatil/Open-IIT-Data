# Geocoding_Engine: Address Geocoder That Learns from Field Visits: EDA, solution, results

All numbers below come from the code in `code/`. Ground truth = the 100 `surveyed_addresses` (15 in the test split). Coordinates are local metres, one frame per town.

## 1. Dataset (Geocoding_Engine + shared folders)

| Table | Rows | Findings |
|---|---|---|
| addresses | 3,117 (2,880 in towns T1-T3, 237 "OUT" villages) | 2,400 accounts; 8.4% contain Kannada/Devanagari script; 9.8% of town addresses name no locality; text pincode present on 94% but wrong/truncated on many (e.g. `90101`, `9601003`) |
| baseline_geocodes | 2,880 | precision mix: locality 2,052, street 504, pincode 274, rooftop 50. Villages have no baseline pin at all |
| field_visits | 5,578 (9 agents, Apr-Jun 2026) | 1,477 of 3,117 addresses visited; 43% of town addresses have a usable visit |
| visit_gps_points | 160,406 | 3-80 fixes per visit, ~30 s apart, accuracy median 10 m |
| surveyed_addresses | 100 | the only true locations; all residences |
| landmarks_poi | 240 | ~6 POIs per type per town, so a landmark name alone is ambiguous |

**Baseline to beat** (on 100 surveyed): median 376 m, p90 839 m, mean 533 m. By precision: pincode 1,957 m mean, locality 439 m, street 100 m, rooftop 26 m.

### Findings that shaped the solution
1. **Addresses have a street grid, and it is learnable.** In T1, `main` maps to x at ~107 m/step and `cross` to y at ~73 m/step. In T2, `gali` maps to y at ~75 m/step. In T3, block letter maps to y at ~124 m/step. Published locality centroids are off by ~200 m on average. Along-street position in T2/T3 is not explained by house number.
2. **A visit's GPS trail beats its check-in point.** Against surveyed houses, check-in fix median 25 m / p90 152 m; accuracy-weighted last 5 trail fixes median 9.6 m / p90 61 m (153 visits).
3. **A "found" visit is location evidence; "not traceable" is negative evidence.** Every outcome except `address_not_traceable` puts the agent at the house (median 15-35 m). Failed visits happen near the commercial pin, and only 9% of failed trails ever pass within 100 m of the true house.
4. **Pin error drives "not traceable".** Rate by commercial-pin error: <100 m 0%, 100-200 m 6%, 200-400 m 20%, 400-800 m 20%, >800 m 54%. 569 addresses have both a found and a not-traceable visit, so the label is often an agent search failure, not a bad address.
5. **One agent's evidence is contaminated.** FA009 has 222 of 610 visits flagged (36%): 123 + 79 check-ins from two fixed spots across 104 + 65 different addresses in different localities, and 162 visits sharing 6 reused photo hashes (23-32 reuses each). On surveyed addresses all 6 flagged found-visits were 2.7-7.2 km from the house. Detection is generic (no agent id in code). The other 8 agents have 0 flags. FA009's unflagged visits are mostly fine (84% within 100 m vs 96% for FA008).
6. **"Same building" matching on text is unsafe.** 27 groups (58 addresses) share locality + grid + house number. Of the 8 pairs where both are located, median separation is 296 m and none within 60 m. House numbers collide by chance, so I did not pool visits across these. A match needs independent location agreement.
7. **"met at shop" remarks look like a useful downweighting signal but weren't.** Widening those visits worsened visited-address p90 from 41.5 to 44.8 m. Rejected.

## 2. Solution

Bayesian grid posterior per address, `posterior()` in `code/geocoder.py`:
- **Text prior:** fuzzy locality match + multilingual landmark lexicon + regex for cross/main/gali/block/road. Per-town trimmed regression for grid slopes with per-locality shrunk slopes and learned locality centres, fit on visit-consensus labels. Heavy-tailed (20% weight at 4x sigma). Fuzzy pincode (edit distance <=1) narrows candidates when no locality is named.
- **Landmark factor:** same-type POIs near a location raise its likelihood (sigma 60 m, 5% "POI missing" rate).
- **Commercial pin** as a noisy observation with sigma by precision (rooftop 30 m, pincode 2 km).
- **Visits:** trail-dwell estimate, per-visit sigma from GPS accuracy, outlier-mixture likelihood, integrity flags and agent discount.
- **Failed visits:** every trail point carves out probability mass where the house was not found.
- **Output:** posterior geometric-median pin, 50%/90% radius, calibrated by regime, tier, landmark directions.

## 3. Results

Median error vs surveyed truth, each row adds one upgrade (n=100; "visited" n=45, "unvisited" n=55):

| Step | All med | All p90 | Visited med | Unvisited med |
|---|---|---|---|---|
| v0 commercial baseline | 376 | 839 | 384 | 363 |
| v1 published locality centroid | 335 | 837 | 336 | 329 |
| v2 + street-grid prior | 232 | 647 | 229 | 240 |
| v3 + commercial pin as observation | 179 | 602 | 185 | 169 |
| v4 + landmark factor | 136 | 528 | 152 | 131 |
| v5 + visit evidence (check-in point, naive) | 58 | 426 | 19 | 131 |
| v6 + trail-dwell estimate | 44 | 426 | 9.2 | 131 |
| v7 + robust outlier mixture + integrity flags | 44 | 426 | 10.6 | 131 |
| v8 + failed-trail negative evidence | 44 | 426 | 10.6 | 131 |
| **v9 + learned visit-quality model (final)** | **43.9** | **425.6** | **9.1** | **131** |

Headline (strict protocol: prior fit on train split only, surveyed addresses excluded from all fitting): **median 43.9 m vs 376 m (8.6x better); 66% of surveyed addresses within 100 m vs 9% for the baseline; 51% within 50 m.** Test split alone (n=15): median 82 m vs 456 m. On the 45 visited addresses: median 9.1 m, p90 35.4 m.

**Be careful with the tails:** mean error is 262 m (baseline 533 m) and p90 is 426 m. Test-split p90 is 2.8 km, driven by addresses with no locality named, where pincode gives only 2-4 candidate localities (surveyed unresolved-locality addresses: n=10, median 412 m, mean 1,566 m). The calibrated radius flags these (tier C).
Steps v7 to v9 barely move the surveyed medians (naive trail v6 is as good: 9.2 m vs 9.1 m). The robust steps matter on bad data, not on this clean 45-address sample: all 6 flagged visits on surveyed addresses were 2.7-7.2 km wrong, and on 362 held-out visits (below) the learned weighting cut median error by a third.

**Learning curve** (time-ordered replay, 7 cutoffs, production prior refit each time): median error 185 m with 436 visits, 101 m with 1,897, 53 m with 5,578. Share within 50 m: 17% to 49%. Never-visited addresses also improve (192 m to 146 m) because the street-grid prior learns from other addresses' visits.

**Confidence radius calibration:** no-visit regime multiplier 1.10, fitted on 347 held-out addresses; coverage of the 90% radius on surveyed truth 0.91 (n=56). Visit regime: raw radius kept (multiplier 1.0), surveyed coverage 0.91 (n=44); the held-out-visit proxy test gave a raw coverage of 0.79 but its targets are single noisy visits, so I did not widen further. 50% radius multiplier 0.6. Median calibrated 90% radius: 30 m with visits, 397 m prior-only.

**Tiers on all 2,880 town addresses:** A visit directly (radius <=75 m) 1,165; B visit with directions (<=200 m) 238; C verify first 1,477; 237 village addresses not geocodable.

**Value-ranked verify queue** (`verify_queue.csv`, 1,172 residence addresses in tier C or not geocodable): ranked by expected value = overdue amount x the dataset's `ability_to_pay_estimate` (an assumption: that estimate stands in for recovery chance). The top 20% of the queue holds 74% of that value, so verification calls can start there.

**Not-traceable (modelled, not measured):** a logistic fit on 200 visits at surveyed addresses implies per-visit not-traceable probability falls from 19.3% (commercial pin) to 7.6% (our pin), about 60% relative. It assumes NT depends only on pin error and rests on 49 addresses. The randomised pilot is what proves it. Productive visits per agent-day is not measurable from this data.

## 4. Limitations
- Only 100 truth points (15 test); confidence intervals are wide. Data is synthetic.
- 1,635 addresses have no clean visit and rely on the text prior (median ~130-190 m, radius ~400 m).
- T2/T3 along-street axis is weakly predicted.
- 237 village addresses are outside this model.
- Prior labels are visit consensus (noisy by ~10-30 m).
- Cross-account building resolution not solved (see finding 6).

## 5. Upgrade backlog (next loops)
1. Ask the borrower or agent for locality/landmark when none is named (tier C driver); a verification call converts C to A.
2. Active exploration: route one visit to the highest-uncertainty cluster per locality per day to improve many unvisited addresses.
3. Per-agent drift monitoring: re-score flags weekly; escalate on flag share >=10%.
4. Landmark DB self-repair: learn missing/offset POIs from visit clusters mentioning the same landmark.
5. Building identity via learned pairwise model once verified pairs exist.
6. Randomised pilot: pins on vs off, measure not-traceable rate and productive visits per agent-day.

## 6. Files
`geocoded_addresses.csv` (pin, calibrated radius, tier, directions, account value fields and suggested field-agent pool for all 3,117), `verify_queue.csv`, `offline_pack_T1/T2/T3.json` (~150-170 KB each, offline use), `visit_integrity.csv`, `agent_integrity.csv`, `ablation.csv`, `learning_curve.csv`, `charts/`, `code/` (run order: `runA.py`, `runB.py`, `runC.py`, `runD.py`, `final_ablation.py`, `charts.py`; `vq.py` builds the visit-quality model, `remark.py` parses remarks).

## 7. Agent remarks (tested after the first version)

`code/remark.py` parses the correction clause agents append to remarks ("address galat likha hai, ghar ganesh mandir ke paas hai", "actual house behind park, 2 lanes ahead", "nija mane ... hinde ide, 2 cross munde") into landmark type, relation (near / opposite / behind / right side) and lane offset.

- **Coverage:** 643 of 5,578 visits (11.5%) carry one. Relations: opposite 173, near 170, behind 150, right side 84; 150 give a lane count. 49 more name landmarks outside the lexicon.
- **Redundant with the visit location:** on the 16 surveyed visits with a correction clause, the agent stood at the house (median 7 m from truth, vs 9.8 m for visits without one). The remark restates a place the GPS already captured.
- **The named landmark is hard to resolve:** the nearest same-type POI to the visit is a median 235 m from the visit, and only 19% of labelled addresses have the remark's landmark within 100 m (text landmarks: 12%). The POI table is incomplete.
- **It does not flag a bad address text:** addresses with a correction remark have a text-prior error of 167 m median vs 219 m without.
- **Ablation:** adding the remark landmark as a probability factor (6 settings) leaves the surveyed medians unchanged: all 44.0 m, visited 10.6 m, p90 425.6 m / 41.5 m, on 13 surveyed addresses with a correction.
- **Search-effort wording** ("kaafi dhunda", "tumba hudukide") on failed visits: failed trails pass within 100 m of the true house in 14% vs 8% of cases (n=21 vs 26), too few to justify a separate weight.
- **Not used in the final model.** Remarks are only written on visits, so they cannot help addresses that have no visit yet.

## 8. Every table, checked for signal

| Table / column | Used? | What I found |
|---|---|---|
| addresses (text, type, source, date) | yes | parsed to locality, grid, landmark, pincode (fuzzy). Skip-trace source: only 27 addresses, 7 visited, too few to model |
| baseline_geocodes | yes | noisy observation with sigma by precision |
| localities, landmarks_poi, towns | yes | prior centres; landmark factor (POI table is incomplete) |
| surveyed_addresses | evaluation only | never used to fit |
| field_visits (outcome, checkin, dwell, accuracy, photo_hash, ptp_id, timestamps) | yes | all feed the visit-quality model; photo hash and check-in clusters drive integrity flags |
| field_visits.remark | tested, no gain | section 7 |
| visit_gps_points | yes | trail-dwell location, failed-trail negative evidence, trail stability features |
| agents (tenure, shift, language) | yes, weak | tenure is a visit-quality feature (not a top one); all field agents work day shift; language team used to build the suggested agent pool. FA009 has 6 months tenure, but FA003 (4 months) and FA004 (3 months) are newer and have zero flags, so tenure does not identify the problem agent |
| accounts (outstanding, overdue, ability to pay, language, portfolio) | yes | value-ranked verify queue; prior error does not differ by portfolio or income type (medians 166-219 m); Kanglish accounts look better (97 m) but that is the T1 town effect |
| lenders (kyc_address_format) | checked, no gain | prior error 175-212 m across formats; unresolved-locality share 9.3-9.9% in all three |
| payments | checked, no link | 92 `cash_collected` visits vs 113 `cash_field` payments: 14 same-account pairs, none within 48 h, so no confirmation signal |
| skip_traces | checked | 27 `new_address_found` results; same 27 addresses as above |
| dial_attempts remarks | checked, no location content | 267 remarks matched location words, all "number off ide eradu tingalinda" (phone switched off) |
| residence vs office of the same account | checked | median 3.75 km apart (24 accounts), so one does not locate the other |
| PS1 / PS2 tables | not applicable | PTP and phone-contact data carry no location |

**Learned visit-quality model** (`code/vq.py`): gradient-boosted classifier predicting whether a visit's location lies within 50 m of the consensus of the other visits at that address. 2,205 training visits, grouped cross-validated AUC 0.916. Most important features: stability of the last 5 trail fixes (tail spread, 0.179 permutation importance), whether the outcome was `met_borrower` (0.035), distance between check-in and trail estimate (0.028), distance from the commercial pin (0.024), check-in lag (0.015). Its probabilities set each visit's outlier probability, and trail spread widens each visit's sigma. On 362 held-out visits: median error 12.3 m to 8.2 m, mean 46.8 to 43.9 m, p90 49.3 to 49.0 m.

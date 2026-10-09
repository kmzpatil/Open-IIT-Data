# CN Synthetic Collections Data 

 collections data for three problem statements. Everything in it is invented.

## Files

| File | What it holds | Used for |
| :---- | :---- | :---- |
| `data/shared/accounts.csv` | One row per loan account: lender, product, bucket, dues and what the lender knows about the borrower | PS1, PS2, Geocoding_Engine |
| `data/shared/splits.csv` | Which accounts are train, validation and test | PS1, PS2, Geocoding_Engine |
| `data/shared/lenders.csv` | The six lenders | PS1, PS2, Geocoding_Engine |
| `data/shared/agents.csv` | Tele-calling agents, field agents and the voice bot | PS1, PS2, Geocoding_Engine |
| `data/shared/dial_attempts.csv` | Every outbound call attempt, with its result, disposition and agent remark | PS1, PS2 |
| `data/shared/payments.csv` | Every payment received | PS1, PS2 |
| `data/shared/addresses.csv` | Every address on file, as written in the lender's system | PS2, Geocoding_Engine |
| `data/shared/field_visits.csv` | Every field visit, with check-in location, outcome and agent remark | PS2, Geocoding_Engine |
| `data/ps1_fake_ptp/ptps.csv` | Every promise to pay recorded on a call or a visit | PS1 |
| `data/ps1_fake_ptp/call_transcripts.csv` | Turn-by-turn transcripts of calls | PS1 |
| `data/ps1_fake_ptp/post_call_events.csv` | What happened after a PTP: payment link sent or opened, WhatsApp confirmations, customer calls | PS1 |
| `data/ps1_fake_ptp/annotated_ptp_sample.csv` | 200 PTPs labelled by type by a reviewer (some labels are wrong) | PS1 |
| `data/ps2_right_party_contact/phones.csv` | Every phone number linked to an account, and where the number came from | PS2 |
| `data/ps2_right_party_contact/skip_traces.csv` | Skip-trace requests and what they found | PS2 |
| `data/ps2_right_party_contact/verified_contact_points.csv` | 250 numbers checked by a verification campaign | PS2 |
| `data/address_geocoder/towns.csv` | The three towns | Geocoding_Engine |
| `data/address_geocoder/localities.csv` | Localities in each town, with pincode and approximate centre | Geocoding_Engine |
| `data/address_geocoder/landmarks_poi.csv` | Landmark database (incomplete and slightly off) | Geocoding_Engine |
| `data/address_geocoder/baseline_geocodes.csv` | Where a commercial geocoder places each address: the baseline to beat | Geocoding_Engine |
| `data/address_geocoder/visit_gps_points.csv` | GPS trail for each field visit | Geocoding_Engine |
| `data/address_geocoder/surveyed_addresses.csv` | 100 addresses with accurately surveyed locations | Geocoding_Engine |

## Files by problem statement

PS1, Real-Time Fake PTP Detection: `ptps.csv`, `call_transcripts.csv`, `post_call_events.csv`, `annotated_ptp_sample.csv`, `dial_attempts.csv`, `payments.csv`, `accounts.csv`, `agents.csv`

PS2, Right-Party Contact Prediction and Skip-Trace Prioritisation: `dial_attempts.csv`, `phones.csv`, `skip_traces.csv`, `verified_contact_points.csv`, `field_visits.csv`, `addresses.csv`, `payments.csv`, `accounts.csv`, `agents.csv`

Geocoding_Engine, Address Geocoder That Learns from Field Visits: `addresses.csv`, `baseline_geocodes.csv`, `field_visits.csv`, `visit_gps_points.csv`, `surveyed_addresses.csv`, `localities.csv`, `landmarks_poi.csv`, `towns.csv`, `accounts.csv`, `agents.csv`

All three also use `splits.csv`.

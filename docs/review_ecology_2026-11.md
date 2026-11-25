# Ecological priors for validating the six behaviour states

Written 2026-11-25. Literature review only. No project data were read, no remote machine was contacted, and no other file was changed.
Web searches used generic topics only (no bird IDs, tag IDs, coordinates or study identifiers).

This document complements, and does not repeat:
- `docs/literature.md`: e-obs ACC format, flight-mode classification methods, SSL design, Hemery 2023, Nourani 2024, Murphy 2017, Poessel 2022, Rus 2017, Katzner 2015, Sur 2017.
- `agent-share/peers/vega/projects/golden-eagle/ecology-and-validation.md` and `age-and-behaviour.md`: the July 1 age origin, floaters vs territorial birds, Preston diet, and the circularity of movement-based age labels.

## Evidence labels (same as `literature.md`)

- **[FT]** full text read during this review.
- **[AB]** abstract, search snippet or repository record only.
- **[RC]** recalled, not checked in this review. Verify before citing.
- **[OURS]** a prediction derived here from the sources. It is not a published result.
- **Preprint** and **dataset record** mark sources that are not peer-reviewed.

## The model outputs being validated

| State | Model summary |
|---|---|
| Rest | 27% of windows |
| Perched-active | 16% |
| Thermal circling | 13%; about 10 m/s, climb about +1.4 m/s, turn about 20°/s |
| Cruising | 20%; about 11 m/s |
| Fast descending glide | 13%; about 18 m/s, vertical speed about −1.9 m/s |
| Flapping | 10%; VeDBA about 0.28 g |

These are shares of recorded 10-s windows. A triggered continuous high-rate mode over-samples active periods, so these shares are not time budgets. Each test below states whether it needs the regular-schedule windows only.

## Correction to `literature.md`

`literature.md` §1.5 and §5 attribute "flapping 2–38% of flight time, 2.2–2.8 Hz" to golden eagles. The full text of Gupta et al. (arXiv 2412.00231, a **preprint**) [FT] gives those ranges across all 12 birds: six golden eagles and six bald eagles. The paper states:
- bald eagles flap for about 38% of flight time;
- golden eagles flap for about 2%;
- golden eagles flap at about 2.5 Hz.

The golden eagles were spring migrants in the eastern US (10 March–2 April 2016), with 40 Hz ACC on CTT tags.

**Consequence for this model [OURS].** The four flight states make up 56% of windows. Flapping is 10% of windows, so it is about 18% of flight windows. That is roughly nine times the published golden eagle figure. There are three possible explanations:
1. Triggered high-rate recording over-samples take-off and manoeuvres.
2. The flapping state absorbs take-off, landing, perch-to-perch hops, prey handling or nest wing-exercise.
3. Resident birds in weak-uplift conditions really do flap more than spring migrants.

Controls C5 and C6 below separate these explanations.

---

## 1. Sex

### Size, wing loading and the flight mechanics they imply

Lish et al. 2016 [FT] measured 33 golden eagles trapped on autumn migration in Montana and Wyoming. Sex was determined by DNA. Wing area excludes the root box.

| | Adult male (n = 15) | Hatch-year male (n = 5) | Adult female (n = 6) | Hatch-year female (n = 7) |
|---|---|---|---|---|
| Mass (g) | 3553 ± 238 | 3730 ± 418 | 5097 ± 564 | 4257 ± 532 |
| Wingspan (cm) | 192.8 | 191.6 | 212.2 | 209.2 |
| Wing area (cm²) | 5029 | 5041 | 5879 | 5899 |
| Wing loading (g/cm²) | 0.71 ± 0.08 | 0.73 ± 0.06 | **0.86 ± 0.09** | 0.72 ± 0.09 |

Summary of the comparisons:
- Adult females were 31% heavier than adult males and had 18% higher wing loading.
- Hatch-year females were lighter than adult females, with wing loading equal to that of males.
- Male mass did not differ by age. A larger sample of 127 birds from the same trapping site confirmed this.
- The authors suggest the lighter wing loading of adult males suits the breeding season, when the male does most of the hunting and carries prey over long distances.

Predictions from standard flight mechanics [OURS]. Each is a ratio, so it does not depend on the unknown absolute calibration.
- **Speed.** Glide and sink speeds scale with √(wing loading). Adult females should glide and sink about 10% faster than adult males: √(0.86/0.71) = 1.10. At 18 m/s, that is about 1.8 m/s.
- **Turning.** At equal bank angle and lift coefficient, turn radius scales with wing loading, so adult females should circle on a radius about 20% larger. At the same airspeed and bank angle, turn rate (g·tanφ/V) does not depend on wing loading. The expected turn rate is about 10% lower only because females fly about 10% faster: turn rate scales with 1/√(wing loading).
- **Caveat on both predictions.** Birds choose glide and circling speeds to suit thermal strength, so the 10% figure is an upper bound. A difference of 1.8 m/s is about the size of wind effects on ground speed, so compare airspeeds (wind-corrected) wherever possible.
- **Hatch-year females should match males** on both speed and turning (0.72 vs 0.71–0.73 g/cm²). The sex effect should therefore appear in adults and disappear in hatch-year birds.
- **Wingbeat frequency should not differ by sex.** The Pennycuick (1990) allometric formula, f = 1.08 m^1/3 g^1/2 b^−1 S^−1/4 ρ^−1/3 (formula [AB], taken from a search snippet), predicts:
  - adult males about 2.97 Hz and adult females about 2.93 Hz at sea level, a difference of about 1%;
  - about 3.1 Hz at a Bighorn Basin air density of about 1.05 kg/m³.
  The absolute values overpredict the measured about 2.5 Hz (Gupta preprint [FT]), so use the formula only for the relative prediction. A sex difference in wingbeat frequency larger than about 5% would be unexpected.

### Nest attendance, hunting and display

- **Incubation and nest attendance.** Collopy 1984, *Auk* 101:753–760, SW Idaho [AB; full text blocked]:
  - Females incubated 67.5% of daylight hours, and eggs were exposed 31.6% of the time.
  - Males incubated only a little, and only by day. Only females incubated at night.
  - Males captured about 1.2 prey/day vs 0.6 for females during incubation and brood rearing.
  - In weeks 7–9 of chick rearing, females supplied about 43% of prey biomass.
  - A Spanish study of 14 pairs found that males contributed significantly to incubation. It is cited in Bassi et al. (accepted manuscript) [AB, secondary citation; the original was not identified].
- **Territorial display.** Harmata 1982, northern Colorado, n = 15 undulating displays [FT]:
  - 73% occurred with an intruder present, and almost all were at low altitude.
  - Resident males performed 66% of displays given by one bird only, mostly in March–May, while females incubated and brooded.
- **Winter roadside use.** Layfield et al. 2025, 51 eagles wintering in Wyoming, 2014–2023 [AB]:
  - Males were 14.2 times more likely than females to select areas close to roads.
  - Adults used roadside areas more than subadults did.
  - Use rose with snow depth, most strongly above 17.9 cm.
  Roadside use means more carrion feeding and more short low flights for males in winter [OURS].

### Range size

- Fielding et al. 2024, Scotland, 69 resident birds [AB]: annual range size did not differ by sex. In the breeding season, female ranges were much smaller than male ranges. Outside it, female ranges were larger. Two abstract versions differed on this point, so treat it as unverified.
- O'Toole et al. 1999 [FT]: fledglings in North Dakota showed no sex difference in distance moved from the nest over the first 6–10 weeks after fledging.

### Juvenile flight by sex

A Zenodo **dataset record** for Alpine juveniles (Hatzl et al. 2026) [AB] reports that, despite strong dimorphism, the sexes differed only in time spent flapping. **The direction of that difference was not found.**

### Why the pooled null result does not test sex

The pooled comparison found no sex difference in state shares across 89 birds. It does not test these predictions [OURS]:
1. The sex predictions are kinematic (speed, sink, turn radius), not state shares.
2. Nest-attendance differences apply only to territorial breeding adults in March–June. Most tagged birds may not be in that group. The roughly 17–20 nestlings cannot be.
3. If high-rate mode is triggered by activity, it under-samples incubation bouts.

| Expectation | Source | How to test with our outputs |
|---|---|---|
| Adult females glide faster than adult males by about 10% (fast-glide median about +1.8 m/s) and sink faster. | Lish 2016 [FT]; scaling [OURS] | Per-bird median speed and vertical speed in the fast-glide state. Compare sexes among adults, using only birds of independently known age. Use a bird-level test, not a window-level test. |
| Adult females circle on a larger radius (about +20%) and turn about 10% more slowly. | Lish 2016 [FT]; scaling [OURS] | In the circling state, compute per-bird median radius (speed ÷ turn rate) and turn rate by sex. |
| The sex difference in speed and radius is absent in hatch-year birds. | Lish 2016 [FT] | Repeat the two tests above on birds tagged as nestlings in their first year. Expect a null result, giving a sex × age interaction. |
| Wingbeat frequency does not differ by sex (less than about 5%). | Pennycuick formula [AB]; [OURS] | Per-bird dominant heave-axis frequency in flapping windows of 10 s or longer, by sex. A difference of 0.1–0.2 Hz or more would point to an artefact such as tag placement or harness. |
| Territorial adult females rest more than males in March–June (incubation, then brooding). | Collopy 1984 [AB] | Regular-schedule windows only. Compare the rest share of adult birds with a nest territory in March–June with the same birds in Sep–Nov, by sex. |
| Males deliver more prey during incubation and early chick rearing: more flight time and more perch-to-perch sequences. | Collopy 1984 [AB] | In the same birds and months, compare flight-state share and the count of rest→flight→rest sequences per day, by sex. |
| Undulating display, mostly by males in March–May and in winter, at low altitude. | Harmata 1982 [FT] | Search for the candidate signature in §4. Expect more candidates in adult males in Mar–May. Report counts; n = 15 in the source, so give no percentages. |
| Males use roadside areas more in Wyoming winters, especially in deep snow. | Layfield 2025 [AB] | Dec–Feb: perched-active (feeding) bouts within a buffer of roads, by sex and snow depth. Needs a road layer. |

---

## 2. Age

### Breeding phenology and when nestlings fledge

- **SW Idaho.** Beecham & Kochert, *Wilson Bull.* 87:506–513 (1975). The data were read from the archive scan [FT]; the journal volume comes from the search listing.
  - Median hatch dates were 11 April (1969), 10 April (1970) and 13 April (1971).
  - The range was 17 March–18 May.
  - 32% of young died before fledging.
- **Wyoming** [AB; the original study could not be identified from the snippet]:
  - Laramie Basin: laying 23 March–4 April, hatch 2–31 May.
  - South-central Wyoming: laying 11 March–4 April, hatch 7 April–18 May.
- **Bighorn Basin.** Preston et al. 2017 [FT]:
  - Occupancy surveys ran from late February to early April.
  - Productivity (fledgling) surveys ran from late June to early July.
  - Annual productivity averaged 0.73 (range 0.38–1.32) fledglings per occupied territory.
- **Age at first flight.** O'Toole et al. 1999, western North Dakota [FT]:
  - Mean age at first flight was 10.1 weeks (about 71 days).
  - Nestlings were radio-tagged at 8–10 weeks.
  - Post-fledging observations ran from mid-July to late August.
  - In France, nestlings were tagged at 48–58 days (Hemery et al. 2024 [FT]).
- **Derived window for the Bighorn Basin [OURS].** Take hatch dates from mid-April to mid-May and add 65–75 days. Fledging should then fall from about the last week of June to the end of July. This agrees with the timing of Preston's fledgling surveys. A tagged nestling's fledge date should be **estimated per bird**, from the first sustained GPS displacement away from the nest, rather than assumed. This replaces the July 1 origin discussed in the peer file.

### Post-fledging dependence, dispersal and survival

- **France.** Hemery et al. 2024 [FT], citing Hemery 2023:
  - The post-fledging dependence period (PFDP) averaged 177.9 days (range 44–395).
  - Fledging was around 20 July (range 21 June–22 August).
  - Juvenile space and habitat use matched adults' about two months after fledging.
  - Only 11% of juveniles died between fledging and the end of the PFDP.
- **Alps.** Juveniles emigrated 4–10 months after fledging [AB; seen in search snippets of the Alpine eLife paper and dataset record].
- **Western US.** Murphy 2017 and Poessel 2022 are summarised in `literature.md`. All 66 juveniles in Murphy 2017 dispersed before 1 year of age.
- **Survival in the western US.** Millsap et al. 2022 [FT]:
  - Annual survival was 0.70 in the first year and 0.90 for adults.
  - The survival analysis used 512 birds with transmitters.
  - About 75% of first-year deaths had natural causes.
  - **Starvation accounted for about 50% of first-year deaths, mostly before or shortly after dispersal from the natal territory.**
  - Anthropogenic causes made up 74% of deaths after the first year, with shooting the leading cause.
  - Note from the peer file: the end of telemetry does not establish death.

### Flight development

- **Golden eagles.** Flight skills improve over about the first 60 days and then plateau (Hemery 2023, in `literature.md`). Alpine juveniles' energetic cost of moving through the landscape falls over the first years (Nourani/Hatzl 2024, in `literature.md`). In the Alpine dataset record [AB], a higher share of soaring rather than flapping after fledging was associated with earlier emigration. The rate of development was not.
- **Griffon vultures, used as an analogue.** Harel et al. 2016 [FT] compared juveniles 0–2 months after fledging with adults:
  - flapping share of flight: 0.05 vs 0.02;
  - climb rate in thermals: 1.26 vs 1.6 m/s;
  - circling radius: 31.8 vs 35.9, smaller in juveniles. The units are not stated, and the difference held only under intermediate wind shear.
- **White storks [AB; KOPS repository record] showed the opposite age trend in flapping.** The direction of age effects is therefore species-specific.

**Floaters vs territorial adults.** The Ornithology study distinguishing territorial birds from floaters (ukaf034) is summarised in the peer file. Poessel 2022 is in `literature.md`. Wallace 2024 [FT] lists resident floaters and pre-adults as part of the Wyoming winter population. Expected pattern [OURS]:
- Territorial adults show nest-centred rest and small daily ranges in March–June.
- Floaters and pre-adults lack that spring contraction and make non-seasonal long trips.
- **Migration by age.** In eastern North America, adults migrated earlier in spring, over fewer days and in more variable weather than pre-adults (Duerr et al. 2014 [FT]). No verified monthly migration timing by age was found for the interior West.

### Expected timeline in months since fledging [OURS]

The timeline assumes fledging at the end of June to July. It is a prior to compare against, not a published sequence.

| Months since fledging | Expected states | Calendar |
|---|---|---|
| −1 to 0 (in the nest, after tagging) | Rest and perched-active only. Possible flapping-like ACC with GPS speed near 0 (wing exercise; [RC]). No circling, cruising or fast glide. | Jun–Jul |
| 0–1 | Short flights. Highest flapping share of flight. Low climb rates. Small, irregular circles. Stays close to the nest. | Jul–Aug |
| 1–2 | Circling share rises. Climb rate rises towards adult values. Flapping share falls. Plateau near day 60. | Aug–Sep |
| 2–6 | Kinematics stable and close to adult. Range grows. Independent hunting begins (O'Toole saw 11 capture attempts in the first 6–10 weeks). | Sep–Dec |
| 4–10 | Emigration: multi-day directed travel (cruising and fast-glide sequences, high daily displacement). | Oct–May |
| First winter | Highest mortality risk; starvation peaks around dispersal. More rest. Weaker uplift (see §3). | Dec–Feb |

| Expectation | Source | How to test with our outputs |
|---|---|---|
| No flight states before first flight in birds tagged as nestlings. | O'Toole 1999 [FT] | Per nestling: share of circling, cruising and fast glide before the estimated fledge date. Expect about 0. Flapping with speed <1 m/s is allowed. |
| First flight at about 65–80 days after hatch, which places fledging from late June to July in the Bighorn Basin. | O'Toole 1999 [FT]; Beecham & Kochert 1975 [FT]; Preston 2017 [FT]; [OURS] | Distribution of per-bird estimated fledge dates. Expect most between 20 June and 31 July. |
| Flapping share of flight falls, and circling climb rate rises, over days 0–60, then plateaus. | Hemery 2023 [AB]; Harel 2016 [FT] (analogue) | Per bird, by 15-day bins since fledging: flapping ÷ all flight windows, and median climb rate in circling. Fit a within-bird trend with a random effect for bird. |
| Circling radius increases with experience. | Harel 2016 [FT] (analogue, only under intermediate wind shear) | Per-bird median radius in circling, by month since fledging. |
| Territorial adults contract their range in spring; floaters and pre-adults do not, and make non-seasonal long trips. | Peer file (ukaf034); Poessel 2022 (in `literature.md`); Wallace 2024 [FT] | Classify territorial status by repeated seasonal occupancy of the same site, not by movement extent, which is circular (peer file). Then compare March–June daily range and rest share between the groups. |
| Emigration 4–10 months after fledging, with high individual variation (France: 44–395 days). | Hemery 2024 [FT]; Alps [AB] | First date of daily net displacement above about 50 km sustained for 2 days or more, per juvenile. Report the distribution, not a mean. |
| Juveniles flap more than adults in comparable months. | Harel 2016 [FT] (analogue); Alpine dataset record [AB] | Flapping ÷ flight by age group within the same months, using birds of independently known age only. |
| First-year telemetry ends concentrate around dispersal. Starvation may show as declining flight share before the track ends. | Millsap 2022 [FT] | Kaplan–Meier curve of track end by months since fledging. Inspect state budgets in the last 14 days before the end. Do not label track ends as deaths without fate records. |

---

## 3. Season

### Breeding cycle in the Bighorn Basin [OURS], from §2 sources

| Months | Breeding stage |
|---|---|
| Feb–early Apr | Territory occupancy and courtship. Undulating displays occur (Harmata 1982 [FT]: displays recorded January–March; single-bird displays by resident males mostly March–May). |
| Mid-March–April | Egg laying. Incubation lasts about 6 weeks ([RC]; Kochert et al. 2002 account not opened). |
| Mid-April–May | Hatching. The female broods; the male provisions (Collopy 1984 [AB]). |
| Late June–July | Fledging. |
| Aug–winter | Post-fledging dependence. |

### Winter

- Wallace et al. 2024 [FT] define winter as **December–February**. Wyoming's wintering population mixes:
  - year-round territorial residents;
  - long-distance migrants from Alaska and northern Canada;
  - short-distance migrants;
  - resident floaters;
  - pre-adults.
  Wind was among the important predictors of winter use.
- Prey availability falls in winter, and carrion and roadside foraging increase (Layfield 2025 [AB]).
- **Expected in winter [OURS]:** more rest, more perched-active (feeding at carcasses), shorter flight days, and a lower circling share because thermals are weaker. Cruising should be relatively enriched on windy days, because slope lift continues.

### Migration

- Twenty birds have "migrant" in their notes (peer file). Western golden eagles are partial migrants (Wallace 2024 [FT]).
- Spring adults travel faster and earlier than pre-adults (Duerr 2014 [FT], eastern migrants).
- Migration speeds are given in `literature.md` (Rus 2017).
- In Duerr 2014 [FT], following Katzner et al. 2012, hourly mean ground speed classes were:
  - **>10 km/h migratory;**
  - **<5 km/h non-migratory;**
  - 5–10 km/h intermediate.
- Migration tracked weather favourable to thermals in every season and age class (Duerr 2014 [FT]).
- **No verified monthly migration peaks for the interior West were found.** Treat any calendar window as unverified.

### Thermal availability by time of day

- Orographic soaring is more common in the morning, the evening and early in the season, and with stronger winds (Katzner 2015, in `literature.md` [AB]).
- With stronger wind, golden eagles flew lower (−20.4 m above ground per +1 m/s of wind) and within a narrower height band (Lanzone 2012 [FT]).
- **Expected [OURS]:**
  - Circling peaks near local solar noon.
  - The circling peak is lower and shorter in Dec–Feb.
  - Cruising and orographic flight are more evenly spread across the day.
  - No flight occurs at solar night.

| Expectation | Source | How to test with our outputs |
|---|---|---|
| Flight states are essentially absent at solar night. | General; Lanzone 2012 switched tags off at night [FT] | State share by solar hour (not UTC). Flight windows between nautical dusk and dawn should be under about 1%. |
| Circling peaks around solar noon. The peak is narrower and lower in Dec–Feb than in May–Aug. | Katzner 2015 [AB]; Duerr 2014 [FT] | Circling share by solar hour × month, using regular-schedule windows. Compare peak height and width between winter and summer. |
| Winter (Dec–Feb) has more rest and perched-active time, and less circling. | Wallace 2024 [FT]; Layfield 2025 [AB]; [OURS] | Per-bird monthly state shares on regular-schedule windows. Compare within birds across months (paired). |
| Cruising (orographic) increases with wind speed, independent of season. | Lanzone 2012 [FT]; Katzner 2015 [AB] | Annotate windows with wind (e.g. ERA5). Model cruising share vs wind speed. Expect a positive slope. Expect the circling share to fall with wind. |
| Territorial adults are tied to the nest in Mar–Jun: small daily ranges and high rest share for females. | Collopy 1984 [AB]; Fielding 2024 [AB] | Daily range (km) and rest share by month for adults identified as territorial. Expect a spring minimum for females. |
| Birds noted as migrants show multi-day directed displacement in spring and autumn that residents lack. | Duerr 2014 [FT]; Wallace 2024 [FT] | Daily net displacement above 50–100 km, or hourly ground speed above 10 km/h, sustained for 2 days or more. Compare migrant-noted with other birds by month. Expect spring adults to migrate before pre-adults. |
| Undulating display is concentrated in late winter and spring. | Harmata 1982 [FT] | Monthly count of display candidates (§4). Expect a Feb–May peak. |

---

## 4. Behaviour repertoire beyond six states

### Published kinematics of golden eagle flight modes [FT]

| Mode | Ground speed (m/s) | Vertical speed (m/s) | Source and data |
|---|---|---|---|
| Thermal soaring | 10.45 ± 0.80 | — | Duerr 2012: 5 spring migrants, 30 s GPS, Appalachians, manual labels |
| Thermal, climbing | 10.17 ± 0.74 | **+1.76 ± 0.19** | Lanzone 2012: same system, fixes less than 60 s apart |
| Glide between thermals | **18.07 ± 1.39** | — | Duerr 2012 |
| Thermal phase, descending | 17.56 ± 1.35 | **−1.91 ± 0.13** | Lanzone 2012 |
| Slope (orographic) soaring | 10.90 ± 0.87 | — | Duerr 2012 |
| Orographic, climbing or sinking | 11.28 / 12.57 | +1.04 / −0.94 | Lanzone 2012 |
| All flight | — | +1.35 climb, −1.49 sink | Lanzone 2012; mean height above ground 965 m (migration) |
| Flapping | — | — | about 2.5 Hz, about 2% of flight time (Gupta **preprint** [FT], 6 eagles); about 2.8 Hz, one female (Laurent 2021, in `literature.md` [AB]) |

**This model's states match these numbers closely [OURS]:**
- Circling at about 10 m/s and +1.4 m/s matches thermal climbing.
- Fast glide at about 18 m/s and −1.9 m/s matches inter-thermal gliding almost exactly.
- Cruising at about 11 m/s matches slope soaring speed.

A turn rate of 20°/s at 10 m/s implies:
- a radius of about 29 m (v/ω);
- a full circle in about 18 s;
- a bank angle of about 20°.

This is consistent with the "full circle within 30 s" rule (Sherub 2016, in `literature.md`). Harel 2016 reports circling radii of 32–36, but without units, so it is not used as a reference here. The published speeds are ground speeds of migrants and include wind. They are a reference band, not exact targets.

### Detectability with 20 Hz ACC and 1 Hz GPS in 10-s windows [OURS unless marked]

| Behaviour | Expected signature | Detectable in a 10-s window? |
|---|---|---|
| Thermal soaring | Turn rate about 15–25°/s sustained; climb +1 to +2 m/s; low VeDBA; no wingbeat peak | Yes (current circling state) |
| Inter-thermal glide | Straight track; 15–20 m/s; −1.5 to −2.5 m/s; low VeDBA | Yes (fast glide) |
| Orographic (slope) soaring | Straight or along a ridge; 10–13 m/s; vertical about ±1 m/s; low above-ground height (≤200 m); wind ≥5 m/s; positive orographic updraft w·sin(slope)·cos(wind − aspect) (Brandes & Ombalski formula as used by Sandhu 2026 [FT]) | Partly. Cruising probably mixes slope soaring with level transit, so wind and DEM annotation are needed to separate them. |
| Flapping (cruise, take-off) | Heave-axis spectral peak at 2.2–2.9 Hz (0.1 Hz resolution in 10 s); high VeDBA | Yes. Take-off flapping should sit next to rest or perched-active windows. |
| Stoop / attack dive | Vertical speed below about −5 to −10 m/s sustained over 3 or more fixes; speed rising; wings tucked (low dynamic ACC), then a high-g pull-out; often ends in a ground contact | Rare and brief. A 10-s window can contain one, but it will be merged into fast glide or flapping. **No published golden eagle ACC/GPS stoop signature was found** (also noted in `literature.md` §2.4). Popular-source stoop speeds are excluded. |
| Contour (low-level) hunting | Low height above ground (tens of metres), terrain-following, frequent turns, flapping and gliding mixed | Only with DEM height above ground; probably split across cruising and flapping. The description of this hunting mode [RC] comes from general accounts that were not opened. |
| Perch hunting | Long perched-active bout, then a short descending flight that ends on the ground and is followed by a long ground bout (prey handling) | As a sequence of windows, not as one window. |
| Undulating display | Altitude oscillating with period of seconds and constant mean height; regular pitch changes with g peaks at each trough; low altitude when an intruder is present (Harmata 1982 [FT], low displays <100 m, more frequent and smaller undulations) | Marginal. 1 Hz GPS vertical error can be about 20 m (CTT test in Lanzone 2012 [FT]; e-obs not tested). A periodic vertical ACC component at below 0.5 Hz in a non-flapping flight window is the better cue. |
| Kleptoparasitism, intraspecific chases | Need the other animal | Not detectable from one tag. |
| Feeding at a carcass | On the ground; repeated head and neck pulling (moderate, irregular dynamic ACC); long stationary bout; in winter often near roads (Layfield 2025 [AB]) | Probably inside perched-active. Can be separated with a ground vs elevated perch cue (DEM, or static pitch). |
| Night roost | Very low VeDBA and stable posture for hours at solar night | Yes (rest). |
| Nest wing-exercise (nestlings) | Flapping-band ACC with GPS speed near 0 at the nest | Should appear in the flapping state for pre-fledging nestlings [OURS; behaviour RC]. |

| Expectation | Source | How to test with our outputs |
|---|---|---|
| Circling ≈ 10 m/s, +1.4–1.8 m/s. Fast glide ≈ 18 m/s, −1.9 m/s. Cruising ≈ 11 m/s, vertical near 0. | Duerr 2012 [FT]; Lanzone 2012 [FT] | Per-bird medians of ground speed and vertical speed by state. Expect the published means to fall inside the inter-bird interquartile range. Report wind-corrected airspeed where wind is available. |
| Circling and fast glide alternate in thermal–glide cycles. | Duerr 2012 [FT] | State transition matrix: circling→fast glide and fast glide→circling should dominate the transitions out of each state. |
| Flapping is about 2% of flight time on unbiased sampling. A share far above that needs explaining. | Gupta preprint [FT] | Flapping ÷ flight windows on regular-schedule windows only, per bird. Then split flapping windows by context: within 30 s of rest or perched-active (take-off or landing) vs mid-flight at altitude. |
| Flapping wingbeat is about 2.2–2.9 Hz. | Gupta preprint [FT]; Laurent 2021 [AB] | Dominant heave-axis frequency per flapping window. Expect a single peak at 2.2–2.9 Hz. A peak at the sampling or harness frequency, or a broad spread, would indicate an artefact or mixed content. |
| Cruising is enriched where orographic updraft is positive, wind ≥5 m/s and height above ground is low. | Sandhu 2026 [FT]; Lanzone 2012 [FT] | Annotate windows with DEM slope and aspect plus wind, and compute the orographic updraft. Compare its distribution in cruising vs circling and fast glide. |
| Stoop candidates are rare (under 1% of flight) and end in ground contact. | [OURS]; no published signature | Flag windows with ≥3 consecutive fixes below −5 m/s and speed rising. Inspect raw ACC and GPS for a sample. Report the count per bird and do not call them stoops without independent validation. |

---

## 5. Wind, terrain and the wind-energy telemetry literature

### What the USGS / West Virginia University / Conservation Science Global group classifies

The group includes Katzner, Miller, Duerr, Lanzone, Brandes and Poessel. **No single published USFWS or USGS behaviour classification was found.** The classes in use are:

- **Hourly movement classes.** Local is <5 km/h, migratory is >10 km/h, and 5–10 km/h is intermediate (Katzner et al. 2012, *J Appl Ecol*, via Duerr 2014 [FT]; abstract [AB]).
  - Local movers turned more often and flew lower than migrants.
  - Eagles flew lower over steep slopes and cliffs (Katzner 2012 [AB]).
- **Manual GPS flight modes.** Thermal soaring, gliding between thermals, and slope soaring, with 30 s fixes. Duerr 2012 [FT] gives no numeric thresholds.
  - Lanzone 2012 [FT] classified by height band, with narrow bands indicating slope soaring.
  - Each +1 m/s of wind lowered the height band by 27 m and height above ground by 20 m.
- **State-space model (Pirotta et al. 2018 [AB], in `literature.md`).** Five states: perching, gliding, orographic soaring and two thermal-soaring variants.
- **Duerr et al. 2019, *Auk* [AB].** In California, the primary drivers of height above ground were tied to thermal formation. Secondary drivers were slope, topographic position and northness, tied to both thermal and orographic lift.
- **Sandhu et al. 2026, *Ecol Evol* [FT].** No discrete modes: a 1-s Markov model of vertical speed, horizontal speed and heading rate.
  - Western data came mainly from Wyoming: 35 mostly resident birds (20 m, 15 f), CTT tags with 3 s–15 min fixes, 2019–2020.
  - **41% of track flight time was at or below 200 m above ground** (the rotor-swept zone).
  - The heading rate turned the bird towards stronger orographic updraft in its line of sight.
  - The model worked best for soaring flights at low altitude in winds of 5 m/s or more.
  - The authors note that discrete flight-mode classification is hard at second scales because eagles switch modes quickly.

**Mapping this model's states onto those classes [OURS]:**

| Model state | Literature class |
|---|---|
| Rest, perched-active | Perching / local |
| Circling | Thermal soaring |
| Fast glide | Gliding between thermals |
| Cruising | Slope soaring or straight transit |
| Flapping | No class in the USGS group's GPS-only schemes; GPS alone cannot see flapping |

Sandhu's 41% figure is a prior for how much flight in Wyoming is low enough to be collision-relevant. Its birds, tags and years differ from this study's.

| Expectation | Source | How to test with our outputs |
|---|---|---|
| A large share of resident Wyoming flight is at or below 200 m above ground (about 41% in a comparable population). | Sandhu 2026 [FT] | Height above ground (GPS altitude minus DEM) per flight window, by state. Report the share at or below 200 m per bird. Expect cruising and flapping to be lower than circling. |
| Height above ground and height band fall as wind rises. | Lanzone 2012 [FT] | Regress per-day median height above ground on wind speed with a random effect for bird. Expect a negative slope of the order of −20 m per m/s. |
| Hourly ground speed separates local from travelling birds. | Katzner 2012 [AB]; Duerr 2014 [FT] | Hourly mean ground speed: share of hours <5 vs >10 km/h by state composition. Hours of migration or emigration should be dominated by fast glide plus circling. |
| Low-level flight is terrain-linked: lower over steep slopes and cliffs. | Katzner 2012 [AB]; Duerr 2019 [AB] | Height above ground vs slope in cruising windows. |
| Heading turns towards stronger updraft. | Sandhu 2026 [FT] | In cruising windows with wind ≥5 m/s, correlate turn direction with the left–right difference in orographic updraft ahead. Expect a positive correlation. |

---

## Positive and negative controls (the ten most decisive checks)

Use regular-schedule windows wherever shares are compared. Use per-bird statistics, then summarise across birds.

Two feasibility notes:
- **C9** needs birds of independently known adult age, and the peer file suggests few or none exist. Until deployment records are checked, compare first-year birds tagged as nestlings with all other birds, and label the second group "age unknown".
- **C5 and every wingbeat test** need windows with at least about 10 s of continuous ACC. The 1.2 s and 2.0 s bursts give only 0.83 Hz and 0.5 Hz frequency resolution (`literature.md` §1.5). For those bursts, use band power in 1.5–4 Hz instead.

| # | Check | Type | Pass condition |
|---|---|---|---|
| C1 | Kinematic match to published modes | Positive | Inter-bird median circling climb +1.2 to +1.8 m/s; fast-glide speed 16–20 m/s and vertical speed −1.5 to −2.3 m/s; cruising about 10–13 m/s (Duerr 2012, Lanzone 2012 [FT]). Already passes at the pooled level; confirm per bird. |
| C2 | No flight at solar night | Negative | Flight states under about 1% of windows between nautical dusk and dawn. |
| C3 | Circling peaks near solar noon and is weaker in Dec–Feb | Positive | Peak within about ±2 h of solar noon; lower winter peak. |
| C4 | Cruising tracks wind and orographic lift | Positive | Cruising share rises with wind speed, and orographic updraft is higher in cruising than in other flight states. |
| C5 | Wingbeat frequency in the flapping state | Positive | A single dominant heave-axis peak at 2.2–2.9 Hz, with no sex difference above about 5%. |
| C6 | Flapping context and share | Diagnostic | Most flapping windows lie within about 30 s of a rest or perched-active window (take-off and landing). The flapping share of flight on regular-schedule windows approaches a few percent. If it stays near 18%, the state is mixing content or the sampling is biased. |
| C7 | Nestlings before first flight | Negative | About 0 circling, cruising and fast glide before each nestling's estimated fledge date. Fledge dates fall mostly between 20 June and 31 July. |
| C8 | Juvenile development over days 0–60 | Positive | Within-bird decline in flapping ÷ flight and rise in circling climb rate, then a plateau. |
| C9 | Sex × age kinematics | Positive / negative | Adult females glide about 10% faster and circle on a larger radius than adult males. The difference is absent in first-year birds. The pooled null in state shares is not a test of this. |
| C10 | Season and migration | Positive | Territorial adult females have a higher rest share in Mar–Jun than in autumn and than males. Migrant-noted birds show spring and autumn multi-day displacement above 50 km/day that residents lack. |

---

## Gaps and items not verified

- **Collopy 1984 full text was blocked (HTTP 403).** The incubation and provisioning figures are [AB]. Collopy & Edwards 1989 (activity budgets, display rates by sex) was also not opened.
- **No published ACC or GPS signature of a golden eagle stoop or attack was found.** Popular-source dive speeds were deliberately excluded.
- **No USFWS behaviour classification for golden eagle telemetry was found.** §5 lists the classes actually used by the USGS group.
- **The direction of the sex difference in juvenile flapping** (Alpine dataset record) was not found.
- **The Wyoming laying and hatch date ranges are [AB].** Their original study was not identified.
- **No monthly migration peaks for the interior West were verified.**
- **The Pennycuick 1990 formula is [AB].** Use it only for the sex ratio, not for absolute frequency.
- **The incubation length (about 6 weeks) and nest wing-exercise are [RC].**
- **The Gupta et al. paper is a preprint.** The Hatzl et al. item is a dataset record. Neither is peer-reviewed.

## References

Labels give what was read.

- Beecham JJ, Kochert MN 1975. Breeding biology of the golden eagle in southwestern Idaho. *Wilson Bull.* 87:506–513. https://archive.org/download/biostor-233281/biostor-233281.pdf [FT scan; volume and pages from the search listing]
- Duerr AE et al. 2012. Testing an emerging paradigm in migration ecology shows surprising differences in efficiency between flight modes. *PLoS ONE* 7:e35548. https://doi.org/10.1371/journal.pone.0035548 [FT]
- Duerr AE et al. 2014. Flight response of slope-soaring birds to seasonal variation in thermal generation. *Funct Ecol*. https://doi.org/10.1111/1365-2435.12381 (PDF: https://www.fs.usda.gov/nrs/pubs/jrnl/2014/nrs_2014_duerr_001.pdf) [FT]
- Duerr AE et al. 2019. Topographic drivers of flight altitude over large spatial and temporal scales. *Auk*. https://pubs.usgs.gov/publication/70204641 [AB]
- Fielding AH et al. 2024. The characteristics and variation of the golden eagle home range. *Diversity* 16:523. https://mdpi.com/1424-2818/16/9/523 [AB]
- Gupta D, Brandes D, Lanzone MJ, Miller T, Bewley GP 2024. Intermittent turbulent gusts lift eagles. arXiv 2412.00231, **preprint**. https://arxiv.org/abs/2412.00231 [FT]
- Harel R, Horvitz N, Nathan R 2016. Adult vultures outperform juveniles in challenging thermal soaring conditions. *Sci Rep* 6:27865. https://doi.org/10.1038/srep27865 [FT]
- Harmata AR 1982. What is the function of undulating flight display in golden eagles? *Raptor Research* 16:103–109. https://archive.org/download/biostor-216610/biostor-216610.pdf [FT]
- Hatzl JS et al. 2026. Supplementary material for "The consequences of food and flight efficiency for the timing of natal dispersal in golden eagles". Zenodo, **dataset record**. https://doi.org/10.5281/zenodo.19002502 [AB]
- Hemery A, Duriez O, Itty C, Henry P-Y, Besnard A 2024. Using juvenile movements as a proxy for adult habitat and space use in long-lived territorial species: a case study on the golden eagle. *J Avian Biol* e03212. https://doi.org/10.1111/jav.03212 [FT]
- Katzner TE et al. 2012. Topography drives migratory flight altitude of golden eagles: implications for on-shore wind energy development. *J Appl Ecol*. https://tethys.pnnl.gov/publications/topography-drives-migratory-flight-altitude-golden-eagles-implications-shore-wind [AB]
- Lanzone MJ et al. 2012. Flight responses by a migratory soaring raptor to changing meteorological conditions. *Biol Lett* 8:710–713. https://doi.org/10.1098/rsbl.2012.0359 [FT]
- Layfield JF et al. 2025. Intrinsic and extrinsic factors simultaneously modulated the use of roadways by golden eagles during winter. *Ecol Evol*. https://doi.org/10.1002/ece3.71955 [AB]
- Lish JW, Domenech R, Bedrosian BE, Ellis DH, Payton M 2016. Wing loading in North American golden eagles. *J Raptor Res* 50:70–75. https://doi.org/10.3356/rapt-50-01-70-75.1 [FT]
- Millsap BA et al. 2022. Age-specific survival rates, causes of death, and allowable take of golden eagles in the western United States. *Ecol Appl* 32:e2544. https://doi.org/10.1002/eap.2544 [FT]
- O'Toole LT, Kennedy PL, Knight RL, McEwen LC 1999. Postfledging behavior of golden eagles. *Wilson Bull.* 111:472–477. https://archive.org/download/biostor-233806/biostor-233806.pdf [FT]
- Pennycuick CJ 1990. Predicting wingbeat frequency and wavelength of birds. *J Exp Biol* 150:171–185. https://cob.silverchair.com/jeb/article/150/1/171/5695/Predicting-Wingbeat-Frequency-and-Wavelength-of [AB; formula from a search snippet]
- Preston CR, Jones RE, Horton NS 2017. Golden eagle diet breadth and reproduction in relation to fluctuations in primary prey abundance in Wyoming's Bighorn Basin. *J Raptor Res* 51:334–346. https://doi.org/10.3356/JRR-16-39.1 [FT]
- Sandhu R et al. 2026. Movement models to predict low-altitude flight of soaring birds using look-ahead environmental factors. *Ecol Evol* 16:e73059. https://doi.org/10.1002/ece3.73059 [FT]
- Wallace ZP et al. 2024. Predicting the spatial distribution of wintering golden eagles to inform full annual cycle conservation in western North America. *PLoS ONE* 19:e0297345. https://doi.org/10.1371/journal.pone.0297345 [FT]
- Bassi E et al. Accepted manuscript on bearded vulture and golden eagle (source of the secondary citation of the Spanish 14-pair study). https://air.unimi.it/bitstream/2434/906210/2/Bassi%20et%20al_Bvulture_Geagle_revision_accepted_ms.pdf [AB, search snippet only]
- White stork age and flight studies (KOPS repository). https://kops.uni-konstanz.de/bitstreams/d3336e14-1430-48ed-a215-3a16d4233ed6/download [AB, search snippet only]
- Collopy MW 1984. Parental care and feeding ecology of golden eagle nestlings. *Auk* 101:753–760. https://digitalcommons.usf.edu/auk/vol101/iss4/11 [AB]
- Already in `literature.md`, cited here without repeating: Hemery et al. 2023 (10.1111/jav.03111), Nourani et al. 2024 (10.7554/eLife.98818), Murphy et al. 2017, Poessel et al. 2022, Rus et al. 2017, Katzner et al. 2015, Pirotta et al. 2018, Sherub et al. 2016, Laurent et al. 2021.

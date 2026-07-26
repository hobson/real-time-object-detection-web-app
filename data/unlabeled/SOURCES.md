# Unlabeled source videos

Raw dashcam-style clips gathered for future ALPR (license plate) testing/
labeling — not yet annotated, not wired into `data/license_plates/`'s
train/val split. Kept here separately until someone labels frames from
them. Video files themselves are gitignored (large, re-downloadable from
the URLs below); only this manifest is tracked.

| File | Source | Duration | Notes |
|---|---|---:|---|
| `pexels-5921059-stockholm-dashcam.mp4` | [Pexels video 5921059](https://www.pexels.com/video/dash-cam-view-of-the-road-5921059/) by Aleks Magnusson | 11.3s | City driving, Stockholm, Sweden (identified from Swedish storefront signage - not US, despite the search intent). Free-to-use Pexels license, no attribution required. |
| `pexels-4608285-us-highway-dashcam.mp4` | [Pexels video 4608285](https://www.pexels.com/video/vehicle-on-highway-with-dash-cam-4608285/) by Kelly | 29.8s | US highway driving (confirmed via green Interstate-style exit signage). Free-to-use Pexels license, no attribution required. |
| `pexels-27974752-ambulance-night-lights.mp4` | [Pexels video 27974752](https://www.pexels.com/video/ambulance-27974752/) by Bubble Media | 10s (trimmed from 30.8s) | Parked ambulance (AMR - American Medical Response, a private EMS provider), night, red/white light bar active. Included for lighting/colormap diversity - saturated red light spill and blown highlights are a different regime than the other daylight clips and are a realistic detector/OCR stress case. Free-to-use Pexels license, no attribution required. |

All three are ordinary public stock footage of vehicles sourced under
each site's free-to-use license, downloaded for local model testing only
(not redistributed) - not footage of any specific person, protest, or
law-enforcement incident.

**Not yet found**: a clip with lights mounted in the front bumper/grille
close enough to the license plate to stress-test glare/occlusion right at
the plate itself (as opposed to a roof light bar, which doesn't sit near
the plate). Several Pexels "police lights" stock clips were checked
(7714436, 7714378, 7714379) but all are close-ups of a roof light bar with
no plate in frame at all, so they were left out - a light bar glowing in
isolation doesn't exercise plate detection/OCR in a meaningful way. Worth
another pass with different search terms (e.g. "push bar spotlight",
"grille lights") or a user-supplied clip if this specific case matters.

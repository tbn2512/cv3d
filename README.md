# Research protocol: iPhone capture → COLMAP → Nerfstudio (NeRF + 3DGS)

**Project:** From Photos to 3D: NeRF and Gaussian Splatting in Practice  
**Capture device:** iPhone 16 Pro Max, iOS 27.2  
**Local workstation:** Windows 11, RTX 3060 Laptop GPU (6 GB VRAM)  
**Training:** Google Colab Pro  
**Shared storage:** Google Drive  
**Primary methods:** Nerfstudio `nerfacto` and `splatfacto`

This is an executable project scaffold. The Python utilities are included in `scripts/`. It is designed to preserve raw data, create one joint COLMAP reconstruction for all capture conditions, and generate fixed evaluation splits shared across experiments.

> **Important scope:** the primary benchmark is RGB-only.

---

## 0. Research design at a glance

### Primary controlled dataset

Capture five video takes/groups of the **same static object in the same unchanged scene**:

| Prefix | Capture | Purpose |
|---|---|---|
| `orbit_` | Three-ring orbit, 1/50 s shutter | Sharp baseline training views and trajectory comparison |
| `spiral_` | One continuous helical trajectory, 1/50 s shutter | Trajectory comparison |
| `blur_moderate_` | Same orbit path/speed, 1/50 s shutter | Moderate motion-blur condition |
| `blur_heavy_` | Same orbit path/speed, 1/25 s shutter | Stronger motion-blur condition |
| `holdout_` | Separate standardized reference pass at 1/50 s | Shared validation and final test cameras |

Use the same physical iPhone, 1x main lens, video resolution, frame rate, focus, scene, object position and lighting for all groups. Change only the intended experimental variable. For the blur study, keep path and walking speed constant. Shutter duration is the primary experimental factor; adjust ISO manually to keep brightness approximately comparable and record the ISO for every take. Because ISO changes can affect noise, report ISO alongside blur results rather than claiming shutter was the only changed camera parameter.

**Why one joint COLMAP reconstruction?** All captures depict one static scene from the same camera/lens. Putting all selected images into one SfM run gives every condition a common coordinate system and shared camera calibration. The condition split files then choose different training images while retaining the same validation/test images. If COLMAP cannot register the blur images, do not silently drop them: report registration rate and either improve capture quality or treat that condition as a failed-registration result.

### Scale

- 60–100 curated training candidates per condition.
- 36–48 holdout candidates from the dedicated reference pass.
- Approximately 250–400 total images in the joint SfM run.
- Keep the number of training images balanced across conditions when comparing trajectory or blur. View-count experiments use nested 100%, 50% and 25% subsets.

These are starting targets, not universal thresholds. The pilot capture determines the final counts.

---

## 1. Physical staging

1. Choose one rigid, matte, textured object. Avoid transparent, reflective, glossy, very dark, or highly repetitive objects for the first experiment.
2. Measure and record its width, height and depth. Photograph the object in its unpowered/unchanging state.
3. Place it on a rigid pedestal/table. **Never rotate the object or move the pedestal between capture conditions.** The camera moves around the object.
4. Make the background static and feature-rich: textured fabric, a detailed room, or a printed feature-rich backdrop. Keep it fixed for all captures. Avoid blank walls, mirrors, windows with changing daylight, moving people, fans, screens and flickering lights.
5. Use broad, diffuse, constant artificial lighting. Close blinds. Allow lights to warm up, then do not change their position or output during the main RGB capture series.
6. The pedestal and any calibration texture must remain stationary. Do not attach a checkerboard/newspaper to the object. COLMAP uses static background features; the foreground mask later excludes the background from model loss.
7. The underside hidden by the support is **not** part of the visible-surface reconstruction. If the research requires a complete 360° model including the underside, design a separate, explicitly documented second capture and solve alignment between the two sessions; do not pretend a single orbit observes hidden surfaces.

### Pilot acceptance

Before the full capture, make a pilot orbit. Extract frames and check:
- the object remains sharp at all elevations;
- the object is large enough in the image but not clipped;
- no highlights are clipped and shadows retain detail;
- no visible LED flicker/banding;
- adjacent frames have overlap but are not nearly identical;
- the background has stable trackable features;
- the main 1x lens remains selected.

Do not proceed until the pilot passes.

---

## 2. iPhone 16 Pro Max capture configuration

The exact control labels can vary with the installed Blackmagic Camera build and iOS release. Verify the displayed values on the phone before each take and record them in the metadata template.

### 2.1 Device settings

1. Clean the main camera lens.
2. In **Settings → Camera**, enable Macro Control if available, then disable automatic macro switching for the capture session. Keep the 1x main camera selected; do not zoom or switch lenses mid-take.
3. Disable Live Photos for stills. Turn off filters/Photographic Styles that alter the appearance between frames.
4. Use airplane mode or Do Not Disturb to prevent notifications interrupting a take.
5. Keep the phone cool; avoid long recordings in direct sun.
6. In Blackmagic Camera, confirm the Record codec, resolution, frame rate, color space and stabilization settings from Section 2.2 before every take.

### 2.2 Blackmagic Camera video settings

Use Blackmagic Camera for the controlled video captures because it exposes manual shutter, ISO, white balance and lens controls.

| Setting | Primary clean capture | Blur extension |
|---|---|---|
| Lens | Rear 1x main camera; never change lens mid-take | Same |
| Resolution | 4K UHD (3840×2160), landscape | Same |
| Frame rate | 25 fps | Same 25 fps |
| Codec | **Apple ProRes 422 HQ** (preferred) | Same ProRes 422 HQ |
| Color space | Rec.709 SDR; do not record Log for the main dataset | Same |
| Shutter | **1/50 s** | **1/50 s** (moderate blur) and **1/25 s** (stronger blur) |
| ISO | Manual; lowest value that gives a well-exposed image; lock and record it | Keep lighting/exposure comparable; record any ISO change |
| White balance | Manual fixed Kelvin matched to lights; lock | Identical |
| Focus | Manually focus on the object; lock focus | Identical |
| Exposure | Manual/locked; protect highlights and avoid underexposure | Match brightness as closely as possible; record any compensation |
| Stabilization | Blackmagic Camera Stabilization: **Off**; keep phone movement smooth | Same |
| Lens correction | Choose one setting (Auto is a reasonable default) and keep it identical | Same |
| Record LUT / proxy | Record LUT Off; proxy Off for the master dataset | Same |

**Why ProRes 422 HQ?** We have substantial cloud storage, and the goal is to preserve the best practical decoded video frames for frame extraction. Blackmagic Camera for iOS lists ProRes 422 HQ, ProRes 422, ProRes 422 LT, ProRes Proxy and HEVC as available recording codecs. Use the same codec for every experimental condition. If ProRes 422 HQ proves too large or the app cannot sustain it, use ProRes 422 for *all* conditions and document the change; do not mix codecs within the benchmark. ProRes is still processed video, not sensor RAW. It reduces compression but cannot restore motion-blurred, defocused or clipped detail. <https://www.blackmagicdesign.com/products/blackmagiccamera/techspecs>

**iPhone 16 Pro Max storage note:** Apple lists 4K ProRes up to 30 fps on internal storage and higher frame rates with compatible external storage. 25 fps is within the internal-storage 4K range for this model. <https://support.apple.com/en-sg/109041>

**Why 25 fps and these shutter settings?** Vietnam uses 50 Hz mains power. Initially, we thought a 1/100 s exposure is synchronized to 50 Hz mains and gives a sharper baseline. Check the pilot for LED banding/flicker; LED drivers can behave differently. The 1/50 s and 1/25 s captures deliberately increase exposure time and motion blur. Keep the camera path and speed the same. Adjust ISO downward in the slower-shutter conditions as needed to keep brightness similar, and record the exact ISO. This means ISO is a documented covariate, not an invisible setting. If cannot match exposure without reaching the app's ISO limit, document the brightness difference rather than silently changing lighting.

**Important stabilization correction:** set the Blackmagic Camera app's stabilization option to Off if exposed, but do not claim the iPhone's physical optical image stabilization (OIS) is disabled. Keep the app setting identical across conditions. Avoid Action Mode and lens switching. The app exposes Off/Standard/Cinematic/Extreme stabilization options; the Off setting is the reproducible app-level choice. <https://www.blackmagicdesign.com/uk/products/blackmagiccamera/techspecs>

PNG frames extracted from ProRes are lossless copies of the *decoded video frames*. They are not RAW sensor frames. Preserve the original `.MOV` masters.

### 2.4 Capture trajectories

Keep the object, pedestal, lights and background fixed. Move the camera around the object.

**A. `orbit_clean`**
- Three 360° rings around the object.
- Ring 1: camera near object mid-height, lens aimed approximately level or slightly upward.
- Ring 2: camera about 35–45° downward.
- Ring 3: camera about 65–75° downward.
- Walk slowly and smoothly; keep camera-to-object distance approximately constant.
- Do not stop and restart at each ring if a continuous orbit is practical. Avoid sudden tilts.

**B. `spiral_clean`**
- One continuous helical path around the object.
- Start low and gradually rise while increasing downward camera pitch.
- Make approximately three revolutions if practical.
- Maintain distance and speed as consistently as possible.

**C. `blur_1_50` and `blur_1_25`**
- Repeat the same orbit path, same direction, same approximate speed and same camera distance as `orbit_clean`.
- Change shutter to 1/50 s or 1/25 s and manually compensate ISO to maintain comparable brightness; record both values for every clip.
- Do not intentionally walk faster. The experiment is shutter/motion-blur sensitivity, not a combined shutter-and-speed experiment.

**D. `holdout` reference pass**
- Capture a separate standardized pass not used for training.
- Aim for 36–48 usable frames covering the same azimuth/elevation envelope as the object’s visible surface.
- A practical method is a slow two-level orbit plus a high-angle pass.
- These images become the shared validation/test cameras for every condition and every view-count ratio.

### 2.5 Capture log

For every take, fill in `configs/capture_metadata_template.yaml`. Record:
- capture ID; -> code when extract frames
- date/time; -> app auto
- operator; -> tbn
- object dimensions/material; -> manual
- app and app version; -> manual fill once
- iOS version; -> manual fill once
- lens, -> 1x fill for all
- resolution, frame rate, codec, -> retrieve from video
- shutter, ISO, WB Kelvin, -> manual/may be retrievable from video
- focus and stabilization options; -> manual
- lighting description and camera distance; -> manual
- trajectory and any deviations; -> manual
- original video filenames and SHA-256 checksums. ->

---

## 3. Windows workstation setup (local preprocessing)

The local workstation is used for file integrity, video decoding, frame auditing, mask drafts and Google Drive synchronization. Run COLMAP/Nerfstudio processing locally only if your Windows installation passes the preflight; Nerfstudio explicitly describes Windows support as less tested and more fragile than Linux. The most reproducible fallback is to run the COLMAP/Nerfstudio processing stage in Colab/Linux, then keep local Windows as the source-data workstation.

### 3.1 Install tools

Install:
- Python 3.12 (or a Python version supported by the installed packages);
- `uv`;
- FFmpeg and ensure `ffmpeg.exe` is on PATH;
- Google Drive for desktop (recommended for uploading and streaming files).

On Windows, install FFmpeg with winget and install Google Drive for desktop from Google's official page:

```powershell
winget install --id Gyan.FFmpeg.Shared -e
```

Install [Google Drive for desktop](https://www.google.com/drive/download/) and sign in with the Google account that owns the project's available storage. In Drive preferences, choose **Stream files** to avoid mirroring the full project onto your Windows disk. Reopen PowerShell after installation so PATH changes take effect.

Open a new PowerShell window and verify:

```powershell
python --version
uv --version
ffmpeg -version
```

Confirm Google Drive for desktop is running and that the project folder appears in File Explorer.

### 3.2 Create the preprocessing environment

From the project root:

```powershell
uv venv .venv --python 3.12
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements-preprocess.txt
```

If PowerShell blocks activation, use the environment interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip --version
```

Verify imports:

```powershell
python -c "import cv2, numpy, PIL; print('OpenCV', cv2.__version__)"
```

### 3.3 Create data directories

```powershell
New-Item -ItemType Directory -Force data\raw, data\candidates, data\curated, data\colmap_input\images, data\masks, data\processed, data\manifests | Out-Null
```

Keep raw files read-only after transfer. Do not put large raw files or trained checkpoints into Git.

---

## 4. Transfer and preserve raw files

1. Transfer original `.MOV` files from iPhone to `data/raw/<capture_id>/` using USB-C or AirDrop/iCloud Drive.
2. Do not transcode or rename the originals. If you rename a file, record the old/new name in the manifest.
3. Copy the original capture metadata YAML beside the files.
4. Generate checksums:

```powershell
python scripts\write_checksums.py --root data\raw --output data\manifests\raw_sha256.csv
```

Upload `data/manifests/raw_sha256.csv` and the YAML capture metadata to `00_documentation/manifests/` and `00_documentation/capture_metadata/` in Drive.

5. Upload the scene folder to the project's Google Drive `01_raw/` folder. Wait until Drive reports that syncing is complete. For a strong integrity check, download the uploaded file back to a temporary folder and compare SHA-256 against the local source using the commands below. Delete the phone copy only after the cloud copy is confirmed. If you cannot temporarily download a second copy, keep the phone original until you can verify it.

PowerShell checksum example (run once on the local original and once on a downloaded Drive copy):

```powershell
Get-FileHash .\data\raw\orbit_clean.mov -Algorithm SHA256
Get-FileHash .\drive_verify\orbit_clean.mov -Algorithm SHA256
```

The two SHA-256 values must match exactly. Drive's video preview is not an integrity test; ProRes may be stored correctly even if browser preview is unavailable.

---

## 5. Extract candidate frames and curate

### 5.1 Extract candidates

Start with 2 frames per second. Do not assume the desired final set is simply every Nth frame; this is only a candidate generator.

```powershell
python scripts\extract_video_frames.py --video data\raw\orbit_clean.mov --output data\candidates\orbit --fps 2
python scripts\extract_video_frames.py --video data\raw\spiral_clean.mov --output data\candidates\spiral --fps 2
python scripts\extract_video_frames.py --video data\raw\blur_1_50.mov --output data\candidates\blur_1_50 --fps 2
python scripts\extract_video_frames.py --video data\raw\blur_1_25.mov --output data\candidates\blur_1_25 --fps 2
python scripts\extract_video_frames.py --video data\raw\holdout.mov --output data\candidates\holdout --fps 2
```

Adjust paths to match your actual filenames. Extracted PNGs preserve decoded frame pixels, not the original sensor RAW data.

### 5.2 Audit image quality

```powershell
python scripts\audit_images.py --input data\candidates\orbit --csv data\manifests\orbit_audit.csv
python scripts\audit_images.py --input data\candidates\spiral --csv data\manifests\spiral_audit.csv
python scripts\audit_images.py --input data\candidates\blur_1_50 --csv data\manifests\blur_1_50_audit.csv
python scripts\audit_images.py --input data\candidates\blur_1_25 --csv data\manifests\blur_1_25_audit.csv
python scripts\audit_images.py --input data\candidates\holdout --csv data\manifests\holdout_audit.csv
```

The script reports Laplacian variance as a *relative sharpness indicator*. It is not a universal blur score and depends on resolution, texture, exposure, noise and focus. Do not automatically delete the lowest 5%. Open the images and inspect them.

### 5.3 Human curation

For each audit CSV:
1. Add a column named `keep`.
2. Mark `true` only for readable frames that are sharp enough, not clipped, not severely occluded, and add useful viewpoint coverage.
3. Avoid keeping long runs of almost identical adjacent frames.
4. Retain a separate rejected-frame list and the reason for rejection where possible.
5. Aim for 60–100 curated images per training condition and 36–48 holdout candidates.

Then copy curated images without altering the source:

```powershell
python scripts\select_frames.py --audit-csv data\manifests\orbit_audit.csv --output data\curated\orbit
python scripts\select_frames.py --audit-csv data\manifests\spiral_audit.csv --output data\curated\spiral
python scripts\select_frames.py --audit-csv data\manifests\blur_1_50_audit.csv --output data\curated\blur_1_50
python scripts\select_frames.py --audit-csv data\manifests\blur_1_25_audit.csv --output data\curated\blur_1_25
python scripts\select_frames.py --audit-csv data\manifests\holdout_audit.csv --output data\curated\holdout
```

For a fair trajectory/blur comparison, use approximately equal training counts across conditions. If one condition has fewer registered images, report that difference rather than silently balancing by deleting successful views after seeing the result.

---

## 6. Build one joint COLMAP input set

The merged folder uses unique names so that source-condition membership is explicit and no frame names collide.

```powershell
python scripts\merge_capture_images.py --output data\colmap_input\images `
  --group orbit=data\curated\orbit `
  --group spiral=data\curated\spiral `
  --group blur_1_50=data\curated\blur_1_50 `
  --group blur_1_25=data\curated\blur_1_25 `
  --group holdout=data\curated\holdout
```

The resulting filenames will begin with `orbit_`, `spiral_`, `blur_1_50_`, `blur_1_25_`, or `holdout_`.

**Critical assumptions for joint SfM:** all images must depict the same static scene; camera lens, resolution, crop, orientation and focal setting must be consistent. Do not merge images recorded with a different lens or a different video crop. If the camera app changes the effective crop between takes, separate calibration may be necessary and the common-coordinate comparison design must be revised.

---

## 7. COLMAP calibration and Nerfstudio conversion

### 7.1 Preferred processing route: Colab/Linux + Google Drive

Use Colab/Linux for COLMAP and Nerfstudio processing. This avoids relying on fragile native Windows Nerfstudio support. Keep the canonical dataset in Google Drive, but copy the working dataset into Colab's local `/content/` disk before COLMAP or training. Do not run image-heavy processing directly from the mounted Drive filesystem.

Create this folder in your Google Drive (the storage-owning account should upload the large files):

```text
MyDrive/CV3D_Final_Project/
├── 00_documentation/
├── 01_raw/
├── 02_candidates/
├── 03_curated/
├── 04_colmap_input/
├── 05_processed/
├── 06_experiments/
├── 07_final_results/
└── 99_archives/
```

In Colab, mount Drive and copy the code bundle and the current dataset archive to the local runtime:

```python
from google.colab import drive
from pathlib import Path
import shutil

drive.mount('/content/drive')
DRIVE = Path('/content/drive/MyDrive/CV3D_Final_Project')
WORK = Path('/content/project')
WORK.mkdir(parents=True, exist_ok=True)

# Keep the protocol bundle (scripts/) in 00_documentation/protocol_bundle/.
shutil.copytree(DRIVE / '00_documentation/protocol_bundle', WORK / 'protocol_bundle', dirs_exist_ok=True)
shutil.copytree(WORK / 'protocol_bundle/scripts', WORK / 'scripts', dirs_exist_ok=True)
```

For COLMAP, make a `.tar` archive of `data/colmap_input/` on Windows after the curated images have been merged. Upload it to Drive as `99_archives/colmap_input_v001.tar`. Also upload the final mask archive as described in Section 8.2. Then in Colab:

```python
archive = DRIVE / '99_archives/colmap_input_v001.tar'
shutil.copy2(archive, '/content/colmap_input_v001.tar')
```

```bash
!mkdir -p /content/project/data/colmap_input
!tar -xf /content/colmap_input_v001.tar -C /content/project/data/colmap_input
!find /content/project/data/colmap_input -maxdepth 2 -type d | head
```

Install COLMAP, FFmpeg and Nerfstudio in the Colab runtime:

```bash
!apt-get update -qq
!apt-get install -y ffmpeg colmap
!python -m pip install --upgrade pip
!python -m pip install nerfstudio
!ffmpeg -version | head -n 2
!colmap -h | head -n 5
!ns-process-data --help
!ns-process-data images --help
!ns-train --help
```

Colab's preinstalled PyTorch/CUDA stack changes over time. If `pip install nerfstudio` fails, follow the current official Nerfstudio installation instructions for that runtime rather than forcing an old CUDA wheel. Record `pip freeze`, `nvidia-smi` and the Colab GPU model in the experiment log.

### 7.2 Run one joint image-sequence reconstruction

From the Colab project directory:

```bash
ns-process-data images \
  --data /content/project/data/colmap_input/images \
  --output-dir /content/project/data/processed/joint_sfm \
  --camera-type perspective \
  --matching-method exhaustive \
  --num-downscales 2
```

`exhaustive` matching is appropriate as a starting point for a few hundred images. For much larger image sets, benchmark the default vocabulary-tree strategy or another supported matcher. Sequential matching is intended for ordered video sequences; because this folder combines multiple capture sequences and a holdout pass, exhaustive matching is the cleaner initial choice.

If COLMAP fails:
- inspect whether the image set contains blank/low-texture backgrounds;
- inspect blurred images and remove only images that are genuinely unusable, recording the reason;
- verify that every input image uses the same resolution/crop/lens;
- check whether COLMAP registered all condition groups and the holdout group;
- do not accept a reconstruction merely because the command exits successfully.

### 7.3 Calibration acceptance checks

Inspect the generated `transforms.json`, `colmap/` directory and Nerfstudio viewer/point cloud. Run the included registration audit to calculate submitted/registered counts overall and by capture condition:

```bash
python scripts/audit_registration.py \
  --input-images /content/project/data/colmap_input/images \
  --transforms /content/project/data/processed/joint_sfm/transforms.json \
  --output-csv /content/project/data/manifests/registration_by_condition.csv
```

Record:
- total images submitted;
- total images registered;
- registered fraction overall and per condition;
- sparse point count;
- reprojection error if available from COLMAP output;
- camera frustum distribution;
- obvious pose outliers or disconnected components.

For the first pilot, target at least 95% registration overall and at least 90% for each condition. These are project QA targets, not universal COLMAP standards. If a blur group is below target, report it as a meaningful failure or recapture; do not hide the failure.

**Do not run COLMAP separately for each training-view ratio.** All ratios reuse the one joint reconstruction so camera poses and world coordinates remain fixed.

---

## 8. Foreground masks

### 8.1 Generate draft masks on local Windows

Install the optional mask dependencies:

```powershell
uv pip install rembg onnxruntime
```

Then generate masks from the **same merged source images** used for COLMAP:

```powershell
python scripts\generate_masks.py --input data\colmap_input\images --output data\masks\draft
```

The script uses `isnet-general-use` and writes binary masks. CPU ONNX Runtime is the robust default; use `onnxruntime-gpu` only if you deliberately configure a compatible CUDA/cuDNN stack.

### 8.2 Inspect and correct masks

Every mask must be inspected. Correct:
- object holes incorrectly filled;
- missing thin structures;
- background fragments marked as foreground;
- pedestal/support included when it is not part of the target object;
- object boundaries that are too eroded or too dilated.

The required Nerfstudio convention is a single-channel binary image: **white (255) = keep/foreground; black (0) = ignore/background**. Mask dimensions must exactly match their corresponding RGB images. All frames must have masks if masking is enabled.

After manual correction, place final masks in `data/masks/final/` with the same filename stems as the merged images, e.g. `orbit_00001.png`. Upload this folder to Drive as `03_curated/masks_final_v001/`. For Colab, create an archive from PowerShell:

```powershell
tar -cf 99_archives\masks_final_v001.tar -C data masks/final
```

In Colab, after mounting Drive as in Section 7.1:

```python
shutil.copy2(DRIVE / '99_archives/masks_final_v001.tar', '/content/masks_final_v001.tar')
```

```bash
!mkdir -p /content/project/data
!tar -xf /content/masks_final_v001.tar -C /content/project/data
```


### 8.3 Attach masks by exact filename

Copy the final mask folder to the Colab project (or run locally if Nerfstudio processing was local), then:

```bash
python scripts/attach_masks.py \
  --dataset /content/project/data/processed/joint_sfm \
  --source-masks /content/project/data/masks/final
```

This script matches masks by exact filename stem. It does **not** assume sorted arrays align. It writes `masks/`, and matching `masks_2/` and `masks_4/` when `images_2/` and `images_4/` exist, then adds `mask_path` to every frame in `transforms.json`.

Run:

```bash
python scripts/validate_dataset.py \
  --dataset /content/project/data/processed/joint_sfm \
  --require-masks
```

Fix all errors before generating splits.

---

## 9. Fixed shared validation/test splits and view-count subsets

### 9.1 Why the split is designed this way

Do not use every eighth sequential video frame as the only evaluation set. Adjacent video frames can be near duplicates, causing an overly optimistic test score. Instead, use the dedicated `holdout_` pass. The split script uses deterministic farthest-point sampling over COLMAP camera centres to select validation cameras; the remaining holdout cameras are the fixed test set. The exact same validation and test filenames are reused across orbit, spiral and blur experiments.

This is stronger than splitting each capture condition independently. It also keeps all conditions in the same COLMAP world coordinate system.

### 9.2 Generate condition-specific view-count JSON files

Run once for each training condition:

```bash
python scripts/create_condition_splits.py --dataset data/processed/joint_sfm --condition-prefix orbit_ --holdout-prefix holdout_ --ratios 1.0,0.5,0.25,0.10
python scripts/create_condition_splits.py --dataset data/processed/joint_sfm --condition-prefix spiral_ --holdout-prefix holdout_ --ratios 1.0,0.5,0.25,0.10
python scripts/create_condition_splits.py --dataset data/processed/joint_sfm --condition-prefix blur_1_50_ --holdout-prefix holdout_ --ratios 1.0,0.5,0.25,0.10
python scripts/create_condition_splits.py --dataset data/processed/joint_sfm --condition-prefix blur_1_25_ --holdout-prefix holdout_ --ratios 1.0,0.5,0.25,0.10
```

The script writes JSON files such as:
- `transforms_orbit_views_100.json`
- `transforms_orbit_views_050.json`
- `transforms_orbit_views_025.json`
- `transforms_orbit_views_010.json`

Each JSON keeps the same `frames`, `val_filenames` and `test_filenames`; only `train_filenames` changes. It uses camera-centre diversity to choose nested training subsets. This is deterministic, but still visually inspect the resulting views: camera-centre spacing is not a guarantee of perfect angular/elevation coverage.

### 9.3 Validate every split

```bash
python scripts/validate_dataset.py --dataset data/processed/joint_sfm --transforms transforms_orbit_views_100.json --require-masks
python scripts/validate_dataset.py --dataset data/processed/joint_sfm --transforms transforms_orbit_views_050.json --require-masks
python scripts/validate_dataset.py --dataset data/processed/joint_sfm --transforms transforms_orbit_views_025.json --require-masks
python scripts/validate_dataset.py --dataset data/processed/joint_sfm --transforms transforms_orbit_views_010.json --require-masks
```

Repeat for spiral and both blur conditions. Check that no train/val/test filename overlaps and all paths exist.

---

## 10. Google Drive storage and group collaboration

### 10.1 Use one canonical Drive repository

Do not create an AWS account or S3 bucket for this project. Your existing Google Drive capacity is sufficient for the expected dataset. Use one account with the large storage allocation as the canonical uploader/owner, and share the project folder with the other three members. Files uploaded by teammates may count against the uploader's own quota, so agree that the storage-owning account (or one designated data manager) performs the large uploads. <https://support.google.com/drive/answer/9312312>

Recommended layout:

```text
MyDrive/CV3D_Final_Project/
├── 00_documentation/
│   ├── dataset_card.yaml
│   ├── capture_metadata/
│   ├── manifests/
│   ├── experiment_log.xlsx
│   └── protocol_bundle/
├── 01_raw/                         # Original ProRes .MOV masters; immutable
│   └── capture_id=YYYYMMDD_object01/
├── 02_candidates/                  # Extracted PNG frames
├── 03_curated/                     # Human-reviewed frames + audit CSVs + masks_final_v001/
├── 04_colmap_input/                # Merged, prefixed image set
├── 05_processed/
│   └── dataset_version=v001/
│       ├── images/ images_2/ images_4/
│       ├── masks/ masks_2/ masks_4/
│       ├── colmap/
│       ├── transforms.json
│       └── transforms_*_views_*.json
├── 06_experiments/
│   ├── nerfacto/
│   └── splatfacto/
├── 07_final_results/
│   ├── renders/
│   ├── metrics/
│   └── presentation/
└── 99_archives/                     # Transfer archives for Colab
```

### 10.2 Upload from iPhone / Windows safely

Preferred reliable sequence:

1. Finish one capture condition/scene on the iPhone.
2. Transfer the original `.MOV` file to Windows via USB-C or the Files app. Do not transcode it.
3. Calculate its SHA-256 checksum with `scripts/write_checksums.py` or `Get-FileHash`.
4. Upload the original into `01_raw/capture_id=.../` using Drive for desktop (streaming mode) or Drive web.
5. Wait for Drive for desktop to report that sync is complete. Download the cloud copy to a temporary verification folder and compare SHA-256. Keep the phone original until the hashes match.
6. Only then delete the phone copy to free space for the next scene. Keep the local PC copy until the next backup/checkpoint is safe.

If you upload directly from the iPhone Google Drive app, wait for the upload to finish and keep the phone original until you can verify a downloaded copy. Do not assume that seeing a filename in Drive means a full upload has completed.

Google Drive supports individual files up to 5 TB, subject to your account's available storage. Your project files should be much smaller than this, but ProRes can still consume space quickly. Check the remaining quota after each scene. <https://support.google.com/drive/answer/37603>

### 10.3 Drive for desktop settings

- Install [Google Drive for desktop](https://www.google.com/drive/download/).
- Choose **Stream files**, not Mirror files, to avoid downloading the entire project onto your local SSD.
- Mark only the current working subset as available offline if needed.
- Do not let multiple teammates independently edit/replace the canonical `05_processed/dataset_version=v001/` folder.
- Treat dataset versions as immutable: use `v002` if images, masks, poses or split files change.

Streaming keeps most files in the cloud and uses much less local disk than mirroring, though accessed files may be cached locally. <https://support.google.com/drive/answer/13401938>

### 10.4 Package datasets for Colab

For a few hundred images, a tar archive makes sequential copying into Colab easier than opening thousands of files over the mounted Drive filesystem. On Windows PowerShell (from the project root; built-in `tar` is available on current Windows versions):

```powershell
New-Item -ItemType Directory -Force 99_archives | Out-Null
tar -cf 99_archives\colmap_input_v001.tar -C data colmap_input
```

After processing, attaching masks, creating all split JSON files and validating on Colab, generate the SHA-256 manifest and archive the complete processed dataset:

```bash
!python /content/project/scripts/write_checksums.py \
  --root /content/project/data/processed/joint_sfm \
  --output /content/processed_sha256.csv
!tar -cf /content/joint_sfm_v001.tar -C /content/project/data/processed joint_sfm
```

Copy both files back to Drive:

```python
shutil.copy2('/content/joint_sfm_v001.tar', DRIVE / '99_archives/joint_sfm_v001.tar')
(DRIVE / '00_documentation/manifests').mkdir(parents=True, exist_ok=True)
shutil.copy2('/content/processed_sha256.csv', DRIVE / '00_documentation/manifests/processed_sha256.csv')
```

Do not repeatedly copy the whole raw-video archive into Colab. COLMAP and training need the curated/processed image dataset, not the original videos. Preserve raw ProRes masters in Drive; keep only selected processed data and experiment artifacts in Colab.

### 10.5 Collaboration and versioning rules

- One designated data manager controls the canonical raw and processed folders.
- `v001` is immutable after approval. Any change to frame selection, masks, camera poses or splits creates `v002`.
- Put `dataset_card.yaml`, checksums and capture metadata beside each approved version.
- Put code in GitHub; do not commit raw videos, large PNG sets or model checkpoints.
- Store each experiment under a unique run ID with config, logs, metrics and checkpoint.
- Avoid duplicate copies of the raw ProRes masters in multiple team members' Drives.

## 11. Dataset checksums and final validation

Generate a processed-file manifest after masks and all split JSON files are in place:

```bash
python scripts/write_checksums.py --root data/processed/joint_sfm --output data/manifests/processed_sha256.csv
```

Upload `data/manifests/processed_sha256.csv` to Drive at `00_documentation/manifests/processed_sha256.csv`. Verify after downloading the processed archive from Google Drive:

```bash
python scripts/verify_checksums.py --root data/processed/joint_sfm --manifest data/manifests/processed_sha256.csv
```

### Release gate: do not train until all are true

- [ ] Raw videos exist and have checksums.
- [ ] Capture metadata is complete for every condition.
- [ ] All images have the same resolution and lens/crop in the joint SfM set.
- [ ] Every curated image has been visually reviewed.
- [ ] `audit_registration.py` reports COLMAP registration rate overall and per condition.
- [ ] Camera frustums form a coherent path; no obvious pose outliers.
- [ ] Sparse reconstruction is connected and has usable points around the object.
- [ ] Every mask is one-channel, binary 0/255, and image-sized.
- [ ] Every image has exactly one correctly matched mask if masking is enabled.
- [ ] Train, validation and test filename sets are disjoint.
- [ ] The exact same holdout validation/test frames are used across conditions.
- [ ] All four view-count splits are validated.
- [ ] `transforms.json`, split JSON files and every referenced image/mask are included in the Google Drive dataset version.
- [ ] SHA-256 checksums pass after download.

---

## 12. Colab Pro: training NeRF and 3DGS

### 12.1 Runtime preparation

Select a GPU runtime in Colab. Record the exact GPU model and VRAM shown by:

```python
!nvidia-smi
!python --version
```

Install Nerfstudio and verify the CLI:

```python
!pip install -U pip
!pip install nerfstudio
!ns-train --help
!ns-eval --help
```

No AWS credentials or `awscli` are required. Mount Google Drive using the Colab Drive integration, then copy the archive into `/content/` before processing/training.

### 12.2 Obtain the immutable dataset

For a fresh Colab session, mount Drive, restore the scripts, copy the archive and checksum manifest to local runtime storage, extract, and verify. Do not train from the mounted Drive path:

```python
from google.colab import drive
from pathlib import Path
import shutil

drive.mount('/content/drive')
DRIVE = Path('/content/drive/MyDrive/CV3D_Final_Project')
WORK = Path('/content/project')
WORK.mkdir(parents=True, exist_ok=True)
shutil.copytree(DRIVE / '00_documentation/protocol_bundle', WORK / 'protocol_bundle', dirs_exist_ok=True)
shutil.copytree(WORK / 'protocol_bundle/scripts', WORK / 'scripts', dirs_exist_ok=True)
shutil.copy2(DRIVE / '99_archives/joint_sfm_v001.tar', '/content/joint_sfm_v001.tar')
shutil.copy2(DRIVE / '00_documentation/manifests/processed_sha256.csv', '/content/processed_sha256.csv')
```

```bash
!mkdir -p /content/project/data/processed
!tar -xf /content/joint_sfm_v001.tar -C /content/project/data/processed
!python /content/project/scripts/verify_checksums.py \
  --root /content/project/data/processed/joint_sfm \
  --manifest /content/processed_sha256.csv
!python /content/project/scripts/validate_dataset.py \
  --dataset /content/project/data/processed/joint_sfm \
  --require-masks
```

Confirm `transforms.json`, `images/`, `masks/`, `colmap/` and all `transforms_*_views_*.json` files are present. Copy the exact approved dataset version; do not silently replace it with a newer folder.

### 12.3 Smoke test first

Train each method for 100–500 iterations on the 100% orbit split before committing to full runs. This checks parser compatibility, masks, GPU memory and training startup.

```bash
ns-train nerfacto \
  --data /content/project/data/processed/joint_sfm/transforms_orbit_views_100.json \
  --max-num-iterations 500 \
  --vis tensorboard \
  nerfstudio-data --downscale-factor 2

ns-train splatfacto \
  --data /content/project/data/processed/joint_sfm/transforms_orbit_views_100.json \
  --max-num-iterations 500 \
  --vis tensorboard \
  nerfstudio-data --downscale-factor 2
```

If the runtime runs out of memory, use `--downscale-factor 4` for both models and document that resolution. Do not compare one model at 1080p and the other at 540p.

### 12.4 Full training commands

After the smoke test succeeds, use the same training budget and image downscale for both methods. Example for orbit 100%:

```bash
ns-train nerfacto \
  --data /content/project/data/processed/joint_sfm/transforms_orbit_views_100.json \
  --max-num-iterations 30000 \
  --vis tensorboard \
  nerfstudio-data --downscale-factor 2

ns-train splatfacto \
  --data /content/project/data/processed/joint_sfm/transforms_orbit_views_100.json \
  --max-num-iterations 30000 \
  --vis tensorboard \
  nerfstudio-data --downscale-factor 2
```

Repeat with the corresponding JSON files for:
- orbit: 100%, 50%, 25%, 10%;
- spiral: 100%, 50%, 25%, 10%;
- blur 1/50: 100%, 50%, 25%, 10%;
- blur 1/25: 100%, 50%, 25%, 10%.

This full factorial matrix can be expensive. If compute is limited, prioritize the core model comparison on orbit 100%, then trajectory 100%, then view-count on orbit, then blur 100%. Record skipped runs explicitly.

### 12.5 Evaluation

For each trained run, locate the generated `config.yml` and run:

```bash
ns-eval --load-config /content/outputs/EXPERIMENT_NAME/METHOD/RUN/config.yml \
  --output-path /content/outputs/EXPERIMENT_NAME/METHOD/RUN/test_metrics.json
```

Use the exact test filenames in the dataset JSON. Save the metrics JSON, training logs, config, runtime GPU, wall-clock time, and final checkpoint path. Nerfstudio's evaluation command provides image metrics supported by the installed version; report the exact metrics present in its JSON. If LPIPS/SSIM are not emitted by your version, calculate them with a separate, fixed evaluation script using the same rendered test images and reference images for both methods.

### 12.6 Fair-comparison rules

For every NeRF-vs-3DGS pair:
- same RGB images and masks;
- same COLMAP poses and camera intrinsics;
- same train/val/test filenames;
- same input resolution/downscale;
- same GPU runtime type where possible;
- same iteration budget (and report wall-clock time separately);
- same evaluation resolution and reference images;
- no depth supervision in the primary comparison;
- record software versions and all command-line arguments.

Do not describe `nerfacto` as the original NeRF implementation or `splatfacto` as the original Inria 3DGS code. They are Nerfstudio method implementations.

---

## 13. Optional LiDAR/depth extension (separate from the RGB baseline)

Do not make LiDAR a dependency for the primary project. Apple exposes LiDAR depth capture through AVFoundation on supported iPhones, but a synchronized RGB+depth capture requires an app/workflow that actually exports paired frames and depth maps. The standard Blackmagic video recording described above should not be assumed to save LiDAR depth maps.

A practical optional route is to make a separate Polycam LiDAR/Room-mode capture and export its raw data with RGB, camera metadata/poses and depth. Nerfstudio's Polycam importer has a `--use-depth` option in versions that support it. Example:

```bash
ns-process-data polycam \
  --data /content/project/data/raw/polycam_export.zip \
  --output-dir /content/project/data/processed/polycam_depth \
  --use-depth
```

Before relying on this extension, inspect the export and confirm that the depth maps are present, paired to the correct RGB frames, in a documented unit, and not merely a colored visualization. Nerfstudio data conventions expect 16-bit or 32-bit depth and use millimetres by default (`depth_unit_scale_factor=0.001`). Verify the actual Polycam export units before training.

Use `depth-nerfacto` only as a separate depth-supervised experiment. Do not compare it directly against RGB-only `splatfacto` as if the inputs were equivalent. A depth-aware 3DGS extension would require a deliberately implemented and documented depth loss or initialization method.

---

## 14. Research result table template

For every run, save one row in `experiments/results.csv` with at least:

| Field | Example |
|---|---|
| dataset_version | v001 |
| condition | orbit / spiral / blur_1_50 / blur_1_25 |
| train_ratio | 1.0 / 0.5 / 0.25 / 0.10 |
| model | nerfacto / splatfacto |
| train_count / val_count / test_count | integer counts |
| image_downscale | 2 or 4 |
| max_iterations | 30000 |
| gpu | exact Colab GPU model |
| wall_clock_seconds | measured |
| peak_vram_mb | measured if available |
| psnr / ssim / lpips | values emitted/calculated |
| render_fps | measured at fixed resolution and camera path |
| run_config_path | config.yml path |
| checkpoint_path | final checkpoint path |
| notes | failures, warnings, changes |

For rendering speed, fix output resolution, camera path, warm-up count and measurement interval. Report whether FPS includes model loading and video encoding. Do not compare a viewer's variable FPS readout across different resolutions.

---

## 15. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| COLMAP registers few images | Low texture, blur, repetitive object, poor overlap, lens/crop changes | Improve static background texture; increase overlap; use sharper frames; verify one lens and one resolution |
| Camera frustums form separate clusters | Disconnected reconstructions or poor loop closure | Check condition-specific registration; use exhaustive matching for this few-hundred-image set; inspect images |
| Mask edges are noisy | Automatic segmentation failure | Manually correct masks; use same corrected masks for both models |
| `attach_masks.py` reports missing stems | Processed images were renamed or source masks do not match exact source names | Inspect `transforms.json` image basenames and generate masks from the merged COLMAP input images; do not pair by sorted index |
| Nerfstudio cannot find images | Wrong `--data` path or relative file paths | Run validator; pass the split JSON file itself as `--data` (the parser supports an explicit JSON path) and preserve its sibling `images/` and `masks/` folders |
| Nerfstudio cannot find masks | Missing `mask_path`, wrong relative path, wrong mask dimensions | Run validator with `--require-masks`; inspect masks and downscale folders |
| CUDA out-of-memory | Image resolution/model batch too large | Use downscale factor 4 consistently; use a larger Colab GPU; do not change only one model's resolution |
| Results vary substantially | Different seeds, splits, resolution, GPU/runtime, preprocessing or lighting | Freeze manifests, JSON split files, configs and runtime details; repeat runs where possible |
| Drive upload appears complete but files differ | Upload interrupted or local/cloud copies differ | Download the Drive copy and compare the project SHA-256 manifest |

---

## 16. Official technical references

- Nerfstudio installation (including Windows caveats): https://docs.nerf.studio/quickstart/installation.html
- Nerfstudio custom data processing: https://docs.nerf.studio/quickstart/custom_dataset.html
- Nerfstudio data conventions (poses, masks and depth): https://docs.nerf.studio/quickstart/data_conventions.html
- Nerfstudio `ns-process-data`: https://docs.nerf.studio/reference/cli/ns_process_data.html
- Nerfstudio `ns-train`: https://docs.nerf.studio/reference/cli/ns_train.html
- Nerfstudio data parser and filename-based train/val/test splits: https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/dataparsers/nerfstudio_dataparser.py
- Nerfstudio `splatfacto`: https://docs.nerf.studio/nerfology/methods/splat.html
- COLMAP tutorial and matching strategies: https://colmap.github.io/tutorial
- Blackmagic Camera technical specifications: https://www.blackmagicdesign.com/products/blackmagiccamera/techspecs
- Apple LiDAR depth capture documentation: https://developer.apple.com/documentation/avfoundation/capturing-depth-using-the-lidar-camera
- Apple ProRes on iPhone: https://support.apple.com/en-sg/109041
- Blackmagic Camera iOS technical specifications: https://www.blackmagicdesign.com/uk/products/blackmagiccamera/techspecs
- Google Drive for desktop streaming and mirroring: https://support.google.com/drive/answer/13401938
- Google Drive file-size limits: https://support.google.com/drive/answer/37603

---

## 17. Exact execution order (short checklist)

1. Pilot capture and verify the camera settings.
2. Stage static object + static textured environment + constant lighting.
3. Record orbit, spiral, two shutter-blur conditions and separate holdout pass.
4. Transfer original `.MOV` files; write raw SHA-256 manifest.
5. Extract 2 fps candidate frames from every video.
6. Audit sharpness/readability; manually curate without deleting originals.
7. Merge curated images with condition prefixes.
8. Run one joint `ns-process-data images` COLMAP reconstruction.
9. Inspect registration rates, sparse cloud and camera frustums.
10. Generate draft masks; manually inspect/correct all masks.
11. Attach masks by exact filename stem.
12. Create condition-specific view-count JSON splits using the shared holdout pass.
13. Run all dataset validators; generate processed SHA-256 manifest.
14. Upload immutable raw/curated/processed versions to the canonical Google Drive folder; never overwrite approved dataset versions.
15. Copy the exact dataset archive from Drive to Colab local disk; extract and verify checksums.
16. Run 500-iteration smoke tests for both methods.
17. Run controlled benchmark matrix; save configs, checkpoints, metrics and logs.
18. Write report with failures and limitations, not only successful reconstructions.

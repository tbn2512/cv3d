# CV3D project protocol — quick start

1. Capture the same static object/scene with iPhone 16 Pro Max + Blackmagic Camera: 4K UHD, 25 fps, Apple ProRes 422 HQ, Rec.709, rear 1x lens, locked focus/WB/ISO, app stabilization Off. Clean baseline shutter 1/100 s; blur extensions 1/50 s and 1/25 s. Keep path/speed identical when testing shutter.
2. Upload each original `.MOV` to the canonical Google Drive `01_raw/` folder. Wait for sync and verify a downloaded copy's SHA-256 before deleting the phone original.
3. On Windows, use `uv` + `requirements-preprocess.txt` + FFmpeg. Extract 2 fps PNG candidates, audit them, curate by hand, and merge all conditions with unique prefixes.
4. Create one joint COLMAP reconstruction in Colab/Linux using `ns-process-data images`. Use the same camera/lens/resolution/crop for all images.
5. Generate masks from merged images, manually inspect them, attach by exact filename stem, and validate.
6. Generate nested 100/50/25/10% training subsets for each condition with the shared holdout pass. Validate every split.
7. Store canonical raw/curated/processed data in Google Drive. Archive curated/processed datasets as `.tar` for copying to Colab `/content/`; do not train directly from mounted Drive.
8. Smoke-test `nerfacto` and `splatfacto` for 500 iterations, then run controlled full experiments at the same resolution, split and iteration budget. Copy configs, metrics, renders and selected checkpoints back to Drive.

See `README.md` for exact commands, folder layout, validation gates, troubleshooting and optional LiDAR extension. No AWS/S3 account is required.

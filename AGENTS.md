# V-FloodNet cloud working instructions

Read `docs/CODEX_CLOUD.md` before installing or running inference.

- This fork's tested environment targets Linux x86_64, Python 3.10.16 and CPU
  PyTorch 2.2.2. Do not replace these with the latest runtime or install the
  upstream root `requirements.txt` into the cloud environment.
- Install: `bash .codex/setup.sh`. Default profile `cpu` installs image
  segmentation and the stop sign estimator, including Detectron2's CPU build.
  `VFLOOD_PROFILE=image bash .codex/setup.sh` installs just the image route.
- Always run Python through `bash .codex/run.sh ...`. Setup-shell activation
  does not carry over to another shell. The wrapper also sets headless plotting
  and limits CPU threads.
- Check: `bash .codex/run.sh .codex/check_environment.py --profile cpu --require-weights --smoke`.
  For the image-only profile, use `--profile image` instead.
- Image masks: `bash .codex/run.sh test_image_seg.py --test-path assets/img_exp --test-name demo`.
- Stop sign depths: `bash .codex/run.sh est_waterlevel.py --test-path assets/img_exp --test-name demo --opt stopsign --device cpu`.
  A negative upstream depth (`-1`) means estimation failed; it is not a water depth.
  Ask for or inspect the reference's actual physical dimensions before using
  estimates as measurements. The built-in dimensions may not match local signs.
- Do not claim the human, video or interactive reference routes are configured
  by the CPU setup. Human meshes require CUDA, MeshTransformer and separately
  licensed SMPL files. The original video entry point requires CUDA. Interactive
  reference tracking needs precomputed calibration files on a headless machine.
- Do not fabricate depths, training results, ground truth or GPU availability.
  Report imports, inference, and accuracy validation separately. Segmentation
  alone provides a water mask, not depth in cm or m.
- Models, input photos, data, generated results, virtual environments, caches
  and credentials must stay out of Git. Use `records/`, `WaterDataset/` and
  `output/` (all ignored), and provide links to actual output files.
- When dependencies change, regenerate `.codex/requirements-cpu.lock` with the
  pinned uv 0.9.7 and run `pip check`, the environment check and a real image
  smoke inference. Do not silently bypass failed installations or downloads.

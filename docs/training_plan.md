# vidHMR: initial training pipeline (plan)

High-level steps only. We fill in the details and the low-level decisions after discussion.
Each task lists what it does, what we decided, and what is still open (my recommendation first).

**Goal:** per-frame SMPL (pose, shape, camera-frame translation) for one selected person, from
frozen video-DiT features, with the ACE-Ego-Hand head structure. A mask prompt on one frame
selects the person. Gravity comes later (GeoCalib post-alignment), not from the model.

**Note:** ACE-Ego-Hand ships inference code only. We take its architecture from
`external/ACE-Ego-Hand/ace_ego_hand/archs/`, but we must write the losses and the training loop.

## What we have

- 3DPW: 60 sequences (24 train, 12 validation, 24 test), GT SMPL, cameras, valid flags, masks.
  One sample = one 81-frame window × one person: 308 train, 134 validation, 502 test items
  (without `outdoors_crosscountry_00` and the items with no valid latent frame; 74,620
  person-frames, 41.5 person-minutes; only 7 different actors). Men: 57 % of the valid train
  person-frames, 43 % of validation, 100 % of test.
- Frozen features for all 60 sequences: Wan2.2-Fun-5B (6 blocks, 3072-d) and Cosmos3-Edge
  (6 layers, 2048-d). 26 × 15 tokens per latent, 1 latent per 4 frames, 81-frame windows.
- SAM 3D Body predictions and metrics on all 60 sequences (the baseline).
- The viewer, which can show new predictions next to the GT.

## Tasks

### 1. Labels and dataset

Do:

- Precompute the targets per person and frame.
- Write a clip dataset that reads the feature memmaps.

Decided:

- Data: 3DPW only, official splits (train on train, select on validation, report on test).
  Other datasets come later; each needs a new feature extraction (≈ 295 GB now for 60 sequences
  and 2 backbones).
- Sample (one dataset item): one extraction window (81 frames, 21 latents) × one person, with
  the features of one layer, K, the targets of the 81 frames, and the prompt sources (GT mask,
  2D joints). A window with two people gives two items. Random sub-clips can come later.
- Target rate: one output per frame for all 81 frames in one forward pass (as ACE), and the
  losses use all 81 frames.
- Body model: the model predicts neutral SMPL (no gender input, like SAM 3D Body's single MHR
  model). The joint losses use the gendered GT joints; the betas loss uses neutral betas
  fitted once to each GT person (`data/smpl/SMPL_NEUTRAL.pkl`). Known conflict: the neutral file
  shares the female joint regressor but not the male one. With the GT pose, the neutral body is
  ≈ 1 mm from the GT joints for women and ≈ 12 mm for men (≈ 7 mm on the posed vertices). So for
  men the parameter targets and the joint targets disagree, and both losses cannot reach zero
  together (step 4).
- Targets are in the camera frame. The 3DPW world frame (body translation t, camera [R | T])
  is used only once, to make the targets; the model never sees it.
- Targets, precomputed in `generated/`:
  - global rotation in the camera frame (R × world global rotation), 23 joint rotations,
    neutral betas;
  - translation target = the pelvis point p in the camera frame, R(J0 + t) + T (the same point
    for any body model). The SMPL translation is then p − J0 of the predicted body;
  - 3D joints, 2D joints, valid flags, GT masks;
  - vertices: not stored; the gendered GT mesh (from `gender`, `betas_gendered` and the GT
    pose) is for the PVE evaluation (task 6), there is no vertex loss (task 4).
- Bad frames (all stay in the input window; only the losses ignore them):
  - invalid frames (3DPW valid = 0): zero loss;
  - camera jumps: zero loss on both frames of a GT camera turn > 10° between two frames. This
    is in 11 of 60 sequences (2 train, 4 validation, 5 test), 1–16 jumps each; most move the
    projected GT body > 50 px off the person, and the valid flags do not mark them;
  - `outdoors_crosscountry_00` (validation): GT behind the camera in all frames; left out, as in
    the SAM 3D Body evaluation;
  - no extra 2D-error mask (it would also drop good frames with bad 2D keypoints);
  - no ±1-frame extension of the jump mask: `outdoors_slalom_00` frame 142 (train, 9.1° turn,
    ≈ 250 px off) and `outdoors_parcours_01` frame 1004 (validation, ≈ 100 px off) stay valid.
- Last windows: kept. The last window of a sequence (33 of 308 train items) overlaps the window
  before it and takes some latents from the earlier DiT pass.
- Built: `vidHMR.datasets.pw3d_targets` (targets, `generated/targets/pw3d/`),
  `vidHMR.datasets.pw3d_windows` (items, loader), `tests/test_pw3d_windows.py`,
  `notebooks/pw3d_dataloader.ipynb`. Items: train 308, validation 134, test 502. An item needs
  at least one valid latent frame (`first_frame + 4k`), for the prompt; this leaves out 4 test
  items with 6 valid frames between latents (step 6 must still score those frames). Losses must
  compare rotations as matrices or 6D (the axis-angle of an upright body flips sign).
- Loading: one backbone and one layer per run (set in the config), loaded once into RAM at the
  start (Wan block 15: 10.7 GB train + 5.1 GB validation; Cosmos layer 14: 7.2 + 3.4 GB). Items
  cut their 21 latents from these arrays; `num_workers=0`. Streaming from disk with a
  DataLoader comes later, only if the data grows.

Note: frozen features allow no image augmentation. Flips would need a second feature pass.

### 2. Model (ACE-Ego-Hand structure, SMPL instead of MANO)

Do: write the decoder below (a reduced copy of ACE's `MemorySegmentBetasTransformer` and
`MemoryAlternatingEncoder`, `external/ACE-Ego-Hand/ace_ego_hand/archs/projector/`).

Decided:

- Frozen features only: the model reads the stored features and the DiT does not run in
  training. A LoRA (as ACE) comes later, only if the frozen results stop improving (it needs the
  5B DiT forward and backward on every window, which does not fit a 4090 easily).
- Known K (GT intrinsics) everywhere: the input ray encoding, the translation decode and the 2D
  reprojection loss. No ray-prediction head. Reason: all 60 3DPW sequences use one camera (44
  portrait, 16 landscape with the same K turned by 90°), so a K-free model would only learn that
  one camera. K-free (as ACE's second config) comes with datasets that have other cameras.
- Terms: a **latent** is one of the 21 time steps of the DiT features of a window (latent k holds
  frames 4k − 3 .. 4k and is aligned with frame `first_frame + 4k`). The **encoder conditioning**
  of a latent is its 26 × 15 DiT tokens (3072 channels), reduced to 384 channels by Linear +
  LayerNorm (as ACE), plus a learned 2D position encoding (ACE's 16 × 16 grid, resized
  bilinearly to the token grid, so a landscape grid is a stretch of it) and the ray encoding
  from K.
- **Mask tokens** (task 3): 390 tokens (384 channels) made from the prompt mask and the encoder
  conditioning of the prompt latent.
- **Query tokens**: 5 per latent, (21, 5, 384) per item: 1 person slot token + 4 register tokens
  (VGGT / "Vision Transformers Need Registers"). All latents start with the same learned tokens.
  No joint query tokens (ACE has 21 per hand; they are its 2D joint head, which we left out).
- 4 alternating layers (hidden size 384, 8 heads), each:
  1. cross-attention per latent: the 5 query tokens read the latent's 390 encoder conditioning
     tokens + the 390 mask tokens (780 keys; the mask tokens get a learned "memory" vector);
     pre-LayerNorm, residual, no feed-forward;
  2. full self-attention over all 105 query tokens of the item (all latents, both directions),
     RoPE with the latent index as the position, then a feed-forward block (384 → 1536 → 384).
  The encoder conditioning and the mask tokens do not change in these layers. The keys and
  values of the mask tokens are computed again for each latent (+55 % decoder cost); sharing
  them is a later speed-up.
- After the 4 layers, a final LayerNorm (ACE `final_norm`). Interpolation: the person slot
  token, (21, 384), is interpolated linearly in time to (81, 384) (latent k → frame 4k), as ACE.
  Then a residual LayerNorm → Linear → GELU on each frame (ACE `hand_fuse`), and the heads.
- Heads (as ACE: Linear 384 → 192, GELU, Linear 192 → out):
  - global rotation 6D and 23 joint rotations 6D per frame → Gram–Schmidt → rotation matrices;
  - camera per frame: u, v (pelvis in the image, −1 .. 1) and log depth; as ACE, the last
    layer starts with zero weights and a bias at the image centre and a start depth (ACE: 0.5 m
    for hands; ours: 3.4 m, the median train pelvis depth); the betas head starts at 0;
  - betas: mean of the 81 frame vectors → LayerNorm → MLP → 10, one set per window.
- Translation decode: inverse projection. The pelvis is the point on the ray (from K) through
  the predicted pixel, at the predicted depth. Mixed PnP (as ACE: keep the depth, fit x and y
  so that all projected joints fall on the predicted 2D joints, inverse projection as the
  fallback) is a later experiment, after the overfit milestone.
- Sizes from the ACE config (`options/ace_ego_hand_k.yml`): tap layer 15, hidden size 384, 4
  alternating layers, 8 heads, 4 register tokens, temporal RoPE.

- Start of the rotation heads: the PyTorch default init, as ACE, so the first rotations are
  arbitrary (decided by the user, 2026-11-26). Why it was a question: a fit of SMPL parameters
  with our losses (no network, `tests/test_model.py::test_losses_fit_a_body_to_the_gt`) reaches
  the GT from starts within 90° of it, but from a fully random start 6 % of the rotations stay
  > 90° off (both rotation terms are flat at 180°, and on the trunk a wrong parent and child
  rotation can cancel in the joint terms; videos from `scripts/fit_video.py`). Experiment: 16
  random portrait train items overfit in one batch, 2000 epochs, ACE start against a mean
  train pose start (rotation head biases at the mean rotations, weights 0). The network does
  not get stuck: with the ACE start, 86 % of the rotations start > 90° off, 0.5 % at epoch 100,
  0 % from epoch 250. The mean pose start is faster early (MPJPE 32 against 57 mm at epoch 250)
  but ends the same (6.9 against 7.1 mm, rotation 1.3° against 1.8°, one seed). Watch
  `rotations_over_90_pct` in the full-split run.

### 3. Prompt

Decided:

- The prompt is a mask, not points (one point is ambiguous: in 4–64 % of the frames of the
  two-person train sequences, the two people share a feature token). For users, one click in SAM 2
  gives the mask.
- On one latent only: the first latent whose frame is valid (the GT mask of an invalid frame can be
  off the person). The model must follow the person through the other latents. Prompt latent > 0 in
  7 train, 4 validation and 44 test items.
- Prompt mask: 0/1 at the DiT input size (832 × 480 portrait, 480 × 832 landscape; the same
  resize as the frames), stored for every 4th frame only (prompt latents are on those frames),
  (≈ 2.4 GB of RAM for train). No −10/+10 and sigmoid
  step (SAM 2 needs it only for its own predicted masks).
- Mask encoder: a copy of SAM 2's memory encoder (`sam2/modeling/memory_encoder.py`, SAM 3 uses
  the same):
  1. mask downsampler: 5 × (conv 3 × 3, stride 2, padding 1; LayerNorm2d; GELU), channels
     1 → 4 → 16 → 64 → 256 → 1024, size 832 × 480 → 26 × 15 (SAM 2 has 4 for stride 16; one
     of our tokens is 32 × 32 mask pixels), then a 1 × 1 conv to 384. As in SAM 2, the
     3 × 3 / stride 2 / padding 1 convs centre each output on the token's top-left corner, so a
     mask token mixes its own block with its neighbours' (half a token off; the network can
     learn it). Kernel 2 or 4 would align exactly; kept as SAM 2 for now;
  2. fuse: 1 × 1 conv on the prompt latent's encoder conditioning, plus the mask result;
  3. fuser: 2 ConvNeXt blocks (depthwise 7 × 7, LayerNorm, 384 → 1536 → 384, layer scale from
     1e-6, residual);
  4. no output projection (SAM 2 reduces to 64 channels only for its 7-frame memory);
  5. no separate position encoding on the mask tokens (SAM 2 adds a sine encoding to its memory
     keys): they already hold the position through the fused encoder conditioning.
- Built: targets key `prompt_masks` (every 4th frame, DiT input size, resized from the
  full-size mask); the item has `prompt_mask` (832 × 480 bool) and `prompt_latent`. Shown in
  `notebooks/pw3d_dataloader.ipynb`, section 8.
- Known risks: the GT mask shows the body outline of that frame; GT masks are the SMPL body
  only, SAM 2 masks also have hair and clothes. The test numbers against SAM 3D Body must use
  SAM 2 masks. Point prompts can come back later.

### 4. Losses

Do:

- Rotations, betas, 3D joints, translation, 2D reprojection (details below).
- Only on valid frames. A temporal smoothness term is optional.

Decided:

- Rotation loss: the 24 rotations of each valid frame (camera-frame global rotation + 23 joint
  rotations) as matrices; geodesic angle + matrix MSE with equal weights, as ACE. All 24
  rotations have the same weight (no extra weight on the global rotation).
- Shape loss: ℓ1 on the 10 neutral betas (one set per window, from the shape head) against
  `betas_neutral`, once per item (the shape is correct on every frame), weight 0.1 as ACE.
  Limit: train has only 3 body shapes (2 women, 1 man) and test has 5 men, 4 of them not in
  train, so the shape head can only recognize the train people. Expect a large shape error on
  test; more shapes need other datasets (for example BEDLAM).
- 3D joint loss: the 24 joints from the neutral SMPL forward pass (predicted rotations and betas)
  against the gendered GT `joints_3d`, ℓ1 on x, y, z, valid frames only, two terms:
  root-relative (pelvis subtracted from both; the pose and bone lengths) and camera frame (with
  the predicted pelvis point added). No separate root term: the pelvis point is the translation
  loss.
- Translation loss: ℓ1 on the predicted pelvis point (camera frame, meters, from the inverse
  projection of the predicted pixel and depth) against the GT `pelvis`, valid frames only, as
  ACE (weight 1). The gradient goes through the decode into the pixel and depth heads. No extra
  pixel or log-depth loss: the 2D loss and the camera-frame joint term also push the pelvis.
- 2D loss: the 24 predicted camera-frame joints projected with K, against `joints_2d`, ℓ1 on
  image coordinates from 0 to 1 (pixel / image width or height, as the ACE code), valid frames
  only. It makes the body fall on the person; the 3D terms handle the depth. No separate 2D joint
  head for now (ACE needs it only for mixed PnP; it comes back with mixed PnP).
- No presence loss (ACE's existence and visibility scores): the prompted person is always in the
  window, and no valid train frame has an empty mask. It can come with datasets where the person
  leaves the image.
- No temporal loss for now (later): the linear interpolation from 21 latents to 81 frames already
  limits the jitter; task 6 measures the acceleration error. Options for later: ACE's ℓ1 on the
  predicted joint acceleration (no GT, but it damps real fast motion), or predicted vs GT
  acceleration (needs three valid frames; breaks at camera jumps and time cuts).
- No ray losses: known K, no ray head (ACE's ray loss trains its ray head; its fit loss is
  K-free only).
- Weights (the ACE values; check them after the first runs): rotation 1 (geodesic) + 1 (matrix
  MSE), betas 0.1, root-relative joints 10, camera-frame joints 5, translation 1, 2D 1.
- No vertex loss for now (later, if needed): ACE and HMR 2.0 do not use one; the rotation and
  betas losses cover twist and shape, and for men the neutral mesh is ≈ 7 mm from the male GT
  mesh. `gender` and `betas_gendered` stay in the item for the PVE evaluation (task 6).

Reference: the ACE losses (arXiv 2608.20308, Eq. 6 and App. B.3), all summed:

- rotations: geodesic angle 1 + rotation-matrix MSE 1 (no loss on the 6D values); betas ℓ1 0.1;
- 3D joints ℓ1: root-relative 10, camera frame 5, wrist (root) 2;
- 2D: soft-argmax 2D joints and projected MANO joints under the GT camera, 1 (wrist 0.5);
- translation ℓ1 1, with the gradient through the PnP solve;
- presence: BCE on hand existence 0.5 and visibility 0.25;
- temporal: ℓ1 on the 3D joint acceleration, 0.5;
- rays: cosine distance of the predicted ray field to the K rays, 1 (K-free adds a fit loss, 5);
- out-of-view hands get the full loss; no vertex loss is stated.
- Training: AdamW, weight decay 0.01, gradient clip 1.0, cosine decay, 200 warmup steps, 20k
  steps, 64 clips per batch; learning rate 2e-4 for the decoder and heads.

### 5. Training loop

Do:

- Config file, AMP, checkpoints and logs in `generated/`, one RTX 4090.
- Overfit milestones, then train on the train split:
  1. One person (the base case): one item, `outdoors_climbing_00` window 0 (a woman, all 81
     frames valid, portrait; a woman, so the neutral body can reach the GT joints, ≈ 1 mm).
  2. Two people: the two items of `courtyard_arguing_00` window 240 (a woman and a man; the same
     features, only the prompt mask differs, so the model must use the prompt).
  Done when: root-relative MPJPE (24 joints, pelvis subtracted) < 10 mm, pelvis error < 2 cm,
  2D error < 5 px, and the bodies are on the correct people (the videos of
  `scripts/prediction_video.py`). The man in 2: MPJPE < 20 mm and 2D < 8 px, because the neutral body
  with his exact GT pose is already 12 mm and 5.4 px off.

Decided:

- Optimizer as ACE (App. B.2): AdamW, weight decay 0.01, learning rate 2e-4 (decoder and heads),
  warmup over the first 10 epochs (ACE: 200 steps), cosine decay to 0, gradient clip 1.0.
- Batch 16: one epoch = 20 batches (19 portrait, 1 landscape with the 7 landscape items).
  250 epochs, validation every 10 epochs, keep the checkpoint with the lowest validation MPJPE.
  The config seed sets the model start and the batch order. Weight decay applies to all
  parameters (LayerNorm, biases, position grid, query tokens too; the ACE repo has only
  inference code, the paper gives only AdamW with 0.01; decided by the user, 2026-11-26).
  (ACE's 20k steps × 64 clips would show each of our 308 train items about 4,150 times.)
- Logging: MLflow, local only (SQLite file and artifacts in `generated/mlflow/`, `mlflow ui`
  through a VS Code port). Each loss term, the total, the learning rate, the gradient norm
  (before the clip), the training errors of `vidHMR.model.losses.metrics` (with the rotation
  errors), each a mean over the batches of a logged epoch; the validation errors; the run
  config and the steps per epoch as parameters.
- Validation: every 10 epochs, all 134 validation items, the joint errors of
  `vidHMR.eval.sam3d_pw3d`. Checkpoints: `best.pt` (lowest validation MPJPE) and `last.pt`
  (saved at each validation and at the last epoch, so at the end the final weights; both with
  the epoch and the validation errors; continuing a stopped run needs the optimizer and
  schedule state later).
- Built (milestone 1): `vidHMR.model.decoder` (the decoder), `vidHMR.model.body` (the decode),
  `vidHMR.model.losses` (losses and metrics), `vidHMR.model.fit` (the direct fit of the loss
  check), `vidHMR.train`, `vidHMR.infer` (a run on all windows of a sequence, joined),
  `tests/test_model.py`, `notebooks/overfit_one_person.ipynb`, `scripts/fit_video.py` (videos
  of the loss check), `scripts/prediction_video.py` (videos of an infer output). The decoder runs in bf16 under autocast, but its frame
  interpolation and heads in float32 (bf16 outputs cost ≈ 9 mm of pelvis error).
- Milestone 1 result (outdoors_climbing_00 window 0): passed, MPJPE 3.5 mm (24 joints, pelvis
  subtracted), pelvis 4.1 mm, 2D 1.8 px after 2000 steps (≈ 4 min on the RTX 4090).
- Milestone 2 result (courtyard_arguing_00 window 240, `configs/overfit_two_people.toml`):
  passed. Woman: MPJPE 1.6 mm, pelvis 1.9 mm, 2D 1.0 px. Man: MPJPE 9.0 mm, pelvis 1.9 mm,
  2D 4.3 px. The two items have the same features and differ only in the prompt mask, so the
  model uses the prompt; the videos show each body on the prompted person.
- Built (full split): one loop in `vidHMR.train` for a config's `items` (the milestones,
  `configs/overfit_*.toml`) or the train split (`configs/train.toml`). The schedule counts
  steps (epochs × batches per epoch). Validation every `validate_every` epochs on all
  validation items with the 12-joint errors of `vidHMR.eval.sam3d_pw3d`; `best.pt` and
  `last.pt`. The run stops if the loss is not finite. Validation is not on the same frames as
  that evaluation: the frames of a sequence's last (overlapping) window count twice, the
  camera-jump frames are out, and the 1–4 frames after the last window are not scored (fine
  for model selection, not for final numbers). Smoke run: one
  epoch ≈ 8–9 s, one validation ≈ 1–2 s, the start (16 GB of features into RAM) ≈ 1 min, peak
  RAM 21 GB, so 250 epochs ≈ 35–40 min. Milestone 2 with this loop gives the same result.

- First full-split runs (2026-11-26, `configs/train.toml` with only `name`, `backbone` and
  `layer` changed; configs and logs in `generated/runs/<name>/`, about 32 min each). Best
  validation errors (12 joints, mm) and the epoch of `best.pt`; the training MPJPE at epoch 250
  is 15–17 mm for every run:

  | run | MPJPE | PA-MPJPE | best epoch |
  |---|---|---|---|
  | Wan layer 5 | 216.9 | 123.1 | 40 |
  | Wan layer 10 | 209.2 | 119.0 | 90 |
  | Wan layer 15 | 209.5 | 119.7 | 80 |
  | Wan layer 20 | 217.7 | 125.8 | 70 |
  | Wan layer 25 | 220.6 | 123.3 | 90 |
  | Wan layer 29 | 220.1 | 126.7 | 30 |
  | Cosmos3-Edge layer 27 | 221.3 | 127.8 | 20 |
  | one constant mean pose (no model) | 239.1 | 123.7 | |
  | SAM 3D Body | 56.5 | 36.4 | |

  Every run memorizes the train items and does not generalize: the validation PA-MPJPE is at
  the level of a constant pose, and the validation error stops falling after 20–90 epochs.
  The layer choice changes the result by only ≈ 12 mm. Next steps: open (the user decides).

  Diagnosis (Wan layer 10, `best.pt`, same 12-joint validation code on each split):
  train 54.4 / 42.9, validation 209.2 / 119.0, test 259.3 / 126.1 (SAM 3D Body: 39.2 / 26.4,
  56.5 / 36.4, 60.7 / 41.4). `last.pt` (epoch 250) reaches 17.3 / 14.2 on train. The error on
  the latent frames (4k) is the same as on the interpolated frames, so the time interpolation is
  not the cause. 3 of the 4 validation people are in train: the model learns the sequences, not
  a general pose. Data: train 24 sequences, 308 items, 3 people; validation 11 / 134 / 4; test
  24 / 502 / 5.

- Train + test experiment (2026-11-26, `generated/runs/wan_layer_10_train_test/`, config key
  `train_splits = ["train", "test"]`, 810 items, batch 32, else as Wan layer 10; the test split
  is training data here, so it is not a clean test set for this model): best validation 188.1 /
  115.5 at epoch 150 (train-only: 209.2 / 119.0); training MPJPE at epoch 250: 29.2 mm. More 3DPW
  data gives a small gain only; the decision is to add a large data set (BEDLAM or BEDLAM 2.0).
  Remove `train_splits` again if this experiment is not kept.
- Training speed (same run): the loading was the limit (one CPU thread stacked 50 MB per item,
  66 items/s). Now: features pinned in RAM, one copy per item to the GPU, the next batch copied
  on a side stream (`on_gpu`), TF32, fused AdamW: 106 items/s, GPU at 100 %, 32 min for 250
  epochs of 810 items (was ≈ 85 min). Worker processes were slower; `torch.compile` not used
  (1.5–2 min per batch shape).
- Feature storage (checked with the `best.pt` of both runs on validation): int8 with one scale
  per channel (per sequence) changes MPJPE and PA-MPJPE by ≤ 0.02 mm; relative error 0.6–0.7 %
  (layers 5, 10, 25, 29), 1.3 % (15), 2.6 % (20). float8 e4m3 changes ≤ 0.04 mm.
  Built (2026-11-26): every stored feature is `layer_XX_int8.npy` + `layer_XX_scale.npy`
  (`vidHMR.features.to_int8`; `run` writes it at the end of a sequence); `train.predict`
  dequantizes on the GPU. All 3DPW layers of both backbones were converted (the float16
  `layer_XX.npy` files are still on disk). Regression: `wan_layer_15` `best.pt` gives
  209.52 / 119.70 (was 209.49 / 119.70). The dataset keeps the prompt masks bit-packed in RAM,
  and the prompt latent must also have a non-empty mask.

- BEDLAM (decisions 2026-11-26, the user): scaling with data. Add one 81-frame window per
  BEDLAM sequence, 3,200 sequences (fits in RAM as int8), a seeded random start (multiple of 4),
  Wan layer 15, train on BEDLAM + 3DPW train (3DPW test stays clean), validate on 3DPW
  validation. Runs: 3DPW only (`wan_layer_15`), + 1,000, + 3,200 windows (config key
  `bedlam_sequences`, which replaced `train_splits`).
  - Data: Hugging Face `Intelligent-Systems/BEDLAM` (MP4 30 fps, masks, camera GT; 51 GB, all
    checksums good) and `smpl_ground_truth.zip` from the BEDLAM site (neutral SMPL, 11 betas).
    10,450 sequences, 1,611,916 frames, 117–264 frames each, 1–10 people. The 5 `zoom` scenes
    (1,600 sequences) are left out: their focal length changes by ≈ 95 % in a sequence, and the
    model takes one K per item.
  - `vidHMR.datasets.bedlam`: a window is named `bedlam/<scene>/<seq>` and stored like a 3DPW
    sequence of 81 frames (targets in `generated/targets/bedlam/`, features in
    `generated/features/<backbone>/bedlam/`). Camera conversion ported from the official
    `df_full_body_smpl.py` (closeup scenes: camera roll + 90°, sensor sides swapped; their masks
    are stored unrotated and are turned 90° clockwise). GT joints from the 11-beta body; the
    target betas are the 10-beta neutral refit (1–4 mm). valid: all joints > 0.1 m in front and
    at least one visible mask pixel. Prompt masks: the BEDLAM visible body + clothing masks
    (they include occlusion by people and objects).
  - Checks: projected GT mesh vs. BEDLAM masks (closeup, orbit, static, pitch-down scenes):
    mask inside the mesh 0.84–0.96, IoU 0.82–0.86, best at frame offset 0 (vs. −1, +1); the
    GT through `decode` gives ≈ 0 loss (joints 0.8–1.6 mm).

Open:

- Curriculum (later): train first on the one-person sequences (14 sequences, 159 items), then
  add the two-person sequences (10 sequences, 149 items), where the prompt must select a person.

### 6. Evaluation

Do:

- Reuse `vidHMR.eval.sam3d_pw3d` (same joints and protocol) for a direct comparison with SAM 3D
  Body. Add PVE and acceleration error.
- Use validation for model selection, test only for final numbers. SAM 3D Body saw 3DPW train.
- Show the predictions in the viewer.

### 7. First experiments

Do:

- Wan vs Cosmos, and which layer(s).
- With vs without temporal context (the main claim of the project).
- The effect of the prompt on the multi-person sequences (3DPW-PC).

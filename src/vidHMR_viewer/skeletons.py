"""Joint names and bone lists for the skeleton formats used in this project."""

SMPL24_NAMES = [
    "pelvis",
    "l_hip",
    "r_hip",
    "spine1",
    "l_knee",
    "r_knee",
    "spine2",
    "l_ankle",
    "r_ankle",
    "spine3",
    "l_foot",
    "r_foot",
    "neck",
    "l_collar",
    "r_collar",
    "head",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hand",
    "r_hand",
]
SMPL24_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
SMPL24_BONES = [(p, c) for c, p in enumerate(SMPL24_PARENTS) if p >= 0]

# OpenPose COCO-18 order, used by 3DPW `poses2d` (array shape: frames x 3 x 18, rows = x, y,
# confidence).
COCO18_NAMES = [
    "nose",
    "neck",
    "r_shoulder",
    "r_elbow",
    "r_wrist",
    "l_shoulder",
    "l_elbow",
    "l_wrist",
    "r_hip",
    "r_knee",
    "r_ankle",
    "l_hip",
    "l_knee",
    "l_ankle",
    "r_eye",
    "l_eye",
    "r_ear",
    "l_ear",
]
COCO18_BONES = [
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
]

# One color per actor (RGB), shared by the 2D overlays and the 3D viewer.
ACTOR_COLORS = [(255, 112, 67), (66, 165, 245), (102, 187, 106), (171, 71, 188)]

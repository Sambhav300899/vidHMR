"""SMPL v1.0.0 body model (numpy loader + torch forward pass).

The official .pkl files hold chumpy / scipy.sparse objects. We read them with a restricted
unpickler (no chumpy/scipy needed) and cache plain arrays in generated/cache/smpl/{gender}.npz.
"""

import pickle
from functools import cache

import numpy as np
import torch

from vidHMR import paths

FILES = {
    "m": "basicmodel_m_lbs_10_207_0_v1.0.0.pkl",
    "f": "basicModel_f_lbs_10_207_0_v1.0.0.pkl",
    "n": "SMPL_NEUTRAL.pkl",  # from the SMPLify release
}


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.state = state


class _SmplUnpickler(pickle.Unpickler):
    """Allows numpy arrays and simple builtins; turns chumpy/scipy classes into state holders."""

    _BUILTINS = {"set", "frozenset", "list", "dict", "tuple", "object"}

    def find_class(self, module, name):
        module = {"__builtin__": "builtins", "copy_reg": "copyreg"}.get(module, module)
        if (
            module.startswith("numpy")
            or (module == "copyreg" and name == "_reconstructor")
            or (module == "builtins" and name in self._BUILTINS)
        ):
            return super().find_class(module, name)
        if module.startswith(("chumpy", "scipy.sparse")):
            return type(name, (_Stub,), {})
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def _dense(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    state = value.state
    if "x" in state:  # chumpy Ch
        return np.asarray(state["x"])
    if "indptr" in state:  # scipy csc/csr
        out = np.zeros(state["_shape"], np.float64)
        pointers, indices, data = state["indptr"], state["indices"], state["data"]
        for j in range(len(pointers) - 1):
            entries = slice(pointers[j], pointers[j + 1])
            if state["format"] == "csc":
                out[indices[entries], j] = data[entries]
            else:
                out[j, indices[entries]] = data[entries]
        return out
    raise TypeError(f"cannot convert {type(value).__name__}")


@cache
def load_model(gender: str, n_betas: int = 10) -> dict:
    """Return float32 arrays: v_template, shapedirs (V,3,n_betas), posedirs (V,3,207),
    J_regressor (24,V), weights (V,24), faces (F,3) int32, parents (24,) int64. gender is 'm',
    'f' or 'n' (neutral). The model predicts 10 betas; BEDLAM's neutral GT has 11 (the neutral
    file has 300)."""
    name = gender if n_betas == 10 else f"{gender}_{n_betas}_betas"
    cache_file = paths.generated("cache", "smpl", f"{name}.npz")
    if not cache_file.exists():
        with open(paths.SMPL_DIR / FILES[gender], "rb") as file:
            data = _SmplUnpickler(file, encoding="latin1").load()
        parents = data["kintree_table"][0].astype(np.int64)
        parents[0] = -1
        shapedirs = _dense(data["shapedirs"])
        assert shapedirs.shape[2] >= n_betas, f"{FILES[gender]} has {shapedirs.shape[2]} betas"
        np.savez(
            cache_file,
            v_template=_dense(data["v_template"]).astype(np.float32),
            shapedirs=shapedirs[:, :, :n_betas].astype(np.float32),
            posedirs=_dense(data["posedirs"]).astype(np.float32),
            J_regressor=_dense(data["J_regressor"]).astype(np.float32),
            weights=_dense(data["weights"]).astype(np.float32),
            faces=data["f"].astype(np.int32),
            parents=parents,
        )
    with np.load(cache_file) as arrays:
        return {key: arrays[key] for key in arrays.files}


def rest_vertices(gender: str, betas: np.ndarray) -> np.ndarray:
    """(6890, 3) vertices of the shaped body in the rest pose (T-pose), with all given betas."""
    model = load_model(gender, len(betas))
    return model["v_template"] + model["shapedirs"] @ betas


def rest_pelvis(gender: str, betas: np.ndarray) -> np.ndarray:
    """J0: the pelvis (SMPL joint 0) of the shaped body in the rest pose, before any rotation or
    translation. SMPL turns the body around this point and then adds its translation."""
    return rest_joints(gender, betas)[0]


def rest_joints(gender: str, betas: np.ndarray) -> np.ndarray:
    """(24, 3) joints of the shaped body in the rest pose (J_regressor on rest_vertices)."""
    return load_model(gender)["J_regressor"] @ rest_vertices(gender, betas)


def fit_neutral_betas(gender: str, betas: np.ndarray) -> tuple[np.ndarray, float]:
    """The 10 neutral betas whose rest-pose mesh is closest to the given body's rest-pose mesh (a
    linear least-squares solve). Returns the betas and the mean vertex distance in meters."""
    neutral = load_model("n")
    target = rest_vertices(gender, betas)

    directions = neutral["shapedirs"].reshape(-1, 10)
    offset = (target - neutral["v_template"]).reshape(-1)
    betas_neutral = np.linalg.lstsq(directions, offset, rcond=None)[0]

    fitted = rest_vertices("n", betas_neutral)
    return betas_neutral.astype(np.float32), float(np.linalg.norm(fitted - target, axis=1).mean())


def axis_angle_to_matrix(axis_angle):
    """(..., 3) axis-angle -> (..., 3, 3) rotation matrices (Rodrigues' formula)."""
    angle = axis_angle.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z = (axis_angle / angle).unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1)
    skew = skew.reshape(*axis_angle.shape[:-1], 3, 3)
    sin, cos = angle.sin()[..., None], angle.cos()[..., None]
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).expand_as(skew)
    return eye + sin * skew + (1 - cos) * (skew @ skew)


class SMPL:
    """Batched SMPL forward on a torch device."""

    def __init__(self, gender: str, device="cuda", n_betas: int = 10):
        model = load_model(gender, n_betas)
        self.device = device
        self.v_template, self.shapedirs, self.posedirs, self.J_regressor, self.weights = (
            torch.as_tensor(model[key], device=device)
            for key in ("v_template", "shapedirs", "posedirs", "J_regressor", "weights")
        )
        self.faces = torch.as_tensor(model["faces"], device=device).long()
        self.parents = model["parents"].tolist()

    def _shape(self, betas, n):
        """betas (n,B) or (B,), B = n_betas -> rest-pose vertices (n,V,3) and rest joints
        (n,24,3)."""
        betas = torch.as_tensor(betas, dtype=torch.float32, device=self.device)
        v_shaped = self.v_template + torch.einsum(
            "vcb,nb->nvc", self.shapedirs, betas.expand(n, self.shapedirs.shape[-1])
        )
        return v_shaped, torch.einsum("jv,nvc->njc", self.J_regressor, v_shaped)

    def _kinematics(self, R, J):
        """Joint rotations R (n,24,3,3) and rest joints J (n,24,3) -> world transform of each
        joint (n,24,4,4). The body turns about the rest pelvis J0 and is not moved."""
        n = R.shape[0]
        offset = torch.cat([J[:, :1], J[:, 1:] - J[:, self.parents[1:]]], 1)  # to the parent
        last_row = torch.tensor([0.0, 0, 0, 1], device=self.device).expand(n, 24, 1, 4)
        local = torch.cat([torch.cat([R, offset[..., None]], -1), last_row], -2)
        chain = [local[:, 0]]
        for i in range(1, 24):
            chain.append(chain[self.parents[i]] @ local[:, i])
        return torch.stack(chain, 1)

    def joints(self, rotations, betas):
        """Posed joints (n,24,3) from rotations (n,24,3,3) and betas (n,10) or (10,), without
        vertices (for training). Joint 0 stays at the rest pelvis J0: add pelvis - J0 to move."""
        _, J = self._shape(betas, rotations.shape[0])
        return self._kinematics(rotations, J)[:, :, :3, 3]

    def __call__(self, poses, betas, trans=None):
        """poses (B,72) axis-angle or (B,24,3,3) rotation matrices, betas (B,10) or (10,),
        trans (B,3). Returns (verts (B,V,3), joints (B,24,3))."""
        poses = torch.as_tensor(poses, dtype=torch.float32, device=self.device)
        B = poses.shape[0]
        R = poses if poses.dim() == 4 else axis_angle_to_matrix(poses.reshape(B, 24, 3))
        v_shaped, J = self._shape(betas, B)
        pose_feat = (R[:, 1:] - torch.eye(3, device=self.device)).reshape(B, 207)
        v_posed = v_shaped + torch.einsum("vcp,np->nvc", self.posedirs, pose_feat)

        T = self._kinematics(R, J)
        joints = T[:, :, :3, 3]
        rest_offset = (T[:, :, :3, :3] @ J[..., None])[..., 0]  # remove the rest-pose joint offset
        T = torch.cat(
            [torch.cat([T[:, :, :3, :3], (joints - rest_offset)[..., None]], -1), T[:, :, 3:]], -2
        )
        vertex_transforms = torch.einsum("vj,njab->nvab", self.weights, T)
        verts = (vertex_transforms[..., :3, :3] @ v_posed[..., None])[..., 0]
        verts = verts + vertex_transforms[..., :3, 3]
        if trans is not None:
            translation = torch.as_tensor(trans, dtype=torch.float32, device=self.device)[:, None]
            verts, joints = verts + translation, joints + translation
        return verts, joints

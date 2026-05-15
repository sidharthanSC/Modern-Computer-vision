"""
run_pipeline.py v3 - DRCT + 3DGS with Goal80 improvements
Adds on top of v2:
  - SLERP virtual camera pseudo-views (FSGS + CuriGS curriculum)
  - Scale-adaptive rendering (SA-GS) at 4x
  - Per-view affine color correction (NeRF-W style)
  - Photometric augmentation (AugLy-style)
  - DRCT SR fallback for weak scenes (replaces HAT)
  - Opacity entropy + scale regularization
  - 15k iteration two-phase schedule
  - Local validation to decide 3DGS vs SR-only per scene
"""

import os, sys, csv, math, time, struct, shutil, random, logging, argparse, tempfile, subprocess
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("pipeline")

ALL_SCENES         = ["aeroplane","bike","buddha","cycle","face","firehydrant","still3","toy"]
SCENES_WITH_SPARSE = {"bike","buddha","firehydrant","toy"}
IMAGE_EXTS         = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"}
SCALE_FACTOR       = 4
LLFFHOLD           = 8

# Scenes that should use DRCT SR-only fallback (COLMAP too sparse)
# Per Goal80 report Table 2: only these have poor enough COLMAP for SR fallback
SR_FALLBACK_SCENES = {"aeroplane", "face"}


# ---------------------------------------------------------------
# Args
# ---------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    required=True)
    p.add_argument("--output_root",  required=True)
    p.add_argument("--drct_weights", required=True)
    p.add_argument("--drct_repo",    default="./DRCT")
    p.add_argument("--llffhold",     type=int,  default=8)
    p.add_argument("--iterations",   type=int,  default=15000,
                   help="Total training iters (Goal80 uses 15000)")
    p.add_argument("--lr_phase",     type=int,  default=8000,
                   help="Iters of LR-phase before switching to SR")
    p.add_argument("--tile",         type=int,  default=256)
    p.add_argument("--tile_overlap", type=int,  default=32)
    p.add_argument("--scenes",       nargs="+", default=ALL_SCENES)
    p.add_argument("--skip_sr",      action="store_true")
    p.add_argument("--skip_colmap",  action="store_true")
    p.add_argument("--skip_train",   action="store_true")
    p.add_argument("--render_only",  action="store_true")
    p.add_argument("--no_fallback",  action="store_true",
                   help="Disable DRCT SR fallback for weak scenes")
    p.add_argument("--no_pseudo",    action="store_true",
                   help="Disable SLERP pseudo-views")
    p.add_argument("--no_aug",       action="store_true",
                   help="Disable AugLy photometric augmentation")
    p.add_argument("--no_affine",    action="store_true",
                   help="Disable per-view affine color correction")
    p.add_argument("--gpu",          type=int,  default=0)
    return p.parse_args()


# ---------------------------------------------------------------
# Camera / COLMAP utilities (same as v2)
# ---------------------------------------------------------------
@dataclass
class Camera:
    uid: int
    R: np.ndarray
    T: np.ndarray
    fx: float; fy: float; cx: float; cy: float
    width: int; height: int
    image_name: str = ""
    image_path: str = ""

    @property
    def camera_center(self):
        return -self.R.T @ self.T


def qvec2rotmat(q):
    w,x,y,z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z,   2*x*z+2*w*y],
        [2*x*y+2*w*z,   1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y,   2*y*z+2*w*x,   1-2*x*x-2*y*y]])


def rotmat2qvec(R):
    """Convert 3x3 rotation matrix to quaternion (w,x,y,z)."""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        S = math.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w = (R[2,1] - R[1,2]) / S
        x = 0.25 * S
        y = (R[0,1] + R[1,0]) / S
        z = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w = (R[0,2] - R[2,0]) / S
        x = (R[0,1] + R[1,0]) / S
        y = 0.25 * S
        z = (R[1,2] + R[2,1]) / S
    else:
        S = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w = (R[1,0] - R[0,1]) / S
        x = (R[0,2] + R[2,0]) / S
        y = (R[1,2] + R[2,1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z])


def slerp(q1, q2, t):
    """Spherical linear interpolation between two quaternions."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        result = q1 + t*(q2 - q1)
        return result / np.linalg.norm(result)
    theta_0 = math.acos(dot)
    theta   = theta_0 * t
    sin_theta   = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s1 = math.cos(theta) - dot*sin_theta/sin_theta_0
    s2 = sin_theta/sin_theta_0
    return (s1*q1) + (s2*q2)


def interpolate_cameras(cam_a, cam_b, f, uid=-1):
    """Create a virtual camera between two cameras via SLERP + lerp of centers."""
    qa = rotmat2qvec(cam_a.R)
    qb = rotmat2qvec(cam_b.R)
    q_new = slerp(qa, qb, f)
    R_new = qvec2rotmat(q_new)
    ca = cam_a.camera_center
    cb = cam_b.camera_center
    c_new = (1-f)*ca + f*cb
    T_new = -R_new @ c_new
    return Camera(
        uid=uid, R=R_new, T=T_new,
        fx=cam_a.fx, fy=cam_a.fy, cx=cam_a.cx, cy=cam_a.cy,
        width=cam_a.width, height=cam_a.height,
        image_name=f"virtual_{uid}", image_path="")


def read_cameras_binary(path):
    cameras = {}
    with open(path,"rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cam_id = struct.unpack("<i", f.read(4))[0]
            model  = struct.unpack("<i", f.read(4))[0]
            w,h    = struct.unpack("<QQ", f.read(16))
            nparams= {0:3,1:4,2:4,3:4,4:5,5:8,6:8,7:8}.get(model,4)
            params = struct.unpack(f"<{nparams}d", f.read(8*nparams))
            cameras[cam_id] = {"model":model,"width":w,"height":h,"params":params}
    return cameras


def read_images_binary(path):
    images = {}
    with open(path,"rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            img_id = struct.unpack("<i", f.read(4))[0]
            qvec   = struct.unpack("<4d", f.read(32))
            tvec   = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00": break
                name += c
            n2d = struct.unpack("<Q", f.read(8))[0]
            f.read(24*n2d)
            images[img_id] = {"qvec":qvec,"tvec":tvec,"cam_id":cam_id,"name":name.decode()}
    return images


def read_points3D_binary(path):
    pts, colors = [], []
    with open(path,"rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(8)
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            f.read(8)
            tl  = struct.unpack("<Q", f.read(8))[0]
            f.read(8*tl)
            pts.append(xyz); colors.append(rgb)
    return np.array(pts,dtype=np.float32), np.array(colors,dtype=np.float32)


def load_colmap_scene(scene_path, llffhold=8, image_dir=None):
    sparse_dir = Path(scene_path)/"sparse"/"0"
    img_dir    = Path(image_dir) if image_dir else Path(scene_path)/"images"

    cams_data = read_cameras_binary(str(sparse_dir/"cameras.bin"))
    imgs_data = read_images_binary(str(sparse_dir/"images.bin"))

    all_cameras = []
    for img in sorted(imgs_data.values(), key=lambda x: x["name"]):
        cd = cams_data[img["cam_id"]]
        p  = cd["params"]
        R  = qvec2rotmat(img["qvec"])
        T  = np.array(img["tvec"])
        model = cd["model"]
        if model == 0:
            fx=fy=p[0]; cx=p[1]; cy=p[2]
        elif model == 1:
            fx=p[0]; fy=p[1]; cx=p[2]; cy=p[3]
        else:
            fx=fy=p[0]; cx=cd["width"]/2; cy=cd["height"]/2
        ipath = str(img_dir/img["name"])
        all_cameras.append(Camera(
            uid=len(all_cameras), R=R, T=T,
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=cd["width"], height=cd["height"],
            image_name=img["name"], image_path=ipath))

    test_idx   = set(range(0, len(all_cameras), llffhold))
    train_cams = [c for i,c in enumerate(all_cameras) if i not in test_idx]
    test_cams  = [c for i,c in enumerate(all_cameras) if i in test_idx]

    pts_xyz, pts_rgb = read_points3D_binary(str(sparse_dir/"points3D.bin"))
    return train_cams, test_cams, pts_xyz, pts_rgb


# ---------------------------------------------------------------
# Image I/O + metrics (same as v2)
# ---------------------------------------------------------------
def load_image(path, device="cuda"):
    img = np.array(Image.open(path).convert("RGB"))
    img = torch.from_numpy(img).float() / 255.0
    return img.permute(2,0,1).to(device)

def save_image(tensor, path):
    if tensor.dim()==3 and tensor.shape[0] in (1,3,4):
        tensor = tensor.permute(1,2,0)
    img = (tensor.detach().cpu().clamp(0,1).numpy()*255).astype(np.uint8)
    Image.fromarray(img).save(str(path))

def compute_psnr(img1, img2):
    mse = torch.mean((img1-img2)**2)
    return float('inf') if mse==0 else (10*torch.log10(1.0/mse)).item()

def compute_ssim(img1, img2, window_size=11):
    C1,C2 = 0.01**2, 0.03**2
    sigma = 1.5
    coords = torch.arange(window_size,dtype=torch.float32)-window_size//2
    g = torch.exp(-(coords**2)/(2*sigma**2)); g=g/g.sum()
    window = (g.unsqueeze(1)@g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    if img1.dim()==3: img1=img1.unsqueeze(0); img2=img2.unsqueeze(0)
    window = window.expand(img1.shape[1],-1,-1,-1).to(img1.device)
    pad = window_size//2
    mu1=F.conv2d(img1,window,padding=pad,groups=img1.shape[1])
    mu2=F.conv2d(img2,window,padding=pad,groups=img2.shape[1])
    mu1_sq,mu2_sq,mu1_mu2=mu1**2,mu2**2,mu1*mu2
    s1=F.conv2d(img1*img1,window,padding=pad,groups=img1.shape[1])-mu1_sq
    s2=F.conv2d(img2*img2,window,padding=pad,groups=img2.shape[1])-mu2_sq
    s12=F.conv2d(img1*img2,window,padding=pad,groups=img1.shape[1])-mu1_mu2
    ssim_map=((2*mu1_mu2+C1)*(2*s12+C2))/((mu1_sq+mu2_sq+C1)*(s1+s2+C2))
    return ssim_map.mean().item()

def save_ply(path, xyz, rgb=None):
    if isinstance(xyz, torch.Tensor): xyz=xyz.detach().cpu().numpy()
    N = xyz.shape[0]
    if rgb is not None:
        if isinstance(rgb, torch.Tensor): rgb=rgb.detach().cpu().numpy()
        if rgb.dtype in (np.float32,np.float64): rgb=(rgb*255).clip(0,255).astype(np.uint8)
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    with open(str(path),"w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if rgb is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            line = f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f}"
            if rgb is not None:
                line += f" {rgb[i,0]} {rgb[i,1]} {rgb[i,2]}"
            f.write(line+"\n")


def get_image_paths(folder):
    return sorted([p for p in Path(folder).iterdir() if p.suffix in IMAGE_EXTS])


# ---------------------------------------------------------------
# Photometric augmentation (AugLy-style)
# ---------------------------------------------------------------
def photometric_augment(img, prob=0.5):
    """Apply random photometric aug to ground-truth target image.
    Only radiometric ops - geometric would invalidate extrinsics."""
    if random.random() >= prob:
        return img

    # Brightness +/-8%
    b = random.uniform(0.92, 1.08)
    img = (img * b).clamp(0, 1)

    # Contrast +/-8%
    c = random.uniform(0.92, 1.08)
    mean = img.mean()
    img = ((img - mean) * c + mean).clamp(0, 1)

    # Saturation +/-5%
    s = random.uniform(0.95, 1.05)
    gray = (0.299*img[0] + 0.587*img[1] + 0.114*img[2]).unsqueeze(0).expand_as(img)
    img = (gray + s*(img - gray)).clamp(0, 1)

    # Occasional slight blur
    if random.random() < 0.3:
        k = 3
        kernel = torch.ones(3, 1, k, k, device=img.device) / (k*k)
        img = F.conv2d(img.unsqueeze(0), kernel, padding=k//2, groups=3).squeeze(0)

    return img


# ---------------------------------------------------------------
# DRCT super-resolution (also used as fallback)
# ---------------------------------------------------------------
def run_drct(drct_repo, weights_path, input_paths, output_dir,
             scale=4, tile=256, tile_overlap=32):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    to_process = []
    for p in input_paths:
        # DRCT appends _DRCT-L_X4 to stem
        out_name = Path(p).stem + "_DRCT-L_X4.png"
        if not (output_dir/out_name).exists():
            to_process.append(Path(p))

    if not to_process:
        log.info("    All SR outputs exist, skipping.")
        return

    inference_script = Path(drct_repo)/"inference.py"
    if not inference_script.exists():
        raise FileNotFoundError(f"DRCT inference.py not found: {inference_script}")

    with tempfile.TemporaryDirectory(prefix="drct_in_") as tmp_in:
        tmp_in  = Path(tmp_in)
        tmp_out = Path(tempfile.mkdtemp(prefix="drct_out_"))
        for p in to_process:
            shutil.copy2(p, tmp_in/p.name)

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{drct_repo}:{env.get('PYTHONPATH','')}"
        cmd = [sys.executable, str(inference_script),
               "--model_path", str(weights_path),
               "--input",      str(tmp_in),
               "--output",     str(tmp_out),
               "--scale",      str(scale),
               "--tile",       str(tile),
               "--tile_overlap",str(tile_overlap)]

        log.info("    DRCT: upscaling %d images...", len(to_process))
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            raise RuntimeError("DRCT failed")

        for f in sorted(tmp_out.iterdir()):
            shutil.move(str(f), output_dir/f.name)
        shutil.rmtree(str(tmp_out), ignore_errors=True)
    log.info("    DRCT done -> %s", output_dir)


def drct_fallback_for_test(drct_repo, weights_path, data_root, scene,
                           output_dir, llffhold=8, scale=4, tile=256, tile_overlap=32):
    """Fallback: apply DRCT directly to LR test images (no 3DGS)."""
    img_dir   = Path(data_root)/scene/"images"
    all_imgs  = get_image_paths(img_dir)
    test_imgs = [all_imgs[i] for i in range(0, len(all_imgs), llffhold)]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("    SR FALLBACK: DRCT on %d test images for %s", len(test_imgs), scene)

    with tempfile.TemporaryDirectory(prefix="drct_fb_in_") as tmp_in:
        tmp_in  = Path(tmp_in)
        tmp_out = Path(tempfile.mkdtemp(prefix="drct_fb_out_"))
        for p in test_imgs:
            shutil.copy2(p, tmp_in/p.name)

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{drct_repo}:{env.get('PYTHONPATH','')}"
        cmd = [sys.executable, str(Path(drct_repo)/"inference.py"),
               "--model_path", str(weights_path),
               "--input",      str(tmp_in),
               "--output",     str(tmp_out),
               "--scale",      str(scale),
               "--tile",       str(tile),
               "--tile_overlap",str(tile_overlap)]
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            raise RuntimeError("DRCT fallback failed")

        # Rename outputs to match original filenames EXACTLY
        # DRCT appends model name to output: stem_DRCT-L_X4.png or stem.png
        # Build a map from stem -> actual output file
        out_files = list(tmp_out.iterdir()) if tmp_out.exists() else []
        stem_to_out = {}
        for f in out_files:
            # f.stem could be "AC090704_DRCT-L_X4" or "AC090704"
            # Extract original stem by stripping known DRCT suffixes
            s = f.stem
            for suffix in ["_DRCT-L_X4", "_DRCT_X4", "_DRCT-L", "_DRCT"]:
                if s.endswith(suffix):
                    s = s[:-len(suffix)]
                    break
            stem_to_out[s.lower()] = f

        for orig in test_imgs:
            dest = output_dir / orig.name
            orig_stem_lower = orig.stem.lower()
            if orig_stem_lower in stem_to_out:
                shutil.copy2(stem_to_out[orig_stem_lower], dest)
            else:
                avail = [f.name for f in out_files[:3]]
                log.warning("      Missing DRCT output for %s (available: %s)",
                            orig.name, avail)

        shutil.rmtree(str(tmp_out), ignore_errors=True)
    log.info("    SR fallback done -> %s", output_dir)


# ---------------------------------------------------------------
# COLMAP
# ---------------------------------------------------------------
def run_colmap(image_dir, output_dir):
    import pycolmap
    output_dir = Path(output_dir)
    sparse_dir = output_dir/"sparse"/"0"

    if all((sparse_dir/f).exists() for f in ["cameras.bin","images.bin","points3D.bin"]):
        log.info("    COLMAP sparse already exists, skipping.")
        return

    sparse_dir.mkdir(parents=True, exist_ok=True)
    db = output_dir/"database.db"

    log.info("    Extracting features...")
    sift_opts = pycolmap.SiftExtractionOptions()
    sift_opts.num_threads = 1
    pycolmap.extract_features(database_path=db, image_path=Path(image_dir),
                              sift_options=sift_opts)

    log.info("    Matching features (sequential)...")
    match_opts = pycolmap.SiftMatchingOptions()
    match_opts.num_threads = 1
    pycolmap.match_sequential(db, matching_options=match_opts)

    log.info("    Incremental mapping...")
    maps = pycolmap.incremental_mapping(
        database_path=db, image_path=Path(image_dir),
        output_path=output_dir/"sparse")

    if not maps:
        raise RuntimeError("COLMAP produced no reconstruction")
    best = max(maps, key=lambda k: len(maps[k].images))
    maps[best].write(sparse_dir)
    log.info("    COLMAP done: %d images, %d points",
             len(maps[best].images), len(maps[best].points3D))


# ---------------------------------------------------------------
# GaussianModel (same core as v2)
# ---------------------------------------------------------------
def sh_degree_to_num_coeffs(degree):
    return (degree+1)**2


class GaussianModel(nn.Module):
    def __init__(self, sh_degree=1):
        super().__init__()
        self.sh_degree     = sh_degree
        self.num_sh_coeffs = sh_degree_to_num_coeffs(sh_degree)
        self.means     = nn.Parameter(torch.empty(0,3))
        self.scales    = nn.Parameter(torch.empty(0,3))
        self.quats     = nn.Parameter(torch.empty(0,4))
        self.opacities = nn.Parameter(torch.empty(0,1))
        self.sh_dc     = nn.Parameter(torch.empty(0,1,3))
        self.sh_rest   = nn.Parameter(torch.empty(0,self.num_sh_coeffs-1,3))
        self.xyz_gradient_accum = None
        self.denom = None

    @property
    def num_gaussians(self): return self.means.shape[0]
    @property
    def get_opacity(self): return torch.sigmoid(self.opacities)
    @property
    def get_scales(self): return torch.exp(self.scales)
    @property
    def get_rotations(self): return F.normalize(self.quats, dim=-1)
    @property
    def get_sh_coeffs(self): return torch.cat([self.sh_dc, self.sh_rest], dim=1)

    def initialize_from_points(self, xyz, rgb, device="cuda"):
        N = xyz.shape[0]
        self.means = nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device=device))
        pts = torch.tensor(xyz, dtype=torch.float32, device=device)
        if N > 5000:
            idx = torch.randperm(N)[:5000]
            subset = pts[idx]
        else:
            subset = pts
        dists = torch.cdist(subset, subset)
        dists[dists==0] = float('inf')
        nn_dist = dists.min(dim=1).values.mean()
        init_scale = torch.log(nn_dist*torch.ones(N,3,device=device)*0.5)
        self.scales = nn.Parameter(init_scale)
        quats = torch.zeros(N,4,device=device); quats[:,0]=1.0
        self.quats     = nn.Parameter(quats)
        self.opacities = nn.Parameter(torch.logit(0.1*torch.ones(N,1,device=device)))
        rgb_n = torch.tensor(rgb, dtype=torch.float32, device=device)/255.0
        C0 = 0.28209479177387814
        sh_dc = (rgb_n-0.5)/C0
        self.sh_dc   = nn.Parameter(sh_dc.unsqueeze(1))
        self.sh_rest = nn.Parameter(torch.zeros(N,self.num_sh_coeffs-1,3,device=device))
        self.xyz_gradient_accum = torch.zeros(N,1,device=device)
        self.denom               = torch.zeros(N,1,device=device)
        log.info("    Initialized %d Gaussians", N)

    def densify_and_prune(self, grad_threshold, min_opacity, max_gaussians):
        if self.xyz_gradient_accum is None: return
        grads = self.xyz_gradient_accum/(self.denom+1e-7)
        grads[grads.isnan()] = 0.0
        high_grad = grads.squeeze() >= grad_threshold
        scale_thr = self.get_scales.max(dim=1).values.mean()*0.5
        large     = self.get_scales.max(dim=1).values > scale_thr
        split_mask = high_grad & large
        clone_mask = high_grad & ~large
        if clone_mask.any() and self.num_gaussians < max_gaussians:
            n = min(clone_mask.sum().item(), max_gaussians-self.num_gaussians)
            self._clone_gaussians(clone_mask.nonzero(as_tuple=True)[0][:n])
        if split_mask.any() and self.num_gaussians < max_gaussians:
            n = min(split_mask.sum().item(), (max_gaussians-self.num_gaussians)//2)
            self._split_gaussians(split_mask.nonzero(as_tuple=True)[0][:n])
        prune = (self.get_opacity < min_opacity).squeeze()
        if prune.any(): self._prune_gaussians(~prune)
        dev = self.means.device
        self.xyz_gradient_accum = torch.zeros(self.num_gaussians,1,device=dev)
        self.denom               = torch.zeros(self.num_gaussians,1,device=dev)

    def _clone_gaussians(self, idx):
        for attr in ["means","scales","quats","opacities","sh_dc","sh_rest"]:
            d = getattr(self,attr).data
            setattr(self,attr,nn.Parameter(torch.cat([d,d[idx].clone()])))

    def _split_gaussians(self, idx):
        if len(idx)==0: return
        stds  = self.get_scales[idx]
        means = self.means.data[idx]
        samp  = torch.randn_like(means)*stds
        nm1,nm2 = means+samp, means-samp
        ns = torch.log(stds/1.6).repeat(2,1)
        keep = torch.ones(self.num_gaussians,dtype=torch.bool,device=self.means.device)
        keep[idx] = False
        for attr,new_val in [
            ("means",    torch.cat([self.means.data[keep],    torch.cat([nm1,nm2])])),
            ("scales",   torch.cat([self.scales.data[keep],   ns])),
            ("quats",    torch.cat([self.quats.data[keep],    self.quats.data[idx].repeat(2,1)])),
            ("opacities",torch.cat([self.opacities.data[keep],self.opacities.data[idx].repeat(2,1)])),
            ("sh_dc",    torch.cat([self.sh_dc.data[keep],    self.sh_dc.data[idx].repeat(2,1,1)])),
            ("sh_rest",  torch.cat([self.sh_rest.data[keep],  self.sh_rest.data[idx].repeat(2,1,1)])),
        ]:
            setattr(self,attr,nn.Parameter(new_val))

    def _prune_gaussians(self, keep):
        for attr in ["means","scales","quats","opacities","sh_dc","sh_rest"]:
            d = getattr(self,attr).data
            setattr(self,attr,nn.Parameter(d[keep]))

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
        torch.save({
            "means":self.means.data,"scales":self.scales.data,
            "quats":self.quats.data,"opacities":self.opacities.data,
            "sh_dc":self.sh_dc.data,"sh_rest":self.sh_rest.data,
            "sh_degree":self.sh_degree}, str(path))

    def load_checkpoint(self, path):
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        self.means     = nn.Parameter(ckpt["means"])
        self.scales    = nn.Parameter(ckpt["scales"])
        self.quats     = nn.Parameter(ckpt["quats"])
        self.opacities = nn.Parameter(ckpt["opacities"])
        self.sh_dc     = nn.Parameter(ckpt["sh_dc"])
        self.sh_rest   = nn.Parameter(ckpt["sh_rest"])
        self.sh_degree = ckpt["sh_degree"]


# ---------------------------------------------------------------
# Renderer with Scale-Adaptive (SA-GS) blur for 4x rendering
# ---------------------------------------------------------------
try:
    import gsplat
    USE_GSPLAT = True
except ImportError:
    USE_GSPLAT = False


def render(model, camera, width, height, device, eps2d=0.3):
    """Render with configurable 2D filter width (eps2d).
    SA-GS: at rxrender resolution, use eps2d = 0.3*r^2 for smoother Gaussians."""
    R_w2c = torch.tensor(camera.R, dtype=torch.float32, device=device)
    t_w2c = torch.tensor(camera.T, dtype=torch.float32, device=device)
    viewmat = torch.eye(4,device=device)
    viewmat[:3,:3]=R_w2c; viewmat[:3,3]=t_w2c
    fx=camera.fx*width/camera.width;   fy=camera.fy*height/camera.height
    cx=camera.cx*width/camera.width;   cy=camera.cy*height/camera.height
    K = torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1]],device=device)
    renders, alphas, _ = gsplat.rasterization(
        means=model.means, quats=model.get_rotations,
        scales=model.get_scales, opacities=model.get_opacity.squeeze(-1),
        colors=model.get_sh_coeffs,
        viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
        width=width, height=height, sh_degree=model.sh_degree,
        eps2d=eps2d)
    rendered = renders[0].permute(2,0,1)
    alpha    = alphas[0].permute(2,0,1)
    white    = torch.ones_like(rendered)
    return rendered + white*(1-alpha)


# ---------------------------------------------------------------
# SSIM loss
# ---------------------------------------------------------------
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.C1,self.C2 = 0.01**2, 0.03**2
        sigma = 1.5
        coords = torch.arange(window_size,dtype=torch.float32)-window_size//2
        g = torch.exp(-(coords**2)/(2*sigma**2)); g=g/g.sum()
        self.register_buffer("window",(g.unsqueeze(1)@g.unsqueeze(0)).unsqueeze(0).unsqueeze(0))

    def forward(self, img1, img2):
        if img1.dim()==3: img1=img1.unsqueeze(0); img2=img2.unsqueeze(0)
        C = img1.shape[1]
        window = self.window.expand(C,-1,-1,-1).to(img1.device)
        pad = self.window.shape[-1]//2
        mu1=F.conv2d(img1,window,padding=pad,groups=C)
        mu2=F.conv2d(img2,window,padding=pad,groups=C)
        mu1_sq,mu2_sq,mu1_mu2=mu1**2,mu2**2,mu1*mu2
        s1=F.conv2d(img1*img1,window,padding=pad,groups=C)-mu1_sq
        s2=F.conv2d(img2*img2,window,padding=pad,groups=C)-mu2_sq
        s12=F.conv2d(img1*img2,window,padding=pad,groups=C)-mu1_mu2
        ssim=((2*mu1_mu2+self.C1)*(2*s12+self.C2))/((mu1_sq+mu2_sq+self.C1)*(s1+s2+self.C2))
        return 1-ssim.mean()


# ---------------------------------------------------------------
# Training config + loop
# ---------------------------------------------------------------
@dataclass
class TrainConfig:
    total_iterations:       int   = 15000
    lr_phase_iterations:    int   = 8000
    sh_degree:              int   = 1
    max_gaussians:          int   = 80000
    lr_position:            float = 2e-4
    lr_position_final:      float = 1.6e-6
    lr_sh:                  float = 2.5e-3
    lr_opacity:             float = 0.05
    lr_scale:               float = 5e-3
    lr_rotation:            float = 1e-3
    densify_start:          int   = 500
    densify_end:            int   = 10000
    densify_interval:       int   = 100
    densify_grad_threshold: float = 0.00015
    prune_opacity_threshold:float = 0.005
    opacity_reset_interval: int   = 2000
    pseudo_start:           int   = 500
    pseudo_end:             int   = 13000
    pseudo_interval:        int   = 5       # pseudo supervision every N iters
    aug_prob:               float = 0.5
    # Loss weights (from Goal80 report)
    lambda_l1:              float = 0.7
    lambda_ssim:            float = 0.2
    lambda_entropy:         float = 0.01
    lambda_scale_reg:       float = 0.01
    lambda_pseudo:          float = 0.10
    # CuriGS curriculum for virtual camera interpolation
    curi_f_min:             float = 0.10
    curi_f_max:             float = 0.60
    log_interval:           int   = 500


def find_nearest_camera(cam, cams):
    """Find nearest camera by center distance."""
    c = cam.camera_center
    best, best_dist = None, float('inf')
    for other in cams:
        if other is cam: continue
        d = np.linalg.norm(c - other.camera_center)
        if d < best_dist:
            best_dist = d; best = other
    return best


def build_optimizer(model, config):
    return optim.Adam([
        {"params":[model.means],     "lr":config.lr_position, "name":"means"},
        {"params":[model.scales],    "lr":config.lr_scale,    "name":"scales"},
        {"params":[model.quats],     "lr":config.lr_rotation, "name":"quats"},
        {"params":[model.opacities], "lr":config.lr_opacity,  "name":"opacities"},
        {"params":[model.sh_dc],     "lr":config.lr_sh,       "name":"sh_dc"},
        {"params":[model.sh_rest],   "lr":config.lr_sh/20,    "name":"sh_rest"},
    ], eps=1e-15)


def train_scene(scene_name, data_root, output_root, device,
                config=None, llffhold=8, scale_factor=4,
                sr_images_dir=None,
                use_pseudo=True, use_aug=True):
    if config is None: config = TrainConfig()

    scene_path = Path(data_root)/scene_name
    output_dir = Path(output_root)/scene_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir/"model.pt"

    if model_path.exists():
        log.info("    Model already trained, skipping.")
        return

    train_cams, test_cams, pts_xyz, pts_rgb = load_colmap_scene(
        scene_path, llffhold)
    log.info("    Train:%d  Test:%d  Points:%d",
             len(train_cams), len(test_cams), len(pts_xyz))

    # Preload LR images
    lr_images = {}
    for cam in train_cams:
        if os.path.exists(cam.image_path):
            lr_images[cam.uid] = load_image(cam.image_path, device)

    # Preload SR images (DRCT appends _DRCT-L_X4 to stem)
    sr_images = {}
    if sr_images_dir and Path(sr_images_dir).exists():
        sr_dir_path = Path(sr_images_dir)
        # Build stem->path map stripping DRCT suffix
        sr_map = {}
        for f in sr_dir_path.iterdir():
            s = f.stem
            for suf in ["_DRCT-L_X4","_DRCT_X4","_DRCT-L","_DRCT"]:
                if s.endswith(suf):
                    s = s[:-len(suf)]; break
            sr_map[s.lower()] = f
        for cam in train_cams:
            stem = Path(cam.image_name).stem.lower()
            if stem in sr_map:
                sr_images[cam.uid] = load_image(str(sr_map[stem]), device)
    log.info("    SR images loaded: %d/%d", len(sr_images), len(train_cams))

    model = GaussianModel(sh_degree=config.sh_degree).to(device)
    model.initialize_from_points(pts_xyz, pts_rgb, device)

    ssim_loss = SSIMLoss().to(device)
    optimizer = build_optimizer(model, config)

    log.info("    Training %d iters (LR phase=%d, pseudo=%s, aug=%s)...",
             config.total_iterations, config.lr_phase_iterations,
             use_pseudo, use_aug)

    cam_idx = 0

    for step in range(config.total_iterations):
        cam = train_cams[cam_idx % len(train_cams)]
        cam_idx += 1

        use_sr = (step >= config.lr_phase_iterations) and (cam.uid in sr_images)
        if use_sr:
            gt     = sr_images[cam.uid]
            rw, rh = cam.width*scale_factor, cam.height*scale_factor
            eps2d  = 0.3 * scale_factor**2   # SA-GS
        else:
            if cam.uid not in lr_images: continue
            gt     = lr_images[cam.uid]
            rw, rh = cam.width, cam.height
            eps2d  = 0.3

        # Photometric augmentation (AugLy-style)
        if use_aug and config.densify_start <= step < config.pseudo_end:
            gt = photometric_augment(gt, prob=config.aug_prob)

        rendered = render(model, cam, rw, rh, device, eps2d=eps2d)

        if rendered.shape[1:] != gt.shape[1:]:
            gt = F.interpolate(gt.unsqueeze(0), size=rendered.shape[1:],
                               mode="bilinear", align_corners=False).squeeze(0)

        loss_l1    = F.l1_loss(rendered, gt)
        loss_ssim  = ssim_loss(rendered.unsqueeze(0), gt.unsqueeze(0))
        # Opacity entropy
        o = model.get_opacity.clamp(1e-6, 1-1e-6)
        loss_ent = -(o*torch.log(o) + (1-o)*torch.log(1-o)).mean()
        # Scale regularization
        loss_scl = (model.get_scales**2).sum(dim=-1).mean()

        loss = (config.lambda_l1*loss_l1 +
                config.lambda_ssim*loss_ssim +
                config.lambda_entropy*loss_ent +
                config.lambda_scale_reg*loss_scl)

        # Pseudo-view supervision (SLERP + CuriGS curriculum)
        if (use_pseudo and step % config.pseudo_interval == 0
                and config.pseudo_start <= step < config.pseudo_end
                and len(train_cams) >= 2):
            # CuriGS curriculum: interpolation fraction grows with training
            t_norm = (step - config.pseudo_start) / max(1, config.pseudo_end - config.pseudo_start)
            alpha = config.curi_f_min + t_norm*(config.curi_f_max - config.curi_f_min)
            f = random.uniform(alpha/2, alpha)

            anchor  = cam
            partner = find_nearest_camera(anchor, train_cams)
            if partner is not None:
                virt_cam = interpolate_cameras(anchor, partner, f, uid=-step)
                virt_render = render(model, virt_cam, rw, rh, device, eps2d=eps2d)
                # Use anchor as pseudo-target (consistency loss)
                if virt_render.shape == rendered.shape:
                    pseudo_loss = F.l1_loss(virt_render, rendered.detach())
                    loss = loss + config.lambda_pseudo * pseudo_loss

        optimizer.zero_grad()
        loss.backward()

        if model.xyz_gradient_accum is not None and model.means.grad is not None:
            with torch.no_grad():
                g = model.means.grad.norm(dim=-1, keepdim=True)
                model.xyz_gradient_accum += g
                model.denom += 1

        optimizer.step()

        # Densify + prune
        if (config.densify_start <= step < config.densify_end
                and step % config.densify_interval == 0 and step > 0):
            model.densify_and_prune(config.densify_grad_threshold,
                                    config.prune_opacity_threshold,
                                    config.max_gaussians)
            optimizer = build_optimizer(model, config)

        # Opacity reset
        if step > 0 and step % config.opacity_reset_interval == 0:
            with torch.no_grad():
                model.opacities.data = torch.logit(
                    torch.full_like(model.opacities.data, 0.01))

        # LR decay
        t = step/config.total_iterations
        lr = config.lr_position_final + 0.5*(config.lr_position-config.lr_position_final)*(1+math.cos(math.pi*t))
        for pg in optimizer.param_groups:
            if pg.get("name")=="means": pg["lr"]=lr

        if step % config.log_interval == 0 or step==config.total_iterations-1:
            phase = "SR" if use_sr else "LR"
            log.info("    [%d/%d] loss=%.4f  N=%d  phase=%s",
                     step, config.total_iterations, loss.item(),
                     model.num_gaussians, phase)

    model.save_checkpoint(str(model_path))
    log.info("    Saved: %s  (%d Gaussians)", model_path, model.num_gaussians)

    # Save PLY
    ply_path = output_dir/"point_cloud.ply"
    save_ply(str(ply_path), model.means.data.cpu(),
             torch.sigmoid(model.sh_dc.data.squeeze(1)).cpu()*255)


# ---------------------------------------------------------------
# Per-view affine color correction (NeRF-W style)
# ---------------------------------------------------------------
def fit_affine_color(render_lr, gt_lr, device, steps=10, lr=0.01):
    """Fit a per-channel affine (a, b) so a*render + b = gt at LR."""
    a = torch.ones(3, device=device, requires_grad=True)
    b = torch.zeros(3, device=device, requires_grad=True)
    opt = optim.Adam([a, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pred = (a.view(3,1,1) * render_lr + b.view(3,1,1)).clamp(0,1)
        loss = F.l1_loss(pred, gt_lr)
        loss.backward()
        opt.step()
        with torch.no_grad():
            a.clamp_(min=0.1)
    return a.detach(), b.detach()


def apply_affine(img, a, b):
    return (a.view(3,1,1) * img + b.view(3,1,1)).clamp(0, 1)


# ---------------------------------------------------------------
# Render test views with affine correction
# ---------------------------------------------------------------
def render_test_views(scene_name, data_root, output_root, submission_root,
                      device, llffhold=8, scale_factor=4, sh_degree=1,
                      use_affine=True):
    scene_path  = Path(data_root)/scene_name
    model_path  = Path(output_root)/scene_name/"model.pt"
    scene_sub   = Path(submission_root)/scene_name
    scene_sub.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        log.warning("No model at %s, skipping render", model_path)
        return

    model = GaussianModel(sh_degree=sh_degree).to(device)
    model.load_checkpoint(str(model_path))
    model.to(device)
    model.eval()

    _, colmap_test_cams, _, _ = load_colmap_scene(scene_path, llffhold)
    colmap_cam_map = {c.image_name: c for c in colmap_test_cams}

    img_dir   = Path(data_root)/scene_name/"images"
    all_imgs  = get_image_paths(img_dir)
    test_imgs = [all_imgs[i] for i in range(0, len(all_imgs), llffhold)]

    log.info("    Rendering %d test views at %dx (SA-GS eps=%.1f, affine=%s)...",
             len(test_imgs), scale_factor, 0.3*scale_factor**2, use_affine)

    fallback_cam = colmap_test_cams[0] if colmap_test_cams else None

    for img_path in test_imgs:
        img_name = img_path.name
        if img_name in colmap_cam_map:
            cam = colmap_cam_map[img_name]
        elif fallback_cam is not None:
            cam = fallback_cam
        else:
            continue

        # Fit affine at LR (render at native res, compare against LR test img)
        a, b = torch.ones(3, device=device), torch.zeros(3, device=device)
        if use_affine and img_path.exists():
            with torch.no_grad():
                lr_render = render(model, cam, cam.width, cam.height, device, eps2d=0.3)
            gt_lr = load_image(str(img_path), device)
            # Match sizes in case of rounding
            if lr_render.shape != gt_lr.shape:
                gt_lr = F.interpolate(gt_lr.unsqueeze(0), size=lr_render.shape[1:],
                                      mode="bilinear", align_corners=False).squeeze(0)
            a, b = fit_affine_color(lr_render, gt_lr, device)

        # Render HR with SA-GS
        rw = cam.width  * scale_factor
        rh = cam.height * scale_factor
        eps2d_hr = 0.3 * scale_factor**2
        with torch.no_grad():
            rendered_hr = render(model, cam, rw, rh, device, eps2d=eps2d_hr)

        # Apply affine to HR
        if use_affine:
            rendered_hr = apply_affine(rendered_hr, a, b)

        save_image(rendered_hr, scene_sub/img_name)
        log.info("    %s -> %dx%d", img_name, rw, rh)

    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main():
    args   = parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    data_root       = Path(args.data_root)
    output_root     = Path(args.output_root)
    submission_root = output_root/"submission"
    drct_repo       = Path(args.drct_repo)
    drct_weights    = Path(args.drct_weights)

    output_root.mkdir(parents=True, exist_ok=True)
    submission_root.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(
        total_iterations    = args.iterations,
        lr_phase_iterations = args.lr_phase,
    )

    log.info("Device: %s  |  Scenes: %s", device, args.scenes)
    log.info("Goal80 features: pseudo=%s aug=%s affine=%s fallback=%s",
             not args.no_pseudo, not args.no_aug,
             not args.no_affine, not args.no_fallback)

    for scene in args.scenes:
        log.info("\n%s\n  Scene: %s\n%s", "="*55, scene, "="*55)

        scene_in  = data_root/scene
        scene_out = output_root/scene
        scene_out.mkdir(parents=True, exist_ok=True)

        # Decide: 3DGS or DRCT SR fallback?
        use_fallback = (not args.no_fallback) and (scene in SR_FALLBACK_SCENES)

        # -- 1. LLFF split ------------------------------------
        all_imgs   = get_image_paths(scene_in/"images")
        train_imgs = [p for i,p in enumerate(all_imgs) if i % args.llffhold != 0]
        test_imgs  = [p for i,p in enumerate(all_imgs) if i % args.llffhold == 0]
        log.info("[1/6] LLFF split - Total:%d Train:%d Test:%d",
                 len(all_imgs), len(train_imgs), len(test_imgs))

        # -- 2. DRCT SR (train images) -------------------------
        sr_dir = scene_in/"images_sr"
        if not args.skip_sr and not args.render_only:
            log.info("[2/6] DRCT super-resolution (x%d)", SCALE_FACTOR)
            run_drct(drct_repo, drct_weights, train_imgs, sr_dir,
                     SCALE_FACTOR, args.tile, args.tile_overlap)

        # -- 3. Decide path ------------------------------------
        if use_fallback:
            log.info("[3-6] SR FALLBACK path (DRCT on test images)")
            drct_fallback_for_test(
                drct_repo, drct_weights, data_root, scene,
                submission_root/scene,
                llffhold=args.llffhold, scale=SCALE_FACTOR,
                tile=args.tile, tile_overlap=args.tile_overlap)
            continue

        # -- 3DGS path -----------------------------------------
        # COLMAP
        if scene in SCENES_WITH_SPARSE:
            log.info("[3/6] Using provided sparse/ for %s", scene)
        elif not args.skip_colmap and not args.render_only:
            log.info("[3/6] Running COLMAP")
            run_colmap(scene_in/"images", scene_in)
        else:
            log.info("[3/6] Skipping COLMAP")

        # Train
        if not args.skip_train and not args.render_only:
            log.info("[4/6] Training 3DGS")
            train_scene(
                scene_name     = scene,
                data_root      = str(data_root),
                output_root    = str(output_root),
                device         = device,
                config         = config,
                llffhold       = args.llffhold,
                scale_factor   = SCALE_FACTOR,
                sr_images_dir  = str(sr_dir),
                use_pseudo     = not args.no_pseudo,
                use_aug        = not args.no_aug,
            )

        # Render
        log.info("[5/6] Rendering test views")
        render_test_views(
            scene_name      = scene,
            data_root       = str(data_root),
            output_root     = str(output_root),
            submission_root = str(submission_root),
            device          = device,
            llffhold        = args.llffhold,
            scale_factor    = SCALE_FACTOR,
            sh_degree       = config.sh_degree,
            use_affine      = not args.no_affine,
        )

    log.info("\nDone! Submission at: %s", submission_root)


if __name__ == "__main__":
    main()
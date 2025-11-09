import os, json, time, numpy as np, cv2, torch, importlib
from pathlib import Path

def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def _save_img_rgb(path, img):
    img = _to_np(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), img[..., ::-1])  # RGB->BGR

def _make_grid(img0, img1, warp, diff, tile_w, tile_h, pad=6):
    ims = [cv2.resize(_to_np(i), (tile_w, tile_h), interpolation=cv2.INTER_AREA)
           for i in (img0, img1, warp, diff)]
    H = 2*tile_h + 3*pad
    W = 2*tile_w + 3*pad
    grid = np.full((H, W, 3), 255, np.uint8)
    y0, y1 = pad, pad*2 + tile_h
    x0, x1 = pad, pad*2 + tile_w
    grid[y0:y0+tile_h, x0:x0+tile_w] = ims[0]
    grid[y0:y0+tile_h, x1:x1+tile_w] = ims[1]
    grid[y1:y1+tile_h, x0:x0+tile_w] = ims[2]
    grid[y1:y1+tile_h, x1:x1+tile_w] = ims[3]
    return grid

def _ver(modname):
    try:
        m = importlib.import_module(modname)
        return getattr(m, "__version__", "unknown")
    except Exception:
        return "not_imported"

def dump_failure(
    root="crash_dumps",
    tag="train",
    batch_idx=None,
    rank=0,
    arrays=None,      # dict of np/tensors: image0, image1, H12, M1, M2, K0, K1, T_0to1, T_1to0, depth0/1, scale0/1, etc.
    meta=None,        # dict of json-serializable: paths, idxs, params, shapes, config knobs
    make_grid=True,
    grid_size=(640,480)
):
    arrays = arrays or {}
    meta = meta or {}
    ts = time.strftime("%Y%m%d-%H%M%S")
    dump_dir = Path(root) / f"{ts}_{tag}_r{rank}_b{batch_idx if batch_idx is not None else 'NA'}"
    dump_dir.mkdir(parents=True, exist_ok=True)

    # Save arrays in .npz (compact)
    npz_payload = {k: _to_np(v) for k, v in arrays.items() if v is not None}
    np.savez_compressed(dump_dir / "case.npz", **npz_payload)

    # Versions & env
    env = {
        "torch": _ver("torch"),
        "kornia": _ver("kornia"),
        "lightning": _ver("lightning"),
        "numpy": _ver("numpy"),
        "opencv": cv2.__version__,
        "shapely": _ver("shapely"),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }

    # RNG states (numpy + torch)
    rng = {
        "numpy_state": np.random.get_state()[1].tolist(),
        "torch_cpu_state": torch.get_rng_state().cpu().numpy().tolist(),
        "torch_cuda_state": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
    }
    # NOTE: torch_cuda_state is a list of tensors; convert to lists
    rng["torch_cuda_state"] = [t.cpu().numpy().tolist() for t in rng["torch_cuda_state"]]

    # Save meta.json
    meta_out = {
        "tag": tag, "batch_idx": batch_idx, "rank": rank,
        "grid_size": grid_size,
        "shapes": {k: tuple(_to_np(v).shape) for k, v in npz_payload.items()},
        "dtypes": {k: str(_to_np(v).dtype) for k, v in npz_payload.items()},
        "meta": meta,
        "env": env,
        "rng": rng,
    }
    with open(dump_dir / "meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    # Optional grid preview
    if make_grid and all(k in arrays for k in ("image0","image1","H12")):
        img0 = _to_np(arrays["image0"])
        img1 = _to_np(arrays["image1"])
        H12  = _to_np(arrays["H12"])
        h, w = img1.shape[:2]
        warped0 = cv2.warpPerspective(img0, H12, (w, h))
        diff = cv2.absdiff(np.clip(warped0,0,255).astype(np.uint8),
                           np.clip(img1,0,255).astype(np.uint8))
        grid = _make_grid(img0, img1, warped0, diff, *grid_size)
        cv2.imwrite(str(dump_dir / "grid.jpg"), grid[..., ::-1])

    print(f"[crash_dump] wrote -> {dump_dir}")
    return dump_dir

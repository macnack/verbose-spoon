import torch
import cv2
import numpy as np
from collections import OrderedDict
from loguru import logger
from kornia.geometry.epipolar import numeric
from kornia.geometry.conversions import convert_points_to_homogeneous


# --- METRICS ---


def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


def symmetric_epipolar_distance(pts0, pts1, E, K0, K1):
    """Squared symmetric epipolar distance.
    This can be seen as a biased estimation of the reprojection error.
    Args:
        pts0 (torch.Tensor): [N, 2]
        E (torch.Tensor): [3, 3]
    """
    pts0 = (pts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    pts1 = (pts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    pts0 = convert_points_to_homogeneous(pts0)
    pts1 = convert_points_to_homogeneous(pts1)

    Ep0 = pts0 @ E.T  # [N, 3]
    p1Ep0 = torch.sum(pts1 * Ep0, -1)  # [N,]
    Etp1 = pts1 @ E  # [N, 3]

    d = p1Ep0**2 * (
        1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2)
        + 1.0 / (Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2)
    )  # N
    return d


def compute_symmetrical_epipolar_errors(data):
    """
    Update:
        data (dict):{"epi_errs": [M]}
    """
    Tx = numeric.cross_product_matrix(data["T_0to1"][:, :3, 3])
    E_mat = Tx @ data["T_0to1"][:, :3, :3]

    m_bids = data["m_bids"].clone().detach()
    pts0 = data["mkpts0_f"].clone().detach()
    pts1 = data["mkpts1_f"].clone().detach()

    epi_errs = []
    for bs in range(Tx.size(0)):
        mask = m_bids == bs
        epi_errs.append(
            symmetric_epipolar_distance(
                pts0[mask], pts1[mask], E_mat[bs], data["K0"][bs], data["K1"][bs]
            )
        )
    epi_errs = torch.cat(epi_errs, dim=0)

    data.update({"epi_errs": epi_errs})


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC
    )
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret

def estimate_homography(kpts0, kpts1, thresh, conf=0.99999):
    """Estimate homography matrix using RANSAC.
    Args:
        kpts0 (np.ndarray): [N, 2]
        kpts1 (np.ndarray): [N, 2]
        thresh (float): RANSAC pixel threshold.
        conf (float): RANSAC confidence.
    Returns:
        H (np.ndarray): [3, 3] homography matrix.
        mask (np.ndarray): [N] boolean inlier mask.
    """
    if len(kpts0) < 4:
        return None, None
    
    H, mask = cv2.findHomography(
        kpts0, kpts1, cv2.RANSAC, ransacReprojThreshold=thresh, confidence=conf
    )
    return H, mask.ravel() > 0

def estimate_lo_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    from .warppers import Camera, Pose
    import poselib

    camera0, camera1 = (
        Camera.from_calibration_matrix(K0).float(),
        Camera.from_calibration_matrix(K1).float(),
    )
    pts0, pts1 = kpts0, kpts1

    M, info = poselib.estimate_relative_pose(
        pts0,
        pts1,
        camera0.to_cameradict(),
        camera1.to_cameradict(),
        {
            "max_epipolar_error": thresh,
        },
    )
    success = M is not None and (
        ((M.t != [0.0, 0.0, 0.0]).all()) or (
            (M.q != [1.0, 0.0, 0.0, 0.0]).all())
    )
    if success:
        M = Pose.from_Rt(torch.tensor(M.R), torch.tensor(M.t))  # .to(pts0)
        # print(M)
    else:
        M = Pose.from_4x4mat(torch.eye(4).numpy())  # .to(pts0)
        # print(M)

    estimation = {
        "success": success,
        "M_0to1": M,
        "inliers": torch.tensor(info.pop("inliers")),  # .to(pts0),
        **info,
    }
    return estimation


def compute_pose_errors(data, config):
    """
    Update:
        data (dict):{
            "R_errs" List[float]: [N]
            "t_errs" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    pixel_thr = config.TRAINER.RANSAC_PIXEL_THR  # 0.5
    conf = config.TRAINER.RANSAC_CONF  # 0.99999
    RANSAC = config.TRAINER.POSE_ESTIMATION_METHOD
    data.update({"R_errs": [], "t_errs": [], "inliers": []})

    m_bids = data["m_bids"].clone().detach().cpu().numpy()
    pts0 = data["mkpts0_f"].clone().detach().cpu().numpy()
    pts1 = data["mkpts1_f"].clone().detach().cpu().numpy()
    K0 = data["K0"].cpu().numpy()
    K1 = data["K1"].cpu().numpy()
    T_0to1 = data["T_0to1"].cpu().numpy()

    for bs in range(K0.shape[0]):
        mask = m_bids == bs
        if config.EDM.EVAL_TIMES >= 1:
            bpts0, bpts1 = pts0[mask], pts1[mask]
            R_list, T_list, inliers_list = [], [], []
            # for _ in range(config.EDM.EVAL_TIMES):
            for _ in range(5):
                shuffling = np.random.permutation(np.arange(len(bpts0)))
                if _ >= config.EDM.EVAL_TIMES:
                    continue
                bpts0 = bpts0[shuffling]
                bpts1 = bpts1[shuffling]

                if RANSAC == "RANSAC":
                    ret = estimate_pose(
                        bpts0, bpts1, K0[bs], K1[bs], pixel_thr, conf=conf
                    )
                    if ret is None:
                        R_list.append(np.inf)
                        T_list.append(np.inf)
                        inliers_list.append(np.array([]).astype(bool))
                    else:
                        R, t, inliers = ret
                        t_err, R_err = relative_pose_error(
                            T_0to1[bs], R, t, ignore_gt_t_thr=0.0
                        )
                        R_list.append(R_err)
                        T_list.append(t_err)
                        inliers_list.append(inliers)

                elif RANSAC == "LO-RANSAC":
                    est = estimate_lo_pose(
                        bpts0, bpts1, K0[bs], K1[bs], pixel_thr, conf=conf
                    )
                    if not est["success"]:
                        R_list.append(90)
                        T_list.append(90)
                        inliers_list.append(np.array([]).astype(bool))
                    else:
                        M = est["M_0to1"]
                        inl = est["inliers"].numpy()
                        t_error, r_error = relative_pose_error(
                            T_0to1[bs], M.R, M.t, ignore_gt_t_thr=0.0
                        )
                        R_list.append(r_error)
                        T_list.append(t_error)
                        inliers_list.append(inl)
                else:
                    raise ValueError(f"Unknown RANSAC method: {RANSAC}")

            data["R_errs"].append(R_list)
            data["t_errs"].append(T_list)
            data["inliers"].append(inliers_list[0])

def compute_homography_errors(data, config):
    """
    Update:
        data (dict):{
            "corner_error": List[float]: [N] - corner reprojection error
            "H_est": List[np.ndarray]: [N] - estimated homography
            "inliers": List[np.ndarray]: [N] - RANSAC inlier mask
            "sym_transfer_err": List[np.ndarray]: [N] - symmetric transfer error for all matches
        }
    """
    pixel_thr = config.TRAINER.RANSAC_PIXEL_THR  # e.g., 0.5, 1.0, etc.
    conf = config.TRAINER.RANSAC_CONF
    
    data.update({"corner_error": [], "inliers": [], "H_est": [], "sym_transfer_err": []})

    m_bids = data["m_bids"].cpu().numpy()
    pts0 = data["mkpts0_f"].cpu().numpy()
    pts1 = data["mkpts1_f"].cpu().numpy()
    H_gt = data["homography"].cpu().numpy()
    
    h, w = data['image0'].shape[2], data['image0'].shape[3]
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    corners = np.expand_dims(corners, axis=0) # Shape: [1, 4, 2]

    all_transfer_errors = []

    for bs in range(data['image0'].shape[0]):
        mask = m_bids == bs
        bpts0, bpts1 = pts0[mask], pts1[mask]
        H_gt = H_gt[bs]

        # --- Calculate Symmetric Transfer Error for this item's matches ---
        if bpts0.shape[0] > 0:
            bpts0_warped = cv2.perspectiveTransform(bpts0.reshape(-1, 1, 2), H_gt).reshape(-1, 2)
            bpts1_warped = cv2.perspectiveTransform(bpts1.reshape(-1, 1, 2), np.linalg.inv(H_gt)).reshape(-1, 2)
            
            err_0to1 = np.linalg.norm(bpts0_warped - bpts1, axis=1)
            err_1to0 = np.linalg.norm(bpts1_warped - bpts0, axis=1)
            
            transfer_errs = (err_0to1 + err_1to0) / 2.0
        else:
            transfer_errs = np.array([])
            
        all_transfer_errors.append(transfer_errs)
        
        # --- Estimate Homography with RANSAC ---
        H_est, inliers_mask = estimate_homography(bpts0, bpts1, pixel_thr, conf=conf)
        
        if H_est is None:
            # Handle RANSAC failure
            data["corner_error"].append(np.inf)
            data["inliers"].append(np.array([]).astype(bool))
            data["H_est"].append(np.eye(3)) # Placeholder
            continue

        # Store RANSAC results
        data["inliers"].append(inliers_mask)
        data["H_est"].append(H_est)
        
        # --- Compute Corner Error ---
        corners_gt_warped = cv2.perspectiveTransform(corners, H_gt)
        corners_est_warped = cv2.perspectiveTransform(corners, H_est)
        
        corner_error = np.mean(np.linalg.norm(corners_gt_warped - corners_est_warped, axis=2))
        data["corner_error"].append(corner_error)
        
    # --- FINAL STEP: Store all transfer errors as a single flat array ---
    # This makes it easy for the plotting function to use.
    if len(all_transfer_errors) > 0:
        data['sym_transfer_err'] = np.concatenate(all_transfer_errors)
    else:
        data['sym_transfer_err'] = np.array([])
# --- METRIC AGGREGATION ---


def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    thresholds = [5, 10, 20]
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index - 1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f"auc@{t}": auc for t, auc in zip(thresholds, aucs)}


def epidist_prec(errors, thresholds, ret_dict=False):
    precs = []
    for thr in thresholds:
        prec_ = []
        for errs in errors:
            correct_mask = errs < thr
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
    if ret_dict:
        return {f"prec@{t:.0e}": prec for t, prec in zip(thresholds, precs)}
    else:
        return precs


def aggregate_metrics(metrics, epi_err_thr=5e-4, config=None):
    """Aggregate metrics for the whole dataset.
    This function is now flexible and handles both pose and homography metrics.
    """
    # filter duplicates
    unq_ids = OrderedDict((iden, id)
                          for id, iden in enumerate(metrics["identifiers"]))
    unq_ids = list(unq_ids.values())
    logger.info(f"Aggregating metrics over {len(unq_ids)} unique items...")

    # Create a dictionary to hold all aggregated results
    aggregated_metrics = {}

    # --- Pose Metrics (if available) ---
    if 'R_errs' in metrics and 't_errs' in metrics:
        pose_errors = np.max(np.stack([
            np.array(metrics["R_errs"], dtype=object)[unq_ids], 
            np.array(metrics["t_errs"], dtype=object)[unq_ids]
        ]), axis=0)
        
        if config and hasattr(config.EDM, 'EVAL_TIMES') and config.EDM.EVAL_TIMES > 1:
             pose_errors = pose_errors.reshape(-1, config.EDM.EVAL_TIMES).reshape(-1)

        angular_thresholds = [5, 10, 20]
        pose_aucs = error_auc(pose_errors, angular_thresholds)
        aggregated_metrics.update({f'pose_{k}': v for k, v in pose_aucs.items()})

    # --- Epipolar Error Metrics (if available) ---
    if 'epi_errs' in metrics:
        # CORRECTED LINE: Use the function argument epi_err_thr
        dist_thresholds = [epi_err_thr]
        precs = epidist_prec(
            np.array(metrics["epi_errs"], dtype=object)[unq_ids], 
            dist_thresholds, True
        )
        aggregated_metrics.update(precs)
    
    # --- Homography Metrics (if corner error data is present) ---
    # `corner_error` is the key we created in `compute_homography_errors`.
    if 'corner_error' in metrics:
        corner_errors = np.array(metrics["corner_error"])[unq_ids]
        # Filter out any 'inf' values that result from RANSAC failures
        corner_errors = corner_errors[np.isfinite(corner_errors)]
        
        # Use pixel-based thresholds for corner error AUC
        pixel_thresholds = [1, 3, 5, 10]
        h_aucs = error_auc(corner_errors, pixel_thresholds)
        # Store with a distinct name, e.g., 'h_auc@1' for "homography auc"
        aggregated_metrics.update({f'h_{k}': v for k, v in h_aucs.items()})
        
        # Also, calculate the Mean Corner Error (MCE)
        aggregated_metrics['MCE'] = np.mean(corner_errors)

    # --- General Metrics ---
    u_num_matches = np.array(metrics["num_matches"], dtype=object)[unq_ids]
    aggregated_metrics['num_matches'] = u_num_matches.mean()
    
    return aggregated_metrics
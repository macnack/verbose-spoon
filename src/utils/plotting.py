import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def _compute_conf_thresh(data):
    dataset_name = data["dataset_name"][0].lower()
    if dataset_name == "scannet":
        thr = 5e-4
    elif dataset_name == "megadepth":
        thr = 1e-4
    elif dataset_name == "synthetichomography":
        thr = 1e-2 # TODO: check what does this value should be and what it means
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return thr


# --- VISUALIZATION --- #


def make_matching_figure(
    img0,
    img1,
    mkpts0,
    mkpts1,
    color,
    kpts0=None,
    kpts1=None,
    text=[],
    dpi=75,
    path=None,
):
    # draw image pair
    assert (
        mkpts0.shape[0] == mkpts1.shape[0]
    ), f"mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}"
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap="gray")
    axes[1].imshow(img1, cmap="gray")
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c="w", s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c="w", s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [
            matplotlib.lines.Line2D(
                (fkpts0[i, 0], fkpts1[i, 0]),
                (fkpts0[i, 1], fkpts1[i, 1]),
                transform=fig.transFigure,
                c=color[i],
                linewidth=1,
            )
            for i in range(len(mkpts0))
        ]

        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = "k" if img0[:100, :200].mean() > 200 else "w"
    fig.text(
        0.01,
        0.99,
        "\n".join(text),
        transform=fig.axes[0].transAxes,
        fontsize=15,
        va="top",
        ha="left",
        color=txt_color,
    )
    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        return fig

def _make_informative_homography_figure(data, b_id, alpha="dynamic"):
    # --- 1. SETUP ---
    b_mask = data["m_bids"] == b_id
    
    # Threshold for a match to be considered a "ground-truth inlier"
    gt_inlier_thr = 3.0 # pixels

    # Prepare images and keypoints
    img0 = (data["image0"][b_id][0].cpu().numpy() * 255).round().astype(np.uint8)
    img1 = (data["image1"][b_id][0].cpu().numpy() * 255).round().astype(np.uint8)
    kpts0 = data["mkpts0_f"][b_mask].cpu()
    kpts1 = data["mkpts1_f"][b_mask].cpu()

    if kpts0.shape[0] == 0:
        # Handle case with no matches
        text = ["#Matches: 0", "Corner Error: N/A"]
        return make_matching_figure(img0, img1, np.array([]), np.array([]), [], text=text)

    # --- 2. DETERMINE MATCH CATEGORIES ---
    # Get ground-truth homography
    H_gt = data.get('homography_gt', data['homography'])[b_id].cpu()
    
    # a) Find ground-truth inliers (True Positives + False Negatives)
    kpts0_warped = warp_points_torch(kpts0, H_gt)
    err = torch.norm(kpts0_warped - kpts1, dim=1)
    is_gt_inlier = (err < gt_inlier_thr).numpy()

    # b) Get RANSAC inliers (True Positives + False Positives)
    ransac_inliers = data["inliers"][b_id]

    # --- 3. ASSIGN COLORS ---
    # Initialize color array (e.g., to blue for True Negatives)
    # RGBA format: [R, G, B, Alpha]
    colors = np.array([[0.5, 0.5, 1.0, 0.6]] * len(kpts0)) # Light blue
    
    # Red: False Positives (RANSAC chose it, but it's wrong)
    colors[ransac_inliers & ~is_gt_inlier] = [1.0, 0.0, 0.0, 1.0]
    
    # Yellow: False Negatives (RANSAC ignored it, but it's correct)
    colors[~ransac_inliers & is_gt_inlier] = [1.0, 1.0, 0.0, 0.8]
    
    # Green: True Positives (RANSAC chose it, and it's correct)
    colors[ransac_inliers & is_gt_inlier] = [0.0, 1.0, 0.0, 1.0]

    # --- 4. PREPARE TEXT ---
    n_matches = len(kpts0)
    n_gt_inliers = np.sum(is_gt_inlier)
    n_ransac_inliers = np.sum(ransac_inliers)
    n_true_positives = np.sum(ransac_inliers & is_gt_inlier)
    
    precision = n_true_positives / n_ransac_inliers if n_ransac_inliers > 0 else 0
    recall = n_true_positives / n_gt_inliers if n_gt_inliers > 0 else 0
    corner_err = data["corner_error"][b_id]

    text = [
        f"Corner Error: {corner_err:.2f} px",
        f"#Matches: {n_matches}",
        f"  - GT Inliers (<{gt_inlier_thr}px): {n_gt_inliers}",
        f"  - RANSAC Inliers: {n_ransac_inliers}",
        f"RANSAC Precision: {precision:.1%}",
        f"RANSAC Recall: {recall:.1%}"
    ]
    
    # --- 5. CREATE FIGURE ---
    # Convert points to numpy for plotting
    kpts0_np = kpts0.numpy()
    kpts1_np = kpts1.numpy()
    figure = make_matching_figure(img0, img1, kpts0_np, kpts1_np, colors, text=text)
    
    return figure

def _make_evaluation_figure(data, b_id, alpha="dynamic"):
    b_mask = data["m_bids"] == b_id
    conf_thr = _compute_conf_thresh(data)

    img0 = (data["image0"][b_id][0].cpu().numpy()
            * 255).round().astype(np.int32)
    img1 = (data["image1"][b_id][0].cpu().numpy()
            * 255).round().astype(np.int32)
    kpts0 = data["mkpts0_f"][b_mask].clone().detach().cpu().numpy()
    kpts1 = data["mkpts1_f"][b_mask].clone().detach().cpu().numpy()
    gt_inlier_thr = 3.0  # pixels

    # for megadepth, we visualize matches on the resized image
    if "scale0" in data:
        kpts0 = kpts0 / data["scale0"][b_id].cpu().numpy()[[1, 0]]
        kpts1 = kpts1 / data["scale1"][b_id].cpu().numpy()[[1, 0]]
        
    if "epi_errs" in data:
        epi_errs = data["epi_errs"][b_mask].cpu().numpy()
        correct_mask = epi_errs < conf_thr
        precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
        n_correct = np.sum(correct_mask)
        n_gt_matches = int(data["conf_matrix_gt"][b_id].sum().cpu())
        recall = 0 if n_gt_matches == 0 else n_correct / (n_gt_matches)
        # matching info
        if alpha == "dynamic":
            alpha = dynamic_alpha(len(correct_mask))
        color = error_colormap(epi_errs, conf_thr, alpha=alpha)
        text = [
            f"#Matches {len(kpts0)}",
            f"Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}",
            f"Recall({conf_thr:.2e}) ({100 * recall:.1f}%): {n_correct}/{n_gt_matches}",
        ]
        # make the figure
        return make_matching_figure(img0, img1, kpts0, kpts1, color, text=text)

    if "corner_error" in data:
        corner_err = data["corner_error"][b_id]
        errors = data["sym_transfer_err"][b_mask.cpu().numpy()]
        correct_mask = errors < gt_inlier_thr
        n_ransac_inliers = np.sum(data["inliers"][b_id])
        precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
        n_correct = np.sum(correct_mask)
        recall = n_correct / n_ransac_inliers if n_ransac_inliers > 0 else 0
        text = [
            f"Corner Error: {corner_err:.2f} px",
            f"#Matches: {len(kpts0)}",
            f"GT Inliers (<{conf_thr:.1f}px): {n_correct}",
            f"Precision (vs GT): {precision:.1%}",
            f"Recall (vs RANSAC): {recall:.1%}"
        ]
        if alpha == "dynamic":
            alpha = dynamic_alpha(len(correct_mask))
        color = error_colormap(errors, conf_thr, alpha=alpha)

        # make the figure
    return make_matching_figure(img0, img1, kpts0, kpts1, color=color, text=text)


def _make_confidence_figure(data, b_id):
    # TODO: Implement confidence figure
    raise NotImplementedError()


def make_matching_figures(data, config, mode="evaluation"):
    """Make matching figures for a batch.

    Args:
        data (Dict): a batch updated by PL_LoFTR.
        config (Dict): matcher config
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    assert mode in ["evaluation", "confidence", "gt"]  # 'confidence'
    figures = {mode: []}
    for b_id in range(data["image0"].size(0)):
        if mode == "evaluation":
            fig = _make_evaluation_figure(
                data, b_id, alpha=config.TRAINER.PLOT_MATCHES_ALPHA
            )
        elif mode == "confidence":
            fig = _make_confidence_figure(data, b_id)
        else:
            raise ValueError(f"Unknown plot mode: {mode}")
        figures[mode].append(fig)
    return figures


def dynamic_alpha(
    n_matches, milestones=[0, 300, 1000, 2000], alphas=[1.0, 0.8, 0.4, 0.2]
):
    if n_matches == 0:
        return 1.0
    ranges = list(zip(alphas, alphas[1:] + [None]))
    loc = bisect.bisect_right(milestones, n_matches) - 1
    _range = ranges[loc]
    if _range[1] is None:
        return _range[0]
    return _range[1] + (milestones[loc + 1] - n_matches) / (
        milestones[loc + 1] - milestones[loc]
    ) * (_range[0] - _range[1])


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2 - x * 2, x * 2, np.zeros_like(x),
                 np.ones_like(x) * alpha], -1),
        0,
        1,
    )

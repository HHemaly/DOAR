"""GENERATED FILE -- do not edit by hand.

Verbatim function bodies extracted by experiments/journal/tools/extract_frozen_e2.py
from DOAR_21_8_26_V1.ipynb (sha256 b7edd22e84ccef6554018ce6883dff20d2445459bd813083dd2d6686dd5f4ac4), cell 22 (E2-A0-R1 pilot (R1-V1)).
Thresholds, constants and logic are unchanged from the runs reported in the
archived notebook; only module imports were added so the functions can run
outside Colab.
"""
# flake8: noqa
# ruff: noqa
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

def get_border_pixels(image_rgb):

    height, width, _ = image_rgb.shape

    border_y = max(
        2,
        int(round(height * 0.04))
    )

    border_x = max(
        2,
        int(round(width * 0.04))
    )

    top = (
        image_rgb[:border_y, :, :]
        .reshape(-1, 3)
    )

    bottom = (
        image_rgb[height-border_y:, :, :]
        .reshape(-1, 3)
    )

    left = (
        image_rgb[:, :border_x, :]
        .reshape(-1, 3)
    )

    right = (
        image_rgb[:, width-border_x:, :]
        .reshape(-1, 3)
    )

    return np.concatenate(
        [
            top,
            bottom,
            left,
            right,
        ],
        axis=0,
    )

def build_a0_mask(image_rgb):

    image_lab = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2LAB,
    ).astype(np.float32)

    border_rgb = get_border_pixels(
        image_rgb
    )

    border_lab = cv2.cvtColor(
        border_rgb.reshape(-1, 1, 3),
        cv2.COLOR_RGB2LAB,
    ).reshape(-1, 3).astype(np.float32)

    background_lab = np.median(
        border_lab,
        axis=0,
    )

    border_distance = np.linalg.norm(
        border_lab
        -
        background_lab[None, :],
        axis=1,
    )

    background_noise = float(
        np.percentile(
            border_distance,
            98,
        )
    )

    threshold = max(
        10.0,
        background_noise + 3.0,
    )

    distance = np.linalg.norm(
        image_lab
        -
        background_lab[
            None,
            None,
            :
        ],
        axis=2,
    )

    raw_mask = (
        distance
        >
        threshold
    ).astype(np.uint8)

    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            raw_mask,
            connectivity=8,
        )
    )

    image_area = (
        raw_mask.shape[0]
        *
        raw_mask.shape[1]
    )

    min_component_area = max(
        3,
        int(
            round(
                image_area
                * 0.000005
            )
        ),
    )

    cleaned = np.zeros_like(
        raw_mask
    )

    for label in range(
        1,
        n_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= min_component_area:

            cleaned[
                labels == label
            ] = 1

    return {
        "mask":
            cleaned,

        "threshold":
            threshold,

        "background_noise":
            background_noise,

        "background_lab":
            background_lab,
    }

def robust_mad(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    median = np.median(
        values
    )

    mad = np.median(
        np.abs(
            values
            -
            median
        )
    )

    # Gaussian-consistent robust scale.
    return float(
        1.4826
        * mad
    )

def build_border_selector(
    height,
    width,
    fraction=0.06,
):

    border_y = max(
        2,
        int(round(height * fraction))
    )

    border_x = max(
        2,
        int(round(width * fraction))
    )

    selector = np.zeros(
        (height, width),
        dtype=bool,
    )

    selector[:border_y, :] = True
    selector[height-border_y:, :] = True
    selector[:, :border_x] = True
    selector[:, width-border_x:] = True

    return selector

def build_r1_mask(image_rgb):

    image_rgb_float = (
        image_rgb.astype(
            np.float32
        )
    )

    image_lab = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2LAB,
    ).astype(
        np.float32
    )

    image_hsv = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2HSV,
    ).astype(
        np.float32
    )

    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY,
    ).astype(
        np.float32
    )

    height, width = gray.shape
    image_area = height * width


    # ----------------------------------------------------------------------------------------------
    # 7A. Candidate paper pixels
    # ----------------------------------------------------------------------------------------------

    value = image_hsv[:, :, 2]
    saturation = image_hsv[:, :, 1]

    # Relative thresholds make this more robust across scans.
    bright_cut = float(
        np.percentile(
            value,
            60
        )
    )

    saturation_cut = float(
        np.percentile(
            saturation,
            45
        )
    )

    # Do not let thresholds become too permissive.
    bright_cut = max(
        bright_cut,
        150.0,
    )

    saturation_cut = min(
        max(
            saturation_cut,
            12.0,
        ),
        70.0,
    )


    likely_paper_global = (
        (value >= bright_cut)
        &
        (saturation <= saturation_cut)
    )


    border_selector = build_border_selector(
        height,
        width,
        fraction=0.06,
    )


    likely_paper_border = (
        likely_paper_global
        &
        border_selector
    )


    minimum_candidate_count = max(
        100,
        int(
            round(
                image_area
                * 0.002
            )
        ),
    )


    if (
        likely_paper_border.sum()
        >= minimum_candidate_count
    ):

        candidate_selector = (
            likely_paper_border
        )

        paper_source = (
            "border_bright_low_saturation"
        )

    elif (
        likely_paper_global.sum()
        >= minimum_candidate_count
    ):

        candidate_selector = (
            likely_paper_global
        )

        paper_source = (
            "global_bright_low_saturation"
        )

    else:

        # Last deterministic fallback:
        # take brightest 10% of image.
        fallback_cut = float(
            np.percentile(
                value,
                90
            )
        )

        candidate_selector = (
            value
            >= fallback_cut
        )

        paper_source = (
            "brightest_10_percent_fallback"
        )


    candidate_lab = (
        image_lab[
            candidate_selector
        ]
    )

    candidate_gray = (
        gray[
            candidate_selector
        ]
    )

    candidate_sat = (
        saturation[
            candidate_selector
        ]
    )


    # ----------------------------------------------------------------------------------------------
    # 7B. Robust background estimate
    # ----------------------------------------------------------------------------------------------

    background_lab = np.median(
        candidate_lab,
        axis=0,
    )

    background_gray = float(
        np.median(
            candidate_gray
        )
    )

    background_saturation = float(
        np.median(
            candidate_sat
        )
    )


    lab_distance = np.linalg.norm(
        image_lab
        -
        background_lab[
            None,
            None,
            :
        ],
        axis=2,
    )


    candidate_distance = np.linalg.norm(
        candidate_lab
        -
        background_lab[
            None,
            :
        ],
        axis=1,
    )


    distance_median = float(
        np.median(
            candidate_distance
        )
    )

    distance_robust_scale = (
        robust_mad(
            candidate_distance
        )
    )


    # ----------------------------------------------------------------------------------------------
    # 7C. Controlled LAB threshold
    #
    # The original A0 could explode above 200.
    #
    # R1:
    #   median + 4 robust SD
    #
    # with explicit conservative limits.
    # ----------------------------------------------------------------------------------------------

    lab_threshold = (
        distance_median
        +
        4.0
        * distance_robust_scale
        +
        3.0
    )

    lab_threshold = float(
        np.clip(
            lab_threshold,
            10.0,
            38.0,
        )
    )


    mask_lab = (
        lab_distance
        >
        lab_threshold
    )


    # ----------------------------------------------------------------------------------------------
    # 7D. Dark-stroke evidence
    # ----------------------------------------------------------------------------------------------

    candidate_gray_scale = (
        robust_mad(
            candidate_gray
        )
    )

    darkness_delta = max(
        15.0,
        4.0
        * candidate_gray_scale,
    )

    darkness_delta = min(
        darkness_delta,
        55.0,
    )


    dark_threshold = (
        background_gray
        -
        darkness_delta
    )


    mask_dark = (
        gray
        <
        dark_threshold
    )


    # ----------------------------------------------------------------------------------------------
    # 7E. Chromatic evidence
    #
    # A coloured mark may still be similar in luminance to the paper.
    # ----------------------------------------------------------------------------------------------

    chromatic_threshold = max(
        35.0,
        background_saturation
        +
        25.0,
    )

    chromatic_threshold = min(
        chromatic_threshold,
        90.0,
    )


    mask_chromatic = (
        saturation
        >
        chromatic_threshold
    )


    # Avoid treating bright coloured scan tint everywhere as foreground:
    # require some colour-distance from estimated paper too.
    mask_chromatic = (
        mask_chromatic
        &
        (
            lab_distance
            >
            max(
                7.0,
                lab_threshold
                * 0.55
            )
        )
    )


    # ----------------------------------------------------------------------------------------------
    # 7F. Combined evidence
    # ----------------------------------------------------------------------------------------------

    raw_mask = (
        mask_lab
        |
        mask_dark
        |
        mask_chromatic
    ).astype(
        np.uint8
    )


    # ----------------------------------------------------------------------------------------------
    # 7G. Small-component cleanup
    #
    # Keep this conservative because pencil lines can be thin.
    # ----------------------------------------------------------------------------------------------

    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            raw_mask,
            connectivity=8,
        )
    )

    min_component_area = max(
        3,
        int(
            round(
                image_area
                * 0.000003
            )
        ),
    )

    cleaned = np.zeros_like(
        raw_mask
    )

    for label in range(
        1,
        n_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= min_component_area:

            cleaned[
                labels
                == label
            ] = 1


    # ----------------------------------------------------------------------------------------------
    # 7H. Very small closing to join tiny gaps.
    #
    # Only 2×2.
    # ----------------------------------------------------------------------------------------------

    kernel = np.ones(
        (2, 2),
        dtype=np.uint8,
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )


    # ----------------------------------------------------------------------------------------------
    # 7I. Paper-confidence diagnostics
    # ----------------------------------------------------------------------------------------------

    candidate_fraction = float(
        candidate_selector.mean()
    )

    border_candidate_fraction = float(
        (
            likely_paper_border
        ).sum()
        /
        max(
            border_selector.sum(),
            1,
        )
    )


    return {
        "mask":
            cleaned.astype(
                np.uint8
            ),

        "raw_mask":
            raw_mask,

        "mask_lab":
            mask_lab.astype(
                np.uint8
            ),

        "mask_dark":
            mask_dark.astype(
                np.uint8
            ),

        "mask_chromatic":
            mask_chromatic.astype(
                np.uint8
            ),

        "paper_source":
            paper_source,

        "background_lab":
            background_lab,

        "background_gray":
            background_gray,

        "background_saturation":
            background_saturation,

        "lab_threshold":
            lab_threshold,

        "dark_threshold":
            float(
                dark_threshold
            ),

        "chromatic_threshold":
            float(
                chromatic_threshold
            ),

        "candidate_fraction":
            candidate_fraction,

        "border_candidate_fraction":
            border_candidate_fraction,

        "candidate_count":
            int(
                candidate_selector.sum()
            ),

        "min_component_area":
            min_component_area,
    }

def qc_status_from_ratio(
    foreground_ratio
):

    if foreground_ratio <= 0:

        return "NO_FOREGROUND"

    if foreground_ratio < 0.001:

        return "VERY_LOW_FOREGROUND"

    if foreground_ratio > 0.80:

        return "VERY_HIGH_FOREGROUND"

    return "OK"

def mask_iou(
    mask_a,
    mask_b
):

    a = (
        mask_a
        > 0
    )

    b = (
        mask_b
        > 0
    )

    union = np.logical_or(
        a,
        b,
    ).sum()

    if union == 0:

        return 1.0

    intersection = np.logical_and(
        a,
        b,
    ).sum()

    return float(
        intersection
        /
        union
    )

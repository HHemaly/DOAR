"""GENERATED FILE -- do not edit by hand.

Verbatim function bodies extracted by experiments/journal/tools/extract_frozen_e2.py
from E2_A0_R1_V2_FROZEN_UNSEEN_HOLDOUT.ipynb (sha256 311031e7548ca0f728d38da27c3090a707019d2ff6f6f1397a10f3478fce683f), cell 3 (frozen A0 / R1-V2 definitions (verbatim from archive cells 22-24)).
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

def robust_mad(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return 0.0

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

    return float(
        1.4826
        * mad
    )

def get_border_pixels(
    image_rgb
):

    h, w, _ = (
        image_rgb.shape
    )

    by = max(
        2,
        int(
            round(
                h
                * 0.04
            )
        ),
    )

    bx = max(
        2,
        int(
            round(
                w
                * 0.04
            )
        ),
    )

    top = (
        image_rgb[
            :by,
            :,
            :
        ]
        .reshape(
            -1,
            3
        )
    )

    bottom = (
        image_rgb[
            h-by:,
            :,
            :
        ]
        .reshape(
            -1,
            3
        )
    )

    left = (
        image_rgb[
            :,
            :bx,
            :
        ]
        .reshape(
            -1,
            3
        )
    )

    right = (
        image_rgb[
            :,
            w-bx:,
            :
        ]
        .reshape(
            -1,
            3
        )
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

def cleanup_components(
    raw_mask,
    minimum_fraction=0.000005,
):

    raw_mask = (
        raw_mask
        > 0
    ).astype(
        np.uint8
    )

    h, w = (
        raw_mask.shape
    )

    image_area = (
        h
        * w
    )

    min_area = max(
        3,
        int(
            round(
                image_area
                * minimum_fraction
            )
        ),
    )

    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            raw_mask,
            connectivity=8,
        )
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

        if area >= min_area:

            cleaned[
                labels
                == label
            ] = 1

    return (
        cleaned,
        min_area,
    )

def build_a0_mask(
    image_rgb
):

    lab = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2LAB,
    ).astype(
        np.float32
    )

    border_rgb = get_border_pixels(
        image_rgb
    )

    border_lab = cv2.cvtColor(
        border_rgb.reshape(
            -1,
            1,
            3
        ),
        cv2.COLOR_RGB2LAB,
    ).reshape(
        -1,
        3
    ).astype(
        np.float32
    )

    bg_lab = np.median(
        border_lab,
        axis=0,
    )

    border_dist = np.linalg.norm(
        border_lab
        -
        bg_lab[
            None,
            :
        ],
        axis=1,
    )

    background_noise = float(
        np.percentile(
            border_dist,
            98
        )
    )

    threshold = max(
        10.0,
        background_noise
        + 3.0,
    )

    distance = np.linalg.norm(
        lab
        -
        bg_lab[
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
    ).astype(
        np.uint8
    )

    cleaned, min_area = (
        cleanup_components(
            raw_mask,
            0.000005,
        )
    )

    return {
        "mask":
            cleaned,

        "threshold":
            float(
                threshold
            ),

        "background_noise":
            background_noise,

        "background_lab":
            bg_lab,

        "min_component_area":
            min_area,
    }

def build_border_selector(
    height,
    width,
    fraction=0.06,
):

    by = max(
        2,
        int(
            round(
                height
                * fraction
            )
        ),
    )

    bx = max(
        2,
        int(
            round(
                width
                * fraction
            )
        ),
    )

    selector = np.zeros(
        (
            height,
            width
        ),
        dtype=bool,
    )

    selector[
        :by,
        :
    ] = True

    selector[
        height-by:,
        :
    ] = True

    selector[
        :,
        :bx
    ] = True

    selector[
        :,
        width-bx:
    ] = True

    return selector

def border_samples_with_side_ids(
    image_rgb,
    fraction=0.08,
):

    h, w, _ = (
        image_rgb.shape
    )

    by = max(
        2,
        int(
            round(
                h
                * fraction
            )
        ),
    )

    bx = max(
        2,
        int(
            round(
                w
                * fraction
            )
        ),
    )

    samples = []
    side_ids = []

    top = (
        image_rgb[
            :by,
            :,
            :
        ]
        .reshape(
            -1,
            3
        )
    )

    samples.append(
        top
    )

    side_ids.extend(
        [0]
        * len(
            top
        )
    )

    bottom = (
        image_rgb[
            h-by:,
            :,
            :
        ]
        .reshape(
            -1,
            3
        )
    )

    samples.append(
        bottom
    )

    side_ids.extend(
        [1]
        * len(
            bottom
        )
    )

    left = (
        image_rgb[
            :,
            :bx,
            :
        ]
        .reshape(
            -1,
            3
        )
    )

    samples.append(
        left
    )

    side_ids.extend(
        [2]
        * len(
            left
        )
    )

    right = (
        image_rgb[
            :,
            w-bx:,
            :
        ]
        .reshape(
            -1,
            3
        )
    )

    samples.append(
        right
    )

    side_ids.extend(
        [3]
        * len(
            right
        )
    )

    rgb = np.concatenate(
        samples,
        axis=0,
    )

    side_ids = np.asarray(
        side_ids,
        dtype=np.int32,
    )

    return (
        rgb,
        side_ids,
    )

def estimate_support_v2(
    image_rgb
):

    border_rgb, side_ids = (
        border_samples_with_side_ids(
            image_rgb,
            fraction=0.08,
        )
    )

    border_lab = cv2.cvtColor(
        border_rgb.reshape(
            -1,
            1,
            3
        ),
        cv2.COLOR_RGB2LAB,
    ).reshape(
        -1,
        3
    ).astype(
        np.float32
    )


    # ----------------------------------------------------------------------------------------------
    # Limit clustering size deterministically for speed.
    # ----------------------------------------------------------------------------------------------

    max_samples = 12000

    if len(
        border_lab
    ) > max_samples:

        indices = np.linspace(
            0,
            len(
                border_lab
            )
            - 1,
            max_samples,
            dtype=int,
        )

        cluster_lab = (
            border_lab[
                indices
            ]
        )

        cluster_sides = (
            side_ids[
                indices
            ]
        )

    else:

        cluster_lab = (
            border_lab
        )

        cluster_sides = (
            side_ids
        )


    # ----------------------------------------------------------------------------------------------
    # Deterministic K-means.
    # ----------------------------------------------------------------------------------------------

    cv2.setRNGSeed(
        20260824
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS
        +
        cv2.TERM_CRITERIA_MAX_ITER,
        40,
        0.1,
    )

    compactness, labels, centers = (
        cv2.kmeans(
            cluster_lab,
            3,
            None,
            criteria,
            1,
            cv2.KMEANS_PP_CENTERS,
        )
    )

    labels = (
        labels
        .reshape(
            -1
        )
    )

    cluster_records = []


    for k in range(
        3
    ):

        selector = (
            labels
            == k
        )

        count = int(
            selector.sum()
        )

        if count == 0:

            continue

        ratio = (
            count
            /
            len(
                labels
            )
        )

        values = (
            cluster_lab[
                selector
            ]
        )

        center = (
            centers[
                k
            ]
        )

        distances = np.linalg.norm(
            values
            -
            center[
                None,
                :
            ],
            axis=1,
        )

        dispersion = float(
            np.median(
                distances
            )
        )

        dispersion_scale = (
            robust_mad(
                distances
            )
        )

        covered_sides = 0

        side_fractions = []

        for side in range(
            4
        ):

            side_selector = (
                cluster_sides
                == side
            )

            side_total = int(
                side_selector.sum()
            )

            if side_total == 0:

                side_fraction = 0.0

            else:

                side_fraction = float(
                    (
                        selector
                        &
                        side_selector
                    ).sum()
                    /
                    side_total
                )

            side_fractions.append(
                side_fraction
            )

            if (
                side_fraction
                >= 0.05
            ):

                covered_sides += 1


        side_factor = (
            0.70
            +
            0.30
            * (
                covered_sides
                /
                4.0
            )
        )

        compactness_factor = (
            1.0
            /
            (
                1.0
                +
                dispersion
                /
                25.0
            )
        )

        score = (
            ratio
            *
            side_factor
            *
            compactness_factor
        )


        cluster_records.append(
            {
                "cluster":
                    k,

                "count":
                    count,

                "ratio":
                    ratio,

                "covered_sides":
                    covered_sides,

                "side_fractions":
                    side_fractions,

                "dispersion":
                    dispersion,

                "dispersion_scale":
                    dispersion_scale,

                "score":
                    score,

                "center":
                    center,
            }
        )


    best = max(
        cluster_records,
        key=lambda x: x[
            "score"
        ],
    )


    support_center = (
        np.asarray(
            best[
                "center"
            ],
            dtype=np.float32,
        )
    )


    # ----------------------------------------------------------------------------------------------
    # Support radius around dominant cluster.
    # ----------------------------------------------------------------------------------------------

    best_selector = (
        labels
        == best[
            "cluster"
        ]
    )

    best_values = (
        cluster_lab[
            best_selector
        ]
    )

    best_distances = np.linalg.norm(
        best_values
        -
        support_center[
            None,
            :
        ],
        axis=1,
    )

    median_distance = float(
        np.median(
            best_distances
        )
    )

    distance_scale = (
        robust_mad(
            best_distances
        )
    )

    support_radius = (
        median_distance
        +
        4.0
        * distance_scale
        +
        2.0
    )

    support_radius = float(
        np.clip(
            support_radius,
            8.0,
            28.0,
        )
    )


    # ----------------------------------------------------------------------------------------------
    # Determine how much of whole image resembles that support.
    # ----------------------------------------------------------------------------------------------

    h, w = (
        image_rgb.shape[
            :2
        ]
    )

    scale = min(
        1.0,
        320.0
        /
        max(
            h,
            w
        ),
    )

    small_w = max(
        32,
        int(
            round(
                w
                * scale
            )
        ),
    )

    small_h = max(
        32,
        int(
            round(
                h
                * scale
            )
        ),
    )

    small_rgb = cv2.resize(
        image_rgb,
        (
            small_w,
            small_h
        ),
        interpolation=cv2.INTER_AREA,
    )

    small_lab = cv2.cvtColor(
        small_rgb,
        cv2.COLOR_RGB2LAB,
    ).astype(
        np.float32
    )

    whole_distance = np.linalg.norm(
        small_lab
        -
        support_center[
            None,
            None,
            :
        ],
        axis=2,
    )

    whole_support_fraction = float(
        (
            whole_distance
            <= support_radius
        ).mean()
    )


    dominant_ratio = float(
        best[
            "ratio"
        ]
    )

    covered_sides = int(
        best[
            "covered_sides"
        ]
    )

    dispersion = float(
        best[
            "dispersion"
        ]
    )


    # ----------------------------------------------------------------------------------------------
    # Support confidence.
    # ----------------------------------------------------------------------------------------------

    if (
        dominant_ratio
        >= 0.55
        and
        covered_sides
        == 4
        and
        whole_support_fraction
        >= 0.20
        and
        dispersion
        <= 18.0
    ):

        confidence = (
            "HIGH"
        )

    elif (
        dominant_ratio
        >= 0.35
        and
        covered_sides
        >= 3
        and
        whole_support_fraction
        >= 0.12
        and
        dispersion
        <= 25.0
    ):

        confidence = (
            "MEDIUM"
        )

    else:

        confidence = (
            "LOW"
        )


    # ----------------------------------------------------------------------------------------------
    # Support regime.
    #
    # OpenCV LAB L is 0–255.
    # ----------------------------------------------------------------------------------------------

    L = float(
        support_center[
            0
        ]
    )

    if confidence == "LOW":

        regime = (
            "AMBIGUOUS_FULL_CANVAS"
        )

    elif L >= 170:

        regime = (
            "LIGHT_SUPPORT"
        )

    elif L <= 90:

        regime = (
            "DARK_SUPPORT"
        )

    else:

        regime = (
            "MID_SUPPORT"
        )


    return {
        "support_center_lab":
            support_center,

        "support_radius":
            support_radius,

        "dominant_border_ratio":
            dominant_ratio,

        "covered_sides":
            covered_sides,

        "whole_support_fraction":
            whole_support_fraction,

        "cluster_dispersion":
            dispersion,

        "support_confidence":
            confidence,

        "support_regime":
            regime,

        "cluster_score":
            float(
                best[
                    "score"
                ]
            ),
    }

def suppress_smooth_border_floods(
    mask,
    local_residual,
    saturation_difference,
    local_threshold,
):

    mask = (
        mask
        > 0
    ).astype(
        np.uint8
    )

    h, w = (
        mask.shape
    )

    image_area = (
        h
        * w
    )

    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    result = (
        mask.copy()
    )


    for label in range(
        1,
        n_labels
    ):

        component = (
            labels
            == label
        )

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA
            ]
        )

        area_fraction = (
            area
            /
            image_area
        )

        x = int(
            stats[
                label,
                cv2.CC_STAT_LEFT
            ]
        )

        y = int(
            stats[
                label,
                cv2.CC_STAT_TOP
            ]
        )

        width = int(
            stats[
                label,
                cv2.CC_STAT_WIDTH
            ]
        )

        height = int(
            stats[
                label,
                cv2.CC_STAT_HEIGHT
            ]
        )

        touches_border = (
            x <= 1
            or
            y <= 1
            or
            (
                x
                +
                width
            )
            >= (
                w
                - 1
            )
            or
            (
                y
                +
                height
            )
            >= (
                h
                - 1
            )
        )


        if (
            not touches_border
            or
            area_fraction
            < 0.15
        ):

            continue


        mean_local = float(
            np.mean(
                local_residual[
                    component
                ]
            )
        )

        mean_sat_difference = float(
            np.mean(
                saturation_difference[
                    component
                ]
            )
        )


        # ------------------------------------------------------------------------------------------
        # Large smooth border component:
        # likely page shadow / lighting / external border rather than graphic content.
        # ------------------------------------------------------------------------------------------

        if (
            mean_local
            <
            (
                local_threshold
                * 1.15
            )
            and
            mean_sat_difference
            <
            18.0
        ):

            result[
                component
            ] = 0


    return result

def build_v2_mask(
    image_rgb
):

    lab = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2LAB,
    ).astype(
        np.float32
    )

    hsv = cv2.cvtColor(
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


    L_channel = (
        lab[
            :,
            :,
            0
        ]
    )

    saturation = (
        hsv[
            :,
            :,
            1
        ]
    )


    support = estimate_support_v2(
        image_rgb
    )

    support_center = (
        support[
            "support_center_lab"
        ]
    )

    support_radius = float(
        support[
            "support_radius"
        ]
    )

    regime = (
        support[
            "support_regime"
        ]
    )

    confidence = (
        support[
            "support_confidence"
        ]
    )


    h, w = (
        gray.shape
    )

    image_area = (
        h
        * w
    )


    # ----------------------------------------------------------------------------------------------
    # Distance from dominant support colour.
    # ----------------------------------------------------------------------------------------------

    distance = np.linalg.norm(
        lab
        -
        support_center[
            None,
            None,
            :
        ],
        axis=2,
    )


    # ----------------------------------------------------------------------------------------------
    # Slowly varying local background.
    #
    # This reduces sensitivity to photo lighting gradients.
    # ----------------------------------------------------------------------------------------------

    sigma = max(
        5.0,
        min(
            h,
            w
        )
        /
        55.0,
    )


    local_L_background = cv2.GaussianBlur(
        L_channel,
        (
            0,
            0
        ),
        sigmaX=sigma,
        sigmaY=sigma,
    )


    local_residual = np.abs(
        L_channel
        -
        local_L_background
    )


    local_sat_background = cv2.GaussianBlur(
        saturation,
        (
            0,
            0
        ),
        sigmaX=sigma,
        sigmaY=sigma,
    )


    local_sat_residual = np.abs(
        saturation
        -
        local_sat_background
    )


    # ==============================================================================================
    # 11A. AMBIGUOUS / FULL-CANVAS MODE
    #
    # We do NOT claim to know the blank-page/background support.
    #
    # Build conservative structural-content mask only for visualization.
    # Final QC remains AMBIGUOUS_SUPPORT.
    # ==============================================================================================

    if regime == "AMBIGUOUS_FULL_CANVAS":

        local_scale = robust_mad(
            local_residual.reshape(
                -1
            )
        )

        local_threshold = float(
            np.clip(
                np.median(
                    local_residual
                )
                +
                4.5
                * local_scale,
                6.0,
                22.0,
            )
        )


        sat_scale = robust_mad(
            local_sat_residual.reshape(
                -1
            )
        )

        sat_threshold = float(
            np.clip(
                np.median(
                    local_sat_residual
                )
                +
                4.0
                * sat_scale,
                8.0,
                35.0,
            )
        )


        gray_u8 = np.clip(
            gray,
            0,
            255,
        ).astype(
            np.uint8
        )


        edges = cv2.Canny(
            gray_u8,
            50,
            150,
        ) > 0


        edge_kernel = np.ones(
            (
                2,
                2
            ),
            dtype=np.uint8,
        )


        edges = (
            cv2.dilate(
                edges.astype(
                    np.uint8
                ),
                edge_kernel,
                iterations=1,
            )
            > 0
        )


        raw = (
            (
                local_residual
                >
                local_threshold
            )
            |
            (
                local_sat_residual
                >
                sat_threshold
            )
            |
            edges
        ).astype(
            np.uint8
        )


        cleaned, min_area = (
            cleanup_components(
                raw,
                0.000003,
            )
        )


        final_mask = (
            cleaned
        )


        return {
            "mask":
                final_mask,

            "support_regime":
                regime,

            "support_confidence":
                confidence,

            "support_radius":
                support_radius,

            "dominant_border_ratio":
                support[
                    "dominant_border_ratio"
                ],

            "covered_sides":
                support[
                    "covered_sides"
                ],

            "whole_support_fraction":
                support[
                    "whole_support_fraction"
                ],

            "cluster_dispersion":
                support[
                    "cluster_dispersion"
                ],

            "global_threshold":
                np.nan,

            "local_threshold":
                local_threshold,

            "sat_threshold":
                sat_threshold,

            "mask_strategy":
                "AMBIGUOUS_LOCAL_CONTENT_ONLY",

            "min_component_area":
                min_area,
        }


    # ==============================================================================================
    # 11B. RELIABLE SUPPORT
    # ==============================================================================================

    support_selector = (
        distance
        <= support_radius
    )


    # ----------------------------------------------------------------------------------------------
    # Guard against too few support pixels.
    # ----------------------------------------------------------------------------------------------

    if (
        support_selector.sum()
        <
        max(
            100,
            int(
                0.002
                * image_area
            ),
        )
    ):

        # Conservative fallback.
        support_selector = (
            distance
            <= (
                support_radius
                * 1.5
            )
        )


    support_L = (
        L_channel[
            support_selector
        ]
    )

    support_sat = (
        saturation[
            support_selector
        ]
    )

    support_local_residual = (
        local_residual[
            support_selector
        ]
    )


    bg_L = float(
        np.median(
            support_L
        )
    )

    bg_sat = float(
        np.median(
            support_sat
        )
    )


    # ----------------------------------------------------------------------------------------------
    # Global support-difference threshold.
    # ----------------------------------------------------------------------------------------------

    global_threshold = float(
        np.clip(
            support_radius
            +
            6.0,
            14.0,
            42.0,
        )
    )


    # ----------------------------------------------------------------------------------------------
    # Local line/stroke threshold.
    # ----------------------------------------------------------------------------------------------

    local_threshold = (
        np.median(
            support_local_residual
        )
        +
        5.0
        * robust_mad(
            support_local_residual
        )
    )

    local_threshold = float(
        np.clip(
            local_threshold,
            5.0,
            18.0,
        )
    )


    # ----------------------------------------------------------------------------------------------
    # Luminance deviation threshold.
    #
    # Symmetric:
    # works for dark ink on light support AND light chalk on dark support.
    # ----------------------------------------------------------------------------------------------

    L_scale = robust_mad(
        support_L
    )

    luminance_delta = float(
        np.clip(
            max(
                22.0,
                4.0
                * L_scale,
            ),
            22.0,
            55.0,
        )
    )


    luminance_difference = np.abs(
        L_channel
        -
        bg_L
    )


    strong_luminance = (
        luminance_difference
        >
        luminance_delta
    )


    # ----------------------------------------------------------------------------------------------
    # Chromatic difference.
    # ----------------------------------------------------------------------------------------------

    sat_scale = robust_mad(
        support_sat
    )

    saturation_delta = float(
        np.clip(
            max(
                22.0,
                4.0
                * sat_scale,
            ),
            22.0,
            60.0,
        )
    )


    saturation_difference = np.abs(
        saturation
        -
        bg_sat
    )


    chromatic_difference = (
        saturation_difference
        >
        saturation_delta
    )


    # ----------------------------------------------------------------------------------------------
    # Local structure.
    # ----------------------------------------------------------------------------------------------

    local_structure = (
        local_residual
        >
        local_threshold
    )


    # ----------------------------------------------------------------------------------------------
    # Candidate foreground.
    #
    # A pixel can become foreground if:
    #
    # A) local line/edge evidence
    #
    # OR
    #
    # B) it is globally different from support AND has strong luminance
    #    or chromatic evidence.
    #
    # This prevents broad mild illumination gradients from entering solely
    # because they differ somewhat from one global support centre.
    # ----------------------------------------------------------------------------------------------

    global_difference = (
        distance
        >
        global_threshold
    )


    raw = (
        local_structure
        |
        (
            global_difference
            &
            (
                strong_luminance
                |
                chromatic_difference
            )
        )
    ).astype(
        np.uint8
    )


    # ----------------------------------------------------------------------------------------------
    # Remove huge smooth border-connected floods.
    # ----------------------------------------------------------------------------------------------

    raw = suppress_smooth_border_floods(
        mask=
            raw,

        local_residual=
            local_residual,

        saturation_difference=
            saturation_difference,

        local_threshold=
            local_threshold,
    )


    cleaned, min_area = (
        cleanup_components(
            raw,
            0.000003,
        )
    )


    # ----------------------------------------------------------------------------------------------
    # Minimal closing only.
    # ----------------------------------------------------------------------------------------------

    kernel = np.ones(
        (
            2,
            2
        ),
        dtype=np.uint8,
    )


    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )


    return {
        "mask":
            cleaned,

        "support_regime":
            regime,

        "support_confidence":
            confidence,

        "support_radius":
            support_radius,

        "dominant_border_ratio":
            support[
                "dominant_border_ratio"
            ],

        "covered_sides":
            support[
                "covered_sides"
            ],

        "whole_support_fraction":
            support[
                "whole_support_fraction"
            ],

        "cluster_dispersion":
            support[
                "cluster_dispersion"
            ],

        "global_threshold":
            global_threshold,

        "local_threshold":
            local_threshold,

        "sat_threshold":
            saturation_delta,

        "mask_strategy":
            "SUPPORT_AWARE_GLOBAL_PLUS_LOCAL",

        "min_component_area":
            min_area,
    }

def border_foreground_fraction(
    mask,
    fraction=0.05,
):

    h, w = (
        mask.shape
    )

    selector = (
        build_border_selector(
            h,
            w,
            fraction,
        )
    )

    return float(
        (
            mask[
                selector
            ]
            > 0
        ).mean()
    )

def v2_qc_status(
    foreground_ratio,
    border_fg_ratio,
    support_regime,
):

    if (
        support_regime
        == "AMBIGUOUS_FULL_CANVAS"
    ):

        return (
            "AMBIGUOUS_SUPPORT"
        )

    if foreground_ratio <= 0:

        return (
            "NO_FOREGROUND"
        )

    if foreground_ratio < 0.001:

        return (
            "VERY_LOW_FOREGROUND"
        )

    if (
        foreground_ratio
        > 0.80
    ):

        return (
            "VERY_HIGH_FOREGROUND"
        )

    if (
        foreground_ratio
        > 0.50
        and
        border_fg_ratio
        > 0.75
    ):

        return (
            "POSSIBLE_BACKGROUND_FLOOD"
        )

    return (
        "OK"
    )

def measurement_policy(
    qc_status,
    support_confidence,
):

    if qc_status in [
        "AMBIGUOUS_SUPPORT",
        "NO_FOREGROUND",
        "VERY_LOW_FOREGROUND",
    ]:

        return (
            "MASK_UNRELIABLE"
        )


    if qc_status in [
        "POSSIBLE_BACKGROUND_FLOOD",
        "VERY_HIGH_FOREGROUND",
    ]:

        return (
            "MASK_REVIEW_REQUIRED"
        )


    if (
        support_confidence
        in [
            "HIGH",
            "MEDIUM",
        ]
        and
        qc_status
        == "OK"
    ):

        return (
            "MASK_USABLE"
        )


    return (
        "MASK_REVIEW_REQUIRED"
    )

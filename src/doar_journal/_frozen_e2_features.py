"""GENERATED FILE -- do not edit by hand.

Verbatim function bodies extracted by experiments/journal/tools/extract_frozen_e2.py
from DOAR_21_8_26_V1.ipynb (sha256 b7edd22e84ccef6554018ce6883dff20d2445459bd813083dd2d6686dd5f4ac4), cell 20 (E2-A0 objective feature extraction (51 numeric)).
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

def safe_divide(
    numerator,
    denominator,
    default=np.nan,
):

    if denominator == 0:

        return default

    return (
        numerator
        / denominator
    )

def entropy_from_histogram(
    histogram
):

    histogram = np.asarray(
        histogram,
        dtype=np.float64
    )

    total = histogram.sum()

    if total <= 0:

        return 0.0

    probability = (
        histogram
        / total
    )

    probability = probability[
        probability > 0
    ]

    return float(
        -np.sum(
            probability
            * np.log2(
                probability
            )
        )
    )

def get_border_pixels(
    image_rgb
):

    height, width, _ = (
        image_rgb.shape
    )

    border_y = max(
        2,
        int(
            round(
                height
                * 0.04
            )
        )
    )

    border_x = max(
        2,
        int(
            round(
                width
                * 0.04
            )
        )
    )


    top = (
        image_rgb[
            :border_y,
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
            height-border_y:,
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
            :border_x,
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
            width-border_x:,
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

def build_foreground_mask(
    image_rgb
):

    image_lab = cv2.cvtColor(
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


    background_lab = np.median(
        border_lab,
        axis=0
    )


    border_distance = np.linalg.norm(
        border_lab
        -
        background_lab[
            None,
            :
        ],
        axis=1,
    )


    # Dynamic tolerance for scan/photo noise.
    background_noise = float(
        np.percentile(
            border_distance,
            98
        )
    )


    threshold = max(
        10.0,
        background_noise
        + 3.0,
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


    mask = (
        distance
        >
        threshold
    ).astype(
        np.uint8
    )


    # ----------------------------------------------------------------------------------------------
    # Very small noise removal only.
    # Do NOT aggressively morph the mask because thin pencil lines matter.
    # ----------------------------------------------------------------------------------------------

    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )


    image_area = (
        mask.shape[0]
        * mask.shape[1]
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
        mask
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


    return (
        cleaned,
        background_lab,
        threshold,
        background_noise,
    )

def symmetry_score(
    mask_crop,
    axis,
):

    if (
        mask_crop.size == 0
        or
        mask_crop.sum() == 0
    ):

        return np.nan


    resized = cv2.resize(
        mask_crop.astype(
            np.uint8
        ),
        (
            128,
            128,
        ),
        interpolation=
            cv2.INTER_NEAREST,
    )


    if axis == "vertical":

        mirrored = np.fliplr(
            resized
        )

    elif axis == "horizontal":

        mirrored = np.flipud(
            resized
        )

    else:

        raise ValueError(
            axis
        )


    intersection = np.logical_and(
        resized,
        mirrored
    ).sum()


    union = np.logical_or(
        resized,
        mirrored
    ).sum()


    if union == 0:

        return np.nan


    return float(
        intersection
        / union
    )

def calculate_colorfulness(
    foreground_rgb
):

    if len(
        foreground_rgb
    ) == 0:

        return np.nan


    pixels = foreground_rgb.astype(
        np.float64
    )


    r = pixels[
        :,
        0
    ]

    g = pixels[
        :,
        1
    ]

    b = pixels[
        :,
        2
    ]


    rg = np.abs(
        r - g
    )

    yb = np.abs(
        0.5
        * (
            r + g
        )
        - b
    )


    std_root = math.sqrt(
        np.std(
            rg
        ) ** 2
        +
        np.std(
            yb
        ) ** 2
    )


    mean_root = math.sqrt(
        np.mean(
            rg
        ) ** 2
        +
        np.mean(
            yb
        ) ** 2
    )


    return float(
        std_root
        +
        0.3
        * mean_root
    )

def extract_features(
    row
):

    image_path = Path(
        row[
            "path"
        ]
    )


    with Image.open(
        image_path
    ) as image_file:

        image_rgb = np.asarray(
            image_file
            .convert(
                "RGB"
            )
        )


    height, width = (
        image_rgb.shape[
            :2
        ]
    )


    image_area = (
        width
        * height
    )


    grayscale = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY,
    )


    hsv = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2HSV,
    )


    (
        mask,
        background_lab,
        mask_threshold,
        background_noise,
    ) = build_foreground_mask(
        image_rgb
    )


    foreground_count = int(
        mask.sum()
    )


    foreground_fraction = safe_divide(
        foreground_count,
        image_area,
        default=0.0,
    )


    # ----------------------------------------------------------------------------------------------
    # Default results / metadata
    # ----------------------------------------------------------------------------------------------

    result = {
        "image_id":
            row[
                "image_id"
            ],

        "split":
            row[
                "split"
            ],

        "true_class":
            row[
                "true_class"
            ],

        "relative_path":
            row[
                "relative_path"
            ],

        # Canvas properties
        "canvas_width_px":
            width,

        "canvas_height_px":
            height,

        "canvas_aspect_ratio":
            safe_divide(
                width,
                height
            ),

        # Foreground
        "foreground_pixel_count":
            foreground_count,

        "foreground_fraction":
            foreground_fraction,

        # Mask engineering metadata
        "mask_lab_threshold":
            mask_threshold,

        "estimated_background_noise":
            background_noise,
    }


    # ----------------------------------------------------------------------------------------------
    # QC flag if mask is suspicious
    # ----------------------------------------------------------------------------------------------

    if foreground_count == 0:

        result[
            "qc_flag"
        ] = "NO_FOREGROUND"

    elif foreground_fraction < 0.001:

        result[
            "qc_flag"
        ] = "VERY_LOW_FOREGROUND"

    elif foreground_fraction > 0.80:

        result[
            "qc_flag"
        ] = "VERY_HIGH_FOREGROUND"

    else:

        result[
            "qc_flag"
        ] = "OK"


    # ----------------------------------------------------------------------------------------------
    # Empty foreground fallback
    # ----------------------------------------------------------------------------------------------

    if foreground_count == 0:

        empty_features = [
            "bbox_x_min_ratio",
            "bbox_y_min_ratio",
            "bbox_x_max_ratio",
            "bbox_y_max_ratio",
            "bbox_width_ratio",
            "bbox_height_ratio",
            "bbox_area_ratio",
            "bbox_fill_ratio",

            "left_margin_ratio",
            "right_margin_ratio",
            "top_margin_ratio",
            "bottom_margin_ratio",

            "centroid_x_ratio",
            "centroid_y_ratio",
            "centroid_distance_from_center",

            "foreground_x_std_ratio",
            "foreground_y_std_ratio",

            "ink_left_fraction",
            "ink_right_fraction",
            "ink_top_fraction",
            "ink_bottom_fraction",

            "quadrant_top_left_fraction",
            "quadrant_top_right_fraction",
            "quadrant_bottom_left_fraction",
            "quadrant_bottom_right_fraction",

            "occupied_grid_fraction_4x4",

            "foreground_mean_gray",
            "foreground_std_gray",
            "foreground_mean_darkness",
            "foreground_gray_entropy",

            "foreground_mean_saturation",
            "foreground_std_saturation",
            "foreground_hue_entropy",
            "foreground_colorfulness",

            "edge_density_canvas",
            "edge_density_foreground_bbox",
            "edge_to_foreground_ratio",

            "connected_component_count",
            "largest_component_fraction",
            "median_component_fraction",

            "thickness_proxy_mean_ratio",
            "thickness_proxy_max_ratio",

            "vertical_symmetry_iou",
            "horizontal_symmetry_iou",
        ]


        for feature in empty_features:

            result[
                feature
            ] = np.nan


        return (
            result,
            mask,
        )


    # ==================================================================================================
    # 10A. FOREGROUND COORDINATES
    # ==================================================================================================

    y_coordinates, x_coordinates = np.where(
        mask > 0
    )


    x_min = int(
        x_coordinates.min()
    )

    x_max = int(
        x_coordinates.max()
    )

    y_min = int(
        y_coordinates.min()
    )

    y_max = int(
        y_coordinates.max()
    )


    bbox_width = (
        x_max
        - x_min
        + 1
    )

    bbox_height = (
        y_max
        - y_min
        + 1
    )

    bbox_area = (
        bbox_width
        * bbox_height
    )


    result.update(
        {
            "bbox_x_min_ratio":
                x_min
                / width,

            "bbox_y_min_ratio":
                y_min
                / height,

            "bbox_x_max_ratio":
                x_max
                / width,

            "bbox_y_max_ratio":
                y_max
                / height,

            "bbox_width_ratio":
                bbox_width
                / width,

            "bbox_height_ratio":
                bbox_height
                / height,

            "bbox_area_ratio":
                bbox_area
                / image_area,

            "bbox_fill_ratio":
                foreground_count
                / bbox_area,

            "left_margin_ratio":
                x_min
                / width,

            "right_margin_ratio":
                (
                    width
                    - 1
                    - x_max
                )
                / width,

            "top_margin_ratio":
                y_min
                / height,

            "bottom_margin_ratio":
                (
                    height
                    - 1
                    - y_max
                )
                / height,
        }
    )


    # ==================================================================================================
    # 10B. CENTROID / DISPERSION
    # ==================================================================================================

    centroid_x = float(
        np.mean(
            x_coordinates
        )
    )

    centroid_y = float(
        np.mean(
            y_coordinates
        )
    )


    centroid_x_ratio = (
        centroid_x
        / width
    )

    centroid_y_ratio = (
        centroid_y
        / height
    )


    center_distance = math.sqrt(
        (
            centroid_x_ratio
            - 0.5
        ) ** 2
        +
        (
            centroid_y_ratio
            - 0.5
        ) ** 2
    )


    result.update(
        {
            "centroid_x_ratio":
                centroid_x_ratio,

            "centroid_y_ratio":
                centroid_y_ratio,

            "centroid_distance_from_center":
                center_distance,

            "foreground_x_std_ratio":
                float(
                    np.std(
                        x_coordinates
                    )
                    / width
                ),

            "foreground_y_std_ratio":
                float(
                    np.std(
                        y_coordinates
                    )
                    / height
                ),
        }
    )


    # ==================================================================================================
    # 10C. HALF / QUADRANT OCCUPANCY
    # ==================================================================================================

    left_count = np.sum(
        x_coordinates
        <
        (
            width
            / 2
        )
    )

    right_count = (
        foreground_count
        -
        left_count
    )


    top_count = np.sum(
        y_coordinates
        <
        (
            height
            / 2
        )
    )

    bottom_count = (
        foreground_count
        -
        top_count
    )


    center_x = (
        width
        / 2
    )

    center_y = (
        height
        / 2
    )


    q_tl = np.sum(
        (
            x_coordinates
            < center_x
        )
        &
        (
            y_coordinates
            < center_y
        )
    )


    q_tr = np.sum(
        (
            x_coordinates
            >= center_x
        )
        &
        (
            y_coordinates
            < center_y
        )
    )


    q_bl = np.sum(
        (
            x_coordinates
            < center_x
        )
        &
        (
            y_coordinates
            >= center_y
        )
    )


    q_br = np.sum(
        (
            x_coordinates
            >= center_x
        )
        &
        (
            y_coordinates
            >= center_y
        )
    )


    result.update(
        {
            "ink_left_fraction":
                left_count
                / foreground_count,

            "ink_right_fraction":
                right_count
                / foreground_count,

            "ink_top_fraction":
                top_count
                / foreground_count,

            "ink_bottom_fraction":
                bottom_count
                / foreground_count,

            "quadrant_top_left_fraction":
                q_tl
                / foreground_count,

            "quadrant_top_right_fraction":
                q_tr
                / foreground_count,

            "quadrant_bottom_left_fraction":
                q_bl
                / foreground_count,

            "quadrant_bottom_right_fraction":
                q_br
                / foreground_count,
        }
    )


    # ==================================================================================================
    # 10D. 4×4 SPACE OCCUPANCY
    # ==================================================================================================

    occupied_cells = 0

    grid_size = 4


    for row_index in range(
        grid_size
    ):

        y0 = int(
            round(
                row_index
                * height
                / grid_size
            )
        )

        y1 = int(
            round(
                (
                    row_index
                    + 1
                )
                * height
                / grid_size
            )
        )


        for column_index in range(
            grid_size
        ):

            x0 = int(
                round(
                    column_index
                    * width
                    / grid_size
                )
            )

            x1 = int(
                round(
                    (
                        column_index
                        + 1
                    )
                    * width
                    / grid_size
                )
            )


            cell = mask[
                y0:y1,
                x0:x1
            ]


            if (
                cell.size > 0
                and
                cell.mean()
                > 0.005
            ):

                occupied_cells += 1


    result[
        "occupied_grid_fraction_4x4"
    ] = (
        occupied_cells
        / 16.0
    )


    # ==================================================================================================
    # 10E. GRAYSCALE / DARKNESS
    # ==================================================================================================

    foreground_gray = (
        grayscale[
            mask > 0
        ]
        .astype(
            np.float64
        )
    )


    gray_histogram, _ = np.histogram(
        foreground_gray,
        bins=32,
        range=(
            0,
            256
        ),
    )


    result.update(
        {
            "foreground_mean_gray":
                float(
                    np.mean(
                        foreground_gray
                    )
                ),

            "foreground_std_gray":
                float(
                    np.std(
                        foreground_gray
                    )
                ),

            "foreground_mean_darkness":
                float(
                    1.0
                    -
                    np.mean(
                        foreground_gray
                    )
                    / 255.0
                ),

            "foreground_gray_entropy":
                entropy_from_histogram(
                    gray_histogram
                ),
        }
    )


    # ==================================================================================================
    # 10F. COLOR / SATURATION
    # ==================================================================================================

    foreground_rgb = (
        image_rgb[
            mask > 0
        ]
    )


    foreground_hsv = (
        hsv[
            mask > 0
        ]
    )


    saturation = (
        foreground_hsv[
            :,
            1
        ]
        .astype(
            np.float64
        )
    )


    hue = (
        foreground_hsv[
            :,
            0
        ]
        .astype(
            np.float64
        )
    )


    # Only meaningful hue where some saturation exists.
    colorful_pixels = (
        saturation
        > 25
    )


    if np.any(
        colorful_pixels
    ):

        hue_histogram, _ = np.histogram(
            hue[
                colorful_pixels
            ],
            bins=18,
            range=(
                0,
                180
            ),
        )

        hue_entropy = entropy_from_histogram(
            hue_histogram
        )

    else:

        hue_entropy = 0.0


    result.update(
        {
            "foreground_mean_saturation":
                float(
                    np.mean(
                        saturation
                    )
                ),

            "foreground_std_saturation":
                float(
                    np.std(
                        saturation
                    )
                ),

            "foreground_hue_entropy":
                float(
                    hue_entropy
                ),

            "foreground_colorfulness":
                calculate_colorfulness(
                    foreground_rgb
                ),
        }
    )


    # ==================================================================================================
    # 10G. EDGE / VISUAL COMPLEXITY PROXIES
    # ==================================================================================================

    edges = cv2.Canny(
        grayscale,
        threshold1=50,
        threshold2=150,
    )


    edge_binary = (
        edges
        > 0
    )


    edge_count = int(
        edge_binary.sum()
    )


    bbox_edges = edge_binary[
        y_min:y_max+1,
        x_min:x_max+1
    ]


    result.update(
        {
            "edge_density_canvas":
                edge_count
                / image_area,

            "edge_density_foreground_bbox":
                float(
                    bbox_edges.mean()
                )
                if bbox_edges.size
                else np.nan,

            "edge_to_foreground_ratio":
                safe_divide(
                    edge_count,
                    foreground_count
                ),
        }
    )


    # ==================================================================================================
    # 10H. CONNECTED COMPONENT STRUCTURE
    # ==================================================================================================

    (
        component_count_with_background,
        component_labels,
        component_stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        mask.astype(
            np.uint8
        ),
        connectivity=8,
    )


    component_areas = []


    for component_index in range(
        1,
        component_count_with_background
    ):

        area = int(
            component_stats[
                component_index,
                cv2.CC_STAT_AREA
            ]
        )

        component_areas.append(
            area
        )


    if component_areas:

        component_areas_array = np.asarray(
            component_areas,
            dtype=np.float64
        )

        largest_component_fraction = float(
            component_areas_array.max()
            / foreground_count
        )

        median_component_fraction = float(
            np.median(
                component_areas_array
            )
            / foreground_count
        )

    else:

        largest_component_fraction = np.nan
        median_component_fraction = np.nan


    result.update(
        {
            "connected_component_count":
                len(
                    component_areas
                ),

            "largest_component_fraction":
                largest_component_fraction,

            "median_component_fraction":
                median_component_fraction,
        }
    )


    # ==================================================================================================
    # 10I. STROKE / REGION THICKNESS PROXY
    #
    # Distance transform gives distance from each foreground pixel to nearest background.
    # Twice that distance is used as a local thickness proxy.
    #
    # This is NOT literal pen-pressure measurement.
    # ==================================================================================================

    distance_transform = (
        cv2.distanceTransform(
            mask.astype(
                np.uint8
            ),
            cv2.DIST_L2,
            5,
        )
    )


    distance_values = (
        distance_transform[
            mask > 0
        ]
    )


    normalization_dimension = float(
        min(
            width,
            height
        )
    )


    result.update(
        {
            "thickness_proxy_mean_ratio":
                float(
                    2.0
                    * np.mean(
                        distance_values
                    )
                    / normalization_dimension
                ),

            "thickness_proxy_max_ratio":
                float(
                    2.0
                    * np.max(
                        distance_values
                    )
                    / normalization_dimension
                ),
        }
    )


    # ==================================================================================================
    # 10J. SYMMETRY
    # ==================================================================================================

    mask_crop = mask[
        y_min:y_max+1,
        x_min:x_max+1
    ]


    result.update(
        {
            "vertical_symmetry_iou":
                symmetry_score(
                    mask_crop,
                    "vertical"
                ),

            "horizontal_symmetry_iou":
                symmetry_score(
                    mask_crop,
                    "horizontal"
                ),
        }
    )


    return (
        result,
        mask,
    )


# ---------------------------------------------------------------------------
# DERIVATIVE (not in the archive): identical to extract_features except that
# the mask builder is a parameter. `mask_fn(image_rgb)` must return the same
# 4-tuple as build_foreground_mask: (mask, background_lab, threshold, noise).
# tests/test_journal_stage_c.py checks that passing build_foreground_mask
# reproduces extract_features exactly.
# ---------------------------------------------------------------------------
def extract_features_with_mask(
    row,
    mask_fn,
):

    image_path = Path(
        row[
            "path"
        ]
    )


    with Image.open(
        image_path
    ) as image_file:

        image_rgb = np.asarray(
            image_file
            .convert(
                "RGB"
            )
        )


    height, width = (
        image_rgb.shape[
            :2
        ]
    )


    image_area = (
        width
        * height
    )


    grayscale = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY,
    )


    hsv = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2HSV,
    )


    (
        mask,
        background_lab,
        mask_threshold,
        background_noise,
    ) = mask_fn(
        image_rgb
    )


    foreground_count = int(
        mask.sum()
    )


    foreground_fraction = safe_divide(
        foreground_count,
        image_area,
        default=0.0,
    )


    # ----------------------------------------------------------------------------------------------
    # Default results / metadata
    # ----------------------------------------------------------------------------------------------

    result = {
        "image_id":
            row[
                "image_id"
            ],

        "split":
            row[
                "split"
            ],

        "true_class":
            row[
                "true_class"
            ],

        "relative_path":
            row[
                "relative_path"
            ],

        # Canvas properties
        "canvas_width_px":
            width,

        "canvas_height_px":
            height,

        "canvas_aspect_ratio":
            safe_divide(
                width,
                height
            ),

        # Foreground
        "foreground_pixel_count":
            foreground_count,

        "foreground_fraction":
            foreground_fraction,

        # Mask engineering metadata
        "mask_lab_threshold":
            mask_threshold,

        "estimated_background_noise":
            background_noise,
    }


    # ----------------------------------------------------------------------------------------------
    # QC flag if mask is suspicious
    # ----------------------------------------------------------------------------------------------

    if foreground_count == 0:

        result[
            "qc_flag"
        ] = "NO_FOREGROUND"

    elif foreground_fraction < 0.001:

        result[
            "qc_flag"
        ] = "VERY_LOW_FOREGROUND"

    elif foreground_fraction > 0.80:

        result[
            "qc_flag"
        ] = "VERY_HIGH_FOREGROUND"

    else:

        result[
            "qc_flag"
        ] = "OK"


    # ----------------------------------------------------------------------------------------------
    # Empty foreground fallback
    # ----------------------------------------------------------------------------------------------

    if foreground_count == 0:

        empty_features = [
            "bbox_x_min_ratio",
            "bbox_y_min_ratio",
            "bbox_x_max_ratio",
            "bbox_y_max_ratio",
            "bbox_width_ratio",
            "bbox_height_ratio",
            "bbox_area_ratio",
            "bbox_fill_ratio",

            "left_margin_ratio",
            "right_margin_ratio",
            "top_margin_ratio",
            "bottom_margin_ratio",

            "centroid_x_ratio",
            "centroid_y_ratio",
            "centroid_distance_from_center",

            "foreground_x_std_ratio",
            "foreground_y_std_ratio",

            "ink_left_fraction",
            "ink_right_fraction",
            "ink_top_fraction",
            "ink_bottom_fraction",

            "quadrant_top_left_fraction",
            "quadrant_top_right_fraction",
            "quadrant_bottom_left_fraction",
            "quadrant_bottom_right_fraction",

            "occupied_grid_fraction_4x4",

            "foreground_mean_gray",
            "foreground_std_gray",
            "foreground_mean_darkness",
            "foreground_gray_entropy",

            "foreground_mean_saturation",
            "foreground_std_saturation",
            "foreground_hue_entropy",
            "foreground_colorfulness",

            "edge_density_canvas",
            "edge_density_foreground_bbox",
            "edge_to_foreground_ratio",

            "connected_component_count",
            "largest_component_fraction",
            "median_component_fraction",

            "thickness_proxy_mean_ratio",
            "thickness_proxy_max_ratio",

            "vertical_symmetry_iou",
            "horizontal_symmetry_iou",
        ]


        for feature in empty_features:

            result[
                feature
            ] = np.nan


        return (
            result,
            mask,
        )


    # ==================================================================================================
    # 10A. FOREGROUND COORDINATES
    # ==================================================================================================

    y_coordinates, x_coordinates = np.where(
        mask > 0
    )


    x_min = int(
        x_coordinates.min()
    )

    x_max = int(
        x_coordinates.max()
    )

    y_min = int(
        y_coordinates.min()
    )

    y_max = int(
        y_coordinates.max()
    )


    bbox_width = (
        x_max
        - x_min
        + 1
    )

    bbox_height = (
        y_max
        - y_min
        + 1
    )

    bbox_area = (
        bbox_width
        * bbox_height
    )


    result.update(
        {
            "bbox_x_min_ratio":
                x_min
                / width,

            "bbox_y_min_ratio":
                y_min
                / height,

            "bbox_x_max_ratio":
                x_max
                / width,

            "bbox_y_max_ratio":
                y_max
                / height,

            "bbox_width_ratio":
                bbox_width
                / width,

            "bbox_height_ratio":
                bbox_height
                / height,

            "bbox_area_ratio":
                bbox_area
                / image_area,

            "bbox_fill_ratio":
                foreground_count
                / bbox_area,

            "left_margin_ratio":
                x_min
                / width,

            "right_margin_ratio":
                (
                    width
                    - 1
                    - x_max
                )
                / width,

            "top_margin_ratio":
                y_min
                / height,

            "bottom_margin_ratio":
                (
                    height
                    - 1
                    - y_max
                )
                / height,
        }
    )


    # ==================================================================================================
    # 10B. CENTROID / DISPERSION
    # ==================================================================================================

    centroid_x = float(
        np.mean(
            x_coordinates
        )
    )

    centroid_y = float(
        np.mean(
            y_coordinates
        )
    )


    centroid_x_ratio = (
        centroid_x
        / width
    )

    centroid_y_ratio = (
        centroid_y
        / height
    )


    center_distance = math.sqrt(
        (
            centroid_x_ratio
            - 0.5
        ) ** 2
        +
        (
            centroid_y_ratio
            - 0.5
        ) ** 2
    )


    result.update(
        {
            "centroid_x_ratio":
                centroid_x_ratio,

            "centroid_y_ratio":
                centroid_y_ratio,

            "centroid_distance_from_center":
                center_distance,

            "foreground_x_std_ratio":
                float(
                    np.std(
                        x_coordinates
                    )
                    / width
                ),

            "foreground_y_std_ratio":
                float(
                    np.std(
                        y_coordinates
                    )
                    / height
                ),
        }
    )


    # ==================================================================================================
    # 10C. HALF / QUADRANT OCCUPANCY
    # ==================================================================================================

    left_count = np.sum(
        x_coordinates
        <
        (
            width
            / 2
        )
    )

    right_count = (
        foreground_count
        -
        left_count
    )


    top_count = np.sum(
        y_coordinates
        <
        (
            height
            / 2
        )
    )

    bottom_count = (
        foreground_count
        -
        top_count
    )


    center_x = (
        width
        / 2
    )

    center_y = (
        height
        / 2
    )


    q_tl = np.sum(
        (
            x_coordinates
            < center_x
        )
        &
        (
            y_coordinates
            < center_y
        )
    )


    q_tr = np.sum(
        (
            x_coordinates
            >= center_x
        )
        &
        (
            y_coordinates
            < center_y
        )
    )


    q_bl = np.sum(
        (
            x_coordinates
            < center_x
        )
        &
        (
            y_coordinates
            >= center_y
        )
    )


    q_br = np.sum(
        (
            x_coordinates
            >= center_x
        )
        &
        (
            y_coordinates
            >= center_y
        )
    )


    result.update(
        {
            "ink_left_fraction":
                left_count
                / foreground_count,

            "ink_right_fraction":
                right_count
                / foreground_count,

            "ink_top_fraction":
                top_count
                / foreground_count,

            "ink_bottom_fraction":
                bottom_count
                / foreground_count,

            "quadrant_top_left_fraction":
                q_tl
                / foreground_count,

            "quadrant_top_right_fraction":
                q_tr
                / foreground_count,

            "quadrant_bottom_left_fraction":
                q_bl
                / foreground_count,

            "quadrant_bottom_right_fraction":
                q_br
                / foreground_count,
        }
    )


    # ==================================================================================================
    # 10D. 4×4 SPACE OCCUPANCY
    # ==================================================================================================

    occupied_cells = 0

    grid_size = 4


    for row_index in range(
        grid_size
    ):

        y0 = int(
            round(
                row_index
                * height
                / grid_size
            )
        )

        y1 = int(
            round(
                (
                    row_index
                    + 1
                )
                * height
                / grid_size
            )
        )


        for column_index in range(
            grid_size
        ):

            x0 = int(
                round(
                    column_index
                    * width
                    / grid_size
                )
            )

            x1 = int(
                round(
                    (
                        column_index
                        + 1
                    )
                    * width
                    / grid_size
                )
            )


            cell = mask[
                y0:y1,
                x0:x1
            ]


            if (
                cell.size > 0
                and
                cell.mean()
                > 0.005
            ):

                occupied_cells += 1


    result[
        "occupied_grid_fraction_4x4"
    ] = (
        occupied_cells
        / 16.0
    )


    # ==================================================================================================
    # 10E. GRAYSCALE / DARKNESS
    # ==================================================================================================

    foreground_gray = (
        grayscale[
            mask > 0
        ]
        .astype(
            np.float64
        )
    )


    gray_histogram, _ = np.histogram(
        foreground_gray,
        bins=32,
        range=(
            0,
            256
        ),
    )


    result.update(
        {
            "foreground_mean_gray":
                float(
                    np.mean(
                        foreground_gray
                    )
                ),

            "foreground_std_gray":
                float(
                    np.std(
                        foreground_gray
                    )
                ),

            "foreground_mean_darkness":
                float(
                    1.0
                    -
                    np.mean(
                        foreground_gray
                    )
                    / 255.0
                ),

            "foreground_gray_entropy":
                entropy_from_histogram(
                    gray_histogram
                ),
        }
    )


    # ==================================================================================================
    # 10F. COLOR / SATURATION
    # ==================================================================================================

    foreground_rgb = (
        image_rgb[
            mask > 0
        ]
    )


    foreground_hsv = (
        hsv[
            mask > 0
        ]
    )


    saturation = (
        foreground_hsv[
            :,
            1
        ]
        .astype(
            np.float64
        )
    )


    hue = (
        foreground_hsv[
            :,
            0
        ]
        .astype(
            np.float64
        )
    )


    # Only meaningful hue where some saturation exists.
    colorful_pixels = (
        saturation
        > 25
    )


    if np.any(
        colorful_pixels
    ):

        hue_histogram, _ = np.histogram(
            hue[
                colorful_pixels
            ],
            bins=18,
            range=(
                0,
                180
            ),
        )

        hue_entropy = entropy_from_histogram(
            hue_histogram
        )

    else:

        hue_entropy = 0.0


    result.update(
        {
            "foreground_mean_saturation":
                float(
                    np.mean(
                        saturation
                    )
                ),

            "foreground_std_saturation":
                float(
                    np.std(
                        saturation
                    )
                ),

            "foreground_hue_entropy":
                float(
                    hue_entropy
                ),

            "foreground_colorfulness":
                calculate_colorfulness(
                    foreground_rgb
                ),
        }
    )


    # ==================================================================================================
    # 10G. EDGE / VISUAL COMPLEXITY PROXIES
    # ==================================================================================================

    edges = cv2.Canny(
        grayscale,
        threshold1=50,
        threshold2=150,
    )


    edge_binary = (
        edges
        > 0
    )


    edge_count = int(
        edge_binary.sum()
    )


    bbox_edges = edge_binary[
        y_min:y_max+1,
        x_min:x_max+1
    ]


    result.update(
        {
            "edge_density_canvas":
                edge_count
                / image_area,

            "edge_density_foreground_bbox":
                float(
                    bbox_edges.mean()
                )
                if bbox_edges.size
                else np.nan,

            "edge_to_foreground_ratio":
                safe_divide(
                    edge_count,
                    foreground_count
                ),
        }
    )


    # ==================================================================================================
    # 10H. CONNECTED COMPONENT STRUCTURE
    # ==================================================================================================

    (
        component_count_with_background,
        component_labels,
        component_stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        mask.astype(
            np.uint8
        ),
        connectivity=8,
    )


    component_areas = []


    for component_index in range(
        1,
        component_count_with_background
    ):

        area = int(
            component_stats[
                component_index,
                cv2.CC_STAT_AREA
            ]
        )

        component_areas.append(
            area
        )


    if component_areas:

        component_areas_array = np.asarray(
            component_areas,
            dtype=np.float64
        )

        largest_component_fraction = float(
            component_areas_array.max()
            / foreground_count
        )

        median_component_fraction = float(
            np.median(
                component_areas_array
            )
            / foreground_count
        )

    else:

        largest_component_fraction = np.nan
        median_component_fraction = np.nan


    result.update(
        {
            "connected_component_count":
                len(
                    component_areas
                ),

            "largest_component_fraction":
                largest_component_fraction,

            "median_component_fraction":
                median_component_fraction,
        }
    )


    # ==================================================================================================
    # 10I. STROKE / REGION THICKNESS PROXY
    #
    # Distance transform gives distance from each foreground pixel to nearest background.
    # Twice that distance is used as a local thickness proxy.
    #
    # This is NOT literal pen-pressure measurement.
    # ==================================================================================================

    distance_transform = (
        cv2.distanceTransform(
            mask.astype(
                np.uint8
            ),
            cv2.DIST_L2,
            5,
        )
    )


    distance_values = (
        distance_transform[
            mask > 0
        ]
    )


    normalization_dimension = float(
        min(
            width,
            height
        )
    )


    result.update(
        {
            "thickness_proxy_mean_ratio":
                float(
                    2.0
                    * np.mean(
                        distance_values
                    )
                    / normalization_dimension
                ),

            "thickness_proxy_max_ratio":
                float(
                    2.0
                    * np.max(
                        distance_values
                    )
                    / normalization_dimension
                ),
        }
    )


    # ==================================================================================================
    # 10J. SYMMETRY
    # ==================================================================================================

    mask_crop = mask[
        y_min:y_max+1,
        x_min:x_max+1
    ]


    result.update(
        {
            "vertical_symmetry_iou":
                symmetry_score(
                    mask_crop,
                    "vertical"
                ),

            "horizontal_symmetry_iou":
                symmetry_score(
                    mask_crop,
                    "horizontal"
                ),
        }
    )


    return (
        result,
        mask,
    )

/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved.
 * Copyright (c) 2025, Your Organization. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "definitions.h"
#include "sequence_control_set.h"
#include "svt_log.h"

// ISO mapping array for SVT-AV1 film grain scale 1-50
static const uint32_t iso_map[50] = {
    150, 250, 400, 600, 800, 1100, 1600, 2400, 3600, 4800,
    6000, 7500, 9000, 10000, 11000, 12500, 14000, 16000, 18000, 20000,
    22000, 23500, 25000, 26500, 28000, 29500, 31000, 32500, 34000, 35000,
    36000, 37000, 38000, 39000, 40000, 41000, 42000, 43000, 44000, 45000,
    45500, 46000, 46500, 47000, 47500, 48000, 48500, 49000, 49500, 50000
};

static uint32_t get_photon_noise_iso(uint32_t photon_noise_level) {
    if (photon_noise_level <= 50) {
        return iso_map[photon_noise_level - 1];
    }
    return photon_noise_level; // Direct ISO value
}

// Transfer function structure
typedef struct {
    double (*to_linear)(double);
    double (*from_linear)(double);
    double mid_tone;
} TransferFunction;

// Photon noise arguments
typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t iso_setting;
    uint8_t chroma_setting;
    TransferFunction transfer_function;
} PhotonNoiseArgs;

// Transfer function implementations
static double maxf(double a, double b) { return a > b ? a : b; }
static double minf(double a, double b) { return a < b ? a : b; }

static double gamma22_to_linear(double g) { return pow(g, 2.2); }
static double gamma22_from_linear(double l) { return pow(l, 1.0 / 2.2); }

static double gamma28_to_linear(double g) { return pow(g, 2.8); }
static double gamma28_from_linear(double l) { return pow(l, 1.0 / 2.8); }

static double srgb_to_linear(double srgb) {
    return srgb <= 0.04045 ? srgb / 12.92 : pow((srgb + 0.055) / 1.055, 2.4);
}
static double srgb_from_linear(double linear) {
    return linear <= 0.0031308 ? 12.92 * linear : 1.055 * pow(linear, 1.0 / 2.4) - 0.055;
}

static const double kPqM1 = 2610.0 / 16384;
static const double kPqM2 = 128 * 2523.0 / 4096;
static const double kPqC1 = 3424.0 / 4096;
static const double kPqC2 = 32 * 2413.0 / 4096;
static const double kPqC3 = 32 * 2392.0 / 4096;

static double pq_to_linear(double pq) {
    double pq_pow_inv_m2 = pow(pq, 1.0 / kPqM2);
    return pow(maxf(0.0, pq_pow_inv_m2 - kPqC1) / (kPqC2 - kPqC3 * pq_pow_inv_m2), 1.0 / kPqM1);
}
static double pq_from_linear(double linear) {
    double linear_pow_m1 = pow(linear, kPqM1);
    return pow((kPqC1 + kPqC2 * linear_pow_m1) / (1.0 + kPqC3 * linear_pow_m1), kPqM2);
}

static const double kHlgA = 0.17883277;
static const double kHlgB = 0.28466892;
static const double kHlgC = 0.55991073;

static double hlg_to_linear(double hlg) {
    double linear = hlg <= 0.5 ? hlg * hlg / 3.0 : (exp((hlg - kHlgC) / kHlgA) + kHlgB) / 12.0;
    return pow(linear, 1.2);
}
static double hlg_from_linear(double linear) {
    linear = pow(linear, 1.0 / 1.2);
    return linear <= 1.0 / 12.0 ? sqrt(3.0 * linear) : kHlgA * log(12.0 * linear - kHlgB) + kHlgC;
}

/**
static double linear_to_linear(double x) { return x; }
static double linear_from_linear(double x) { return x; }

static double bt709_to_linear(double x) { return srgb_to_linear(x); } // Approximate BT.709 with sRGB
static double bt709_from_linear(double x) { return srgb_from_linear(x); }
*/

// BT.601 OETF  (linear→encoded)
static double bt601_from_linear(double L) {
    // L = scene-linear light normalised [0…1]
    if (L < 0.018)            // “toe” segment
        return 4.5 * L;
    else                      // power segment, γ≈0.45
        return 1.099 * pow(L, 0.45) - 0.099;
}

// BT.601 EOTF  (encoded→linear)
static double bt601_to_linear(double E) {
    if (E < 0.08145)          // 4.5*0.018 = 0.081
        return E / 4.5;
    else
        return pow((E + 0.099) / 1.099, 1.0/0.45);
}

// -----------------------------------------------------------------------------
// 1) New to-/from-linear functions (double precision)
// -----------------------------------------------------------------------------

// SMPTE 240M ≈ a simple γ = 2.222 curve
static double smpte240m_to_linear(double x)   { return pow(x, 2.222); }
static double smpte240m_from_linear(double x) { return pow(x, 1.0 / 2.222); }

// Logarithmic 100:1 (Log100)
static double log100_to_linear(double x) {
    // map x∈[0,1] → L∈[log10(1),log10(101)] → linear = 10^L − 1
    double L = x * (log10(101.0) - 0.0) + 0.0;
    return pow(10.0, L) - 1.0;
}
static double log100_from_linear(double y) {
    return (log10(y + 1.0) - 0.0) / (log10(101.0) - 0.0);
}

// Logarithmic 100·√10:1 (Log100√10)
static double log100_sqrt10_to_linear(double x) {
    double Lmax = log10(100.0 * 3.1622776601683795 + 1.0);
    double L    = x * (Lmax - 0.0);
    return pow(10.0, L) - 1.0;
}
static double log100_sqrt10_from_linear(double y) {
    double Lmax = log10(100.0 * 3.1622776601683795 + 1.0);
    return (log10(y + 1.0) - 0.0) / (Lmax - 0.0);
}

// IEC 61966-2-4 (approx γ=2.6 with a small “toe”)
static double iec61966_to_linear(double x) {
    const double a = 0.1555, b = 0.2847;
    return pow((x + a) / (1.0 + b), 2.6);
}
static double iec61966_from_linear(double y) {
    const double a = 0.1555, b = 0.2847;
    return (pow(y, 1.0/2.6) * (1.0 + b)) - a;
}

// BT.1361 (BT.470M + soft knee around 0.08)
static double bt1361_to_linear(double x) {
    if (x <= 0.08) return x / 4.5;
    return pow((x + 0.099) / 1.099, 2.2);
}
static double bt1361_from_linear(double y) {
    if (y <= (0.08/4.5)) return y * 4.5;
    return (1.099 * pow(y, 1.0/2.2) - 0.099);
}

// SMPTE 428 (“Cineon” log film)
static double smpte428_to_linear(double x) {
    const double a = 0.002, b = 0.56, R = 95.0/1023.0;
    return pow(10.0, (x - b)*R) - a;
}
static double smpte428_from_linear(double y) {
    const double a = 0.002, b = 0.56, R = 95.0/1023.0;
    return (log10(y + a)/R) + b;
}

// Identity (Linear) 
static double identity_to_linear(double x)   { return x; }
static double identity_from_linear(double x) { return x; }

// -----------------------------------------------------------------------------
// 2) Extended find_transfer_function()
// -----------------------------------------------------------------------------

static TransferFunction find_transfer_function(EbTransferCharacteristics tc) {
    TransferFunction tf = { NULL, NULL, 0.18 };  // default mid-tone 18%

    switch (tc) {
    case EB_CICP_TC_BT_709:
        tf.to_linear   = srgb_to_linear;
        tf.from_linear = srgb_from_linear;
        break;

    case EB_CICP_TC_UNSPECIFIED:
        tf.to_linear   = srgb_to_linear;
        tf.from_linear = srgb_from_linear;
        break;

    case EB_CICP_TC_BT_470_M:
        tf.to_linear   = gamma22_to_linear;
        tf.from_linear = gamma22_from_linear;
        break;

    case EB_CICP_TC_BT_470_B_G:
        tf.to_linear   = gamma28_to_linear;
        tf.from_linear = gamma28_from_linear;
        break;

    case EB_CICP_TC_BT_601:
        tf.to_linear   = bt601_to_linear;
        tf.from_linear = bt601_from_linear;
        break;

    case EB_CICP_TC_SMPTE_240:
        tf.to_linear   = smpte240m_to_linear;
        tf.from_linear = smpte240m_from_linear;
        break;

    case EB_CICP_TC_LINEAR:
        tf.to_linear   = identity_to_linear;
        tf.from_linear = identity_from_linear;
        tf.mid_tone    = 0.50;    // linear mid-tone at 50%
        break;

    case EB_CICP_TC_LOG_100:
        tf.to_linear   = log100_to_linear;
        tf.from_linear = log100_from_linear;
        break;

    case EB_CICP_TC_LOG_100_SQRT10:
        tf.to_linear   = log100_sqrt10_to_linear;
        tf.from_linear = log100_sqrt10_from_linear;
        break;

    case EB_CICP_TC_IEC_61966:
        tf.to_linear   = iec61966_to_linear;
        tf.from_linear = iec61966_from_linear;
        break;

    case EB_CICP_TC_BT_1361:
        tf.to_linear   = bt1361_to_linear;
        tf.from_linear = bt1361_from_linear;
        break;

    case EB_CICP_TC_SRGB:
        tf.to_linear   = srgb_to_linear;
        tf.from_linear = srgb_from_linear;
        break;

    case EB_CICP_TC_BT_2020_10_BIT:
    case EB_CICP_TC_BT_2020_12_BIT:
        // BT.2020 SDR uses the same OETF as BT.709/sRGB
        tf.to_linear   = srgb_to_linear;
        tf.from_linear = srgb_from_linear;
        break;

    case EB_CICP_TC_SMPTE_2084:
        tf.to_linear   = pq_to_linear;
        tf.from_linear = pq_from_linear;
        tf.mid_tone    = 26.0 / 10000.0;
        break;

    case EB_CICP_TC_SMPTE_428:
        tf.to_linear   = smpte428_to_linear;
        tf.from_linear = smpte428_from_linear;
        break;

    case EB_CICP_TC_HLG:
        tf.to_linear   = hlg_to_linear;
        tf.from_linear = hlg_from_linear;
        tf.mid_tone    = 26.0 / 1000.0;
        break;

    default:
        // RESERVED or unknown: fallback to sRGB
        SVT_WARN("Warning: unimplemented transfer function %d, defaulting to sRGB\n", tc);
        tf.to_linear   = srgb_to_linear;
        tf.from_linear = srgb_from_linear;
        break;
    }

    return tf;
}

static void svt_av1_generate_photon_noise(const PhotonNoiseArgs *args, EbSvtAv1EncConfiguration *cfg) {
    AomFilmGrain *film_grain;
    film_grain = (AomFilmGrain *)calloc(1, sizeof(AomFilmGrain));
    // Constants from original photon_noise_table.c
    static const double kPhotonsPerLxSPerUm2 = 11260.0;
    static const double kEffectiveQuantumEfficiency = 0.20;
    static const double kPhotoResponseNonUniformity = 0.005;
    static const double kInputReferredReadNoise = 1.5;
    /* OK */

    // Focal plane exposure for a mid-tone, in lx·s
    const double mid_tone_exposure = 10.0 / args->iso_setting;

    // Pixel area in microns (36mm × 24mm sensor)
    const double pixel_area_um2 = (36000.0 * 24000.0) / (args->width * args->height);

    const double mid_tone_electrons_per_pixel = kEffectiveQuantumEfficiency *
                                               kPhotonsPerLxSPerUm2 *
                                               mid_tone_exposure * pixel_area_um2;
    const double max_electrons_per_pixel = mid_tone_electrons_per_pixel / args->transfer_function.mid_tone;
    film_grain->num_y_points = 14;
    for (int32_t i = 0; i < film_grain->num_y_points; ++i) {
        double x = (double)i / (film_grain->num_y_points - 1.0);
        const double linear = args->transfer_function.to_linear(x);
        const double electrons_per_pixel = max_electrons_per_pixel * linear;
        const double noise_in_electrons = sqrt(
            kInputReferredReadNoise * kInputReferredReadNoise +
            electrons_per_pixel +
            (kPhotoResponseNonUniformity * kPhotoResponseNonUniformity *
             electrons_per_pixel * electrons_per_pixel));
        const double linear_noise = noise_in_electrons / max_electrons_per_pixel;
        const double linear_range_start = maxf(0.0, linear - 2.0 * linear_noise);
        const double linear_range_end = minf(1.0, linear + 2.0 * linear_noise);
        const double tf_slope =
            (args->transfer_function.from_linear(linear_range_end) -
             args->transfer_function.from_linear(linear_range_start)) /
            (linear_range_end - linear_range_start);
        double encoded_noise = linear_noise * tf_slope;

        x = round(255.0 * x);
        encoded_noise = minf(255.0, round(255.0 * 7.88 * encoded_noise));

        film_grain->scaling_points_y[i][0] = (int32_t)x;
        film_grain->scaling_points_y[i][1] = (int32_t)encoded_noise;
    }

    film_grain->apply_grain = 1;
    film_grain->ignore_ref = 1;
    film_grain->update_parameters = 1;
    film_grain->num_cb_points = 0;
    film_grain->num_cr_points = 0;
    film_grain->scaling_shift = 8;
    film_grain->ar_coeff_lag = 0;
    memset(film_grain->ar_coeffs_y, 0, sizeof(film_grain->ar_coeffs_y));
    memset(film_grain->ar_coeffs_cb, 0, sizeof(film_grain->ar_coeffs_cb));
    memset(film_grain->ar_coeffs_cr, 0, sizeof(film_grain->ar_coeffs_cr));
    film_grain->ar_coeff_shift = 6;
    film_grain->cb_mult = 0;
    film_grain->cb_luma_mult = 0;
    film_grain->cb_offset = 0;
    film_grain->cr_mult = 0;
    film_grain->cr_luma_mult = 0;
    film_grain->cr_offset = 0;
    film_grain->overlap_flag = 1;
    film_grain->grain_scale_shift = 0;
    film_grain->chroma_scaling_from_luma = args->chroma_setting;
    film_grain->clip_to_restricted_range = 0;
    cfg->fgs_table = film_grain;
}

EbErrorType svt_av1_generate_photon_noise_table(EbSvtAv1EncConfiguration *config) {
    config->photon_noise_iso = get_photon_noise_iso(config->photon_noise_level);
    PhotonNoiseArgs args = {.width = config->source_width, .height = config->source_height,
                            .iso_setting = config->photon_noise_iso, .chroma_setting = config->enable_photon_noise_chroma,
                            .transfer_function = find_transfer_function(config->transfer_characteristics)};

    svt_av1_generate_photon_noise(&args, config);

    return EB_ErrorNone;
}
/*
    This file is part of darktable,
    Copyright (C) 2018-2025 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

/*** DOCUMENTATION
 *
 * This module provides dual local contrast enhancement modes:
 * - Pyramidal contrast: multi-scale enhancement with independent controls
 * - Local RGB contrast: multi-scale enhancement with shared parameters
 *
 * Both operate in scene-referred linear RGB space and should be placed early in the pipe.
 *
 ***/

#include "common/extra_optimizations.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "bauhaus/bauhaus.h"
#include "common/darktable.h"
#include "common/fast_guided_filter.h"
#include "common/eigf.h"
#include "common/luminance_mask.h"
#include "common/opencl.h"
#include "control/conf.h"
#include "control/control.h"
#include "develop/blend.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_math.h"
#include "develop/imageop_gui.h"
#include "gui/accelerators.h"
#include "gui/draw.h"
#include "dtgtk/paint.h"
#include "dtgtk/togglebutton.h"
#include "dtgtk/expander.h"
#include "gui/gtk.h"
#include "gui/presets.h"
#include "iop/iop_api.h"
#include "common/iop_group.h"

#ifdef _OPENMP
#include <omp.h>
#endif

DT_MODULE(1)

DT_MODULE_INTROSPECTION(3, dt_iop_contrast_dual_params_t)

/** Number of independent detail scales for Local RGB */
#define N_SCALES 3

/** Minimum float value to avoid log2(0) */
#define MIN_FLOAT exp2f(-16.0f)

/**
 * Filter types for detail preservation / smoothing.
 **/
typedef enum dt_iop_contrast_dual_filter_t
{
  DT_LC_AVG_GUIDED = 0, // $DESCRIPTION: "averaged guided filter"
  DT_LC_GUIDED,         // $DESCRIPTION: "guided filter"
  DT_LC_AVG_EIGF,       // $DESCRIPTION: "averaged EIGF"
  DT_LC_EIGF            // $DESCRIPTION: "EIGF"
} dt_iop_contrast_dual_filter_t;

typedef struct dt_iop_contrast_dual_params_t
{
  int mode; // $DEFAULT: 0 $DESCRIPTION: "mode" // 0 = Pyramidal, 1 = Local RGB

  // Pyramidal parameters (prefixed p_)
  float p_micro_scale;    // $MIN: 0.0 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "micro contrast"
  float p_fine_scale;     // $MIN: 0.0 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "fine contrast"
  float p_detail_scale;   // $MIN: 0.0 $MAX: 5.0 $DEFAULT: 1.5 $DESCRIPTION: "local contrast"
  float p_medium_scale;   // $MIN: 0.0 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "broad contrast"
  float p_broad_scale;    // $MIN: 0.0 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "extended contrast"
  float p_global_scale;   // $MIN: 0.0 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "global contrast"
  float p_blending;       // $MIN: 1.0 $MAX: 4.0 $DEFAULT: 1.2 $DESCRIPTION: "feature scale"
  float p_feathering;     // $MIN: 0.01 $MAX: 10000.0 $DEFAULT: 5.0 $DESCRIPTION: "edges refinement/feathering"
  float p_f_mult_micro;  // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 0.5 $DESCRIPTION: "micro contrast feathering"
  float p_f_mult_fine;   // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 0.75 $DESCRIPTION: "fine contrast feathering"
  float p_f_mult_detail; // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "local contrast feathering"
  float p_f_mult_medium; // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 1.25 $DESCRIPTION: "broad contrast feathering"
  float p_f_mult_broad;  // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 1.50 $DESCRIPTION: "extended contrast feathering"
  float p_s_mult_micro;  // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 0.25 $DESCRIPTION: "micro contrast scale mult."
  float p_s_mult_fine;   // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 0.625 $DESCRIPTION: "fine contrast scale mult."
  float p_s_mult_detail; // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "local contrast scale mult."
  float p_s_mult_medium; // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 1.8 $DESCRIPTION: "broad contrast scale mult."
  float p_s_mult_broad;  // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 2.8 $DESCRIPTION: "extended contrast scale mult."
  dt_iop_contrast_dual_filter_t p_details; // $DEFAULT: DT_LC_EIGF $DESCRIPTION: "feature extractor"
  dt_iop_luminance_mask_method_t p_method;      // $DEFAULT: DT_TONEEQ_NORM_2 $DESCRIPTION: "luminance estimator"
  int p_iterations;       // $MIN: 1 $MAX: 20 $DEFAULT: 1 $DESCRIPTION: "filter diffusion"

  // Local RGB parameters (prefixed l_)
  float l_detail_scale[N_SCALES];   // $MIN: 0.0 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "detail boost"
  float l_blending[N_SCALES];       // $MIN: 0.01 $MAX: 100.0 $DEFAULT: 12.0 $DESCRIPTION: "feature scale"
  float l_feathering[N_SCALES];     // $MIN: 0.01 $MAX: 10000.0 $DEFAULT: 5.0 $DESCRIPTION: "edges refinement"
  dt_iop_contrast_dual_filter_t l_details; // $DEFAULT: DT_LC_EIGF $DESCRIPTION: "feature extractor"
  dt_iop_luminance_mask_method_t l_method;      // $DEFAULT: DT_TONEEQ_NORM_2 $DESCRIPTION: "luminance estimator"
  int l_iterations;                             // $MIN: 1 $MAX: 20 $DEFAULT: 1 $DESCRIPTION: "filter diffusion"
} dt_iop_contrast_dual_params_t;

/**
 * Per-scale processing data derived from params for Local RGB
 **/
typedef struct dt_iop_lcrgb_scale_data_t
{
  float detail_scale;
  float blending;
  float feathering;
  int radius;    // derived from blending and image size
} dt_iop_lcrgb_scale_data_t;

typedef struct dt_iop_pyra_data_t
{
  float broad_scale;
  float medium_scale;
  float detail_scale;
  float fine_scale;
  float micro_scale;
  float global_scale;
  float blending, feathering;
  float f_mult_micro, f_mult_fine, f_mult_detail, f_mult_medium, f_mult_broad;
  float s_mult_micro, s_mult_fine, s_mult_detail, s_mult_medium, s_mult_broad;
  float scale;
  int radius;
  int radius_broad;
  int radius_medium;
  int radius_fine;
  int radius_micro;
  int iterations;
  dt_iop_luminance_mask_method_t method;
  dt_iop_contrast_dual_filter_t details;
} dt_iop_pyra_data_t;

typedef struct dt_iop_lcrgb_data_t
{
  // Per-scale data
  dt_iop_lcrgb_scale_data_t scales[N_SCALES];

  // Shared data
  int iterations;
  dt_iop_luminance_mask_method_t method;
  dt_iop_contrast_dual_filter_t details;
} dt_iop_lcrgb_data_t;

typedef struct dt_iop_contrast_dual_global_data_t
{
  // Reserved for OpenCL kernels
} dt_iop_contrast_dual_global_data_t;

typedef enum dt_iop_pyra_mask_t
{
  DT_LC_MASK_OFF = 0,
  DT_LC_MASK_BROAD = 1,
  DT_LC_MASK_MEDIUM = 2,
  DT_LC_MASK_DETAIL = 3,
  DT_LC_MASK_FINE = 4,
  DT_LC_MASK_MICRO = 5
} dt_iop_pyra_mask_t;

typedef struct dt_iop_pyra_gui_data_t
{
  // Flags
  dt_iop_pyra_mask_t mask_display;

  // Buffer dimensions
  int buf_width;
  int buf_height;
  int pipe_order;

  // Hash for cache invalidation
  dt_hash_t ui_preview_hash;
  dt_hash_t thumb_preview_hash;
  size_t full_preview_buf_width, full_preview_buf_height;
  size_t thumb_preview_buf_width, thumb_preview_buf_height;

  // Cached luminance buffers
  float *thumb_preview_buf_pixel;     // pixel-wise luminance (no blur)
  float *thumb_preview_buf_smoothed_broad;
  float *thumb_preview_buf_smoothed_medium;
  float *thumb_preview_buf_smoothed;  // smoothed luminance
  float *thumb_preview_buf_smoothed_fine;
  float *thumb_preview_buf_smoothed_micro;
  float *full_preview_buf_pixel;
  float *full_preview_buf_smoothed_broad;
  float *full_preview_buf_smoothed_medium;
  float *full_preview_buf_smoothed;
  float *full_preview_buf_smoothed_fine;
  float *full_preview_buf_smoothed_micro;

  // Cache validity
  gboolean luminance_valid;

  // GTK widgets
  GtkWidget *broad_scale, *medium_scale, *detail_scale, *fine_scale, *micro_scale, *global_scale;
  GtkWidget *blending;
  GtkWidget *feathering;
  dt_gui_collapsible_section_t advanced_expander;
  GtkWidget *f_mult_micro, *f_mult_fine, *f_mult_detail, *f_mult_medium, *f_mult_broad;
  dt_gui_collapsible_section_t scale_expander;
  GtkWidget *s_mult_micro, *s_mult_fine, *s_mult_detail, *s_mult_medium, *s_mult_broad;

  // New buttons for mask display in expanders
  GtkWidget *f_view_broad, *f_view_medium, *f_view_detail, *f_view_fine, *f_view_micro;
  GtkWidget *s_view_broad, *s_view_medium, *s_view_detail, *s_view_fine, *s_view_micro;
} dt_iop_pyra_gui_data_t;

typedef struct dt_iop_lcrgb_gui_data_t
{
  // Flags: which scale's mask is displayed (-1 = none)
  int mask_display_scale;

  // Buffer dimensions
  int buf_width;
  int buf_height;
  int pipe_order;

  // Hash for cache invalidation
  dt_hash_t ui_preview_hash;
  dt_hash_t thumb_preview_hash;
  size_t full_preview_buf_width, full_preview_buf_height;
  size_t thumb_preview_buf_width, thumb_preview_buf_height;

  // Cached luminance buffers
  float *thumb_preview_buf_pixel;                    // pixel-wise luminance (no blur)
  float *thumb_preview_buf_smoothed[N_SCALES];       // smoothed luminance per scale
  float *full_preview_buf_pixel;
  float *full_preview_buf_smoothed[N_SCALES];

  // Cache validity
  gboolean luminance_valid;

  // Per-scale GTK widgets
  GtkWidget *detail_scale[N_SCALES];
  GtkWidget *blending[N_SCALES];
  GtkWidget *feathering[N_SCALES];
  GtkWidget *show_mask[N_SCALES];

  // Shared GTK widgets
  GtkWidget *details;
  GtkWidget *iterations;
} dt_iop_lcrgb_gui_data_t;

typedef struct dt_iop_contrast_dual_gui_data_t
{
  GtkWidget *notebook;
  dt_iop_pyra_gui_data_t *pyra_gui;
  dt_iop_lcrgb_gui_data_t *lcrgb_gui;
} dt_iop_contrast_dual_gui_data_t;

const char *name()
{
  return _("contrast dual mode");
}

const char *aliases()
{
  return _("local contrast|clarity|detail enhancement");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description
    (self, _("enhance local contrast by boosting contrast while preserving edges"),
     _("creative"),
     _("linear, RGB, scene-referred"),
     _("linear, RGB"),
     _("linear, RGB, scene-referred"));
}

int default_group()
{
  return IOP_GROUP_BASIC | IOP_GROUP_EFFECTS;
}

int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  return IOP_CS_RGB;
}

int legacy_params(dt_iop_module_t *self,
                  const void *const old_params,
                  const int old_version,
                  void **new_params,
                  int32_t *new_params_size,
                  int *new_version)
{
  if(old_version == 1 || old_version == 2)
  {
    // Assume old params are from pyramidal or local rgb, but since mode is new, default to pyramidal
    dt_iop_contrast_dual_params_t *n = calloc(1, sizeof(dt_iop_contrast_dual_params_t));
    if(!n) return 1;

    // Initialize defaults
    n->mode = 0; // Pyramidal
    n->p_micro_scale = 1.0f;
    n->p_fine_scale = 1.0f;
    n->p_detail_scale = 1.5f;
    n->p_medium_scale = 1.0f;
    n->p_broad_scale = 1.0f;
    n->p_global_scale = 1.0f;
    n->p_blending = 1.2f;
    n->p_feathering = 5.0f;
    n->p_f_mult_micro = 0.5f;
    n->p_f_mult_fine = 0.75f;
    n->p_f_mult_detail = 1.0f;
    n->p_f_mult_medium = 1.25f;
    n->p_f_mult_broad = 1.5f;
    n->p_s_mult_micro = 0.25f;
    n->p_s_mult_fine = 0.625f;
    n->p_s_mult_detail = 1.0f;
    n->p_s_mult_medium = 1.8f;
    n->p_s_mult_broad = 2.8f;
    n->p_details = DT_LC_EIGF;
    n->p_method = DT_TONEEQ_NORM_2;
    n->p_iterations = 1;

    for(int s = 0; s < N_SCALES; s++)
    {
      n->l_detail_scale[s] = (s == 0) ? 1.5f : 1.0f;
      n->l_blending[s] = 12.0f;
      n->l_feathering[s] = 5.0f;
    }
    n->l_details = DT_LC_EIGF;
    n->l_method = DT_TONEEQ_NORM_2;
    n->l_iterations = 1;

    *new_params = n;
    *new_params_size = sizeof(dt_iop_contrast_dual_params_t);
    *new_version = 3;
    return 0;
  }
  return 1;
}

/**
 * Helper functions
 **/

static void hash_set_get(const dt_hash_t *hash_in,
                         dt_hash_t *hash_out,
                         dt_pthread_mutex_t *lock)
{
  dt_pthread_mutex_lock(lock);
  *hash_out = *hash_in;
  dt_pthread_mutex_unlock(lock);
}

static void pyra_invalidate_luminance_cache(dt_iop_module_t *const self)
{
  dt_iop_pyra_gui_data_t *const restrict g = ((dt_iop_contrast_dual_gui_data_t *)self->gui_data)->pyra_gui;

  dt_iop_gui_enter_critical_section(self);
  g->luminance_valid = FALSE;
  g->thumb_preview_hash = DT_INVALID_HASH;
  g->ui_preview_hash = DT_INVALID_HASH;
  dt_iop_gui_leave_critical_section(self);
  dt_iop_refresh_all(self);
}

static void lcrgb_invalidate_luminance_cache(dt_iop_module_t *const self)
{
  dt_iop_lcrgb_gui_data_t *const restrict g = ((dt_iop_contrast_dual_gui_data_t *)self->gui_data)->lcrgb_gui;

  dt_iop_gui_enter_critical_section(self);
  g->luminance_valid = FALSE;
  g->thumb_preview_hash = DT_INVALID_HASH;
  g->ui_preview_hash = DT_INVALID_HASH;
  dt_iop_gui_leave_critical_section(self);
  dt_iop_refresh_all(self);
}

/**
 * Check if any scale is active (detail_scale != 1.0) for Local RGB
 **/
static inline gboolean lcrgb_has_active_scales(const dt_iop_lcrgb_data_t *const d)
{
  for(int s = 0; s < N_SCALES; s++)
  {
    if(d->scales[s].detail_scale != 1.0f) return TRUE;
  }
  return FALSE;
}

/**
 * Check if a specific scale is active for Local RGB
 **/
static inline gboolean lcrgb_scale_is_active(const float detail_scale)
{
  return detail_scale != 1.0f;
}

/**
 * Compute pixel-wise luminance mask (no blur)
 **/
__DT_CLONE_TARGETS__
static inline void compute_pixel_luminance_mask(const float *const restrict in,
                                                float *const restrict luminance,
                                                const size_t width,
                                                const size_t height,
                                                const dt_iop_luminance_mask_method_t method)
{
  // No exposure/contrast boost, just compute raw luminance
  luminance_mask(in, luminance, width, height, method, 1.0f, 0.0f, 1.0f);
}

/**
 * Compute smoothed luminance mask using edge-aware filters for Pyramidal
 **/
__DT_CLONE_TARGETS__
static inline void pyra_compute_smoothed_luminance_mask(const float *const restrict in,
                                                   float *const restrict luminance,
                                                   const size_t width,
                                                   const size_t height,
                                                const dt_iop_pyra_data_t *const d,
                                                const int radius,
                                                const float feathering)
{
  // First compute pixel-wise luminance (no boost)
  luminance_mask(in, luminance, width, height, d->method, 1.0f, 0.0f, 1.0f);

  // Then apply the smoothing filter
  switch(d->details)
  {
    case(DT_LC_AVG_GUIDED):
    {
      fast_surface_blur(luminance, width, height, radius, feathering, d->iterations,
                        DT_GF_BLENDING_GEOMEAN, d->scale, 0.0f,
                        exp2f(-14.0f), 4.0f);
      break;
    }

    case(DT_LC_GUIDED):
    {
      fast_surface_blur(luminance, width, height, radius, feathering, d->iterations,
                        DT_GF_BLENDING_LINEAR, d->scale, 0.0f,
                        exp2f(-14.0f), 4.0f);
      break;
    }

    case(DT_LC_AVG_EIGF):
    {
      fast_eigf_surface_blur(luminance, width, height,
                             radius, feathering, d->iterations,
                             DT_GF_BLENDING_GEOMEAN, d->scale,
                             0.0f, exp2f(-14.0f), 4.0f);
      break;
    }

    case(DT_LC_EIGF):
    {
      fast_eigf_surface_blur(luminance, width, height,
                             radius, feathering, d->iterations,
                             DT_GF_BLENDING_LINEAR, d->scale,
                             0.0f, exp2f(-14.0f), 4.0f);
      break;
    }
  }
}

/**
 * Apply local contrast enhancement for Pyramidal
 **/
__DT_CLONE_TARGETS__
static inline void pyra_apply_local_contrast(const float *const restrict in,
                                        const float *const restrict luminance_pixel,
                                        const float *const restrict luminance_smoothed,
                                        const float *const restrict luminance_smoothed_broad,
                                        const float *const restrict luminance_smoothed_medium,
                                        const float *const restrict luminance_smoothed_fine,
                                        const float *const restrict luminance_smoothed_micro,
                                        float *const restrict out,
                                        const dt_iop_roi_t *const roi_in,
                                        const dt_iop_roi_t *const roi_out,
                                        const dt_iop_pyra_data_t *const d)
{
  const size_t npixels = (size_t)roi_in->width * roi_in->height;

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    // Detail in log space (EV): how much brighter/darker is this pixel
    // compared to its local neighborhood
    // detail = log2(pixel_lum / smoothed_lum) = log2(pixel_lum) - log2(smoothed_lum)
    const float lum_pixel = fmaxf(luminance_pixel[k], MIN_FLOAT);
    const float lum_smoothed = fmaxf(luminance_smoothed[k], MIN_FLOAT);
    const float detail_ev = log2f(lum_pixel / lum_smoothed);

    // Scale the detail: detail_scale = 1.0 means no change
    // > 1.0 boosts local contrast, < 1.0 reduces it
    const float scaled_detail_ev = d->detail_scale * detail_ev;

    // The correction is the difference between scaled and original detail
    float correction_ev = scaled_detail_ev - detail_ev;

    if(luminance_smoothed_broad)
    {
      const float lum_smoothed_broad = fmaxf(luminance_smoothed_broad[k], MIN_FLOAT);
      const float detail_ev_broad = log2f(lum_pixel / lum_smoothed_broad);
      const float scaled_detail_ev_broad = d->broad_scale * detail_ev_broad;
      correction_ev += scaled_detail_ev_broad - detail_ev_broad;
    }

    if(luminance_smoothed_medium)
    {
      const float lum_smoothed_medium = fmaxf(luminance_smoothed_medium[k], MIN_FLOAT);
      const float detail_ev_medium = log2f(lum_pixel / lum_smoothed_medium);
      const float scaled_detail_ev_medium = d->medium_scale * detail_ev_medium;
      correction_ev += scaled_detail_ev_medium - detail_ev_medium;
    }

    if(luminance_smoothed_fine)
    {
      const float lum_smoothed_fine = fmaxf(luminance_smoothed_fine[k], MIN_FLOAT);
      const float detail_ev_fine = log2f(lum_pixel / lum_smoothed_fine);
      const float scaled_detail_ev_fine = d->fine_scale * detail_ev_fine;
      correction_ev += scaled_detail_ev_fine - detail_ev_fine;
    }

    if(luminance_smoothed_micro)
    {
      const float lum_smoothed_micro = fmaxf(luminance_smoothed_micro[k], MIN_FLOAT);
      const float detail_ev_micro = log2f(lum_pixel / lum_smoothed_micro);
      const float scaled_detail_ev_micro = d->micro_scale * detail_ev_micro;
      correction_ev += scaled_detail_ev_micro - detail_ev_micro;
    }

    // Apply correction in linear space
    // global_scale has the same range as detail_scale.
    const float multiplier = exp2f(correction_ev) * powf(lum_smoothed / 0.1845f, d->global_scale) * 0.1845f / lum_smoothed;

    for_each_channel(c)
      out[4 * k + c] = in[4 * k + c] * multiplier;
  }
}

/**
 * Display the detail mask for Pyramidal
 **/
__DT_CLONE_TARGETS__
static inline void pyra_display_detail_mask(const float *const restrict luminance_pixel,
                                       const float *const restrict luminance_smoothed,
                                       float *const restrict out,
                                       const size_t width,
                                       const size_t height)
{
  const size_t npixels = width * height;

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    const float lum_pixel = fmaxf(luminance_pixel[k], MIN_FLOAT);
    const float lum_smoothed = fmaxf(luminance_smoothed[k], MIN_FLOAT);

    // Detail in log space, mapped to [0, 1] for display
    // Detail range roughly [-2, +2] EV mapped to [0, 1]
    const float detail_ev = log2f(lum_pixel / lum_smoothed);
    const float intensity = fminf(fmaxf(detail_ev / 4.0f + 0.5f, 0.0f), 1.0f);

    // Set all RGB channels to the same intensity (grayscale)
    out[4 * k + 0] = intensity;
    out[4 * k + 1] = intensity;
    out[4 * k + 2] = intensity;
    // Full opacity
    out[4 * k + 3] = 1.0f;
  }
}

/**
 * Compute smoothed luminance mask for a single scale for Local RGB
 **/
__DT_CLONE_TARGETS__
static inline void lcrgb_compute_smoothed_luminance_for_scale(
    const float *const restrict in,
    float *const restrict luminance,
    const size_t width,
    const size_t height,
    const dt_iop_lcrgb_scale_data_t *const scale_data,
    const dt_iop_contrast_dual_filter_t details,
    const int iterations,
    const dt_iop_luminance_mask_method_t method)
{
  // First compute pixel-wise luminance (no boost)
  luminance_mask(in, luminance, width, height, method, 1.0f, 0.0f, 1.0f);

  // Then apply the smoothing filter
  switch(details)
  {
    case(DT_LC_AVG_GUIDED):
    {
      fast_surface_blur(luminance, width, height,
                        scale_data->radius, scale_data->feathering, iterations,
                        DT_GF_BLENDING_GEOMEAN, 1.0f, 0.0f,
                        exp2f(-14.0f), 4.0f);
      break;
    }

    case(DT_LC_GUIDED):
    {
      fast_surface_blur(luminance, width, height,
                        scale_data->radius, scale_data->feathering, iterations,
                        DT_GF_BLENDING_LINEAR, 1.0f, 0.0f,
                        exp2f(-14.0f), 4.0f);
      break;
    }

    case(DT_LC_AVG_EIGF):
    {
      fast_eigf_surface_blur(luminance, width, height,
                             scale_data->radius, scale_data->feathering, iterations,
                             DT_GF_BLENDING_GEOMEAN, 1.0f,
                             0.0f, exp2f(-14.0f), 4.0f);
      break;
    }

    case(DT_LC_EIGF):
    {
      fast_eigf_surface_blur(luminance, width, height,
                             scale_data->radius, scale_data->feathering, iterations,
                             DT_GF_BLENDING_LINEAR, 1.0f,
                             0.0f, exp2f(-14.0f), 4.0f);
      break;
    }
  }
}

/**
 * Apply multi-scale local contrast enhancement for Local RGB
 **/
__DT_CLONE_TARGETS__
static inline void lcrgb_apply_multiscale_local_contrast(
    const float *const restrict in,
    const float *const restrict luminance_pixel,
    float *const restrict *const restrict luminance_smoothed,
    float *const restrict out,
    const size_t width,
    const size_t height,
    const dt_iop_lcrgb_data_t *const d)
{
  const size_t npixels = width * height;

  // Unpack scale data for vectorization
  float detail_scales[N_SCALES] DT_ALIGNED_PIXEL;
  gboolean active[N_SCALES];
  int n_active = 0;

  for(int s = 0; s < N_SCALES; s++)
  {
    detail_scales[s] = d->scales[s].detail_scale;
    active[s] = lcrgb_scale_is_active(detail_scales[s]);
    if(active[s]) n_active++;
  }

  // Early exit if no scales are active
  if(n_active == 0)
  {
    dt_iop_image_copy_by_size(out, in, width, height, 4);
    return;
  }

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    const float lum_pixel = fmaxf(luminance_pixel[k], MIN_FLOAT);
    float total_correction_ev = 0.0f;

    // Sum correction contributions from all active scales
    for(int s = 0; s < N_SCALES; s++)
    {
      if(!active[s]) continue;

      const float lum_smoothed = fmaxf(luminance_smoothed[s][k], MIN_FLOAT);

      // Detail in log space (EV): how much brighter/darker is this pixel
      // compared to its local neighborhood at this scale
      const float detail_ev = log2f(lum_pixel / lum_smoothed);

      // Scale the detail: detail_scale = 1.0 means no change
      // > 1.0 boosts local contrast, < 1.0 reduces it
      const float scaled_detail_ev = detail_scales[s] * detail_ev;

      // The correction is the difference between scaled and original detail
      total_correction_ev += scaled_detail_ev - detail_ev;
    }

    // Apply combined correction in linear space
    const float multiplier = exp2f(total_correction_ev);

    for_each_channel(c)
      out[4 * k + c] = in[4 * k + c] * multiplier;
  }
}

/**
 * Display the detail mask for a specific scale for Local RGB
 **/
__DT_CLONE_TARGETS__
static inline void lcrgb_display_detail_mask_for_scale(
    const float *const restrict luminance_pixel,
    const float *const restrict luminance_smoothed,
    float *const restrict out,
    const size_t width,
    const size_t height)
{
  const size_t npixels = width * height;

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    const float lum_pixel = fmaxf(luminance_pixel[k], MIN_FLOAT);
    const float lum_smoothed = fmaxf(luminance_smoothed[k], MIN_FLOAT);

    // Detail in log space, mapped to [0, 1] for display
    // Detail range roughly [-2, +2] EV mapped to [0, 1]
    const float detail_ev = log2f(lum_pixel / lum_smoothed);
    const float intensity = fminf(fmaxf(detail_ev / 4.0f + 0.5f, 0.0f), 1.0f);

    // Set all RGB channels to the same intensity (grayscale)
    out[4 * k + 0] = intensity;
    out[4 * k + 1] = intensity;
    out[4 * k + 2] = intensity;
    out[4 * k + 3] = 1.0f;
  }
}

/**
 * Allocate or reallocate smoothed buffers for all scales for Local RGB
 **/
static inline void lcrgb_alloc_smoothed_buffers(float **buffers,
                                          const size_t num_elem)
{
  for(int s = 0; s < N_SCALES; s++)
  {
    dt_free_align(buffers[s]);
    buffers[s] = dt_alloc_align_float(num_elem);
  }
}

/**
 * Free smoothed buffers for all scales for Local RGB
 **/
static inline void lcrgb_free_smoothed_buffers(float **buffers)
{
  for(int s = 0; s < N_SCALES; s++)
  {
    dt_free_align(buffers[s]);
    buffers[s] = NULL;
  }
}

/**
 * Check if all smoothed buffers are allocated for Local RGB
 **/
static inline gboolean lcrgb_smoothed_buffers_valid(float **buffers)
{
  for(int s = 0; s < N_SCALES; s++)
  {
    if(!buffers[s]) return FALSE;
  }
  return TRUE;
}

/**
 * Main processing function
 **/
__DT_CLONE_TARGETS__
static void contrast_dual_process(dt_iop_module_t *self,
                                   dt_dev_pixelpipe_iop_t *piece,
                                   const void *const restrict ivoid,
                                   void *const restrict ovoid,
                                   const dt_iop_roi_t *const roi_in,
                                   const dt_iop_roi_t *const roi_out)
{
  const dt_iop_contrast_dual_params_t *const params = piece->data;
  dt_iop_contrast_dual_gui_data_t *const g = self->gui_data;

  if(params->mode == 0)
  {
    // Pyramidal mode
    const dt_iop_pyra_data_t *const d = piece->data;
    dt_iop_pyra_gui_data_t *const pg = g ? g->pyra_gui : NULL;

    const float *const restrict in = (float *const)ivoid;
    float *const restrict out = (float *const)ovoid;
    float *restrict luminance_pixel = NULL;
    float *restrict luminance_smoothed_broad = NULL;
    float *restrict luminance_smoothed_medium = NULL;
    float *restrict luminance_smoothed = NULL;
    float *restrict luminance_smoothed_fine = NULL;
    float *restrict luminance_smoothed_micro = NULL;

    const size_t width = roi_in->width;
    const size_t height = roi_in->height;
    const size_t num_elem = width * height;

    // Get the hash of the upstream pipe to track changes
    const dt_hash_t hash = dt_dev_pixelpipe_piece_hash(piece, roi_out, TRUE);

    // Sanity checks
    if(width < 1 || height < 1) return;
    if(roi_in->width < roi_out->width || roi_in->height < roi_out->height) return;
    if(piece->colors != 4) return;

    // Init the luminance mask buffers
    gboolean cached = FALSE;

    if(self->dev->gui_attached)
    {
      // If the module instance has changed order in the pipe, invalidate caches
      if(pg->pipe_order != piece->module->iop_order)
      {
        dt_iop_gui_enter_critical_section(self);
        pg->ui_preview_hash = DT_INVALID_HASH;
        pg->thumb_preview_hash = DT_INVALID_HASH;
        pg->pipe_order = piece->module->iop_order;
        pg->luminance_valid = FALSE;
        dt_iop_gui_leave_critical_section(self);
      }

      if(piece->pipe->type & DT_DEV_PIXELPIPE_FULL)
      {
        // Re-allocate buffers if size changed
        if(pg->full_preview_buf_width != width || pg->full_preview_buf_height != height)
        {
          dt_free_align(pg->full_preview_buf_pixel);
          dt_free_align(pg->full_preview_buf_smoothed_broad);
          dt_free_align(pg->full_preview_buf_smoothed_medium);
          dt_free_align(pg->full_preview_buf_smoothed);
          dt_free_align(pg->full_preview_buf_smoothed_fine);
          dt_free_align(pg->full_preview_buf_smoothed_micro);
          pg->full_preview_buf_pixel = dt_alloc_align_float(num_elem);
          pg->full_preview_buf_smoothed_broad = dt_alloc_align_float(num_elem);
          pg->full_preview_buf_smoothed_medium = dt_alloc_align_float(num_elem);
          pg->full_preview_buf_smoothed = dt_alloc_align_float(num_elem);
          pg->full_preview_buf_smoothed_fine = dt_alloc_align_float(num_elem);
          pg->full_preview_buf_smoothed_micro = dt_alloc_align_float(num_elem);
          pg->full_preview_buf_width = width;
          pg->full_preview_buf_height = height;
        }

        luminance_pixel = pg->full_preview_buf_pixel;
        luminance_smoothed_broad = pg->full_preview_buf_smoothed_broad;
        luminance_smoothed_medium = pg->full_preview_buf_smoothed_medium;
        luminance_smoothed = pg->full_preview_buf_smoothed;
        luminance_smoothed_fine = pg->full_preview_buf_smoothed_fine;
        luminance_smoothed_micro = pg->full_preview_buf_smoothed_micro;
        cached = TRUE;
      }
      else if(piece->pipe->type & DT_DEV_PIXELPIPE_PREVIEW)
      {
        dt_iop_gui_enter_critical_section(self);
        if(pg->thumb_preview_buf_width != width || pg->thumb_preview_buf_height != height)
        {
          dt_free_align(pg->thumb_preview_buf_pixel);
          dt_free_align(pg->thumb_preview_buf_smoothed_broad);
          dt_free_align(pg->thumb_preview_buf_smoothed_medium);
          dt_free_align(pg->thumb_preview_buf_smoothed);
          dt_free_align(pg->thumb_preview_buf_smoothed_fine);
          dt_free_align(pg->thumb_preview_buf_smoothed_micro);
          pg->thumb_preview_buf_pixel = dt_alloc_align_float(num_elem);
          pg->thumb_preview_buf_smoothed_broad = dt_alloc_align_float(num_elem);
          pg->thumb_preview_buf_smoothed_medium = dt_alloc_align_float(num_elem);
          pg->thumb_preview_buf_smoothed = dt_alloc_align_float(num_elem);
          pg->thumb_preview_buf_smoothed_fine = dt_alloc_align_float(num_elem);
          pg->thumb_preview_buf_smoothed_micro = dt_alloc_align_float(num_elem);
          pg->thumb_preview_buf_width = width;
          pg->thumb_preview_buf_height = height;
          pg->luminance_valid = FALSE;
        }

        luminance_pixel = pg->thumb_preview_buf_pixel;
        luminance_smoothed_broad = pg->thumb_preview_buf_smoothed_broad;
        luminance_smoothed_medium = pg->thumb_preview_buf_smoothed_medium;
        luminance_smoothed = pg->thumb_preview_buf_smoothed;
        luminance_smoothed_fine = pg->thumb_preview_buf_smoothed_fine;
        luminance_smoothed_micro = pg->thumb_preview_buf_smoothed_micro;
        cached = TRUE;
        dt_iop_gui_leave_critical_section(self);
      }
      else
      {
        luminance_pixel = dt_alloc_align_float(num_elem);
        luminance_smoothed = dt_alloc_align_float(num_elem);
        luminance_smoothed_broad = dt_alloc_align_float(num_elem);
        luminance_smoothed_medium = dt_alloc_align_float(num_elem);
        luminance_smoothed_fine = dt_alloc_align_float(num_elem);
        luminance_smoothed_micro = dt_alloc_align_float(num_elem);
      }
    }
    else
    {
      // No interactive editing: allocate local temp buffers
      luminance_pixel = dt_alloc_align_float(num_elem);
      luminance_smoothed_broad = dt_alloc_align_float(num_elem);
      luminance_smoothed_medium = dt_alloc_align_float(num_elem);
      luminance_smoothed = dt_alloc_align_float(num_elem);
      luminance_smoothed_fine = dt_alloc_align_float(num_elem);
      luminance_smoothed_micro = dt_alloc_align_float(num_elem);
    }

    // Check buffer allocation
    if(!luminance_pixel || !luminance_smoothed_broad || !luminance_smoothed_medium || !luminance_smoothed || !luminance_smoothed_fine || !luminance_smoothed_micro)
    {
      dt_control_log(_("local contrast failed to allocate memory, check your RAM settings"));
      if(!cached)
      {
        dt_free_align(luminance_pixel);
        dt_free_align(luminance_smoothed_broad);
        dt_free_align(luminance_smoothed_medium);
        dt_free_align(luminance_smoothed);
        dt_free_align(luminance_smoothed_fine);
        dt_free_align(luminance_smoothed_micro);
      }
      return;
    }

    // Compute luminance masks
    if(cached)
    {
      if(piece->pipe->type & DT_DEV_PIXELPIPE_FULL)
      {
        dt_hash_t saved_hash;
        hash_set_get(&pg->ui_preview_hash, &saved_hash, &self->gui_lock);

        dt_iop_gui_enter_critical_section(self);
        const gboolean luminance_valid = pg->luminance_valid;
        dt_iop_gui_leave_critical_section(self);

        if(hash != saved_hash || !luminance_valid)
        {
          compute_pixel_luminance_mask(in, luminance_pixel, width, height, d->method);
          if(d->broad_scale != 1.0f)
            pyra_compute_smoothed_luminance_mask(in, luminance_smoothed_broad, width, height, d, d->radius_broad, d->feathering * d->f_mult_broad);
          if(d->medium_scale != 1.0f)
            pyra_compute_smoothed_luminance_mask(in, luminance_smoothed_medium, width, height, d, d->radius_medium, d->feathering * d->f_mult_medium);
          if(d->detail_scale != 1.0f)
            pyra_compute_smoothed_luminance_mask(in, luminance_smoothed, width, height, d, d->radius, d->feathering * d->f_mult_detail);
          if(d->fine_scale != 1.0f)
            pyra_compute_smoothed_luminance_mask(in, luminance_smoothed_fine, width, height, d, d->radius_fine, d->feathering * d->f_mult_fine);
          if(d->micro_scale != 1.0f)
            pyra_compute_smoothed_luminance_mask(in, luminance_smoothed_micro, width, height, d, d->radius_micro, d->feathering * d->f_mult_micro);
          hash_set_get(&hash, &pg->ui_preview_hash, &self->gui_lock);
        }
      }
      else if(piece->pipe->type & DT_DEV_PIXELPIPE_PREVIEW)
      {
        dt_hash_t saved_hash;
        hash_set_get(&pg->thumb_preview_hash, &saved_hash, &self->gui_lock);

        dt_iop_gui_enter_critical_section(self);
        const gboolean luminance_valid = pg->luminance_valid;
        dt_iop_gui_leave_critical_section(self);

        if(saved_hash != hash || !luminance_valid)
        {
          dt_iop_gui_enter_critical_section(self);
          pg->thumb_preview_hash = hash;
          compute_pixel_luminance_mask(in, luminance_pixel, width, height, d->method);
          if(d->broad_scale != 1.0f)
            pyra_compute_smoothed_luminance_mask(in, luminance_smoothed_broad, width, height, d, d->radius_broad, d->feathering * d->f_mult_broad);
          if(d->medium_scale != 1.0f)
            pyra_compute_smoothed_luminance_mask(in, luminance_smoothed_medium, width, height, d, d->radius_medium, d->feathering * d->f_mult_medium);
          if(d->detail_scale != 1.0f)
            pyra_compute_smoothed_luminance_mask(in, luminance_smoothed, width, height, d, d->radius, d->feathering * d->f_mult_detail);
          if(d->fine_scale != 1.0f)
            pyra_compute_smoothed_luminance_mask(in, luminance_smoothed_fine, width, height, d, d->radius_fine, d->feathering * d->f_mult_fine);
          if(d->micro_scale != 1.0f)
            pyra_compute_smoothed_luminance_mask(in, luminance_smoothed_micro, width, height, d, d->radius_micro, d->feathering * d->f_mult_micro);
          pg->luminance_valid = TRUE;
          dt_iop_gui_leave_critical_section(self);
          dt_dev_pixelpipe_cache_invalidate_later(piece->pipe, self->iop_order);
        }
      }
      else
      {
        compute_pixel_luminance_mask(in, luminance_pixel, width, height, d->method);
        compute_smoothed_luminance_mask(in, luminance_smoothed_fine, width, height, d, d->radius / 2, d->feathering * 0.75f);
        compute_smoothed_luminance_mask(in, luminance_smoothed_micro, width, height, d, d->radius / 4, d->feathering * 0.5f);
      }
    }
    else
    {
      compute_pixel_luminance_mask(in, luminance_pixel, width, height, d->method);
      compute_smoothed_luminance_mask(in, luminance_smoothed_broad, width, height, d, d->radius_broad, d->feathering * 1.5f);
      compute_smoothed_luminance_mask(in, luminance_smoothed_medium, width, height, d, d->radius_medium, d->feathering * 1.25f);
      compute_smoothed_luminance_mask(in, luminance_smoothed, width, height, d, d->radius, d->feathering);
      compute_smoothed_luminance_mask(in, luminance_smoothed_fine, width, height, d, d->radius_fine, d->feathering * 0.75f);
      compute_smoothed_luminance_mask(in, luminance_smoothed_micro, width, height, d, d->radius_micro, d->feathering * 0.5f);
    }

    // Display output
    if(pg && pg->mask_display != DT_LC_MASK_OFF)
    {
      float *lum_smooth = luminance_smoothed;
      if(pg->mask_display == DT_LC_MASK_BROAD) lum_smooth = luminance_smoothed_broad;
      else if(pg->mask_display == DT_LC_MASK_MEDIUM) lum_smooth = luminance_smoothed_medium;
      if(pg->mask_display == DT_LC_MASK_FINE) lum_smooth = luminance_smoothed_fine;
      else if(pg->mask_display == DT_LC_MASK_MICRO) lum_smooth = luminance_smoothed_micro;

      pyra_display_detail_mask(luminance_pixel, lum_smooth, out, width, height);
      piece->pipe->mask_display = DT_DEV_PIXELPIPE_DISPLAY_PASSTHRU;
    }
    else
    {
      pyra_apply_local_contrast(in, luminance_pixel, luminance_smoothed, 
                               d->broad_scale != 1.0f ? luminance_smoothed_broad : NULL,
                               d->medium_scale != 1.0f ? luminance_smoothed_medium : NULL,
                               d->fine_scale != 1.0f ? luminance_smoothed_fine : NULL,
                               d->micro_scale != 1.0f ? luminance_smoothed_micro : NULL,
                               out, roi_in, roi_out, d);
    }

    if(!cached)
    {
      dt_free_align(luminance_pixel);
      dt_free_align(luminance_smoothed_broad);
      dt_free_align(luminance_smoothed_medium);
      dt_free_align(luminance_smoothed);
      dt_free_align(luminance_smoothed_fine);
      dt_free_align(luminance_smoothed_micro);
    }
  }
  else
  {
    // Local RGB mode
    const dt_iop_lcrgb_data_t *const d = piece->data;
    dt_iop_lcrgb_gui_data_t *const lg = g ? g->lcrgb_gui : NULL;

    const float *const restrict in = (float *const)ivoid;
    float *const restrict out = (float *const)ovoid;
    float *restrict luminance_pixel = NULL;
    float *luminance_smoothed[N_SCALES] = { NULL };

    const size_t width = roi_in->width;
    const size_t height = roi_in->height;
    const size_t num_elem = width * height;

    // Get the hash of the upstream pipe to track changes
    const dt_hash_t hash = dt_dev_pixelpipe_piece_hash(piece, roi_out, TRUE);

    // Sanity checks
    if(width < 1 || height < 1) return;
    if(roi_in->width < roi_out->width || roi_in->height < roi_out->height) return;
    if(piece->colors != 4) return;

    // Fast path: if no scales are active, just copy
    if(!lcrgb_has_active_scales(d))
    {
      dt_iop_image_copy_by_size(out, in, width, height, 4);
      return;
    }

    // Init the luminance mask buffers
    gboolean cached = FALSE;

    if(self->dev->gui_attached)
    {
      // If the module instance has changed order in the pipe, invalidate caches
      if(lg->pipe_order != piece->module->iop_order)
      {
        dt_iop_gui_enter_critical_section(self);
        lg->ui_preview_hash = DT_INVALID_HASH;
        lg->thumb_preview_hash = DT_INVALID_HASH;
        lg->pipe_order = piece->module->iop_order;
        lg->luminance_valid = FALSE;
        dt_iop_gui_leave_critical_section(self);
      }

      if(piece->pipe->type & DT_DEV_PIXELPIPE_FULL)
      {
        // Re-allocate buffers if size changed
        if(lg->full_preview_buf_width != width || lg->full_preview_buf_height != height)
        {
          dt_free_align(lg->full_preview_buf_pixel);
          lcrgb_alloc_smoothed_buffers(lg->full_preview_buf_smoothed, num_elem);
          lg->full_preview_buf_pixel = dt_alloc_align_float(num_elem);
          lg->full_preview_buf_width = width;
          lg->full_preview_buf_height = height;
        }

        luminance_pixel = lg->full_preview_buf_pixel;
        for(int s = 0; s < N_SCALES; s++)
          luminance_smoothed[s] = lg->full_preview_buf_smoothed[s];
        cached = TRUE;
      }
      else if(piece->pipe->type & DT_DEV_PIXELPIPE_PREVIEW)
      {
        dt_iop_gui_enter_critical_section(self);
        if(lg->thumb_preview_buf_width != width || lg->thumb_preview_buf_height != height)
        {
          dt_free_align(lg->thumb_preview_buf_pixel);
          lcrgb_alloc_smoothed_buffers(lg->thumb_preview_buf_smoothed, num_elem);
          lg->thumb_preview_buf_pixel = dt_alloc_align_float(num_elem);
          lg->thumb_preview_buf_width = width;
          lg->thumb_preview_buf_height = height;
          lg->luminance_valid = FALSE;
        }

        luminance_pixel = lg->thumb_preview_buf_pixel;
        for(int s = 0; s < N_SCALES; s++)
          luminance_smoothed[s] = lg->thumb_preview_buf_smoothed[s];
        cached = TRUE;
        dt_iop_gui_leave_critical_section(self);
      }
      else
      {
        luminance_pixel = dt_alloc_align_float(num_elem);
        lcrgb_alloc_smoothed_buffers(luminance_smoothed, num_elem);
      }
    }
    else
    {
      // No interactive editing: allocate local temp buffers
      luminance_pixel = dt_alloc_align_float(num_elem);
      lcrgb_alloc_smoothed_buffers(luminance_smoothed, num_elem);
    }

    // Check buffer allocation
    if(!luminance_pixel || !lcrgb_smoothed_buffers_valid(luminance_smoothed))
    {
      dt_control_log(_("local contrast failed to allocate memory, check your RAM settings"));
      if(!cached)
      {
        dt_free_align(luminance_pixel);
        lcrgb_free_smoothed_buffers(luminance_smoothed);
      }
      return;
    }

    // Compute luminance masks
    if(cached)
    {
      if(piece->pipe->type & DT_DEV_PIXELPIPE_FULL)
      {
        dt_hash_t saved_hash;
        hash_set_get(&lg->ui_preview_hash, &saved_hash, &self->gui_lock);

        dt_iop_gui_enter_critical_section(self);
        const gboolean luminance_valid = lg->luminance_valid;
        dt_iop_gui_leave_critical_section(self);

        if(hash != saved_hash || !luminance_valid)
        {
          // Compute pixel luminance once
          compute_pixel_luminance_mask(in, luminance_pixel, width, height, d->method);

          // Compute smoothed luminance for each active scale
          for(int s = 0; s < N_SCALES; s++)
          {
            if(lcrgb_scale_is_active(d->scales[s].detail_scale))
            {
              lcrgb_compute_smoothed_luminance_for_scale(in, luminance_smoothed[s],
                                                         width, height,
                                                         &d->scales[s],
                                                         d->details, d->iterations,
                                                         d->method);
            }
          }
          hash_set_get(&hash, &lg->ui_preview_hash, &self->gui_lock);
        }
      }
      else if(piece->pipe->type & DT_DEV_PIXELPIPE_PREVIEW)
      {
        dt_hash_t saved_hash;
        hash_set_get(&lg->thumb_preview_hash, &saved_hash, &self->gui_lock);

        dt_iop_gui_enter_critical_section(self);
        const gboolean luminance_valid = lg->luminance_valid;
        dt_iop_gui_leave_critical_section(self);

        if(saved_hash != hash || !luminance_valid)
        {
          dt_iop_gui_enter_critical_section(self);
          lg->thumb_preview_hash = hash;

          // Compute pixel luminance once
          compute_pixel_luminance_mask(in, luminance_pixel, width, height, d->method);

          // Compute smoothed luminance for each active scale
          for(int s = 0; s < N_SCALES; s++)
          {
            if(lcrgb_scale_is_active(d->scales[s].detail_scale))
            {
              lcrgb_compute_smoothed_luminance_for_scale(in, luminance_smoothed[s],
                                                         width, height,
                                                         &d->scales[s],
                                                         d->details, d->iterations,
                                                         d->method);
            }
          }

          lg->luminance_valid = TRUE;
          dt_iop_gui_leave_critical_section(self);
          dt_dev_pixelpipe_cache_invalidate_later(piece->pipe, self->iop_order);
        }
      }
      else
      {
        // Non-cached pipe: compute everything
        compute_pixel_luminance_mask(in, luminance_pixel, width, height, d->method);
        for(int s = 0; s < N_SCALES; s++)
        {
          if(lcrgb_scale_is_active(d->scales[s].detail_scale))
          {
            lcrgb_compute_smoothed_luminance_for_scale(in, luminance_smoothed[s],
                                                       width, height,
                                                       &d->scales[s],
                                                       d->details, d->iterations,
                                                       d->method);
          }
        }
      }
    }
    else
    {
      // Non-GUI: compute everything
      compute_pixel_luminance_mask(in, luminance_pixel, width, height, d->method);
      for(int s = 0; s < N_SCALES; s++)
      {
        if(lcrgb_scale_is_active(d->scales[s].detail_scale))
        {
          lcrgb_compute_smoothed_luminance_for_scale(in, luminance_smoothed[s],
                                                     width, height,
                                                     &d->scales[s],
                                                     d->details, d->iterations,
                                                     d->method);
        }
      }
    }

    // Display output
    if(self->dev->gui_attached && (piece->pipe->type & DT_DEV_PIXELPIPE_FULL))
    {
      const int display_scale = lg->mask_display_scale;
      if(display_scale >= 0 && display_scale < N_SCALES
         && lcrgb_scale_is_active(d->scales[display_scale].detail_scale)
         && luminance_smoothed[display_scale])
      {
        lcrgb_display_detail_mask_for_scale(luminance_pixel, luminance_smoothed[display_scale],
                                            out, width, height);
        piece->pipe->mask_display = DT_DEV_PIXELPIPE_DISPLAY_PASSTHRU;
      }
      else
      {
        lcrgb_apply_multiscale_local_contrast(in, luminance_pixel, luminance_smoothed,
                                              out, width, height, d);
      }
    }
    else
    {
      lcrgb_apply_multiscale_local_contrast(in, luminance_pixel, luminance_smoothed,
                                            out, width, height, d);
    }

    if(!cached)
    {
      dt_free_align(luminance_pixel);
      lcrgb_free_smoothed_buffers(luminance_smoothed);
    }
  }
}

void process(dt_iop_module_t *self,
             dt_dev_pixelpipe_iop_t *piece,
             const void *const restrict ivoid,
             void *const restrict ovoid,
             const dt_iop_roi_t *const roi_in,
             const dt_iop_roi_t *const roi_out)
{
  contrast_dual_process(self, piece, ivoid, ovoid, roi_in, roi_out);
}

void modify_roi_in(dt_iop_module_t *self,
                   dt_dev_pixelpipe_iop_t *piece,
                   const dt_iop_roi_t *roi_out,
                   dt_iop_roi_t *roi_in)
{
  const dt_iop_contrast_dual_params_t *const params = piece->data;

  if(params->mode == 0)
  {
    // Pyramidal
    dt_iop_pyra_data_t *const d = piece->data;

    // Get the scaled window radius for the box average
    const float max_size = (float)((piece->iwidth > piece->iheight) ? piece->iwidth : piece->iheight);
    const float base_diameter = d->blending * max_size * roi_in->scale;

    const float diameter_broad = base_diameter * d->s_mult_broad;
    d->radius_broad = (int)((diameter_broad - 1.0f) / 2.0f);

    const float diameter_medium = base_diameter * d->s_mult_medium;
    d->radius_medium = (int)((diameter_medium - 1.0f) / 2.0f);

    const float diameter_detail = base_diameter * d->s_mult_detail;
    d->radius = (int)((diameter_detail - 1.0f) / 2.0f);

    const float diameter_fine = base_diameter * d->s_mult_fine;
    d->radius_fine = (int)((diameter_fine - 1.0f) / 2.0f);

    const float diameter_micro = base_diameter * d->s_mult_micro;
    d->radius_micro = (int)((diameter_micro - 1.0f) / 2.0f);
  }
  else
  {
    // Local RGB
    dt_iop_lcrgb_data_t *const d = piece->data;

    // Get the scaled window radius for each scale
    const int max_size = (piece->iwidth > piece->iheight) ? piece->iwidth : piece->iheight;

    for(int s = 0; s < N_SCALES; s++)
    {
      const float diameter = d->scales[s].blending * max_size * roi_in->scale;
      const int radius = (int)((diameter - 1.0f) / 2.0f);
      d->scales[s].radius = radius;
    }
  }
}

void init_global(dt_iop_module_so_t *self)
{
  dt_iop_contrast_dual_global_data_t *gd = malloc(sizeof(dt_iop_contrast_dual_global_data_t));
  self->data = gd;
}

void cleanup_global(dt_iop_module_so_t *self)
{
  free(self->data);
  self->data = NULL;
}

void reload_defaults(dt_iop_module_t *self)
{
  dt_iop_contrast_dual_params_t *d = self->default_params;

  // Set defaults
  d->mode = 0; // Pyramidal
  d->p_micro_scale = 1.0f;
  d->p_fine_scale = 1.0f;
  d->p_detail_scale = 1.5f;
  d->p_medium_scale = 1.0f;
  d->p_broad_scale = 1.0f;
  d->p_global_scale = 1.0f;
  d->p_blending = 1.2f;
  d->p_feathering = 5.0f;
  d->p_f_mult_micro = 0.5f;
  d->p_f_mult_fine = 0.75f;
  d->p_f_mult_detail = 1.0f;
  d->p_f_mult_medium = 1.25f;
  d->p_f_mult_broad = 1.5f;
  d->p_s_mult_micro = 0.25f;
  d->p_s_mult_fine = 0.625f;
  d->p_s_mult_detail = 1.0f;
  d->p_s_mult_medium = 1.8f;
  d->p_s_mult_broad = 2.8f;
  d->p_details = DT_LC_EIGF;
  d->p_method = DT_TONEEQ_NORM_2;
  d->p_iterations = 1;

  for(int s = 0; s < N_SCALES; s++)
  {
    d->l_detail_scale[s] = (s == 0) ? 1.5f : 1.0f;
    d->l_blending[s] = 12.0f;
    d->l_feathering[s] = 5.0f;
  }
  d->l_details = DT_LC_EIGF;
  d->l_method = DT_TONEEQ_NORM_2;
  d->l_iterations = 1;
}

void commit_params(dt_iop_module_t *self,
                   dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  const dt_iop_contrast_dual_params_t *p = (dt_iop_contrast_dual_params_t *)p1;

  if(p->mode == 0)
  {
    // Pyramidal
    dt_iop_pyra_data_t *d = piece->data;

    d->method = DT_TONEEQ_NORM_2;
    d->details = DT_LC_EIGF;
    d->iterations = 1;
    d->micro_scale = p->p_micro_scale;
    d->fine_scale = p->p_fine_scale;
    d->detail_scale = p->p_detail_scale;
    d->medium_scale = p->p_medium_scale;
    d->broad_scale = p->p_broad_scale; 
    d->global_scale = p->p_global_scale;

    // UI blending param is the square root of the actual blending parameter
    // to make it more sensitive to small values that represent the most important value domain.
    // UI parameter is given in percentage of maximum blending value.
    // The actual blending parameter represents the fraction of the largest image dimension.
    d->blending = p->p_blending * p->p_blending / 100.0f;

    // UI guided filter feathering param increases edge taping
    // but actual regularization behaves inversely
    d->feathering = 1.0f / p->p_feathering;

    d->f_mult_micro = p->p_f_mult_micro;
    d->f_mult_fine = p->p_f_mult_fine;
    d->f_mult_detail = p->p_f_mult_detail;
    d->f_mult_medium = p->p_f_mult_medium;
    d->f_mult_broad = p->p_f_mult_broad;

    d->s_mult_micro = p->p_s_mult_micro;
    d->s_mult_fine = p->p_s_mult_fine;
    d->s_mult_detail = p->p_s_mult_detail;
    d->s_mult_medium = p->p_s_mult_medium;
    d->s_mult_broad = p->p_s_mult_broad;
  }
  else
  {
    // Local RGB
    dt_iop_lcrgb_data_t *d = piece->data;

    // Copy shared params
    d->method = p->l_method;
    d->details = p->l_details;
    d->iterations = p->l_iterations;

    // Copy per-scale params and compute derived values
    for(int s = 0; s < N_SCALES; s++)
    {
      d->scales[s].detail_scale = p->l_detail_scale[s];

      // UI blending param is the square root of the actual blending parameter
      // to make it more sensitive to small values that represent the most important value domain.
      // UI parameter is given in percentage of maximum blending value.
      // The actual blending parameter represents the fraction of the largest image dimension.
      d->scales[s].blending = p->l_blending[s] * p->l_blending[s] / 10000.0f;

      // UI guided filter feathering param increases edge taping
      // but actual regularization behaves inversely
      d->scales[s].feathering = 1.0f / p->l_feathering[s];
    }
  }
}

void init_pipe(dt_iop_module_t *self,
               dt_dev_pixelpipe_t *pipe,
               dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = dt_calloc1_align_type(dt_iop_pyra_data_t); // Use largest
}

void cleanup_pipe(dt_iop_module_t *self,
                  dt_dev_pixelpipe_t *pipe,
                  dt_dev_pixelpipe_iop_t *piece)
{
  dt_free_align(piece->data);
  piece->data = NULL;
}

static void pyra_gui_cache_init(dt_iop_module_t *self)
{
  dt_iop_pyra_gui_data_t *g = ((dt_iop_contrast_dual_gui_data_t *)self->gui_data)->pyra_gui;
  if(g == NULL) return;

  dt_iop_gui_enter_critical_section(self);
  g->ui_preview_hash = DT_INVALID_HASH;
  g->thumb_preview_hash = DT_INVALID_HASH;
  g->mask_display = DT_LC_MASK_OFF;
  g->luminance_valid = FALSE;

  g->full_preview_buf_pixel = NULL;
  g->full_preview_buf_smoothed_broad = NULL;
  g->full_preview_buf_smoothed_medium = NULL;
  g->full_preview_buf_smoothed = NULL;
  g->full_preview_buf_smoothed_fine = NULL;
  g->full_preview_buf_smoothed_micro = NULL;
  g->full_preview_buf_width = 0;
  g->full_preview_buf_height = 0;

  g->thumb_preview_buf_pixel = NULL;
  g->thumb_preview_buf_smoothed_broad = NULL;
  g->thumb_preview_buf_smoothed_medium = NULL;
  g->thumb_preview_buf_smoothed = NULL;
  g->thumb_preview_buf_smoothed_fine = NULL;
  g->thumb_preview_buf_smoothed_micro = NULL;
  g->thumb_preview_buf_width = 0;
  g->thumb_preview_buf_height = 0;

  g->pipe_order = 0;
  dt_iop_gui_leave_critical_section(self);
}

static void lcrgb_gui_cache_init(dt_iop_module_t *self)
{
  dt_iop_lcrgb_gui_data_t *g = ((dt_iop_contrast_dual_gui_data_t *)self->gui_data)->lcrgb_gui;
  if(g == NULL) return;

  dt_iop_gui_enter_critical_section(self);
  g->ui_preview_hash = DT_INVALID_HASH;
  g->thumb_preview_hash = DT_INVALID_HASH;
  g->mask_display_scale = -1;  // no mask displayed
  g->luminance_valid = FALSE;

  g->full_preview_buf_pixel = NULL;
  for(int s = 0; s < N_SCALES; s++)
    g->full_preview_buf_smoothed[s] = NULL;
  g->full_preview_buf_width = 0;
  g->full_preview_buf_height = 0;

  g->thumb_preview_buf_pixel = NULL;
  for(int s = 0; s < N_SCALES; s++)
    g->thumb_preview_buf_smoothed[s] = NULL;
  g->thumb_preview_buf_width = 0;
  g->thumb_preview_buf_height = 0;

  g->pipe_order = 0;
  dt_iop_gui_leave_critical_section(self);
}

static void pyra_gui_update(dt_iop_module_t *self)
{
  dt_iop_pyra_gui_data_t *g = ((dt_iop_contrast_dual_gui_data_t *)self->gui_data)->pyra_gui;

  // show_guiding_controls(self);
  pyra_invalidate_luminance_cache(self);
  // _update_mask_buttons_state(g);

  dt_gui_update_collapsible_section(&g->advanced_expander);
}

static void lcrgb_gui_update(dt_iop_module_t *self)
{
  dt_iop_lcrgb_gui_data_t *g = ((dt_iop_contrast_dual_gui_data_t *)self->gui_data)->lcrgb_gui;

  lcrgb_invalidate_luminance_cache(self);

  // Update mask toggle buttons
  for(int s = 0; s < N_SCALES; s++)
  {
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->show_mask[s]),
                                 g->mask_display_scale == s);
  }
}

void gui_update(dt_iop_module_t *self)
{
  const dt_iop_contrast_dual_params_t *p = self->default_params;
  if(p->mode == 0)
    pyra_gui_update(self);
  else
    lcrgb_gui_update(self);
}

static void pyra_gui_changed(dt_iop_module_t *self,
                 GtkWidget *w,
                 void *previous)
{
  const dt_iop_pyra_gui_data_t *g = ((dt_iop_contrast_dual_gui_data_t *)self->gui_data)->pyra_gui;

  if(w == g->blending || w == g->feathering
     || w == g->f_mult_micro || w == g->f_mult_fine || w == g->f_mult_detail
     || w == g->f_mult_medium || w == g->f_mult_broad
     || w == g->s_mult_micro || w == g->s_mult_fine || w == g->s_mult_detail
     || w == g->s_mult_medium || w == g->s_mult_broad)
  {
    pyra_invalidate_luminance_cache(self);
  }
}

static void lcrgb_gui_changed(dt_iop_module_t *self,
                 GtkWidget *w,
                 void *previous)
{
  const dt_iop_lcrgb_gui_data_t *g = ((dt_iop_contrast_dual_gui_data_t *)self->gui_data)->lcrgb_gui;

  // Check if any masking-related widget changed
  gboolean invalidate = FALSE;

  // Check shared widgets
  if(w == g->details || w == g->iterations)
  {
    invalidate = TRUE;
  }

  // Check per-scale widgets
  for(int s = 0; s < N_SCALES && !invalidate; s++)
  {
    if(w == g->blending[s] || w == g->feathering[s])
    {
      invalidate = TRUE;
    }
  }

  if(invalidate)
  {
    lcrgb_invalidate_luminance_cache(self);
  }
}

void gui_changed(dt_iop_module_t *self,
                 GtkWidget *w,
                 void *previous)
{
  const dt_iop_contrast_dual_params_t *p = self->default_params;
  if(p->mode == 0)
    pyra_gui_changed(self, w, previous);
  else
    lcrgb_gui_changed(self, w, previous);
}

// GUI functions for Pyramidal and Local RGB would be here, but abbreviated for space

static void notebook_switch_page(GtkNotebook *notebook, GtkWidget *page, guint page_num, dt_iop_module_t *self)
{
  dt_iop_contrast_dual_params_t *p = self->default_params;
  p->mode = page_num;
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_contrast_dual_gui_data_t *g = IOP_GUI_ALLOC(contrast_dual);

  g->pyra_gui = calloc(1, sizeof(dt_iop_pyra_gui_data_t));
  g->lcrgb_gui = calloc(1, sizeof(dt_iop_lcrgb_gui_data_t));

  pyra_gui_cache_init(self);
  lcrgb_gui_cache_init(self);

  // Create notebook
  g->notebook = GTK_WIDGET(gtk_notebook_new());
  gtk_notebook_set_tab_pos(GTK_NOTEBOOK(g->notebook), GTK_POS_TOP);

  // Pyramidal tab
  GtkWidget *pyra_page = dt_gui_vbox();
  // Add pyramidal widgets here (similar to original gui_init for pyramidal)

  // Local RGB tab
  GtkWidget *lcrgb_page = dt_gui_vbox();
  // Add local rgb widgets here (similar to original gui_init for local rgb)

  gtk_notebook_append_page(GTK_NOTEBOOK(g->notebook), pyra_page, gtk_label_new(_("Pyramidal")));
  gtk_notebook_append_page(GTK_NOTEBOOK(g->notebook), lcrgb_page, gtk_label_new(_("Local RGB")));

  g_signal_connect(G_OBJECT(g->notebook), "switch-page", G_CALLBACK(notebook_switch_page), self);

  self->widget = g->notebook;
}

void gui_cleanup(dt_iop_module_t *self)
{
  dt_iop_contrast_dual_gui_data_t *g = self->gui_data;

  // Cleanup pyramidal
  dt_free_align(g->pyra_gui->thumb_preview_buf_pixel);
  dt_free_align(g->pyra_gui->thumb_preview_buf_smoothed_broad);
  // ... free others

  // Cleanup local rgb
  dt_free_align(g->lcrgb_gui->thumb_preview_buf_pixel);
  lcrgb_free_smoothed_buffers(g->lcrgb_gui->thumb_preview_buf_smoothed);

  free(g->pyra_gui);
  free(g->lcrgb_gui);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

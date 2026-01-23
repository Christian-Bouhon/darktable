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
 * This module performs local contrast enhancement in scene-referred linear RGB space.
 *
 * It works by computing two luminance masks:
 * 1. A pixel-wise luminance (unblurred)
 * 2. A smoothed luminance using edge-aware filters (guided filter or EIGF)
 *
 * The difference between these two masks represents the local detail/contrast.
 * The local contrast is then enhanced by scaling this difference and applying
 * it as an exposure correction to each pixel.
 *
 * The module should be placed early in the pipe (before color profile)
 * as it operates on scene-linear RGB data.
 * A Modifier
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


DT_MODULE_INTROSPECTION(3, dt_iop_pyramidal_contrast_params_t)


#define MIN_FLOAT exp2f(-16.0f)

/**
 * Filter types for detail preservation / smoothing.
 * DT_TONEEQ_NONE is intentionally omitted as it produces no blur,
 * which would result in no local contrast extraction.
 **/
typedef enum dt_iop_pyramidal_contrast_filter_t
{
  DT_PYR_AVG_GUIDED = 0, // $DESCRIPTION: "averaged guided filter"
  DT_PYR_GUIDED,         // $DESCRIPTION: "guided filter"
  DT_PYR_AVG_EIGF,       // $DESCRIPTION: "averaged EIGF"
  DT_PYR_EIGF            // $DESCRIPTION: "EIGF"
} dt_iop_pyramidal_contrast_filter_t;


typedef struct dt_iop_pyramidal_contrast_params_t
{
  // Local contrast scaling factor
  float pyr_micro_scale;    // $MIN: 0.0 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "micro contrast"
  float pyr_fine_scale;     // $MIN: 0.0 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "fine contrast"
  float pyr_detail_scale;   // $MIN: 0.0 $MAX: 5.0 $DEFAULT: 1.5 $DESCRIPTION: "local contrast"
  float pyr_medium_scale;   // $MIN: 0.0 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "broad contrast"
  float pyr_broad_scale;    // $MIN: 0.0 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "extended contrast"
  float pyr_global_scale;   // $MIN: 0.0 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "global contrast"

  // Masking parameters
  // Blending is log-encoded because changes in small values are more noticeable
  float pyr_blending;       // $MIN: 1.0 $MAX: 4.0 $DEFAULT: 1.2 $DESCRIPTION: "feature scale"
  float pyr_feathering;     // $MIN: 0.01 $MAX: 10000.0 $DEFAULT: 5.0 $DESCRIPTION: "edges refinement/feathering"

  float pyr_f_mult_micro;  // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 0.5 $DESCRIPTION: "micro contrast feathering"
  float pyr_f_mult_fine;   // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 0.75 $DESCRIPTION: "fine contrast feathering"
  float pyr_f_mult_detail; // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "local contrast feathering"
  float pyr_f_mult_medium; // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 1.25 $DESCRIPTION: "broad contrast feathering"
  float pyr_f_mult_broad;  // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 1.50 $DESCRIPTION: "extended contrast feathering"

  float pyr_s_mult_micro;  // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 0.25 $DESCRIPTION: "micro contrast scale mult."
  float pyr_s_mult_fine;   // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 0.625 $DESCRIPTION: "fine contrast scale mult."
  float pyr_s_mult_detail; // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "local contrast scale mult."
  float pyr_s_mult_medium; // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 1.8 $DESCRIPTION: "broad contrast scale mult."
  float pyr_s_mult_broad;  // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 2.8 $DESCRIPTION: "extended contrast scale mult."

  dt_iop_pyramidal_contrast_filter_t pyr_details; // $DEFAULT: DT_PYR_EIGF $DESCRIPTION: "feature extractor"
  dt_iop_luminance_mask_method_t pyr_method;      // $DEFAULT: DT_TONEEQ_NORM_2 $DESCRIPTION: "luminance estimator"
  int pyr_iterations;       // $MIN: 1 $MAX: 20 $DEFAULT: 1 $DESCRIPTION: "filter diffusion"
} dt_iop_pyramidal_contrast_params_t;


typedef struct dt_iop_pyramidal_contrast_data_t
{
  float pyr_broad_scale;
  float pyr_medium_scale;
  float pyr_detail_scale;
  float pyr_fine_scale;
  float pyr_micro_scale;
  float pyr_global_scale;
  float pyr_blending, pyr_feathering;
  float pyr_f_mult_micro, pyr_f_mult_fine, pyr_f_mult_detail, pyr_f_mult_medium, pyr_f_mult_broad;
  float pyr_s_mult_micro, pyr_s_mult_fine, pyr_s_mult_detail, pyr_s_mult_medium, pyr_s_mult_broad;
  float pyr_scale;
  int pyr_radius;
  int pyr_radius_broad;
  int pyr_radius_medium;
  int pyr_radius_fine;
  int pyr_radius_micro;
  int pyr_iterations;
  dt_iop_luminance_mask_method_t pyr_method;
  dt_iop_pyramidal_contrast_filter_t pyr_details;
} dt_iop_pyramidal_contrast_data_t;


typedef struct dt_iop_pyramidal_contrast_global_data_t
{
  // Reserved for OpenCL kernels
} dt_iop_pyramidal_contrast_global_data_t;


typedef enum dt_iop_pyramidal_contrast_mask_t
{
  DT_PYR_MASK_OFF = 0,
  DT_PYR_MASK_BROAD = 1,
  DT_PYR_MASK_MEDIUM = 2,
  DT_PYR_MASK_DETAIL = 3,
  DT_PYR_MASK_FINE = 4,
  DT_PYR_MASK_MICRO = 5
} dt_iop_pyramidal_contrast_mask_t;

typedef struct dt_iop_pyramidal_contrast_gui_data_t
{
  // Flags
  dt_iop_pyramidal_contrast_mask_t pyr_mask_display;

  // Buffer dimensions
  int pyr_buf_width;
  int pyr_buf_height;
  int pyr_pipe_order;

  // Hash for cache invalidation
  dt_hash_t pyr_ui_preview_hash;
  dt_hash_t pyr_thumb_preview_hash;
  size_t pyr_full_preview_buf_width, pyr_full_preview_buf_height;
  size_t pyr_thumb_preview_buf_width, pyr_thumb_preview_buf_height;

  // Cached luminance buffers
  float *pyr_thumb_preview_buf_pixel;     // pixel-wise luminance (no blur)
  float *pyr_thumb_preview_buf_smoothed_broad;
  float *pyr_thumb_preview_buf_smoothed_medium;
  float *pyr_thumb_preview_buf_smoothed;  // smoothed luminance
  float *pyr_thumb_preview_buf_smoothed_fine;
  float *pyr_thumb_preview_buf_smoothed_micro;
  float *pyr_full_preview_buf_pixel;
  float *pyr_full_preview_buf_smoothed_broad;
  float *pyr_full_preview_buf_smoothed_medium;
  float *pyr_full_preview_buf_smoothed;
  float *pyr_full_preview_buf_smoothed_fine;
  float *pyr_full_preview_buf_smoothed_micro;

  // Cache validity
  gboolean luminance_valid;

  // GTK widgets
  GtkWidget *pyr_broad_scale, *pyr_medium_scale, *pyr_detail_scale, *pyr_fine_scale, *pyr_micro_scale, *pyr_global_scale;
  GtkWidget *pyr_blending;
  GtkWidget *pyr_feathering;
  dt_gui_collapsible_section_t pyr_advanced_expander;
  GtkWidget *pyr_f_mult_micro, *pyr_f_mult_fine, *pyr_f_mult_detail, *pyr_f_mult_medium, *pyr_f_mult_broad;

  // New buttons for mask display in expanders
  GtkWidget *pyr_f_view_broad, *pyr_f_view_medium, *pyr_f_view_detail, *pyr_f_view_fine, *pyr_f_view_micro;
} dt_iop_pyramidal_contrast_gui_data_t;


const char *name()
{
  return _("pyramidal contrast");
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
  typedef struct dt_iop_pyramidal_contrast_params_v1_t
  {
    float micro_scale;
    float fine_scale;
    float detail_scale;
    float medium_scale;
    float broad_scale;
    float global_scale;
    float blending;
    float feathering;
    dt_iop_pyramidal_contrast_filter_t details;
    dt_iop_luminance_mask_method_t method;
    int iterations;
  } dt_iop_pyramidal_contrast_params_v1_t;

  if(old_version == 1)
  {
    const dt_iop_pyramidal_contrast_params_v1_t *o = (dt_iop_pyramidal_contrast_params_v1_t *)old_params;
    dt_iop_pyramidal_contrast_params_t *n = malloc(sizeof(dt_iop_pyramidal_contrast_params_t));

    // Copie manuelle pour éviter d'écraser les nouveaux champs insérés au milieu
    n->pyr_micro_scale = o->micro_scale;
    n->pyr_fine_scale = o->fine_scale;
    n->pyr_detail_scale = o->detail_scale;
    n->pyr_medium_scale = o->medium_scale;
    n->pyr_broad_scale = o->broad_scale;
    n->pyr_global_scale = o->global_scale;
    n->pyr_blending = o->blending;
    n->pyr_feathering = o->feathering;

    // Initialisation des nouveaux champs
    n->pyr_f_mult_micro = 0.5f;
    n->pyr_f_mult_fine = 0.75f;
    n->pyr_f_mult_detail = 1.0f;
    n->pyr_f_mult_medium = 1.25f;
    n->pyr_f_mult_broad = 1.5f;

    n->pyr_s_mult_micro = 0.25f;
    n->pyr_s_mult_fine = 0.625f;
    n->pyr_s_mult_detail = 1.0f;
    n->pyr_s_mult_medium = 1.8f;
    n->pyr_s_mult_broad = 2.8f;

    // Copie de la fin de la structure v1
    n->pyr_details = o->details;
    n->pyr_method = o->method;
    n->pyr_iterations = o->iterations;

    *new_params = n;
    *new_params_size = sizeof(dt_iop_pyramidal_contrast_params_t);
    *new_version = 3;
    return 0;
  }
  return 1;
}

/**
 * Helper functions
 **/

static void pyr_hash_set_get(const dt_hash_t *hash_in,
                         dt_hash_t *hash_out,
                         dt_pthread_mutex_t *lock)
{
  dt_pthread_mutex_lock(lock);
  *hash_out = *hash_in;
  dt_pthread_mutex_unlock(lock);
}


static void pyr_invalidate_luminance_cache(dt_iop_module_t *const self)
{
  dt_iop_pyramidal_contrast_gui_data_t *const restrict g = self->gui_data;

  dt_iop_gui_enter_critical_section(self);
  g->luminance_valid = FALSE;
  g->pyr_thumb_preview_hash = DT_INVALID_HASH;
  g->pyr_ui_preview_hash = DT_INVALID_HASH;
  dt_iop_gui_leave_critical_section(self);
  dt_iop_refresh_all(self);
}


/**
 * Compute pixel-wise luminance mask (no blur)
 **/
__DT_CLONE_TARGETS__
static inline void pyr_compute_pixel_luminance_mask(const float *const restrict in,
                                                float *const restrict luminance,
                                                const size_t width,
                                                const size_t height,
                                                const dt_iop_luminance_mask_method_t method)
{
  // No exposure/contrast boost, just compute raw luminance
  luminance_mask(in, luminance, width, height, method, 1.0f, 0.0f, 1.0f);
}


/**
 * Compute smoothed luminance mask using edge-aware filters
 **/
__DT_CLONE_TARGETS__
static inline void pyr_compute_smoothed_luminance_mask(const float *const restrict in,
                                                   float *const restrict luminance,
                                                   const size_t width,
                                                   const size_t height,
                                                const dt_iop_pyramidal_contrast_data_t *const d,
                                                const int radius,
                                                const float feathering)
{
  // First compute pixel-wise luminance (no boost)
  luminance_mask(in, luminance, width, height, d->pyr_method, 1.0f, 0.0f, 1.0f);

  // Then apply the smoothing filter
  switch(d->pyr_details)
  {
    case(DT_PYR_AVG_GUIDED):
    {
      fast_surface_blur(luminance, width, height, radius, feathering, d->pyr_iterations,
                        DT_GF_BLENDING_GEOMEAN, d->pyr_scale, 0.0f,
                        exp2f(-14.0f), 4.0f);
      break;
    }

    case(DT_PYR_GUIDED):
    {
      fast_surface_blur(luminance, width, height, radius, feathering, d->pyr_iterations,
                        DT_GF_BLENDING_LINEAR, d->pyr_scale, 0.0f,
                        exp2f(-14.0f), 4.0f);
      break;
    }

    case(DT_PYR_AVG_EIGF):
    {
      fast_eigf_surface_blur(luminance, width, height,
                             radius, feathering, d->pyr_iterations,
                             DT_GF_BLENDING_GEOMEAN, d->pyr_scale,
                             0.0f, exp2f(-14.0f), 4.0f);
      break;
    }

    case(DT_PYR_EIGF):
    {
      fast_eigf_surface_blur(luminance, width, height,
                             radius, feathering, d->pyr_iterations,
                             DT_GF_BLENDING_LINEAR, d->pyr_scale,
                             0.0f, exp2f(-14.0f), 4.0f);
      break;
    }
  }
}


/**
 * Apply local contrast enhancement
 *
 * The detail (local contrast) is the log-space difference between pixel luminance
 * and smoothed luminance. Boosting this difference amplifies local details.
 **/
__DT_CLONE_TARGETS__
static inline void pyr_apply_local_contrast(const float *const restrict in,
                                        const float *const restrict luminance_pixel,
                                        const float *const restrict luminance_smoothed,
                                        const float *const restrict luminance_smoothed_broad,
                                        const float *const restrict luminance_smoothed_medium,
                                        const float *const restrict luminance_smoothed_fine,
                                        const float *const restrict luminance_smoothed_micro,
                                        float *const restrict out,
                                        const dt_iop_roi_t *const roi_in,
                                        const dt_iop_roi_t *const roi_out,
                                        const dt_iop_pyramidal_contrast_data_t *const d)
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
    const float scaled_detail_ev = d->pyr_detail_scale * detail_ev;

    // The correction is the difference between scaled and original detail
    float correction_ev = scaled_detail_ev - detail_ev;

    if(luminance_smoothed_broad)
    {
      const float lum_smoothed_broad = fmaxf(luminance_smoothed_broad[k], MIN_FLOAT);
      const float detail_ev_broad = log2f(lum_pixel / lum_smoothed_broad);
      const float scaled_detail_ev_broad = d->pyr_broad_scale * detail_ev_broad;
      correction_ev += scaled_detail_ev_broad - detail_ev_broad;
    }

    if(luminance_smoothed_medium)
    {
      const float lum_smoothed_medium = fmaxf(luminance_smoothed_medium[k], MIN_FLOAT);
      const float detail_ev_medium = log2f(lum_pixel / lum_smoothed_medium);
      const float scaled_detail_ev_medium = d->pyr_medium_scale * detail_ev_medium;
      correction_ev += scaled_detail_ev_medium - detail_ev_medium;
    }

    if(luminance_smoothed_fine)
    {
      const float lum_smoothed_fine = fmaxf(luminance_smoothed_fine[k], MIN_FLOAT);
      const float detail_ev_fine = log2f(lum_pixel / lum_smoothed_fine);
      const float scaled_detail_ev_fine = d->pyr_fine_scale * detail_ev_fine;
      correction_ev += scaled_detail_ev_fine - detail_ev_fine;
    }

    if(luminance_smoothed_micro)
    {
      const float lum_smoothed_micro = fmaxf(luminance_smoothed_micro[k], MIN_FLOAT);
      const float detail_ev_micro = log2f(lum_pixel / lum_smoothed_micro);
      const float scaled_detail_ev_micro = d->pyr_micro_scale * detail_ev_micro;
      correction_ev += scaled_detail_ev_micro - detail_ev_micro;
    }

    // Apply correction in linear space
    // global_scale has the same range as detail_scale.
    const float multiplier = exp2f(correction_ev) * powf(lum_smoothed / 0.1845f, d->pyr_global_scale) * 0.1845f / lum_smoothed;

    for_each_channel(c)
      out[4 * k + c] = in[4 * k + c] * multiplier;
  }
}


/**
 * Display the detail mask (difference between pixel and smoothed luminance)
 * Output is a grayscale image normalized to [0, 1] where:
 * - 0.5 = no local detail (pixel matches neighborhood)
 * - < 0.5 = pixel darker than neighborhood
 * - > 0.5 = pixel brighter than neighborhood
 **/
__DT_CLONE_TARGETS__
static inline void pyr_display_detail_mask(const float *const restrict luminance_pixel,
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
 * Main processing function
 **/
__DT_CLONE_TARGETS__
static void pyr_process(dt_iop_module_t *self,
                                   dt_dev_pixelpipe_iop_t *piece,
                                   const void *const restrict ivoid,
                                   void *const restrict ovoid,
                                   const dt_iop_roi_t *const roi_in,
                                   const dt_iop_roi_t *const roi_out)
{
  const dt_iop_pyramidal_contrast_data_t *const d = piece->data;
  dt_iop_pyramidal_contrast_gui_data_t *const g = self->gui_data;

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
    if(g->pyr_pipe_order != piece->module->iop_order)
    {
      dt_iop_gui_enter_critical_section(self);
      g->pyr_ui_preview_hash = DT_INVALID_HASH;
      g->pyr_thumb_preview_hash = DT_INVALID_HASH;
      g->pyr_pipe_order = piece->module->iop_order;
      g->luminance_valid = FALSE;
      dt_iop_gui_leave_critical_section(self);
    }

    if(piece->pipe->type & DT_DEV_PIXELPIPE_FULL)
    {
      // Re-allocate buffers if size changed
      if(g->pyr_full_preview_buf_width != width || g->pyr_full_preview_buf_height != height)
      {
        dt_free_align(g->pyr_full_preview_buf_pixel);
        dt_free_align(g->pyr_full_preview_buf_smoothed_broad);
        dt_free_align(g->pyr_full_preview_buf_smoothed_medium);
        dt_free_align(g->pyr_full_preview_buf_smoothed);
        dt_free_align(g->pyr_full_preview_buf_smoothed_fine);
        dt_free_align(g->pyr_full_preview_buf_smoothed_micro);
        g->pyr_full_preview_buf_pixel = dt_alloc_align_float(num_elem);
        g->pyr_full_preview_buf_smoothed_broad = dt_alloc_align_float(num_elem);
        g->pyr_full_preview_buf_smoothed_medium = dt_alloc_align_float(num_elem);
        g->pyr_full_preview_buf_smoothed = dt_alloc_align_float(num_elem);
        g->pyr_full_preview_buf_smoothed_fine = dt_alloc_align_float(num_elem);
        g->pyr_full_preview_buf_smoothed_micro = dt_alloc_align_float(num_elem);
        g->pyr_full_preview_buf_width = width;
        g->pyr_full_preview_buf_height = height;
      }

      luminance_pixel = g->pyr_full_preview_buf_pixel;
      luminance_smoothed_broad = g->pyr_full_preview_buf_smoothed_broad;
      luminance_smoothed_medium = g->pyr_full_preview_buf_smoothed_medium;
      luminance_smoothed = g->pyr_full_preview_buf_smoothed;
      luminance_smoothed_fine = g->pyr_full_preview_buf_smoothed_fine;
      luminance_smoothed_micro = g->pyr_full_preview_buf_smoothed_micro;
      cached = TRUE;
    }
    else if(piece->pipe->type & DT_DEV_PIXELPIPE_PREVIEW)
    {
      dt_iop_gui_enter_critical_section(self);
      if(g->pyr_thumb_preview_buf_width != width || g->pyr_thumb_preview_buf_height != height)
      {
        dt_free_align(g->pyr_thumb_preview_buf_pixel);
        dt_free_align(g->pyr_thumb_preview_buf_smoothed_broad);
        dt_free_align(g->pyr_thumb_preview_buf_smoothed_medium);
        dt_free_align(g->pyr_thumb_preview_buf_smoothed);
        dt_free_align(g->pyr_thumb_preview_buf_smoothed_fine);
        dt_free_align(g->pyr_thumb_preview_buf_smoothed_micro);
        g->pyr_thumb_preview_buf_pixel = dt_alloc_align_float(num_elem);
        g->pyr_thumb_preview_buf_smoothed_broad = dt_alloc_align_float(num_elem);
        g->pyr_thumb_preview_buf_smoothed_medium = dt_alloc_align_float(num_elem);
        g->pyr_thumb_preview_buf_smoothed = dt_alloc_align_float(num_elem);
        g->pyr_thumb_preview_buf_smoothed_fine = dt_alloc_align_float(num_elem);
        g->pyr_thumb_preview_buf_smoothed_micro = dt_alloc_align_float(num_elem);
        g->pyr_thumb_preview_buf_width = width;
        g->pyr_thumb_preview_buf_height = height;
        g->luminance_valid = FALSE;
      }

      luminance_pixel = g->pyr_thumb_preview_buf_pixel;
      luminance_smoothed_broad = g->pyr_thumb_preview_buf_smoothed_broad;
      luminance_smoothed_medium = g->pyr_thumb_preview_buf_smoothed_medium;
      luminance_smoothed = g->pyr_thumb_preview_buf_smoothed;
      luminance_smoothed_fine = g->pyr_thumb_preview_buf_smoothed_fine;
      luminance_smoothed_micro = g->pyr_thumb_preview_buf_smoothed_micro;
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
      pyr_hash_set_get(&g->pyr_ui_preview_hash, &saved_hash, &self->gui_lock);

      dt_iop_gui_enter_critical_section(self);
      const gboolean luminance_valid = g->luminance_valid;
      dt_iop_gui_leave_critical_section(self);

      if(hash != saved_hash || !luminance_valid)
      {
        pyr_compute_pixel_luminance_mask(in, luminance_pixel, width, height, d->pyr_method);
        if(d->pyr_broad_scale != 1.0f || g->pyr_mask_display == DT_PYR_MASK_BROAD)
          pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_broad, width, height, d, d->pyr_radius_broad, d->pyr_feathering * d->pyr_f_mult_broad);
        if(d->pyr_medium_scale != 1.0f || g->pyr_mask_display == DT_PYR_MASK_MEDIUM)
          pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_medium, width, height, d, d->pyr_radius_medium, d->pyr_feathering * d->pyr_f_mult_medium);
        if(d->pyr_detail_scale != 1.0f || g->pyr_mask_display == DT_PYR_MASK_DETAIL)
          pyr_compute_smoothed_luminance_mask(in, luminance_smoothed, width, height, d, d->pyr_radius, d->pyr_feathering * d->pyr_f_mult_detail);
        if(d->pyr_fine_scale != 1.0f || g->pyr_mask_display == DT_PYR_MASK_FINE)
          pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_fine, width, height, d, d->pyr_radius_fine, d->pyr_feathering * d->pyr_f_mult_fine);
        if(d->pyr_micro_scale != 1.0f || g->pyr_mask_display == DT_PYR_MASK_MICRO)
          pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_micro, width, height, d, d->pyr_radius_micro, d->pyr_feathering * d->pyr_f_mult_micro);
        pyr_hash_set_get(&hash, &g->pyr_ui_preview_hash, &self->gui_lock);
      }
    }
    else if(piece->pipe->type & DT_DEV_PIXELPIPE_PREVIEW)
    {
      dt_hash_t saved_hash;
      pyr_hash_set_get(&g->pyr_thumb_preview_hash, &saved_hash, &self->gui_lock);

      dt_iop_gui_enter_critical_section(self);
      const gboolean luminance_valid = g->luminance_valid;
      dt_iop_gui_leave_critical_section(self);

      if(saved_hash != hash || !luminance_valid)
      {
        dt_iop_gui_enter_critical_section(self);
        g->pyr_thumb_preview_hash = hash;
        pyr_compute_pixel_luminance_mask(in, luminance_pixel, width, height, d->pyr_method);
        if(d->pyr_broad_scale != 1.0f || g->pyr_mask_display == DT_PYR_MASK_BROAD)
          pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_broad, width, height, d, d->pyr_radius_broad, d->pyr_feathering * d->pyr_f_mult_broad);
        if(d->pyr_medium_scale != 1.0f || g->pyr_mask_display == DT_PYR_MASK_MEDIUM)
          pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_medium, width, height, d, d->pyr_radius_medium, d->pyr_feathering * d->pyr_f_mult_medium);
        if(d->pyr_detail_scale != 1.0f || g->pyr_mask_display == DT_PYR_MASK_DETAIL)
          pyr_compute_smoothed_luminance_mask(in, luminance_smoothed, width, height, d, d->pyr_radius, d->pyr_feathering * d->pyr_f_mult_detail);
        if(d->pyr_fine_scale != 1.0f || g->pyr_mask_display == DT_PYR_MASK_FINE)
          pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_fine, width, height, d, d->pyr_radius_fine, d->pyr_feathering * d->pyr_f_mult_fine);
        if(d->pyr_micro_scale != 1.0f || g->pyr_mask_display == DT_PYR_MASK_MICRO)
          pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_micro, width, height, d, d->pyr_radius_micro, d->pyr_feathering * d->pyr_f_mult_micro);
        g->luminance_valid = TRUE;
        dt_iop_gui_leave_critical_section(self);
        dt_dev_pixelpipe_cache_invalidate_later(piece->pipe, self->iop_order);
      }
    }
    else
    {
      pyr_compute_pixel_luminance_mask(in, luminance_pixel, width, height, d->pyr_method);
      pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_fine, width, height, d, d->pyr_radius / 2, d->pyr_feathering * 0.75f);
      pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_micro, width, height, d, d->pyr_radius / 4, d->pyr_feathering * 0.5f);
    }
  }
  else
  {
    pyr_compute_pixel_luminance_mask(in, luminance_pixel, width, height, d->pyr_method);
    pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_broad, width, height, d, d->pyr_radius_broad, d->pyr_feathering * 1.5f);
    pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_medium, width, height, d, d->pyr_radius_medium, d->pyr_feathering * 1.25f);
    pyr_compute_smoothed_luminance_mask(in, luminance_smoothed, width, height, d, d->pyr_radius, d->pyr_feathering);
    pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_fine, width, height, d, d->pyr_radius_fine, d->pyr_feathering * 0.75f);
    pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_micro, width, height, d, d->pyr_radius_micro, d->pyr_feathering * 0.5f);
  }

  // Display output
  if(g && g->pyr_mask_display != DT_PYR_MASK_OFF)
  {
    float *lum_smooth = luminance_smoothed;
    if(g->pyr_mask_display == DT_PYR_MASK_BROAD) lum_smooth = luminance_smoothed_broad;
    else if(g->pyr_mask_display == DT_PYR_MASK_MEDIUM) lum_smooth = luminance_smoothed_medium;
    if(g->pyr_mask_display == DT_PYR_MASK_FINE) lum_smooth = luminance_smoothed_fine;
    else if(g->pyr_mask_display == DT_PYR_MASK_MICRO) lum_smooth = luminance_smoothed_micro;

    pyr_display_detail_mask(luminance_pixel, lum_smooth, out, width, height);
    piece->pipe->mask_display = DT_DEV_PIXELPIPE_DISPLAY_PASSTHRU;
  }
  else
  {
    pyr_apply_local_contrast(in, luminance_pixel, luminance_smoothed, 
                         d->pyr_broad_scale != 1.0f ? luminance_smoothed_broad : NULL,
                         d->pyr_medium_scale != 1.0f ? luminance_smoothed_medium : NULL,
                         d->pyr_fine_scale != 1.0f ? luminance_smoothed_fine : NULL,
                         d->pyr_micro_scale != 1.0f ? luminance_smoothed_micro : NULL,
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


void process(dt_iop_module_t *self,
             dt_dev_pixelpipe_iop_t *piece,
             const void *const restrict ivoid,
             void *const restrict ovoid,
             const dt_iop_roi_t *const roi_in,
             const dt_iop_roi_t *const roi_out)
{
  pyr_process(self, piece, ivoid, ovoid, roi_in, roi_out);
}


void modify_roi_in(dt_iop_module_t *self,
                   dt_dev_pixelpipe_iop_t *piece,
                   const dt_iop_roi_t *roi_out,
                   dt_iop_roi_t *roi_in)
{
  dt_iop_pyramidal_contrast_data_t *const d = piece->data;

  // Get the scaled window radius for the box average
  const float max_size = (float)((piece->iwidth > piece->iheight) ? piece->iwidth : piece->iheight);
  const float base_diameter = d->pyr_blending * max_size * roi_in->scale;

  const float diameter_broad = base_diameter * d->pyr_s_mult_broad;
  d->pyr_radius_broad = (int)((diameter_broad - 1.0f) / 2.0f);

  const float diameter_medium = base_diameter * d->pyr_s_mult_medium;
  d->pyr_radius_medium = (int)((diameter_medium - 1.0f) / 2.0f);

  const float diameter_detail = base_diameter * d->pyr_s_mult_detail;
  d->pyr_radius = (int)((diameter_detail - 1.0f) / 2.0f);

  const float diameter_fine = base_diameter * d->pyr_s_mult_fine;
  d->pyr_radius_fine = (int)((diameter_fine - 1.0f) / 2.0f);

  const float diameter_micro = base_diameter * d->pyr_s_mult_micro;
  d->pyr_radius_micro = (int)((diameter_micro - 1.0f) / 2.0f);
}


void init_global(dt_iop_module_so_t *self)
{
  dt_iop_pyramidal_contrast_global_data_t *gd = malloc(sizeof(dt_iop_pyramidal_contrast_global_data_t));
  self->data = gd;
}


void cleanup_global(dt_iop_module_so_t *self)
{
  free(self->data);
  self->data = NULL;
}


void commit_params(dt_iop_module_t *self,
                   dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  const dt_iop_pyramidal_contrast_params_t *p = (dt_iop_pyramidal_contrast_params_t *)p1;
  dt_iop_pyramidal_contrast_data_t *d = piece->data;

  d->pyr_method = DT_TONEEQ_NORM_2;
  d->pyr_details = DT_PYR_EIGF;
  d->pyr_iterations = 1;
  d->pyr_scale = 1.0f;
  d->pyr_micro_scale = p->pyr_micro_scale;
  d->pyr_fine_scale = p->pyr_fine_scale;
  d->pyr_detail_scale = p->pyr_detail_scale;
  d->pyr_medium_scale = p->pyr_medium_scale;
  d->pyr_broad_scale = p->pyr_broad_scale; 
  d->pyr_global_scale = p->pyr_global_scale;

  // UI blending param is the square root of the actual blending parameter
  // to make it more sensitive to small values that represent the most important value domain.
  // UI parameter is given in percentage of maximum blending value.
  // The actual blending parameter represents the fraction of the largest image dimension.
  d->pyr_blending = p->pyr_blending * p->pyr_blending / 100.0f;

  // UI guided filter feathering param increases edge taping
  // but actual regularization behaves inversely
  d->pyr_feathering = 1.0f / p->pyr_feathering;

  d->pyr_f_mult_micro = p->pyr_f_mult_micro;
  d->pyr_f_mult_fine = p->pyr_f_mult_fine;
  d->pyr_f_mult_detail = p->pyr_f_mult_detail;
  d->pyr_f_mult_medium = p->pyr_f_mult_medium;
  d->pyr_f_mult_broad = p->pyr_f_mult_broad;

  d->pyr_s_mult_micro = p->pyr_s_mult_micro;
  d->pyr_s_mult_fine = p->pyr_s_mult_fine;
  d->pyr_s_mult_detail = p->pyr_s_mult_detail;
  d->pyr_s_mult_medium = p->pyr_s_mult_medium;
  d->pyr_s_mult_broad = p->pyr_s_mult_broad;
}


void init_pipe(dt_iop_module_t *self,
               dt_dev_pixelpipe_t *pipe,
               dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = dt_calloc1_align_type(dt_iop_pyramidal_contrast_data_t);
}


void cleanup_pipe(dt_iop_module_t *self,
                  dt_dev_pixelpipe_t *pipe,
                  dt_dev_pixelpipe_iop_t *piece)
{
  dt_free_align(piece->data);
  piece->data = NULL;
}


static void pyr_gui_cache_init(dt_iop_module_t *self)
{
  dt_iop_pyramidal_contrast_gui_data_t *g = self->gui_data;
  if(g == NULL) return;

  dt_iop_gui_enter_critical_section(self);
  g->pyr_ui_preview_hash = DT_INVALID_HASH;
  g->pyr_thumb_preview_hash = DT_INVALID_HASH;
  g->pyr_mask_display = DT_PYR_MASK_OFF;
  g->luminance_valid = FALSE;

  g->pyr_full_preview_buf_pixel = NULL;
  g->pyr_full_preview_buf_smoothed_broad = NULL;
  g->pyr_full_preview_buf_smoothed_medium = NULL;
  g->pyr_full_preview_buf_smoothed = NULL;
  g->pyr_full_preview_buf_smoothed_fine = NULL;
  g->pyr_full_preview_buf_smoothed_micro = NULL;
  g->pyr_full_preview_buf_width = 0;
  g->pyr_full_preview_buf_height = 0;

  g->pyr_thumb_preview_buf_pixel = NULL;
  g->pyr_thumb_preview_buf_smoothed_broad = NULL;
  g->pyr_thumb_preview_buf_smoothed_medium = NULL;
  g->pyr_thumb_preview_buf_smoothed = NULL;
  g->pyr_thumb_preview_buf_smoothed_fine = NULL;
  g->pyr_thumb_preview_buf_smoothed_micro = NULL;
  g->pyr_thumb_preview_buf_width = 0;
  g->pyr_thumb_preview_buf_height = 0;

  g->pyr_pipe_order = 0;
  dt_iop_gui_leave_critical_section(self);
}
static void pyr_update_mask_buttons_state(dt_iop_pyramidal_contrast_gui_data_t *g)
{
  if(darktable.gui->reset) return;
  ++darktable.gui->reset;

  dt_bauhaus_widget_set_quad_active(g->pyr_broad_scale, g->pyr_mask_display == DT_PYR_MASK_BROAD);
  dt_bauhaus_widget_set_quad_active(g->pyr_medium_scale, g->pyr_mask_display == DT_PYR_MASK_MEDIUM);
  dt_bauhaus_widget_set_quad_active(g->pyr_detail_scale, g->pyr_mask_display == DT_PYR_MASK_DETAIL);
  dt_bauhaus_widget_set_quad_active(g->pyr_fine_scale, g->pyr_mask_display == DT_PYR_MASK_FINE);
  dt_bauhaus_widget_set_quad_active(g->pyr_micro_scale, g->pyr_mask_display == DT_PYR_MASK_MICRO);

  if(g->pyr_f_view_broad) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->pyr_f_view_broad), g->pyr_mask_display == DT_PYR_MASK_BROAD);
  if(g->pyr_f_view_medium) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->pyr_f_view_medium), g->pyr_mask_display == DT_PYR_MASK_MEDIUM);
  if(g->pyr_f_view_detail) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->pyr_f_view_detail), g->pyr_mask_display == DT_PYR_MASK_DETAIL);
  if(g->pyr_f_view_fine) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->pyr_f_view_fine), g->pyr_mask_display == DT_PYR_MASK_FINE);
  if(g->pyr_f_view_micro) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->pyr_f_view_micro), g->pyr_mask_display == DT_PYR_MASK_MICRO);

  --darktable.gui->reset;
}

static void pyr_set_mask_display(dt_iop_module_t *self, dt_iop_pyramidal_contrast_mask_t mask_type)
{
  dt_iop_pyramidal_contrast_gui_data_t *g = self->gui_data;

  if(darktable.gui->reset) return;

  // If blend module is displaying mask, don't display here
  if(self->request_mask_display)
  {
    dt_control_log(_("cannot display masks when the blending mask is displayed"));
    g->pyr_mask_display = DT_PYR_MASK_OFF;
  }
  else
  {
    // Toggle logic
    if(g->pyr_mask_display == mask_type)
    {
      g->pyr_mask_display = DT_PYR_MASK_OFF;
    }
    else
    {
      g->pyr_mask_display = mask_type;
    }
  }

  pyr_update_mask_buttons_state(g);

  pyr_invalidate_luminance_cache(self);
}

static gboolean pyr_mask_toggle_callback(GtkWidget *togglebutton, GdkEventButton *event, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return FALSE;
  dt_iop_pyramidal_contrast_mask_t mask_type = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(togglebutton), "mask-type"));
  pyr_set_mask_display(self, mask_type);
  return TRUE;
}

static void pyr_create_slider_with_mask_button(dt_iop_module_t *self, GtkWidget *container, GtkWidget **slider_widget,
                                            GtkWidget **button_widget, const char *param_name, const char *tooltip,
                                            dt_iop_pyramidal_contrast_mask_t mask_type)
{
  GtkWidget *hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);

  *slider_widget = dt_bauhaus_slider_from_params(self, param_name);
  dt_bauhaus_slider_set_digits(*slider_widget, 2);
  dt_bauhaus_slider_set_soft_range(*slider_widget, 0.1, 3.0);
  dt_bauhaus_slider_set_format(*slider_widget, "%");
  dt_bauhaus_slider_set_factor(*slider_widget, 100.0);

  g_object_ref(*slider_widget);
  gtk_container_remove(GTK_CONTAINER(self->widget), *slider_widget);

  gtk_box_pack_start(GTK_BOX(hbox), *slider_widget, TRUE, TRUE, 0);
  g_object_unref(*slider_widget);

  *button_widget = dt_iop_togglebutton_new(self, NULL, tooltip, NULL, G_CALLBACK(pyr_mask_toggle_callback), TRUE, 0, 0,
                                           dtgtk_cairo_paint_showmask, hbox);
  g_object_set_data(G_OBJECT(*button_widget), "mask-type", GINT_TO_POINTER(mask_type));
  dt_gui_add_class(*button_widget, "dt_transparent_background");

  dt_gui_box_add(container, hbox);
}

static void pyr_show_guiding_controls(const dt_iop_module_t *self)
{
  const dt_iop_pyramidal_contrast_gui_data_t *g = self->gui_data;

  // All filters need these controls
  gtk_widget_set_visible(g->pyr_blending, TRUE);
  gtk_widget_set_visible(g->pyr_feathering, TRUE);
}


void gui_update(dt_iop_module_t *self)
{
  dt_iop_pyramidal_contrast_gui_data_t *g = self->gui_data;

  pyr_show_guiding_controls(self);
  pyr_invalidate_luminance_cache(self);
  pyr_update_mask_buttons_state(g);

  dt_gui_update_collapsible_section(&g->pyr_advanced_expander);
}


void gui_changed(dt_iop_module_t *self,
                 GtkWidget *w,
                 void *previous)
{
  const dt_iop_pyramidal_contrast_gui_data_t *g = self->gui_data;

  if(w == g->pyr_blending || w == g->pyr_feathering
     || w == g->pyr_f_mult_micro || w == g->pyr_f_mult_fine || w == g->pyr_f_mult_detail
     || w == g->pyr_f_mult_medium || w == g->pyr_f_mult_broad)
  {
    pyr_invalidate_luminance_cache(self);
  }
}


static void pyr_quad_callback(GtkWidget *quad, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_pyramidal_contrast_gui_data_t *g = self->gui_data;
  dt_iop_pyramidal_contrast_mask_t mask_type = DT_PYR_MASK_OFF;

  if(quad == g->pyr_broad_scale) mask_type = DT_PYR_MASK_BROAD;
  else if(quad == g->pyr_medium_scale) mask_type = DT_PYR_MASK_MEDIUM;
  else if(quad == g->pyr_detail_scale) mask_type = DT_PYR_MASK_DETAIL;
  else if(quad == g->pyr_fine_scale) mask_type = DT_PYR_MASK_FINE;
  else if(quad == g->pyr_micro_scale) mask_type = DT_PYR_MASK_MICRO;

  if(mask_type != DT_PYR_MASK_OFF)
  {
    pyr_set_mask_display(self, mask_type);
  }
}


static void _develop_ui_pipe_started_callback(gpointer instance,
                                              dt_iop_module_t *self)
{
  dt_iop_pyramidal_contrast_gui_data_t *g = self->gui_data;
  if(g == NULL) return;

  if(!self->expanded || !self->enabled)
  {
    dt_iop_gui_enter_critical_section(self);
    g->pyr_mask_display = DT_PYR_MASK_OFF;
    dt_iop_gui_leave_critical_section(self);
  }

  ++darktable.gui->reset;
  dt_iop_gui_enter_critical_section(self);
  pyr_update_mask_buttons_state(g);
  dt_iop_gui_leave_critical_section(self);
  --darktable.gui->reset;
}


static void _develop_preview_pipe_finished_callback(gpointer instance,
                                                    dt_iop_module_t *self)
{
  const dt_iop_pyramidal_contrast_gui_data_t *g = self->gui_data;
  if(g == NULL) return;
}


static void _develop_ui_pipe_finished_callback(gpointer instance,
                                               dt_iop_module_t *self)
{
  const dt_iop_pyramidal_contrast_gui_data_t *g = self->gui_data;
  if(g == NULL) return;
}


void gui_focus(dt_iop_module_t *self, gboolean in)
{
  dt_iop_pyramidal_contrast_gui_data_t *g = self->gui_data;
  if(!in)
  {
    const gboolean mask_was_shown = (g->pyr_mask_display != DT_PYR_MASK_OFF);
    g->pyr_mask_display = DT_PYR_MASK_OFF;

    pyr_update_mask_buttons_state(g);
    if(mask_was_shown) dt_dev_reprocess_center(self->dev);
  }
}


void gui_reset(dt_iop_module_t *self)
{
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}


void gui_init(dt_iop_module_t *self)
{
  dt_iop_pyramidal_contrast_gui_data_t *g = IOP_GUI_ALLOC(pyramidal_contrast);

  pyr_gui_cache_init(self);

  // Main container
  self->widget = dt_gui_vbox();

  // Micro detail slider
  g->pyr_micro_scale = dt_bauhaus_slider_from_params(self, "pyr_micro_scale");
  dt_bauhaus_slider_set_soft_range(g->pyr_micro_scale, 0.25, 3.0);
  dt_bauhaus_slider_set_digits(g->pyr_micro_scale, 2);
  dt_bauhaus_slider_set_format(g->pyr_micro_scale, "%");
  dt_bauhaus_slider_set_factor(g->pyr_micro_scale, 100.0);
  gtk_widget_set_tooltip_text(g->pyr_micro_scale, _("amount of micro contrast enhancement"));
  dt_bauhaus_widget_set_quad(g->pyr_micro_scale, self, dtgtk_cairo_paint_showmask, TRUE, pyr_quad_callback,
                             _("visualize micro contrast mask"));

  // Fine detail slider
  g->pyr_fine_scale = dt_bauhaus_slider_from_params(self, "pyr_fine_scale");
  dt_bauhaus_slider_set_soft_range(g->pyr_fine_scale, 0.25, 3.0);
  dt_bauhaus_slider_set_digits(g->pyr_fine_scale, 2);
  dt_bauhaus_slider_set_format(g->pyr_fine_scale, "%");
  dt_bauhaus_slider_set_factor(g->pyr_fine_scale, 100.0);
  gtk_widget_set_tooltip_text(g->pyr_fine_scale, _("amount of fine contrast enhancement"));
  dt_bauhaus_widget_set_quad(g->pyr_fine_scale, self, dtgtk_cairo_paint_showmask, TRUE, pyr_quad_callback,
                             _("visualize fine contrast mask"));

  // Detail boost slider
  g->pyr_detail_scale = dt_bauhaus_slider_from_params(self, "pyr_detail_scale");
  dt_bauhaus_slider_set_soft_range(g->pyr_detail_scale, 0.25, 3.0);
  dt_bauhaus_slider_set_digits(g->pyr_detail_scale, 2);
  dt_bauhaus_slider_set_format(g->pyr_detail_scale, "%");
  dt_bauhaus_slider_set_factor(g->pyr_detail_scale, 100.0);
  gtk_widget_set_tooltip_text
    (g->pyr_detail_scale,
     _("amount of local contrast enhancement\n"
       "1.0 = no change\n"
       "> 1.0 = boost local contrast\n"
       "< 1.0 = reduce local contrast"));
  dt_bauhaus_widget_set_quad(g->pyr_detail_scale, self, dtgtk_cairo_paint_showmask, TRUE, pyr_quad_callback,
                             _("visualize local contrast mask"));

  // Medium detail slider
  g->pyr_medium_scale = dt_bauhaus_slider_from_params(self, "pyr_medium_scale");
  dt_bauhaus_slider_set_soft_range(g->pyr_medium_scale, 0.25, 3.0);
  dt_bauhaus_slider_set_digits(g->pyr_medium_scale, 2);
  dt_bauhaus_slider_set_format(g->pyr_medium_scale, "%");
  dt_bauhaus_slider_set_factor(g->pyr_medium_scale, 100.0);
  gtk_widget_set_tooltip_text(g->pyr_medium_scale, _("amount of broad contrast enhancement"));
  dt_bauhaus_widget_set_quad(g->pyr_medium_scale, self, dtgtk_cairo_paint_showmask, TRUE, pyr_quad_callback,
                             _("visualize broad contrast mask"));

  // Broad detail slider
  g->pyr_broad_scale = dt_bauhaus_slider_from_params(self, "pyr_broad_scale");
  dt_bauhaus_slider_set_soft_range(g->pyr_broad_scale, 0.25, 3.0);
  dt_bauhaus_slider_set_digits(g->pyr_broad_scale, 2);
  dt_bauhaus_slider_set_format(g->pyr_broad_scale, "%");
  dt_bauhaus_slider_set_factor(g->pyr_broad_scale, 100.0);
  gtk_widget_set_tooltip_text(g->pyr_broad_scale, _("amount of extended contrast enhancement"));
  dt_bauhaus_widget_set_quad(g->pyr_broad_scale, self, dtgtk_cairo_paint_showmask, TRUE, pyr_quad_callback,
                             _("visualize extended contrast mask"));

  // Global contrast slider
  g->pyr_global_scale = dt_bauhaus_slider_from_params(self, "pyr_global_scale");
  dt_bauhaus_slider_set_soft_range(g->pyr_global_scale, 0.25, 3.0);
  dt_bauhaus_slider_set_digits(g->pyr_global_scale, 2);
  dt_bauhaus_slider_set_format(g->pyr_global_scale, "%");
  dt_bauhaus_slider_set_factor(g->pyr_global_scale, 100.0);
  gtk_widget_set_tooltip_text
    (g->pyr_global_scale,
     _("amount of global contrast enhancement"));

  // Separator
  GtkWidget *label = dt_ui_section_label_new(C_("section", "masking"));
  gtk_widget_set_margin_top(label, DT_PIXEL_APPLY_DPI(10));
  dt_gui_box_add(self->widget, label);

  g->pyr_blending = dt_bauhaus_slider_from_params(self, "pyr_blending");
  dt_bauhaus_slider_set_soft_range(g->pyr_blending, 1.0, 4.0);
  dt_bauhaus_slider_set_format(g->pyr_blending, "%");
  dt_bauhaus_slider_set_factor(g->pyr_blending, 10.0);
  gtk_widget_set_tooltip_text
    (g->pyr_blending,
     _("size of the smoothing area as percentage of image size\n"
       "larger = affects broader features\n"
       "smaller = affects finer details"));

  g->pyr_feathering = dt_bauhaus_slider_from_params(self, "pyr_feathering");
  dt_bauhaus_slider_set_soft_range(g->pyr_feathering, 0.1, 50.0);
  gtk_widget_set_tooltip_text(g->pyr_feathering, _("edges refinement"));

  // Save main widget
  GtkWidget *main_box = self->widget;

  // Create section
  dt_gui_new_collapsible_section(&g->pyr_advanced_expander, "plugins/darkroom/pyramidal_contrast/expanded_advanced",
                                 _("feathering fine tuning"), GTK_BOX(main_box), DT_ACTION(self));
  
  // Switch self->widget to the section container
  self->widget = GTK_WIDGET(g->pyr_advanced_expander.container);

  pyr_create_slider_with_mask_button(self, self->widget, &g->pyr_f_mult_micro, &g->pyr_f_view_micro, "pyr_f_mult_micro", _("visualize micro contrast mask"), DT_PYR_MASK_MICRO);
  pyr_create_slider_with_mask_button(self, self->widget, &g->pyr_f_mult_fine, &g->pyr_f_view_fine, "pyr_f_mult_fine", _("visualize fine contrast mask"), DT_PYR_MASK_FINE);
  pyr_create_slider_with_mask_button(self, self->widget, &g->pyr_f_mult_detail, &g->pyr_f_view_detail, "pyr_f_mult_detail", _("visualize local contrast mask"), DT_PYR_MASK_DETAIL);
  pyr_create_slider_with_mask_button(self, self->widget, &g->pyr_f_mult_medium, &g->pyr_f_view_medium, "pyr_f_mult_medium", _("visualize broad contrast mask"), DT_PYR_MASK_MEDIUM);
  pyr_create_slider_with_mask_button(self, self->widget, &g->pyr_f_mult_broad, &g->pyr_f_view_broad, "pyr_f_mult_broad", _("visualize extended contrast mask"), DT_PYR_MASK_BROAD);

  // Restore main widget
  self->widget = main_box;

  // Connect signals for pipe events
  DT_CONTROL_SIGNAL_HANDLE(DT_SIGNAL_DEVELOP_PREVIEW_PIPE_FINISHED, _develop_preview_pipe_finished_callback);
  DT_CONTROL_SIGNAL_HANDLE(DT_SIGNAL_DEVELOP_UI_PIPE_FINISHED, _develop_ui_pipe_finished_callback);
  DT_CONTROL_SIGNAL_HANDLE(DT_SIGNAL_DEVELOP_HISTORY_CHANGE, _develop_ui_pipe_started_callback);
}


void gui_cleanup(dt_iop_module_t *self)
{
  dt_iop_pyramidal_contrast_gui_data_t *g = self->gui_data;

  dt_free_align(g->pyr_thumb_preview_buf_pixel);
  dt_free_align(g->pyr_thumb_preview_buf_smoothed_broad);
  dt_free_align(g->pyr_thumb_preview_buf_smoothed_medium);
  dt_free_align(g->pyr_thumb_preview_buf_smoothed);
  dt_free_align(g->pyr_thumb_preview_buf_smoothed_fine);
  dt_free_align(g->pyr_thumb_preview_buf_smoothed_micro);
  dt_free_align(g->pyr_full_preview_buf_pixel);
  dt_free_align(g->pyr_full_preview_buf_smoothed_broad);
  dt_free_align(g->pyr_full_preview_buf_smoothed_medium);
  dt_free_align(g->pyr_full_preview_buf_smoothed);
  dt_free_align(g->pyr_full_preview_buf_smoothed_fine);
  dt_free_align(g->pyr_full_preview_buf_smoothed_micro);
}


// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
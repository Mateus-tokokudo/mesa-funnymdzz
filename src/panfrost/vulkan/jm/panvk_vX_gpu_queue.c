/*
 * Copyright © 2021 Collabora Ltd.
 *
 * Derived from tu_device.c which is:
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "genxml/gen_macros.h"

#include "decode.h"

#include "panvk_cmd_buffer.h"
#include "panvk_device.h"
#include "panvk_entrypoints.h"
#include "panvk_event.h"
#include "panvk_image.h"
#include "panvk_image_view.h"
#include "panvk_instance.h"
#include "panvk_physical_device.h"
#include "panvk_priv_bo.h"
#include "panvk_queue.h"

#include "vk_drm_syncobj.h"
#include "vk_framebuffer.h"

#include "drm-uapi/panfrost_drm.h"

#ifdef HAVE_PAN_KMOD_KBASE
#include <inttypes.h>
#include <unistd.h>
#include "drm-uapi/mali_kbase_ioctl.h"
#include "kmod/kbase_kmod.h"
#endif

static bool
gpu_queue_uses_kbase(const struct panvk_device *dev)
{
   return to_panvk_physical_device(dev->vk.physical)->kbase_node_path[0] != '\0';
}

static void
panvk_queue_submit_batch(struct panvk_gpu_queue *queue,
                         struct panvk_cmd_buffer *cmdbuf,
                         struct panvk_batch *batch, uint32_t *bos,
                         unsigned nr_bos, uint32_t *in_fences,
                         unsigned nr_in_fences)
{
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);
   struct panvk_physical_device *phys_dev =
      to_panvk_physical_device(dev->vk.physical);
   int ret;

   if (batch->issued) {
      util_dynarray_foreach(&batch->jobs, void *, job)
         memset((*job), 0, 4 * 4);

      if (batch->tiler.ctx_descs.cpu) {
         memcpy(batch->tiler.heap_desc.cpu, &batch->tiler.heap_templ,
                sizeof(batch->tiler.heap_templ));

         struct mali_tiler_context_packed *ctxs = batch->tiler.ctx_descs.cpu;

         for (uint32_t i = 0; i < batch->fb.layer_count; i++)
            memcpy(&ctxs[i], &batch->tiler.ctx_templ, sizeof(*ctxs));
      }

      panvk_pool_flush_maps(&cmdbuf->desc_pool);
   }

   pan_kmod_flush_bo_map_syncs(dev->kmod.dev);

   if (batch->vtc_jc.first_job) {
      struct drm_panfrost_submit submit = {
         .bo_handles = (uintptr_t)bos,
         .bo_handle_count = nr_bos,
         .in_syncs = (uintptr_t)in_fences,
         .in_sync_count = nr_in_fences,
         .out_sync = queue->sync,
         .jc = batch->vtc_jc.first_job,
      };

      ret = pan_kmod_ioctl(dev->drm_fd, DRM_IOCTL_PANFROST_SUBMIT, &submit);
      if (ret < 0) {
         mesa_logw("panvk: DRM submission notification for panfrost vtc in the kbase backend: %s",
                   strerror(errno));
      }

      if ((PANVK_DEBUG(TRACE) || PANVK_DEBUG(SYNC)) && ret == 0) {
         int wait_ret = drmSyncobjWait(dev->drm_fd, &submit.out_sync, 1, INT64_MAX, 0, NULL);
         if (wait_ret == 0) {
            panvk_pool_invalidate_maps(&cmdbuf->desc_pool);
            pan_kmod_flush_bo_map_syncs(dev->kmod.dev);
         }
      }

      if (PANVK_DEBUG(TRACE)) {
         pandecode_jc(dev->debug.decode_ctx, batch->vtc_jc.first_job,
                      phys_dev->kmod.dev->props.gpu_id);
      }

      if (PANVK_DEBUG(DUMP))
         pandecode_dump_mappings(dev->debug.decode_ctx);

      if (PANVK_DEBUG(SYNC))
         pandecode_abort_on_fault(dev->debug.decode_ctx, submit.jc,
                                  phys_dev->kmod.dev->props.gpu_id);
   }

   /* Processamento dos Jobs de Fragmentos (frag_jc) */
   if (batch->frag_jc.first_job) {
      struct drm_panfrost_submit submit = {
         .bo_handles = (uintptr_t)bos,
         .bo_handle_count = nr_bos,
         .out_sync = queue->sync,
         .jc = batch->frag_jc.first_job,
         .requirements = PANFROST_JD_REQ_FS,
      };

      if (batch->vtc_jc.first_job) {
         submit.in_syncs = (uintptr_t)(&queue->sync);
         submit.in_sync_count = 1;
      } else {
         submit.in_syncs = (uintptr_t)in_fences;
         submit.in_sync_count = nr_in_fences;
      }

      ret = pan_kmod_ioctl(dev->drm_fd, DRM_IOCTL_PANFROST_SUBMIT, &submit);
      if (ret < 0) {
         mesa_logw("panvk: DRM submission warning for panfrost frag in the kbase backend %s",
                   strerror(errno));
      }

      if ((PANVK_DEBUG(TRACE) || PANVK_DEBUG(SYNC)) && ret == 0) {
         int wait_ret = drmSyncobjWait(dev->drm_fd, &submit.out_sync, 1, INT64_MAX, 0, NULL);
         if (wait_ret == 0) {
            panvk_pool_invalidate_maps(&cmdbuf->desc_pool);
            pan_kmod_flush_bo_map_syncs(dev->kmod.dev);
         }
      }

      if (PANVK_DEBUG(TRACE))
         pandecode_jc(dev->debug.decode_ctx, batch->frag_jc.first_job,
                      phys_dev->kmod.dev->props.gpu_id);

      if (PANVK_DEBUG(DUMP))
         pandecode_dump_mappings(dev->debug.decode_ctx);

      if (PANVK_DEBUG(SYNC))
         pandecode_abort_on_fault(dev->debug.decode_ctx, submit.jc,
                                  phys_dev->kmod.dev->props.gpu_id);
   }

   if (PANVK_DEBUG(TRACE))
      pandecode_next_frame(dev->debug.decode_ctx);

   batch->issued = true;
}

static void
panvk_queue_transfer_sync(struct panvk_gpu_queue *queue, uint32_t syncobj)
{
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);
   int ret;

   struct drm_syncobj_handle handle = {
      .handle = queue->sync,
      .flags = DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_EXPORT_SYNC_FILE,
      .fd = -1,
   };

   if (dev->drm_fd < 0 || !syncobj || gpu_queue_uses_kbase(dev))
      return;

   ret = pan_kmod_ioctl(dev->drm_fd, DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD, &handle);
   if (ret < 0 || handle.fd < 0)
      return;

   handle.handle = syncobj;
   ret = pan_kmod_ioctl(dev->drm_fd, DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE, &handle);
   (void)ret;

   if (handle.fd >= 0)
      close(handle.fd);
}

static void
panvk_add_wait_event_syncobjs(struct panvk_batch *batch, uint32_t *in_fences,
                              unsigned *nr_in_fences)
{
   util_dynarray_foreach(&batch->event_ops, struct panvk_cmd_event_op, op) {
      switch (op->type) {
      case PANVK_EVENT_OP_SET:
         break;
      case PANVK_EVENT_OP_RESET:
         break;
      case PANVK_EVENT_OP_WAIT:
         in_fences[(*nr_in_fences)++] = op->event->syncobj;
         break;
      default:
         UNREACHABLE("type of panvk_cmd_event_op invalid\n");
      }
   }
}

static void
panvk_signal_event_syncobjs(struct panvk_gpu_queue *queue,
                            struct panvk_batch *batch)
{
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);

   util_dynarray_foreach(&batch->event_ops, struct panvk_cmd_event_op, op) {
      switch (op->type) {
      case PANVK_EVENT_OP_SET: {
         panvk_queue_transfer_sync(queue, op->event->syncobj);
         break;
      }
      case PANVK_EVENT_OP_RESET: {
         struct panvk_event *event = op->event;

         struct drm_syncobj_array objs = {
            .handles = (uint64_t)(uintptr_t)&event->syncobj,
            .count_handles = 1};

         if (!gpu_queue_uses_kbase(dev)) {
            int ret = pan_kmod_ioctl(dev->drm_fd, DRM_IOCTL_SYNCOBJ_RESET, &objs);
            (void)ret;
         }
         break;
      }
      case PANVK_EVENT_OP_WAIT:
         break;
      default:
         UNREACHABLE("type of panvk_cmd_event_op invalid\n");
      }
   }
}

VkResult
panvk_per_arch(gpu_queue_submit)(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct panvk_gpu_queue *queue = container_of(vk_queue, struct panvk_gpu_queue, vk);
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);

   unsigned nr_semaphores = submit->wait_count + 1;
   uint32_t semaphores[nr_semaphores];

   semaphores[0] = queue->sync;

   for (unsigned i = 0; i < submit->wait_count; i++) {
      if (submit->waits[i].sync && vk_sync_type_is_drm_syncobj(submit->waits[i].sync->type)) {
         struct vk_drm_syncobj *syncobj = vk_sync_as_drm_syncobj(submit->waits[i].sync);
         if (syncobj)
            semaphores[i + 1] = syncobj->syncobj;
      } else {
         semaphores[i + 1] = 0;
      }
   }

   for (uint32_t j = 0; j < submit->command_buffer_count; ++j) {
      struct panvk_cmd_buffer *cmdbuf =
         container_of(submit->command_buffers[j], struct panvk_cmd_buffer, vk);

      list_for_each_entry(struct panvk_batch, batch, &cmdbuf->batches, node) {
         unsigned nr_bos = panvk_pool_num_bos(&cmdbuf->desc_pool) +
                           panvk_pool_num_bos(&cmdbuf->varying_pool) +
                           panvk_pool_num_bos(&cmdbuf->tls_pool) +
                           batch->fb.bo_count + (batch->blit.src ? 1 : 0) +
                           (batch->blit.dst ? 1 : 0) +
                           (batch->vtc_jc.first_tiler ? 1 : 0) + 1;
         unsigned bo_idx = 0;
         uint32_t bos[nr_bos];

         panvk_pool_get_bo_handles(&cmdbuf->desc_pool, &bos[bo_idx]);
         bo_idx += panvk_pool_num_bos(&cmdbuf->desc_pool);

         panvk_pool_get_bo_handles(&cmdbuf->varying_pool, &bos[bo_idx]);
         bo_idx += panvk_pool_num_bos(&cmdbuf->varying_pool);

         panvk_pool_get_bo_handles(&cmdbuf->tls_pool, &bos[bo_idx]);
         bo_idx += panvk_pool_num_bos(&cmdbuf->tls_pool);

         for (unsigned i = 0; i < batch->fb.bo_count; i++)
            bos[bo_idx++] = pan_kmod_bo_handle(batch->fb.bos[i]);

         if (batch->blit.src)
            bos[bo_idx++] = pan_kmod_bo_handle(batch->blit.src);

         if (batch->blit.dst)
            bos[bo_idx++] = pan_kmod_bo_handle(batch->blit.dst);

         if (batch->vtc_jc.first_tiler)
            bos[bo_idx++] = pan_kmod_bo_handle(dev->tiler_heap->bo);

         bos[bo_idx++] = pan_kmod_bo_handle(dev->sample_positions->bo);
         assert(bo_idx == nr_bos);

         for (unsigned x = 0; x < nr_bos; x++) {
            for (unsigned y = x + 1; y < nr_bos;) {
               if (bos[x] == bos[y])
                  bos[y] = bos[--nr_bos];
               else
                  y++;
            }
         }

         unsigned nr_in_fences = 0;
         unsigned max_wait_event_syncobjs = util_dynarray_num_elements(
            &batch->event_ops, struct panvk_cmd_event_op);
         uint32_t in_fences[nr_semaphores + max_wait_event_syncobjs];
         memcpy(in_fences, semaphores, nr_semaphores * sizeof(*in_fences));
         nr_in_fences += nr_semaphores;

         panvk_add_wait_event_syncobjs(batch, in_fences, &nr_in_fences);

         panvk_queue_submit_batch(queue, cmdbuf, batch, bos, nr_bos, in_fences,
                                  nr_in_fences);

         panvk_signal_event_syncobjs(queue, batch);
      }
   }

   for (unsigned i = 0; i < submit->signal_count; i++) {
      if (submit->signals[i].sync && vk_sync_type_is_drm_syncobj(submit->signals[i].sync->type)) {
         struct vk_drm_syncobj *syncobj = vk_sync_as_drm_syncobj(submit->signals[i].sync);
         if (syncobj)
            panvk_queue_transfer_sync(queue, syncobj->syncobj);
      }
   }

   return VK_SUCCESS;
}

VkResult
panvk_per_arch(create_gpu_queue)(struct panvk_device *device,
                                 const VkDeviceQueueCreateInfo *create_info,
                                 uint32_t queue_idx,
                                 struct vk_queue **out_queue)
{
   ASSERTED const VkDeviceQueueGlobalPriorityCreateInfoKHR *priority_info =
      vk_find_struct_const(create_info->pNext,
                           DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR);
   ASSERTED const VkQueueGlobalPriorityKHR priority =
      priority_info ? priority_info->globalPriority
                    : VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR;

   assert(priority == VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR);

   struct panvk_gpu_queue *queue = vk_zalloc(&device->vk.alloc, sizeof(*queue), 8,
                                         VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!queue)
      return panvk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result =
      vk_queue_init(&queue->vk, &device->vk, create_info, queue_idx);
   if (result != VK_SUCCESS)
      goto err_free_queue;

   if (!gpu_queue_uses_kbase(device)) {
      int ret = drmSyncobjCreate(device->drm_fd, DRM_SYNCOBJ_CREATE_SIGNALED,
                                 &queue->sync);
      if (ret < 0) {
         mesa_logw("panvk: drmSyncobjCreate not supported");
         queue->sync = 0;
      }
   } else {
      queue->sync = 0;
   }

   queue->vk.driver_submit = panvk_per_arch(gpu_queue_submit);
   *out_queue = &queue->vk;
   return VK_SUCCESS;

err_free_queue:
   vk_free(&device->vk.alloc, queue);
   return result;
}

void panvk_per_arch(destroy_gpu_queue)(struct vk_queue *vk_queue)
{
   struct panvk_gpu_queue *queue = container_of(vk_queue, struct panvk_gpu_queue, vk);
   struct panvk_device *dev = to_panvk_device(vk_queue->base.device);

   vk_queue_finish(&queue->vk);
   if (queue->sync != 0 && !gpu_queue_uses_kbase(dev))
      drmSyncobjDestroy(dev->drm_fd, queue->sync);
   vk_free(&dev->vk.alloc, queue);
}

VkResult
panvk_per_arch(gpu_queue_check_status)(struct vk_queue *vk_queue)
{
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(QueueWaitIdle)(VkQueue _queue)
{
   VK_FROM_HANDLE(panvk_gpu_queue, queue, _queue);
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);

   assert(queue->vk.submit.mode != VK_QUEUE_SUBMIT_MODE_THREADED);

   if (vk_device_is_lost(&dev->vk)) {
      u_printf_with_ctx(stdout, &dev->printf.ctx);
      return VK_ERROR_DEVICE_LOST;
   }

   if (queue->sync != 0 && !gpu_queue_uses_kbase(dev)) {
      int ret = drmSyncobjWait(dev->drm_fd, &queue->sync, 1,
                               INT64_MAX, DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL,
                               NULL);
      (void)ret;
   }

   return VK_SUCCESS;
}

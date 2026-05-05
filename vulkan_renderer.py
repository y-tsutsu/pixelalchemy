import math
import os
import subprocess
from pathlib import Path

import glfw
import numpy as np
import vulkan as vk


UINT64_MAX = 0xffffffffffffffff


class VulkanRenderer:
    def __init__(self, pipeline, width=800, height=600):
        self.pipeline = pipeline
        self.width = width
        self.height = height
        self.clear_test = os.environ.get('PIXELALCHEMY_CLEAR_TEST') == '1'
        self.window = None

        self.instance = None
        self.surface = None
        self.physical_device = None
        self.device = None
        self.queue_family_index = None
        self.queue = None

        self.swapchain = None
        self.swapchain_format = None
        self.swapchain_extent = None
        self.swapchain_images = []
        self.swapchain_views = []
        self.swapchain_layouts = []

        self.output_image = None
        self.output_image_memory = None
        self.output_image_view = None
        self.output_image_layout = vk.VK_IMAGE_LAYOUT_UNDEFINED

        self.src_image = None
        self.src_image_memory = None
        self.src_image_view = None
        self.src_image_size = None
        self.src_image_layout = vk.VK_IMAGE_LAYOUT_UNDEFINED

        self.staging_buffer = None
        self.staging_memory = None
        self.staging_size = 0

        self.descriptor_set_layout = None
        self.descriptor_pool = None
        self.descriptor_sets = []
        self.pipeline_layout = None
        self.compute_pipeline = None
        self.shader_module = None

        self.command_pool = None
        self.command_buffer = None
        self.image_available = None
        self.compute_finished = None
        self.in_flight = None

        self.vkGetPhysicalDeviceSurfaceSupportKHR = None
        self.vkGetPhysicalDeviceSurfaceCapabilitiesKHR = None
        self.vkGetPhysicalDeviceSurfaceFormatsKHR = None
        self.vkGetPhysicalDeviceSurfacePresentModesKHR = None
        self.vkCreateSwapchainKHR = None
        self.vkDestroySwapchainKHR = None
        self.vkGetSwapchainImagesKHR = None
        self.vkAcquireNextImageKHR = None
        self.vkQueuePresentKHR = None

    def run(self):
        self._init_window()
        self._init_vulkan()

        try:
            while not glfw.window_should_close(self.window):
                glfw.poll_events()
                frame = self.pipeline.get_frame()
                if frame is not None:
                    self.draw(frame)
        finally:
            self.close()

    def draw(self, frame):
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        h, w, channels = frame.shape
        if channels != 4:
            raise RuntimeError(f'Expected RGBA frame, got {channels} channels')

        self._ensure_source_resources(w, h)
        self._upload_frame(frame)

        vk.vkWaitForFences(self.device, 1, [self.in_flight], vk.VK_TRUE, UINT64_MAX)
        vk.vkResetFences(self.device, 1, [self.in_flight])

        image_index = self.vkAcquireNextImageKHR(
            self.device,
            self.swapchain,
            UINT64_MAX,
            self.image_available,
            vk.VK_NULL_HANDLE,
        )

        self._record_command_buffer(image_index, w, h)

        submit_info = vk.VkSubmitInfo(
            pWaitSemaphores=[self.image_available],
            pWaitDstStageMask=[vk.VK_PIPELINE_STAGE_TRANSFER_BIT],
            pCommandBuffers=[self.command_buffer],
            pSignalSemaphores=[self.compute_finished],
        )
        vk.vkQueueSubmit(self.queue, 1, [submit_info], self.in_flight)

        present_info = vk.VkPresentInfoKHR(
            pWaitSemaphores=[self.compute_finished],
            pSwapchains=[self.swapchain],
            pImageIndices=[image_index],
        )
        self.vkQueuePresentKHR(self.queue, present_info)

    def close(self):
        if self.device:
            vk.vkDeviceWaitIdle(self.device)

        self._destroy_source_resources()
        self._destroy_output_resources()
        self._destroy_staging_resources()

        if self.compute_pipeline:
            vk.vkDestroyPipeline(self.device, self.compute_pipeline, None)
        if self.pipeline_layout:
            vk.vkDestroyPipelineLayout(self.device, self.pipeline_layout, None)
        if self.shader_module:
            vk.vkDestroyShaderModule(self.device, self.shader_module, None)
        if self.descriptor_pool:
            vk.vkDestroyDescriptorPool(self.device, self.descriptor_pool, None)
        if self.descriptor_set_layout:
            vk.vkDestroyDescriptorSetLayout(self.device, self.descriptor_set_layout, None)
        if self.image_available:
            vk.vkDestroySemaphore(self.device, self.image_available, None)
        if self.compute_finished:
            vk.vkDestroySemaphore(self.device, self.compute_finished, None)
        if self.in_flight:
            vk.vkDestroyFence(self.device, self.in_flight, None)
        if self.command_pool:
            vk.vkDestroyCommandPool(self.device, self.command_pool, None)
        for view in self.swapchain_views:
            vk.vkDestroyImageView(self.device, view, None)
        if self.swapchain:
            self.vkDestroySwapchainKHR(self.device, self.swapchain, None)
        if self.device:
            vk.vkDestroyDevice(self.device, None)
        if self.surface:
            destroy_surface = vk.vkGetInstanceProcAddr(self.instance, 'vkDestroySurfaceKHR')
            destroy_surface(self.instance, self.surface, None)
        if self.instance:
            vk.vkDestroyInstance(self.instance, None)
        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()

    def _init_window(self):
        if not glfw.init():
            raise RuntimeError('Failed to initialize GLFW')
        if not glfw.vulkan_supported():
            raise RuntimeError('GLFW reports Vulkan is not supported')

        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        self.window = glfw.create_window(
            self.width,
            self.height,
            'GStreamer + Vulkan Compute Edge detection',
            None,
            None,
        )
        if not self.window:
            raise RuntimeError('Failed to create GLFW window')

    def _init_vulkan(self):
        self._create_instance()
        self._create_surface()
        self._load_instance_extensions()
        self._pick_physical_device()
        self._create_device()
        self._load_device_extensions()
        self._create_swapchain()
        self._create_descriptor_set_layout()
        self._create_compute_pipeline()
        self._create_command_pool()
        self._create_sync_objects()

    def _create_instance(self):
        extensions = glfw.get_required_instance_extensions()
        app_info = vk.VkApplicationInfo(
            pApplicationName='pixelalchemy',
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName='pixelalchemy',
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_0,
        )
        create_info = vk.VkInstanceCreateInfo(
            pApplicationInfo=app_info,
            ppEnabledExtensionNames=extensions,
        )
        self.instance = vk.vkCreateInstance(create_info, None)

    def _create_surface(self):
        surface_ptr = vk.ffi.new('VkSurfaceKHR*')
        result = glfw.create_window_surface(self.instance, self.window, None, surface_ptr)
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f'glfwCreateWindowSurface failed: {result}')
        self.surface = surface_ptr[0]

    def _load_instance_extensions(self):
        self.vkGetPhysicalDeviceSurfaceSupportKHR = vk.vkGetInstanceProcAddr(
            self.instance, 'vkGetPhysicalDeviceSurfaceSupportKHR')
        self.vkGetPhysicalDeviceSurfaceCapabilitiesKHR = vk.vkGetInstanceProcAddr(
            self.instance, 'vkGetPhysicalDeviceSurfaceCapabilitiesKHR')
        self.vkGetPhysicalDeviceSurfaceFormatsKHR = vk.vkGetInstanceProcAddr(
            self.instance, 'vkGetPhysicalDeviceSurfaceFormatsKHR')
        self.vkGetPhysicalDeviceSurfacePresentModesKHR = vk.vkGetInstanceProcAddr(
            self.instance, 'vkGetPhysicalDeviceSurfacePresentModesKHR')

    def _pick_physical_device(self):
        for physical_device in vk.vkEnumeratePhysicalDevices(self.instance):
            queue_index = self._find_queue_family(physical_device)
            if queue_index is None:
                continue
            if not self._supports_device_extension(physical_device, vk.VK_KHR_SWAPCHAIN_EXTENSION_NAME):
                continue
            self.physical_device = physical_device
            self.queue_family_index = queue_index
            return
        raise RuntimeError('No Vulkan device with compute + present + swapchain support found')

    def _find_queue_family(self, physical_device):
        families = vk.vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
        for index, props in enumerate(families):
            has_compute = props.queueFlags & vk.VK_QUEUE_COMPUTE_BIT
            present = self.vkGetPhysicalDeviceSurfaceSupportKHR(
                physical_device, index, self.surface)
            if has_compute and present:
                return index
        return None

    def _supports_device_extension(self, physical_device, extension_name):
        extensions = list(vk.vkEnumerateDeviceExtensionProperties(physical_device, None))
        return any(ext.extensionName == extension_name for ext in extensions)

    def _create_device(self):
        priority = [1.0]
        queue_info = vk.VkDeviceQueueCreateInfo(
            queueFamilyIndex=self.queue_family_index,
            pQueuePriorities=priority,
        )
        device_info = vk.VkDeviceCreateInfo(
            pQueueCreateInfos=[queue_info],
            ppEnabledExtensionNames=[vk.VK_KHR_SWAPCHAIN_EXTENSION_NAME],
        )
        self.device = vk.vkCreateDevice(self.physical_device, device_info, None)
        self.queue = vk.vkGetDeviceQueue(self.device, self.queue_family_index, 0)

    def _load_device_extensions(self):
        self.vkCreateSwapchainKHR = vk.vkGetDeviceProcAddr(self.device, 'vkCreateSwapchainKHR')
        self.vkDestroySwapchainKHR = vk.vkGetDeviceProcAddr(self.device, 'vkDestroySwapchainKHR')
        self.vkGetSwapchainImagesKHR = vk.vkGetDeviceProcAddr(self.device, 'vkGetSwapchainImagesKHR')
        self.vkAcquireNextImageKHR = vk.vkGetDeviceProcAddr(self.device, 'vkAcquireNextImageKHR')
        self.vkQueuePresentKHR = vk.vkGetDeviceProcAddr(self.device, 'vkQueuePresentKHR')

    def _create_swapchain(self):
        caps = self.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(self.physical_device, self.surface)
        formats = self.vkGetPhysicalDeviceSurfaceFormatsKHR(self.physical_device, self.surface)
        present_modes = self.vkGetPhysicalDeviceSurfacePresentModesKHR(
            self.physical_device, self.surface)

        image_format, color_space = self._choose_surface_format(formats)
        present_mode = self._choose_present_mode(present_modes)
        extent = self._choose_extent(caps)
        image_count = caps.minImageCount + 1
        if caps.maxImageCount and image_count > caps.maxImageCount:
            image_count = caps.maxImageCount

        required_usage = vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
        if not caps.supportedUsageFlags & required_usage:
            raise RuntimeError('Swapchain images do not support VK_IMAGE_USAGE_TRANSFER_DST_BIT')

        swapchain_info = vk.VkSwapchainCreateInfoKHR(
            surface=self.surface,
            minImageCount=image_count,
            imageFormat=image_format,
            imageColorSpace=color_space,
            imageExtent=extent,
            imageArrayLayers=1,
            imageUsage=vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            imageSharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            preTransform=caps.currentTransform,
            compositeAlpha=vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            presentMode=present_mode,
            clipped=vk.VK_TRUE,
            oldSwapchain=vk.VK_NULL_HANDLE,
        )
        self.swapchain = self.vkCreateSwapchainKHR(self.device, swapchain_info, None)
        self.swapchain_format = image_format
        self.swapchain_extent = extent
        self.swapchain_images = list(self.vkGetSwapchainImagesKHR(self.device, self.swapchain))
        self.swapchain_views = [
            self._create_image_view(image, self.swapchain_format)
            for image in self.swapchain_images
        ]
        self.swapchain_layouts = [vk.VK_IMAGE_LAYOUT_UNDEFINED for _ in self.swapchain_images]
        self._create_output_resources()

    def _choose_surface_format(self, formats):
        for fmt in formats:
            if (fmt.format == vk.VK_FORMAT_R8G8B8A8_UNORM and
                    fmt.colorSpace == vk.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR):
                return int(fmt.format), int(fmt.colorSpace)
        for fmt in formats:
            if fmt.format == vk.VK_FORMAT_B8G8R8A8_UNORM:
                return int(fmt.format), int(fmt.colorSpace)
        return int(formats[0].format), int(formats[0].colorSpace)

    def _choose_present_mode(self, present_modes):
        modes = [int(mode) for mode in present_modes]
        if vk.VK_PRESENT_MODE_MAILBOX_KHR in modes:
            return vk.VK_PRESENT_MODE_MAILBOX_KHR
        return vk.VK_PRESENT_MODE_FIFO_KHR

    def _choose_extent(self, caps):
        if caps.currentExtent.width not in (-1, 0xffffffff):
            return vk.VkExtent2D(
                width=caps.currentExtent.width,
                height=caps.currentExtent.height,
            )
        width, height = glfw.get_framebuffer_size(self.window)
        width = min(max(width, caps.minImageExtent.width), caps.maxImageExtent.width)
        height = min(max(height, caps.minImageExtent.height), caps.maxImageExtent.height)
        return vk.VkExtent2D(width=width, height=height)

    def _create_descriptor_set_layout(self):
        bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=0,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            vk.VkDescriptorSetLayoutBinding(
                binding=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
        ]
        layout_info = vk.VkDescriptorSetLayoutCreateInfo(pBindings=bindings)
        self.descriptor_set_layout = vk.vkCreateDescriptorSetLayout(
            self.device, layout_info, None)

    def _create_compute_pipeline(self):
        shader_path = Path('shaders/vulkan_edge.comp')
        spv = self._compile_shader(shader_path)
        shader_info = vk.VkShaderModuleCreateInfo(codeSize=len(spv), pCode=spv)
        self.shader_module = vk.vkCreateShaderModule(self.device, shader_info, None)

        layout_info = vk.VkPipelineLayoutCreateInfo(
            pSetLayouts=[self.descriptor_set_layout])
        self.pipeline_layout = vk.vkCreatePipelineLayout(self.device, layout_info, None)

        stage_info = vk.VkPipelineShaderStageCreateInfo(
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=self.shader_module,
            pName='main',
        )
        pipeline_info = vk.VkComputePipelineCreateInfo(
            stage=stage_info,
            layout=self.pipeline_layout,
        )
        pipelines = vk.vkCreateComputePipelines(
            self.device, vk.VK_NULL_HANDLE, 1, [pipeline_info], None)
        self.compute_pipeline = pipelines[0]

    def _compile_shader(self, shader_path):
        spv_path = Path('/tmp') / f'{shader_path.name}.spv'
        subprocess.run(
            ['glslangValidator', '-V', str(shader_path), '-o', str(spv_path)],
            check=True,
        )
        return spv_path.read_bytes()

    def _create_command_pool(self):
        pool_info = vk.VkCommandPoolCreateInfo(
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=self.queue_family_index,
        )
        self.command_pool = vk.vkCreateCommandPool(self.device, pool_info, None)
        alloc_info = vk.VkCommandBufferAllocateInfo(
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        self.command_buffer = vk.vkAllocateCommandBuffers(self.device, alloc_info)[0]

    def _create_sync_objects(self):
        semaphore_info = vk.VkSemaphoreCreateInfo()
        self.image_available = vk.vkCreateSemaphore(self.device, semaphore_info, None)
        self.compute_finished = vk.vkCreateSemaphore(self.device, semaphore_info, None)
        fence_info = vk.VkFenceCreateInfo(flags=vk.VK_FENCE_CREATE_SIGNALED_BIT)
        self.in_flight = vk.vkCreateFence(self.device, fence_info, None)

    def _ensure_source_resources(self, width, height):
        byte_size = width * height * 4
        if self.src_image_size != (width, height):
            vk.vkDeviceWaitIdle(self.device)
            self._destroy_source_resources()
            self.src_image_size = (width, height)
            self.src_image, self.src_image_memory = self._create_image(
                width,
                height,
                vk.VK_FORMAT_R8G8B8A8_UNORM,
                vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT | vk.VK_IMAGE_USAGE_STORAGE_BIT,
                vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            )
            self.src_image_view = self._create_image_view(
                self.src_image, vk.VK_FORMAT_R8G8B8A8_UNORM)
            self.src_image_layout = vk.VK_IMAGE_LAYOUT_UNDEFINED
            self._recreate_descriptor_sets()
        if byte_size > self.staging_size:
            self._destroy_staging_resources()
            self.staging_buffer, self.staging_memory = self._create_buffer(
                byte_size,
                vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            )
            self.staging_size = byte_size

    def _upload_frame(self, frame):
        data = frame.tobytes()
        mapped = vk.vkMapMemory(self.device, self.staging_memory, 0, len(data), 0)
        mapped[:] = data
        vk.vkUnmapMemory(self.device, self.staging_memory)

    def _recreate_descriptor_sets(self):
        if self.descriptor_pool:
            vk.vkDestroyDescriptorPool(self.device, self.descriptor_pool, None)

        pool_size = vk.VkDescriptorPoolSize(
            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            descriptorCount=len(self.swapchain_images) * 2,
        )
        pool_info = vk.VkDescriptorPoolCreateInfo(
            maxSets=len(self.swapchain_images),
            pPoolSizes=[pool_size],
        )
        self.descriptor_pool = vk.vkCreateDescriptorPool(self.device, pool_info, None)
        alloc_info = vk.VkDescriptorSetAllocateInfo(
            descriptorPool=self.descriptor_pool,
            pSetLayouts=[self.descriptor_set_layout] * len(self.swapchain_images),
        )
        self.descriptor_sets = list(vk.vkAllocateDescriptorSets(self.device, alloc_info))

        for descriptor_set in self.descriptor_sets:
            src_info = vk.VkDescriptorImageInfo(
                imageView=self.src_image_view,
                imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            )
            dst_info = vk.VkDescriptorImageInfo(
                imageView=self.output_image_view,
                imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            )
            writes = [
                vk.VkWriteDescriptorSet(
                    dstSet=descriptor_set,
                    dstBinding=0,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    pImageInfo=[src_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=descriptor_set,
                    dstBinding=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    pImageInfo=[dst_info],
                ),
            ]
            vk.vkUpdateDescriptorSets(self.device, len(writes), writes, 0, None)

    def _record_command_buffer(self, image_index, src_width, src_height):
        vk.vkResetCommandBuffer(self.command_buffer, 0)
        begin_info = vk.VkCommandBufferBeginInfo(
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
        vk.vkBeginCommandBuffer(self.command_buffer, begin_info)

        self._image_barrier(
            self.src_image,
            self.src_image_layout,
            vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0,
            vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
        )
        copy_region = vk.VkBufferImageCopy(
            bufferOffset=0,
            bufferRowLength=0,
            bufferImageHeight=0,
            imageSubresource=vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0,
                baseArrayLayer=0,
                layerCount=1,
            ),
            imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
            imageExtent=vk.VkExtent3D(width=src_width, height=src_height, depth=1),
        )
        vk.vkCmdCopyBufferToImage(
            self.command_buffer,
            self.staging_buffer,
            self.src_image,
            vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            [copy_region],
        )
        self._image_barrier(
            self.src_image,
            vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            vk.VK_IMAGE_LAYOUT_GENERAL,
            vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            vk.VK_ACCESS_SHADER_READ_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        )
        self.src_image_layout = vk.VK_IMAGE_LAYOUT_GENERAL

        output_src_access = 0
        output_src_stage = vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
        if self.output_image_layout == vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            output_src_access = vk.VK_ACCESS_TRANSFER_READ_BIT
            output_src_stage = vk.VK_PIPELINE_STAGE_TRANSFER_BIT

        self._image_barrier(
            self.output_image,
            self.output_image_layout,
            vk.VK_IMAGE_LAYOUT_GENERAL,
            output_src_access,
            vk.VK_ACCESS_SHADER_WRITE_BIT,
            output_src_stage,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        )
        self.output_image_layout = vk.VK_IMAGE_LAYOUT_GENERAL

        vk.vkCmdBindPipeline(
            self.command_buffer,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.compute_pipeline,
        )
        vk.vkCmdBindDescriptorSets(
            self.command_buffer,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.pipeline_layout,
            0,
            1,
            [self.descriptor_sets[image_index]],
            0,
            None,
        )
        vk.vkCmdDispatch(
            self.command_buffer,
            math.ceil(self.swapchain_extent.width / 16),
            math.ceil(self.swapchain_extent.height / 16),
            1,
        )

        self._image_barrier(
            self.output_image,
            vk.VK_IMAGE_LAYOUT_GENERAL,
            vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            vk.VK_ACCESS_SHADER_WRITE_BIT,
            vk.VK_ACCESS_TRANSFER_READ_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
        )
        self.output_image_layout = vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL

        self._image_barrier(
            self.swapchain_images[image_index],
            self.swapchain_layouts[image_index],
            vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0,
            vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
        )

        if self.clear_test:
            color = vk.VkClearColorValue(float32=[1.0, 0.0, 0.0, 1.0])
            vk.vkCmdClearColorImage(
                self.command_buffer,
                self.swapchain_images[image_index],
                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                color,
                1,
                [self._color_subresource_range()],
            )
        else:
            blit = vk.VkImageBlit(
                srcSubresource=vk.VkImageSubresourceLayers(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    mipLevel=0,
                    baseArrayLayer=0,
                    layerCount=1,
                ),
                srcOffsets=[
                    vk.VkOffset3D(x=0, y=0, z=0),
                    vk.VkOffset3D(
                        x=self.swapchain_extent.width,
                        y=self.swapchain_extent.height,
                        z=1,
                    ),
                ],
                dstSubresource=vk.VkImageSubresourceLayers(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    mipLevel=0,
                    baseArrayLayer=0,
                    layerCount=1,
                ),
                dstOffsets=[
                    vk.VkOffset3D(x=0, y=0, z=0),
                    vk.VkOffset3D(
                        x=self.swapchain_extent.width,
                        y=self.swapchain_extent.height,
                        z=1,
                    ),
                ],
            )
            vk.vkCmdBlitImage(
                self.command_buffer,
                self.output_image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                self.swapchain_images[image_index],
                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                [blit],
                vk.VK_FILTER_NEAREST,
            )

        self._image_barrier(
            self.swapchain_images[image_index],
            vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            0,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        )
        self.swapchain_layouts[image_index] = vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR

        vk.vkEndCommandBuffer(self.command_buffer)

    def _image_barrier(
            self,
            image,
            old_layout,
            new_layout,
            src_access,
            dst_access,
            src_stage,
            dst_stage):
        barrier = vk.VkImageMemoryBarrier(
            oldLayout=old_layout,
            newLayout=new_layout,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            image=image,
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0,
                levelCount=1,
                baseArrayLayer=0,
                layerCount=1,
            ),
            srcAccessMask=src_access,
            dstAccessMask=dst_access,
        )
        vk.vkCmdPipelineBarrier(
            self.command_buffer,
            src_stage,
            dst_stage,
            0,
            0,
            None,
            0,
            None,
            1,
            [barrier],
        )

    def _color_subresource_range(self):
        return vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1,
        )

    def _create_buffer(self, size, usage, properties):
        info = vk.VkBufferCreateInfo(
            size=size,
            usage=usage,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        buffer = vk.vkCreateBuffer(self.device, info, None)
        requirements = vk.vkGetBufferMemoryRequirements(self.device, buffer)
        alloc_info = vk.VkMemoryAllocateInfo(
            allocationSize=requirements.size,
            memoryTypeIndex=self._find_memory_type(requirements.memoryTypeBits, properties),
        )
        memory = vk.vkAllocateMemory(self.device, alloc_info, None)
        vk.vkBindBufferMemory(self.device, buffer, memory, 0)
        return buffer, memory

    def _create_image(self, width, height, image_format, usage, properties):
        image_info = vk.VkImageCreateInfo(
            imageType=vk.VK_IMAGE_TYPE_2D,
            format=image_format,
            extent=vk.VkExtent3D(width=width, height=height, depth=1),
            mipLevels=1,
            arrayLayers=1,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=usage,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        )
        image = vk.vkCreateImage(self.device, image_info, None)
        requirements = vk.vkGetImageMemoryRequirements(self.device, image)
        alloc_info = vk.VkMemoryAllocateInfo(
            allocationSize=requirements.size,
            memoryTypeIndex=self._find_memory_type(requirements.memoryTypeBits, properties),
        )
        memory = vk.vkAllocateMemory(self.device, alloc_info, None)
        vk.vkBindImageMemory(self.device, image, memory, 0)
        return image, memory

    def _create_output_resources(self):
        self.output_image, self.output_image_memory = self._create_image(
            self.swapchain_extent.width,
            self.swapchain_extent.height,
            vk.VK_FORMAT_R8G8B8A8_UNORM,
            vk.VK_IMAGE_USAGE_STORAGE_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )
        self.output_image_view = self._create_image_view(
            self.output_image, vk.VK_FORMAT_R8G8B8A8_UNORM)
        self.output_image_layout = vk.VK_IMAGE_LAYOUT_UNDEFINED

    def _create_image_view(self, image, image_format):
        view_info = vk.VkImageViewCreateInfo(
            image=image,
            viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
            format=image_format,
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0,
                levelCount=1,
                baseArrayLayer=0,
                layerCount=1,
            ),
        )
        return vk.vkCreateImageView(self.device, view_info, None)

    def _find_memory_type(self, type_filter, properties):
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)
        for i in range(mem_props.memoryTypeCount):
            if (type_filter & (1 << i) and
                    (mem_props.memoryTypes[i].propertyFlags & properties) == properties):
                return i
        raise RuntimeError('Failed to find suitable Vulkan memory type')

    def _destroy_source_resources(self):
        if self.src_image_view:
            vk.vkDestroyImageView(self.device, self.src_image_view, None)
            self.src_image_view = None
        if self.src_image:
            vk.vkDestroyImage(self.device, self.src_image, None)
            self.src_image = None
        if self.src_image_memory:
            vk.vkFreeMemory(self.device, self.src_image_memory, None)
            self.src_image_memory = None
        self.src_image_size = None
        self.src_image_layout = vk.VK_IMAGE_LAYOUT_UNDEFINED

    def _destroy_output_resources(self):
        if self.output_image_view:
            vk.vkDestroyImageView(self.device, self.output_image_view, None)
            self.output_image_view = None
        if self.output_image:
            vk.vkDestroyImage(self.device, self.output_image, None)
            self.output_image = None
        if self.output_image_memory:
            vk.vkFreeMemory(self.device, self.output_image_memory, None)
            self.output_image_memory = None
        self.output_image_layout = vk.VK_IMAGE_LAYOUT_UNDEFINED

    def _destroy_staging_resources(self):
        if self.staging_buffer:
            vk.vkDestroyBuffer(self.device, self.staging_buffer, None)
            self.staging_buffer = None
        if self.staging_memory:
            vk.vkFreeMemory(self.device, self.staging_memory, None)
            self.staging_memory = None
        self.staging_size = 0

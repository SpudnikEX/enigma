// Restructured window.rs into here

use cgmath::num_traits::ToPrimitive;
use cgmath::{InnerSpace, Matrix, Matrix3, Matrix4, Point3, Rad, SquareMatrix, Vector3};
use glam::{vec3, Mat4, Vec3Swizzles};
use gltf::json::extensions::texture;
use gltf::json::serialize::to_vec;
use vulkano::device::Features;
use vulkano::pipeline::graphics::color_blend::{AttachmentBlend, ColorComponents};
use vulkano::DeviceSize;
use std::ops::Index;
use std::{default, string, sync::Arc, time::Instant};
use vulkano::command_buffer::{CopyBufferToImageInfo, ImageBlit, PrimaryCommandBufferAbstract};
use vulkano::image::sampler::{self, Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode};
use vulkano::image::ImageLayout;
use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, BlitImageInfo,
        CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo,
        SubpassContents,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Queue,
        QueueCreateInfo, QueueFlags,
    },
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions},
    memory::allocator::{
        AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryTypeFilter,
        StandardMemoryAllocator,
    },
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::{CullMode, FrontFace, PolygonMode, RasterizationState},
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::{
        acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
    Validated, VulkanError, VulkanLibrary,
};
use winit::{
    dpi::{LogicalSize, PhysicalSize},
    event::{self, Event, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    monitor::VideoMode,
    platform::windows::WindowBuilderExtWindows,
    window::{CursorGrabMode, Fullscreen, Window, WindowBuilder},
};

use crate::camera::Camera;
use crate::render;

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Declarations & Implementations
//////////////////////////////////////////////////////////////////////////////////////////////////

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/default.vert" //raymarch.vert
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/default.frag", //raymarch.frag
    }
}

/*
// DO NOT IMPLEMENT YET
pub enum Resolutions {
    Optimal,
    MAX,
    GBA,
    NDS,
}
impl Resolutions {
    fn get_render_resolution(res: Resolutions) -> [f32; 2] {
        let new_res = match res {
            Resolutions::MAX => [2560, 1440],
            Resolutions::GBA => [240,160], //240 (w) × 160 (h) pixels (3:2 aspect ratio)
            Resolutions::NDS => [256,192], //256 × 192 pixels (4:3 aspect ratio) for each screen
            Resolutions::Optimal => [1920, 1080],
            _ => [640,360] // Default, Max / 4
        };
        [new_res[0] as f32, new_res[1] as f32]
    }
}
*/

// dont remember what I was doing with this? possibly continuous data?
pub struct ModelBuffers {
    // save the model data or save the vertex buffer and index buffer here? need to check taid on their multiple model loading
    vertex_buffer: Subbuffer<[crate::model::Vertex_3D]>, //vertex_buffer: Subbuffer<[crate::model::Vertex_3D]>, //vertex_buffer: Subbuffer<[CustomVertex]>, // Temporary until i add more buffers per model
    index_buffer: Subbuffer<[u16]>,
    // indirect buffers?
}

// values cannot be saved into .rs file, they exist either inside a function for the duration of the function
// or they live inside a struct, for the duration of the struct

/// Cached types that are reused during the render loop
/// Restructued by following https://taidaesal.github.io/vulkano_tutorial/section_11.html
pub struct Renderer {
    // Create struct for values commonly used
    // cache types reused within the render loop?

    // In order of being called / used
    // Hover over variables to view their types inside functions

    //event_loop: EventLoop<()>, // Already Initalized inside main game loop
    //library: Arc<VulkanLibrary>, // Un-needed outside of initalization
    required_extensions: InstanceExtensions,
    instance: Arc<Instance>,
    window: Arc<Window>,

    // Create different pipelines later on
    pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,
    device: Arc<Device>,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    swapchain_images: Vec<Arc<Image>>,
    swapchain_framebuffers: Vec<Arc<Framebuffer>>,
    uniform_buffer: SubbufferAllocator, // change the name to desriptor set buffer??

    internal_image: Arc<Image>,
    internal_framebuffer: Arc<Framebuffer>,

    memory_allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    descriptor_set_allocator: StandardDescriptorSetAllocator,
    command_buffer_allocator: StandardCommandBufferAllocator,
    /// "Render pass tells Vulkan what is available. However to tell Vulkan how to use it, need to update the pipeline declaration." -Taid 
    /// - See RenderPass for more information from vulkano
    render_pass: Arc<RenderPass>,

    //vertex_buffer: Subbuffer<[crate::model::Vertex_3D]>, //vertex_buffer: Subbuffer<[crate::model::Vertex_3D]>, //vertex_buffer: Subbuffer<[CustomVertex]>, // Temporary until i add more buffers per model
    //index_buffer: Subbuffer<[u16]>,
    //depth_buffer: Arc<ImageView>,
    pub recreate_swapchain: bool, // Expose public for now
    previous_frame_end: Option<Box<dyn GpuFuture>>,

    // Non Vulkan Setup Variables
    pub camera: Camera, // Expose public for now, move into ecs system later?
    render_settings: crate::settings::VideoSettings,


    pub models: Vec<crate::model::Model>, // contiguous array holding model data
    //builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,

    /*
    // Moved texture loading into texture.rs
    texture_imageview_sampler : Vec<(Arc<ImageView>, Arc<Sampler>)>,
    */
    // ❗ load one
    //texture_imageview: Arc<ImageView>,
    //texture_sampler: Arc<Sampler>,

    /// Contains the ImageView & Sampler for the loaded textures, in that order.
    pub textures: crate::texture::Texture,

}

impl Renderer {
    ///Initalize like normal
    /// Reference "blit" vulkano tutorial for normal layout execution order
    /// Variables not cached in the struct will exit scope one function ends
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        ///////////////////////////////////////////////////////////////////////////////////////////
        // Non-Vulkan Setup
        ///////////////////////////////////////////////////////////////////////////////////////////

        // struct or a function with enum that holds the values?
        let render_settings = crate::settings::VideoSettings::new();

        //❗ Change this later, when window resizes, recalculate aspect_ratio and projection matrix
        let mut camera = crate::camera::Camera::new();
        let aspect_ratio = 1080.0 / 720.0; //2560.0/1440.0;
        camera.set_projection(110.0, aspect_ratio, 1.0, 500.0);

        ///////////////////////////////////////////////////////////////////////////////////////////
        /// Vulkan Setup
        ///////////////////////////////////////////////////////////////////////////////////////////

        // Moved event_loop initalization into main game loop
        //let event_loop = EventLoop::new();
        let library = VulkanLibrary::new().unwrap();

        let required_extensions = Surface::required_extensions(&event_loop);

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        // Create window, set as fullscreen or not
        let window = Arc::new(
            WindowBuilder::new()
                .with_inner_size(LogicalSize::new(1920, 1080)) // LogicalSize vs PhysicalSize
                .with_title("Enigma Engine")
                //.with_taskbar_icon(taskbar_icon)
                //.with_window_icon(window_icon)
                //.with_fullscreen(Some(Fullscreen::Borderless(None)))
                .build(&event_loop)
                .unwrap(),
        );

        //Used to select whether or not VSync is used
        // see window.rs WindowDescriptor present_mode
        // default value is FIFO, which enables a form of vsync. Need to check capabilties before trying other modes

        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        // Add features for wireframe rendering (see vulkano multiview/main.rs)
            // 🟠 Edit physical_device & (device, queues)
        let features = Features {
            fill_mode_non_solid: true, // Fill mode non solid is required for wireframe display
            wide_lines: true, // Either use thin lines or wide lines 🟠 Change Pipeline:rasterization_state:line_width
            ..Features::empty()
        };

        // Physical device? Queue family index?
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter(|p| p.supported_features().contains(&features))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        // pick first queue_familiy_index that handles graphics and can draw on the surface created by winit
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                // lower score for preferred device types
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .expect("No suitable physical device found");

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        // device and queueS? revisit description
        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                enabled_features: features, // "You only need to enable the features that you need."
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();

        // queue?? difference from queue?
        let queue = queues.next().unwrap();

        let (swapchain, images) = {
            let surface_capabilities = device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();

            let image_format = device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0;

            // This function returns the swapchain plus a list of the images that belong to the swapchain.
            // The order in which the images are returned is important for the acquire_next_image and present functions.
            Swapchain::new(
                device.clone(),
                surface,
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2), //setup the minimum number of images that are present in the swapchain
                    image_format,
                        // Set the size of the swapchain image
                        // Set to the window size
                    image_extent: window.inner_size().into(), 
                    image_usage: ImageUsage::COLOR_ATTACHMENT
                        | ImageUsage::TRANSFER_SRC
                        | ImageUsage::TRANSFER_DST, // How the image can be used (COLOR_ATTACHMENT, TRANSFER_SRC)
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    ..Default::default()
                },
            )
            .unwrap()
        };

        // Create buffer allocators
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator =
            StandardDescriptorSetAllocator::new(device.clone(), Default::default());
        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());

        // "Uniforms" are another name for "inputs" to the shader. I have named them "uniform"
        let uniform_buffer = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        /*
        let uniform_subbuffer = {
            // glam::Mat4 is in column order, needs to be converted into f32 array
            // uniform glsl is in an array of f32 -> [[f32;4];4]
            let uniform_data = vs::MvpData {
                model_matrix: temp_model.model_matrix.to_cols_array_2d(), //camera.mvp.model.to_cols_array_2d(), //Default::default(),
                view_matrix: camera.mvp.view.to_cols_array_2d(), //camera.mvp.view.to_cols_array_2d(), //Default::default(),
                projection_matrix: camera.mvp.projection.to_cols_array_2d(), //Default::default(),
                time: 0.0,
            };

            let subbuffer = uniform_buffer.allocate_sized().unwrap();
            *subbuffer.write().unwrap() = uniform_data;

            subbuffer
        };
        */

        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            // 🟠 Add the following attachments to the pass (below) [dwadw]
            attachments: { 
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
                // 🟠 Make sure to adjust in pipeline & frambuffer attachments
                depth: {
                    format: Format::D16_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
                }
            },
            pass: { 
                color: [color],
                depth_stencil: {depth}
            }
        )
        .unwrap();

        // Subpasses later ??
        // Different Pipelines later? reference taid for the different pipelines later

        // Define shader entrypoints
        let vs = vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        /*
        // ❗ remove verts and index?
        // Moved vertices into model.rs
        // temporary second vertex buffer for second model
        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            temp_model.vertex_data.clone(), //temp_model.vertices, // using iter, get vertices //.iter.cloned() or clone() ?
        )
        .unwrap();

        let index_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            temp_model.indices.clone(),
        )
        .unwrap();
        */

        let vertex_input_state = crate::model::Vertex_3D::per_vertex() // The inputs for vertex fields, uploaded to shaders.
            .definition(&vs.info().input_interface)
            .unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(vs.clone()),
            PipelineShaderStageCreateInfo::new(fs.clone()),
        ]
        .into_iter()
        .collect();

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        // Flip the Vulkan viewport on the Y-Axis. (So that models will be oriented correctly during rendering)
            // ❗ first finish solidifying camera axis , first & third person
            // Will need to remove 
         // ❗ Needs to be true for final render, false while debugging
        let viewport: Viewport = match render_settings.flip_viewport {
            false => { 
                Viewport { // Default, not flipped
                    offset: [0.0, 0.0], // "Coordinates in pixels of the top-left hand corner of the viewport."
                    extent: render_settings.viewport_resolution.into(), //[1920.0,1080.0], // "Dimensions in pixels of the viewport." The size of the screen to be rendered to.
                    depth_range: 0.0..=1.0, // "The default value is 0.0..=1.0."    Note this is where Vulkan maps the Z-component to complete the Clip-Space / NDC-Space (0 is at screen, 1 is max depth forwards)
                        // for example, a vertex position at (0,0,0) or (0,0,1) would be visible, but (0,0,2) would not render
                        // Maximum depth is 1.0, minimum depth is 0.0
                }
            },
            true => {
                Viewport { // following Sascha Willems blog post: 
                    offset: [0.0, render_settings.viewport_resolution.index(1).clone()], // Need to change the origin when flipping viewport
                    extent: [render_settings.viewport_resolution.index(0).clone(), -render_settings.viewport_resolution.index(1).clone()], // need to change y value to be negative
                    depth_range: 0.0..=1.0, 
                }
            }
        };

        // 🟠 Pipeline depends on viewport, viewport MUST be manually set and not ::default()
        // Pipeline describes how things will look when drawn to the viewport, ex: lines vs tris
        let pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages,
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState {
                    topology: PrimitiveTopology::TriangleList, // "Describes how vertices must be grouped together to form primitives."  Default: PrimitiveTopology::TriangleList
                    primitive_restart_enable: false,           //The default value is false
                    ..Default::default()
                }), // Sets primitive toploogy type (point, lines, triangle list)
                viewport_state: Some(ViewportState {
                    // "dynamic viewport allows chaing the viewport per draw call, at the cost of performance"  Changed inside GraphicsPipelineCreateInfo (below)
                    viewports: [viewport.clone()].into_iter().collect(),
                    ..Default::default()
                }),
                //viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState {
                    // Sets primitive polygon mode, cull mode, winding order, etc.
                    front_face: FrontFace::CounterClockwise, // Default is CounterClockwise
                    depth_clamp_enable: false,
                    rasterizer_discard_enable: false,
                    polygon_mode: PolygonMode::Fill, // Polygonmode fill, lines wireframe, or points. 🟠 Change render feature
                    cull_mode: CullMode::Front, // Front face, Back face, or None
                    line_width: 1.0, // Line width for drawing lines. "The default value is `1.0`" 🟠 Requires PolygonFillMode=LINE & Feature:nonSolidMode=true
                    ..Default::default()
                }),
                depth_stencil_state: Some(DepthStencilState {
                    // 🟠 Sets depth testing, make sure to setup in render_pass & framebuffer attachments
                    depth: Some(DepthState::simple()),
                    ..Default::default()
                }),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    Subpass::from(render_pass.clone(), 0)
                        .unwrap()
                        .num_color_attachments(),
                    ColorBlendAttachmentState {
                        blend: Some(AttachmentBlend::alpha()), // Enables texture transparency using texture alpha (See vulkano image/main.rs)
                        color_write_mask: ColorComponents::all(),
                        color_write_enable: true,
                        ..Default::default()
                    },
                )),
                //dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(Subpass::from(render_pass.clone(), 0).unwrap().into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap();

        let depth_buffer = ImageView::new_default( // "Imageview: A wrapper around an image that makes it available to shaders or framebuffers."
            Image::new(
                memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::D16_UNORM,
                        // 🟠 Change(d) to use internal blit image size instead of swapchain image size (would be full window)
                    extent: [
                        render_settings.viewport_resolution[0] as u32,
                        render_settings.viewport_resolution[1] as u32,
                        1
                    ], 
                    usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();

        // the framebuffers, and subsequently the number of framebuffers, are constructed from the amount of images that are constructed when making the swapchain
        // Create framebuffers using swapchain images, to draw directly to the swapchain
        let framebuffers = images
            .iter()
            .map(|image| {
                let image_view = ImageView::new_default(image.clone()).unwrap();
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![image_view, depth_buffer.clone()],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        // Internal image(s) to render into, instead of rendering directly to swapchain.
        // Draw to this image, then blit to swapchain
        // For more examples, check vulkano "offscreen" example.
        let internal_image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: swapchain.image_format(), // Default is Undefined
                    // Use the internal viewport resolution size instead of swapchain size
                extent: [
                    render_settings.viewport_resolution[0] as u32,
                    render_settings.viewport_resolution[1] as u32,
                    1
                ], // Resuse the swapchain image size. Used to use [window.inner_size().width, window.inner_size().height, 1], // [1920, 1080, 1], //
                usage: ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::TRANSFER_SRC
                    | ImageUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();

        // Create internal image view and internal framebuffer, to draw to internal image. Used to blit to swapchain image
        let internal_image_view = ImageView::new_default(internal_image.clone()).unwrap();
        let internal_framebuffer = Framebuffer::new(
            render_pass.clone(),
            FramebufferCreateInfo {
                // Attach the offscreen image to the framebuffer.
                attachments: vec![internal_image_view, depth_buffer.clone()],
                ..Default::default()
            },
        )
        .unwrap();

        // Flag if swapchain is invalid, or windowsize changes.
        let recreate_swapchain = false;

        let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

        /*         let builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap(); */


        ////////////////////////////////////////////////////////////////////////////////
        // Textures
        // Refer to FBY for tutorial & vulkano/image/main.rs examples
        // Refer to Taid for alternative example
        ////////////////////////////////////////////////////////////////////////////////

        // Temp? List of textures to render
        // move into a globals struct for contiguous data?
        let tex: Vec<(Arc<ImageView>, Arc<Sampler>)> = Vec::new();



        

        ///////////////////////////////////////////////////////////////////////////////////////////
        // Return values for struct
        // Get all the values before starting event_loop
        ///////////////////////////////////////////////////////////////////////////////////////////
        Renderer {
            // Struct variable: local variable value //
            //event_loop, // Already initalized in main game loop
            //library, //Un-needed outside of initilization
            required_extensions,
            instance,
            window,
            device: device,
            queue: queue,
            swapchain,
            swapchain_images: images,
            swapchain_framebuffers: framebuffers,
            uniform_buffer: uniform_buffer,

            memory_allocator: memory_allocator,
            descriptor_set_allocator: descriptor_set_allocator,
            command_buffer_allocator: command_buffer_allocator,
            render_pass: render_pass,
            //vertex_buffer: vertex_buffer, // Temporary until I add more vertex buffers per model
            //index_buffer: index_buffer,
            pipeline: pipeline,
            viewport: viewport,

            internal_image: internal_image,
            internal_framebuffer: internal_framebuffer,
            recreate_swapchain: recreate_swapchain,
            previous_frame_end: previous_frame_end,

            // Non Vulkan Setup Variables //
            camera: camera,
            render_settings: render_settings,
            models: Vec::new(), // ❗ Temp? List of models to render
            //builder: builder,

            // First Iteration Texture and Sampler for Textures
            //texture_imageview_sampler: tex
            //texture_imageview: 
            //texture_sampler: Vec::new(),
            textures: crate::texture::Texture::new(),

        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Main Rendering Loop
    ///////////////////////////////////////////////////////////////////////////////////////////////
    
    /// Main Rendering Loop
    /// Call in `MainEventsCleared` event
    pub fn start(&mut self, delta_time: &f32) {
        // Size of the image that is rendered in the window surface.
        // Updated automatically each swapchain execution
        let image_extent: [u32; 2] = self.window.inner_size().into();

        // Both values must be greater than zero. Note that on some platforms, SurfaceCapabilities::current_extent will be ZERO if the surface is minimized.
        // Care must be taken to check for this, to avoid trying to create a zero-size swapchain.
        // The default value is [0, 0], which must be overridden.
        if image_extent.contains(&0) {
            println!("Not Rendering");
            return;
        }

        // It is highly recommended to call cleanup_finished from time to time. Doing so will prevent memory usage from increasing over time, and will also destroy the locks on resources used by the GPU.
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        if self.recreate_swapchain {
            // Creates a new swapchain from the previous one.
            // Use this when a swapchain has become invalidated, such as due to window resizes.
            (self.swapchain, self.swapchain_images) = self
                .swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent: image_extent, //window.inner_size().into();
                    image_usage: ImageUsage::COLOR_ATTACHMENT
                        | ImageUsage::TRANSFER_SRC
                        | ImageUsage::TRANSFER_DST,
                    ..self.swapchain.create_info()
                })
                .expect("failed to recreate swapchain");

            let depth_buffer = ImageView::new_default(
                Image::new(
                    self.memory_allocator.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format: Format::D16_UNORM,
                        extent: [
                            self.render_settings.viewport_resolution[0] as u32,
                            self.render_settings.viewport_resolution[1] as u32,
                            1
                        ], //self.swapchain_images[0].extent(),
                        usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT
                            | ImageUsage::TRANSIENT_ATTACHMENT,
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                )
                .unwrap(),
            )
            .unwrap();

            // Recreate the framebuffers
            // The image views that are attached to a render pass during drawing.
            // A framebuffer is a collection of images, and supplies the actual inputs and outputs of each attachment within a render pass. Each attachment point in the render pass must have a matching image in the framebuffer
            self.swapchain_framebuffers = self
                .swapchain_images
                .iter()
                .map(|image| {
                    let view = ImageView::new_default(image.clone()).unwrap();
                    Framebuffer::new(
                        self.render_pass.clone(),
                        FramebufferCreateInfo {
                            attachments: vec![view, depth_buffer.clone()],
                            ..Default::default()
                        },
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();

            // Internal image(s) to render into, instead of rendering directly to swapchain.
            // Draw to this image, then blit to swapchain
            // For more examples, check vulkano "offscreen" example.
            self.internal_image = Image::new(
                self.memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: self.swapchain.image_format(), // Default is Undefined
                    extent: [
                        self.render_settings.viewport_resolution[0] as u32,
                        self.render_settings.viewport_resolution[1] as u32,
                        1
                    ], // Resuse the swapchain image size. Used to use [window.inner_size().width, window.inner_size().height, 1], // [1920, 1080, 1], //
                    usage: ImageUsage::COLOR_ATTACHMENT
                        | ImageUsage::TRANSFER_SRC
                        | ImageUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap();

            // Create internal image view and internal framebuffer, to draw to internal image. Used to blit to swapchain image
            let internal_image_view = ImageView::new_default(self.internal_image.clone()).unwrap();
            self.internal_framebuffer = Framebuffer::new(
                self.render_pass.clone(),
                FramebufferCreateInfo {
                    // Attach the offscreen image to the framebuffer.
                    attachments: vec![internal_image_view, depth_buffer.clone()],
                    ..Default::default()
                },
            )
            .unwrap();

            /////////// Resizes window on re-create (can do without) ///////////////////////////////////////////////////
            /////////// Technically, resizes the rendered part of the image (could be incorrect)
            // Define shader entrypoints
            let vs = vs::load(self.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = fs::load(self.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let vertex_input_state = crate::model::Vertex_3D::per_vertex()
                .definition(&vs.info().input_interface)
                .unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vs.clone()),
                PipelineShaderStageCreateInfo::new(fs.clone()),
            ]
            .into_iter()
            .collect();

            let pipeline_layout = PipelineLayout::new(
                self.device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(self.device.clone())
                    .unwrap(),
            )
            .unwrap();

            /////////// Resizes window on re-create (can do without) ///////////////////////////////////////////////////
            // Dimensions in pixels of the viewport.
            // The default value is [1.0; 2], which you probably want to override if you are not using dynamic state.
                // THE RENDER SIZE ON THE IMAGE

            // ❗ change this for when resizing the window
                // Either limit the amount that the window can resize to (clamp)
                // Or get the smaller of the two values, from viewport_resolutoion or window.inner_size() (minimum of the two)
            self.viewport.extent = self.render_settings.viewport_resolution; //self.window.inner_size().into(); // Size of the viewport (rendered area in image) //[128.0, 128.0];//

            self.pipeline = GraphicsPipeline::new(
                self.device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages,
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState {
                        // dynamic viewport allows chaing the viewport per draw call, at the cost of performance
                        viewports: [self.viewport.clone()].into_iter().collect(),
                        ..Default::default()
                    }),
                    rasterization_state: Some(RasterizationState::default()),
                    depth_stencil_state: Some(DepthStencilState {
                        depth: Some(DepthState::simple()),
                        ..Default::default()
                    }),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        Subpass::from(self.render_pass.clone(), 0)
                            .unwrap()
                            .num_color_attachments(),
                            ColorBlendAttachmentState {
                                blend: Some(AttachmentBlend::alpha()), // Enables texture transparency using texture alpha (See vulkano image/main.rs)
                                color_write_mask: ColorComponents::all(),
                                color_write_enable: true,
                                ..Default::default()
                            },
                    )),
                    // When [DynamicState::Viewport] is used, the values of each viewport are ignored and must be set dynamically, but the number of viewports is fixed and must be matched when setting the dynamic value.
                    //dynamic_state: [DynamicState::Viewport].into_iter().collect(), //
                    subpass: Some(Subpass::from(self.render_pass.clone(), 0).unwrap().into()),
                    ..GraphicsPipelineCreateInfo::layout(pipeline_layout)
                },
            )
            .unwrap();
            ///////////////////////////////////////////////////////////////////////////////////////////////////

            self.recreate_swapchain = false;
        }

        // Tries to take ownership of an image in order to draw on it.
        // The function returns the index of the image in the array of images that was returned when creating the swapchain,
        // plus a future that represents the moment when the image will become available from the GPU (which may not be immediately).
        // If you try to draw on an image without acquiring it first, the execution will block. (TODO behavior may change).
        // The second field in the tuple in the Ok result is a bool represent if the acquisition was suboptimal.
        // In this case the acquired image is still usable, but the swapchain should be recreated as the Surface's properties no longer match the swapchain.
        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(self.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    self.recreate_swapchain = true;
                    return;
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            };

        if suboptimal {
            self.recreate_swapchain = true;
            println!("Swapchain needs recreate");
        }


        ////////////////////////////////////////////////////////////////////////////////
        // Update Command Buffers
        ////////////////////////////////////////////////////////////////////////////////
        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Execute render pass & draw to the framebuffer, as well as additional commands
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    // Set values for each attachment (depth, color, etc.)
                    clear_values: vec![
                        // The default clear color. Upon beginning the render pass, the image is cleared with this color.
                        Some(self.render_settings.clear_color.into()), // values for color attachemnt
                        Some(1f32.into()), // values for depth attachment
                        /*
                        because our depth buffer is in a format that takes a single value per vertex rather than an array we can get away with using 1.0 as the clear value rather than a color vector
                        the clear values must be listed in the same order as the buffer attachments. This is an easy thing to get wrong, so just keep it in mind.
                         */
                    ],
                        //render_area_extent: framebuffers[image_index as usize].extent(), //[u32; 2]
                    // Select the framebuffer to draw to for this render_pass.
                    ..RenderPassBeginInfo::framebuffer(
                        // Assign which image to draw to
                            // Draw to internal image instead of swapchain.
                            // Framebuffer has internal_image attached
                            match self.render_settings.debug_disable_blit {
                                true => self.swapchain_framebuffers[image_index as usize].clone(), // Draw directly to the final swapchain image, to presented to the screen.
                                false => self.internal_framebuffer.clone(),  // Draw to the internal_framebuffer, for use with blitting
                            }
                    )
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap()
            .set_viewport(0, [self.viewport.clone()].into_iter().collect())
            .unwrap()
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap();

        // Loop through models inside of model struct
            // Setup their respective pipelines (❓setup enum for different piplines or create different Vecs for different stages)
                // ❗ Probably should have seperate loops for each pipeline, setup pipeline and models within, which are inside a Vec<>
            // Execute draw commands for each model
        for model in &self.models {
            // ❓ not sure if seperating this into different functions will increase speed.

            ///////////////////////////////////////////////////////////////////////////////////
            // Update Uniforms & Descriptor
            ///////////////////////////////////////////////////////////////////////////////////
            let uniform_subbuffer = {
                // Moved code into camera.rs

                let uniform_data = vs::MvpData {
                    // ❗ RENAME THIS LATER
                    model_matrix: model.model_matrix.to_cols_array_2d(), //self.temp_model.model_matrix.to_cols_array_2d(),//camera.mvp.model.to_cols_array_2d(), //Default::default(),
                    view_matrix: 
                        self.camera.get_view().to_cols_array_2d(), //self.camera.mvp.view.to_cols_array_2d(), //camera.mvp.view.to_cols_array_2d(), //Default::default(),
                    projection_matrix: self.camera.mvp.projection.to_cols_array_2d(), //Default::default(),
                    time: 0.0, //self.instant.elapsed().as_secs_f32().into(),
                };

                let subbuffer = self.uniform_buffer.allocate_sized().unwrap();
                *subbuffer.write().unwrap() = uniform_data;

                subbuffer
            };


            let texture_index = model.texture_index_data.clone();
            let i = texture_index.index(0).to_usize().unwrap();

            // Create descriptor set that holds uniform buffers
            let descriptor_layout = self.pipeline.layout().set_layouts().get(0).unwrap();
            let descriptor_set = PersistentDescriptorSet::new(
                &self.descriptor_set_allocator, 
                descriptor_layout.clone(), 
                [
                    WriteDescriptorSet::buffer(0, uniform_subbuffer.clone()),
                        // 🔴 Remember to update the shader to accept new bindings
                    // Send a single texture to the GPU
                    // Send per-model texture index.
                    WriteDescriptorSet::image_view_sampler(
                        1, 
                        self.textures.global_textures.index(i).0.clone(), //❗ will need to implement more than 1 texture per model loading
                        self.textures.global_textures.index(i).1.clone()
                    ),


                    //WriteDescriptorSet::image_view_sampler(1, self.texture_imageview.clone(), self.texture_sampler.clone()), // Work backwards from the end result! ❓ Use seperate sampler and image (reference vulkano image/main.rs example). 
                    // Send an array of textures to the GPU
                        //https://kylehalladay.com/blog/tutorial/vulkan/2018/01/28/Textue-Arrays-Vulkan.html
                    //WriteDescriptorSet::image_view_sampler_array(1, 0, ) // ❗ I believe this is used for multiple textures for ONE MODEL
                    ],
                [],
            )
            .unwrap();

            // Moved vertices into model.rs
            // temporary second vertex buffer for second model
            let vertex_buffer = Buffer::from_iter(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                model.vertex_data.clone(), //temp_model.vertex_data.clone(), //temp_model.vertices, // using iter, get vertices //.iter.cloned() or clone() ?
            )
            .unwrap();

            let index_buffer = Buffer::from_iter(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::INDEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                model.indices.clone(), //temp_model.indices.clone(),
            )
            .unwrap();

            // what about different pipelines? do later?
            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.pipeline.layout().clone(),
                    0,
                    descriptor_set,
                )
                .unwrap()
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .unwrap()
                .bind_index_buffer(index_buffer.clone())
                .unwrap()
                .draw_indexed(index_buffer.len() as u32, 1, 0, 0, 0)
                .unwrap();
        }

        // I don't belive the following is nesecarry anymore, more a reference for the future
        /*
        .set_viewport(0, [self.viewport.clone()].into_iter().collect())
        .unwrap()
        .bind_pipeline_graphics(self.pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            self.pipeline.layout().clone(),
            0,
            descriptor_set,
        )
        .unwrap()

        .bind_vertex_buffers(0, self.vertex_buffer.clone())
        .unwrap()

        // The "draw" command is executed on the framebuffer
        //.draw(self.vertex_buffer.len() as u32, 1, 0, 0)
        //.unwrap()

        // IMPORTANT! Either use .draw (above) or .draw_indexed, DO NOT USE BOTH
        .bind_index_buffer(self.index_buffer.clone())
        .unwrap()
        .draw_indexed(self.index_buffer.len() as u32, 1, 0, 0, 0)
        .unwrap()
        */

        match self.render_settings.debug_disable_blit {
            true => { // Do not blit, present to swapchain. 🟠 Change frambuffer destitnation when beginning drawing.
                builder.end_render_pass(Default::default()).unwrap();
            }, 
            false => { // Blit to swapchain
                builder
                // Stop drawing to the selected drawable image
                .end_render_pass(Default::default())
                .unwrap()
                // 🔴 When removing blit, change the destination to swapchain_framebuffer in RenderPassBeginInfo
                // Grabbed from the vulkano "copy-blit-image" example. Blit was added using the example from vulkano offscreen example
                // Blit happens after the render pass "draw", will throw error if called during.
                .blit_image(BlitImageInfo {
                    src_image_layout: ImageLayout::General, //TransferSrcOptimal
                    dst_image_layout: ImageLayout::General, //TransferDstOptimal
                        // Nearest for pixel perfect rendering
                        // Linear for aliasing
                        // Cubic needs to have device extension enabled
                    filter: Filter::Nearest,                
                    regions: [ImageBlit {
                        src_subresource: self.internal_image.subresource_layers(), //images[image_index as usize].subresource_layers(),
                        src_offsets: [
                            [0, 0, 0], // source, top left corner, in pixels (should be the viewport size)
                            //[128, 128, 1], // source, bottom right corner, in pixels (should be the viewport size)
                            //self.internal_image.extent(),
                            [
                                self.render_settings.viewport_resolution[0] as u32,
                                self.render_settings.viewport_resolution[1] as u32,
                                1,
                            ], // The viewport render resolution
                        ], //[[u32; 3]; 2]
                        dst_subresource: self.swapchain_images[image_index as usize]
                            .subresource_layers(),
                        dst_offsets: [
                            [0, 0, 0], // destination, top left corner, in pixels
                            // The swapchain image.
                            // Trying to use window.image_extent can throw an error when going between full screen and windowed.
                            // Using the swapchain's image extent gaurentees the image extent are correct.
                            // Use swapchain's dimensions to prevent errors
                            self.swapchain_images[image_index as usize].extent(), // destination, bottom right corner, in pixels
                        ], //[[u32; 3]; 2]
                        ..Default::default()
                    }]
                    .into(),
                    ..BlitImageInfo::images(
                        self.internal_image.clone(),
                        self.swapchain_images[image_index as usize].clone(),
                    )
                })
                .unwrap();
            },
        }

        

        let command_buffer = builder.build().unwrap();

        ///////////////////////////////////////////////////////////////////////////////////
        // Fences & Futures
        ///////////////////////////////////////////////////////////////////////////////////

        // Send the build command buffer to the GPU to be executed
        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                self.previous_frame_end = Some(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                self.recreate_swapchain = true;
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                panic!("failed to flush future: {e}");
            }
        }
    }

    /// Call when all rendering is done for this frame
    pub fn end(&mut self) {
        // Have the command buffer split in two?
        // Or keep as one, keeping the models stored in a vec?
    }

    pub fn geometry(&mut self, model: &mut crate::model::Model) {
        /*
        ///////////////////////////////////////////////////////////////////////////////////
        // Update Uniforms & Descriptor
        ///////////////////////////////////////////////////////////////////////////////////
        let uniform_subbuffer = {
            // Moved code into camera.rs

            let uniform_data = vs::MvpData {
                // ❗ RENAME THIS LATER
                model_matrix: model.model_matrix.to_cols_array_2d(), //self.temp_model.model_matrix.to_cols_array_2d(),//camera.mvp.model.to_cols_array_2d(), //Default::default(),
                view_matrix: self.camera.mvp.view.to_cols_array_2d(), //camera.mvp.view.to_cols_array_2d(), //Default::default(),
                projection_matrix: self.camera.mvp.projection.to_cols_array_2d(), //Default::default(),
                time: 0.0, //self.instant.elapsed().as_secs_f32().into(),
            };

            let subbuffer = self.uniform_buffer.allocate_sized().unwrap();
            *subbuffer.write().unwrap() = uniform_data;

            subbuffer
        };

        // Create descriptor set that holds uniform buffers
        let descriptor_layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let descriptor_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            descriptor_layout.clone(),
            [WriteDescriptorSet::buffer(0, uniform_subbuffer.clone())],
            [],
        )
        .unwrap();

        // Moved vertices into model.rs
        // temporary second vertex buffer for second model
        let vertex_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            model.vertex_data.clone(), //temp_model.vertex_data.clone(), //temp_model.vertices, // using iter, get vertices //.iter.cloned() or clone() ?
        )
        .unwrap();

        let index_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            model.indices.clone(), //temp_model.indices.clone(),
        )
        .unwrap();

        self.builder
            .set_viewport(0, [self.viewport.clone()].into_iter().collect())
            .unwrap()
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap()
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .unwrap()
            .bind_index_buffer(self.index_buffer.clone())
            .unwrap()
            .draw_indexed(self.index_buffer.len() as u32, 1, 0, 0, 0)
            .unwrap();
        */
    }

    // Pass texture path or texture data?
    // for custom texture loading, texture path
    // for embedded texture loading, texture data as Vec<u8>

    /// Create and return `ImageView` & `Sampler` for the Global Texture Vec.
    /// 
    /// NOTE: This function is kept here because it relies on so much of other things inside the renderer.
    /// 
    /// ❓ Pass either path name, or image data (Vec<u8>)
    pub fn create_imageview_sampler(&mut self, texture_path: &str) -> (Arc<ImageView>, Arc<Sampler>) {

        let mut uploads = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Import texture (external for now)
            // Remember to set the UV's per vertex on the model
        // load the image data and dimensions before event loop
        // ❗❗ To implement per-model texture loading, need to move this outside of this function
        // Load one one instance of a texture for use in memory, as to not have duplicates (wasted memory)
        let texture_imageview = {
            // Macros provided by Rust Standard Lib
            // Example provided by vulkano/image/main.rs

            // ❓ as_slice or as_vec? GLTF data is read as this natively.    vulkano example & fby uses as_slice while taid uses to_vec?
            //let png_bytes = include_bytes!("../models/tests/steve-classic.png").as_slice(); // &[u8]
            //let png_vec = include_bytes!("../models/tests/old-grass.png").to_vec(); //Vec<u8>

            // include_bytes! for paths known at compile time
                // path starts from calling file directory
            // std::fs::read for paths known at runtime
                // path starts from project root 

            let cursor = std::io::Cursor::new(std::fs::read(texture_path).unwrap()); //Taid addition

            let decoder = png::Decoder::new(cursor); // png_bytes OR png_vec + cursor
            //decoder.set_transformations(png::Transformations::)
            let mut reader = decoder.read_info().unwrap();
            let info = reader.info();
            let dimensions = [info.width, info.height, 1]; // "If image_type is ImageType::Dim2d, then the depth must be 1. If image_type is ImageType::Dim1d, then the height and depth must be 1." -image_extent
            println!("dimensions: {:?} | {:?}",info.width, info.height);
            //std::process::exit(0);
    
            let bit_depth: u32 = match info.bit_depth {
                png::BitDepth::One => 1,
                png::BitDepth::Two => 2,
                png::BitDepth::Four => 4,
                png::BitDepth::Eight => 8,
                png::BitDepth::Sixteen => 16,
                _ => 4 // bit_depth is default 4 in vulkano example
            };

            let upload_buffer = Buffer::new_slice(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                (info.width * info.height * 4) as DeviceSize, // * bit_depth
            )
            .unwrap();
    
            reader
                .next_frame(&mut upload_buffer.write().unwrap())
                .unwrap();
    
            let image = Image::new(
                self.memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R8G8B8A8_UNORM, // Default R8G8B8A8_SRGB, but gives incorrect colors
                    extent: dimensions,
                    usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                    //tiling: vulkano::image::ImageTiling::Optimal, //The default value is ImageTiling::Optimal.
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap();
    
            uploads
                .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                    upload_buffer,
                    image.clone(),
                ))
                .unwrap();
    
            ImageView::new_default(image).unwrap()
        };

        // Create Sampler for the image
        let texture_sampler = Sampler::new(
            self.device.clone(),
            SamplerCreateInfo {
                // mag_filter: Filter::Nearest, // Nearest, Cubic, Linear
                // min_filter: Filter::Nearest, // Nearest, Cubic, Linear
                // mipmap_mode: SamplerMipmapMode::Nearest,
                // mip_lod_bias: 0.0,
                // address_mode: [SamplerAddressMode::ClampToEdge; 3],//[SamplerAddressMode::Repeat; 3], // defualt: ClampToEdge
                ..Default::default()
            },
        )
        .unwrap();

        // ❗ for every texture, does previous _frame_end need to be appended?
            // if so, when do these need to be appended?
        self.previous_frame_end = Some(
            uploads
                .build()
                .unwrap()
                .execute(self.queue.clone())
                .unwrap()
                .boxed(),
        );

        // For multiple texture loading
        //let texture_imageview_sampler: Vec<(Arc<ImageView>, Arc<Sampler>)> = Vec::new();
        (texture_imageview, texture_sampler)
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
pub fn main() {



    let renderer = Renderer::new();


    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Buffers & Allocators
    ///////////////////////////////////////////////////////////////////////////////////////////////



    // let index_data: Vec<u32> = vec!(
    //     0, 1, 2, 2, 3, 0
    // );

    // let index_buffer = Buffer::from_iter(
    //     memory_allocator.clone(),
    //     BufferCreateInfo {
    //         usage: BufferUsage::INDEX_BUFFER,
    //         ..Default::default()
    //     },
    //     AllocationCreateInfo {
    //         memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
    //             | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
    //         ..Default::default()
    //     },
    //     index_data,
    // )
    // .unwrap();

    // let depth_buffer = ImageView::new_default(
    //     Image::new(
    //         memory_allocator.clone(),
    //         ImageCreateInfo {
    //             image_type: ImageType::Dim2d,
    //             format: Format::D16_UNORM,
    //             extent: images[0].extent(),
    //             usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
    //             ..Default::default()
    //         },
    //         AllocationCreateInfo::default(),
    //     )
    //     .unwrap(),
    // )
    // .unwrap();

    // create camera from my camera struct



*/

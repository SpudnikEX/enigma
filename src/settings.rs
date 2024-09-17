/*
https://stackoverflow.com/questions/27791532/how-do-i-create-a-global-mutable-singleton
Avoid global state in general. 
Instead, construct the object somewhere early (perhaps in main), then pass mutable references to that object into the places that need it. 
This will usually make your code easier to reason about and doesn't require as much bending over backwards.
*/

pub struct GlobalSettings {
    pub video_settings: VideoSettings,
    pub audio_settings: AudioSettings,
    pub control_settings: ControlSettings,
    pub access_settings: AccessibiltySettings,
}

// impl GlobalSettings {
//     pub fn new() -> Self {
//         Self {
//             video_settings: Default::default(),
//             audio_settings: Default::default(),
//             control_settings: Default::default(),
//             access_settings: Default::default(),
//         }
//     }
// }

// Video or Graphics Settings?
pub struct VideoSettings {
    // Internal Render Settings

    /// The viewport resolution used on the internal image. Used to blit lower resolution drawn image to the swapchain.
    /// The viewport resolution in pixels. Used in blitting the internal rendered image to the swapchain image
    pub viewport_resolution: [f32; 2], // Format used when creating swapchain / depth / internal / images
    pub clear_color: [f32; 4],
    pub mouse_visible: bool,
    pub fullscreen: bool,
    pub aspect_ratio: f32,

    /// Toggle for debug menu
    pub debug_mode: bool,

    /// Set to `True` to disable blitting to swapchain
    /// Turn off blitting for debug
    /// - Default is `False`
    pub debug_disable_blit: bool,

    /// Flip the viewport horizontally to orient Y+ as up. Use when not debugging, and for final release.
    pub flip_viewport: bool,
}

impl VideoSettings {
    pub fn new() -> Self {
        //match statement can go here for resolutions

        let mut resolution = [2560.0, 1440.0]; //[1080.0, 720.0], //[1080.0, 720.0], //[320.0, 180.0],//[1080.0,720.0], //[320.0, 180.0],//[640.0, 360.0],

        // aspect ratio = screen height / screen width
        let ratio = resolution[0] / resolution[1];

        Self {
            viewport_resolution: resolution,
            aspect_ratio: ratio,
            clear_color: [0.45, 0.45, 0.45, 1.0], // The clear color if nothing (mesh) is rendered to the screen
            mouse_visible: true,
            fullscreen: false,

            // Debug settings
            debug_mode: false,
            debug_disable_blit: true,
            flip_viewport: false,
            
        }
    }
}

pub struct AudioSettings {
    pub volume_master: u8,
    pub volume_music: u8,
}

pub struct ControlSettings {
    //Invert controls
}

pub struct AccessibiltySettings {
    // disable / replace eating sounds (Daniel)
}

/*
To Do:
Save and load settings to / from file
*/

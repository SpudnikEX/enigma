use cgmath::num_traits::AsPrimitive;
use glam::vec3;
use winit::event::{self, Event, VirtualKeyCode};
use winit_input_helper::WinitInputHelper;
// static mut input: WinitInputHelper = WinitInputHelper::new();

pub struct KeyBindings {
    forward: (VirtualKeyCode),
    backward:(VirtualKeyCode),
    left:(VirtualKeyCode),
    right:(VirtualKeyCode),
    up:(VirtualKeyCode),
    down:(VirtualKeyCode),

    menu:VirtualKeyCode,
    fullscreen:VirtualKeyCode,
    debug_menu:VirtualKeyCode,
    mouse_toggle:VirtualKeyCode,
}


impl KeyBindings {
    /// Create default Keybindings
    fn new() -> Self {
        // instead of writing self.etc, encompass all inside of Self{}
        Self {
            forward: VirtualKeyCode::W,
            backward: VirtualKeyCode::S,
            left: VirtualKeyCode::A,
            right: VirtualKeyCode::D,
            up: VirtualKeyCode::Space,
            down: VirtualKeyCode::LControl,

            menu: VirtualKeyCode::Escape,
            fullscreen: VirtualKeyCode::F11,
            debug_menu: VirtualKeyCode::F3,
            mouse_toggle: VirtualKeyCode::LAlt,
        }
    }

    /// Change the keybinding for a specific key
    pub fn set_value(&mut self, key: VirtualKeyCode) {
        self.forward = key;
        
    }

}

/// Refer to https://docs.rs/winit/latest/winit/ for setup of the input crate
/// Multiply by delta time for frame rate independance
pub fn update_input(input: &WinitInputHelper, camera: &mut crate::camera::Camera, delta_time: f32) {

    // // DEBUG!!! REMOVE LATER!! 
    // // make sure nothing is hijacking input (soundpad)
/*     if winit_input.key_held(winit::event::VirtualKeyCode::W) {
        renderer.temp_model.translate(glam::vec3(0.0, 0.0, 1.0), speed);
        //renderer.camera.translate_local(glam::vec3(0.0, 0.0, 1.0), 0.01);
        println!("MOVE");
    }

    if winit_input.key_held(winit::event::VirtualKeyCode::S) {
        renderer.temp_model.translate(glam::vec3(0.0, 0.0, -1.0), speed);
        //renderer.camera.translate_local(glam::vec3(0.0, 0.0, -1.0), 0.01);
    }
    if winit_input.key_held(winit::event::VirtualKeyCode::A) {
        renderer.temp_model.translate(glam::vec3(-1.0, 0.0, 0.0), speed);
        //renderer.camera.translate_local(glam::vec3(-1.0, 0.0, 0.0), 0.01);
    }
    if winit_input.key_held(winit::event::VirtualKeyCode::D) {
        renderer.temp_model.translate(glam::vec3(1.0, 0.0, 0.0), speed);
        //renderer.camera.translate_local(glam::vec3(1.0, 0.0, 0.0), 0.01);
    }
    if winit_input.key_held(winit::event::VirtualKeyCode::Space) {
        renderer.temp_model.translate(glam::vec3(0.0, -1.0, 0.0), speed);
        //renderer.camera.translate_local(glam::vec3(0.0, -1.0, 0.0), 0.01);
    }
    if winit_input.key_held(winit::event::VirtualKeyCode::C) {
        renderer.temp_model.translate(glam::vec3(0.0, 1.0, 0.0), speed);
        
    }
    if winit_input.key_pressed(winit::event::VirtualKeyCode::R) {
        renderer.temp_model.reset();
        renderer.camera.reset();
    } */

    let mut move_speed = 0.0;
    if input.held_shift() {
        move_speed = 25.0;
    } else {
        move_speed = 1.0;
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Camera Movement
    ///////////////////////////////////////////////////////////////////////////////////////////////
    
    // Calculate the position movement relative to the camera's orientation
    // Apply the movment to all objects in the opposite direction of the camera (they should exist in the camera's local space)

    // each vector corresponds to the axis it will rotate on
        // Ex: x = rotate on x axis
    let mut rot = glam::Vec3::new(0.0, 0.0, 0.0);

    // Rotate around Y axis (UP) (Look left / right)
    if input.key_held(event::VirtualKeyCode::Left) {
        rot.y = -1.0;
    }
    if input.key_held(event::VirtualKeyCode::Right) {
        rot.y = 1.0;
    }
    // Rotate around X axis (RIGHT) (Look up / down)
    if (input.key_held(event::VirtualKeyCode::Down)) {
        rot.x = -1.0;
    }
    if (input.key_held(event::VirtualKeyCode::Up)) {
        rot.x = 1.0;
    }
    // Rotate around Z axis (FORWARD) (Tilt left / right)
    if input.key_held(event::VirtualKeyCode::Q) {
        rot.z = -1.0;
    }
    if input.key_held(event::VirtualKeyCode::E) {
        rot.z = 1.0;
    }


    let invert_x = false;
    let invert_y = true;
    // query the change in cursor this update
    // VISIT THIS, CHANGE METHOD NAME AND ETC ETC
    // instead of setting the view direction, change the increment each frame?
    if input.mouse_held(0) {
        let cursor_diff = input.mouse_diff();
        if cursor_diff != (0.0, 0.0) {
            //camera.set_view_direction(vec3((cursor_diff.0), (cursor_diff.1), 0.0));
            rot.x = match invert_x {
                true => cursor_diff.1,
                false => -cursor_diff.1,
            };
            rot.y = match invert_y {
                true => cursor_diff.0,
                false => -cursor_diff.0,
            };

            //println!("The cursor diff is: {:?}", cursor_diff); // uses vulkan coordinates, y+ is down, y- is up
            //println!("The cursor position is: {:?}", input.mouse()); // Return mouse coordinates in pixels
            // Where window width & height are the size of the window (in pixels)
            //          (0,0)    (window width,0)
            //             ------------
            //            |     |      |
            //            |     |      |
            //            |     |      |
            //             ------------
            // (0,window height) (window width, window height)
        }
    }

    let temp_camera_speed = delta_time * 2000.0;

/*     if input.key_held(event::VirtualKeyCode::Up) {
        // cursor lock or hide or somthn idk
        camera.set_view_direction(vec3(0.0, -1.0 * temp_camera_speed, 0.0));
    }
    if input.key_held(event::VirtualKeyCode::Down) {
        // cursor lock or hide or somthn idk
        camera.set_view_direction(vec3(0.0, 1.0 * temp_camera_speed, 0.0));
    }
    if input.key_held(event::VirtualKeyCode::Left) {
        // cursor lock or hide or somthn idk
        camera.set_view_direction(vec3(-1.0 * temp_camera_speed, 0.0, 0.0));
    }
    if input.key_held(event::VirtualKeyCode::Right) {
        // cursor lock or hide or somthn idk
        camera.set_view_direction(vec3(1.0 * temp_camera_speed, 0.0, 0.0));
    } */

    if (rot != glam::Vec3::ZERO) {
        camera.rotate_quat(rot,false);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    /// Linear Movement
    ///////////////////////////////////////////////////////////////////////////////////////////////

    let mut translation = glam::Vec3::ZERO;

    if input.key_held(winit::event::VirtualKeyCode::W) {
        translation.z += 1.0;
        //camera.translate(vec3(0.0, 0.0, 1.0));
    }

    if input.key_held(winit::event::VirtualKeyCode::S) {
        translation.z -= 1.0;
        //camera.translate(vec3(0.0, 0.0, -1.0));
    }

    if input.key_held(winit::event::VirtualKeyCode::A) {
        //camera.translate(vec3(-1.0, 0.0, 0.0));
        translation.x -= 1.0;
    }

    if input.key_held(winit::event::VirtualKeyCode::D) {
        //camera.translate(vec3(1.0, 0.0, 0.0));
        translation.x += 1.0;
    }

    // In Vulkan clip-space, y- is up
    if input.key_held(winit::event::VirtualKeyCode::Space) {
        //camera.translate(vec3(0.0, 1.0, 0.0));
        translation.y += 1.0;
    }

    // Vulkan clip-space, y+ is down
    if input.key_held(winit::event::VirtualKeyCode::C) {
        //camera.translate(vec3(0.0, -1.0, 0.0));
        translation.y -= 1.0;
    }

    // ❗ Delta frame movement, or per frame movement?
    camera.translate(translation * 0.01 * move_speed); 

    if input.key_pressed(winit::event::VirtualKeyCode::R) {
        camera.reset();
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// OLD METHODS
///////////////////////////////////////////////////////////////////////////////////////////////////

// Look at input.update as a reference, from winit_input_helper
fn update<T>(input: &mut WinitInputHelper, event: &Event<T>) {
    if input.update(&event) {
        if input.key_pressed(winit::event::VirtualKeyCode::W) {
            println!("The 'W' key (US layout) was pressed on the keyboard");
        }

        if input.key_held(winit::event::VirtualKeyCode::R) {
            println!("The 'R' key (US layout) key is held");
        }

        // query the change in cursor this update
        let cursor_diff = input.mouse_diff();
        if cursor_diff != (0.0, 0.0) {
            println!("The cursor diff is: {:?}", cursor_diff);
            println!("The cursor position is: {:?}", input.mouse()); // Return mouse coordinates in pixels
        }
    }
}

// From Easy Input
// https://dhghomon.github.io/easy_rust/Chapter_63.html
fn sample_input() {
    println!("Please type something, or x to escape:");
    let mut input_string = String::new();

    while input_string.trim() != "x" {
        input_string.clear();
        std::io::stdin().read_line(&mut input_string).unwrap();
        println!("You wrote {}", input_string);
    }
    println!("See you later!");
}

fn get_text_input() -> String {
    println!("Please type something, or x to escape:");
    let mut input_string = String::new();

    while input_string.trim() != "x" {
        input_string.clear();
        std::io::stdin().read_line(&mut input_string).unwrap();
        println!("You wrote {}", input_string);
    }
    println!("See you later!");
    input_string
}

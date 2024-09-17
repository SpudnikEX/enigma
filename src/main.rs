// Files need to be defined here?

use std::{
    ops::AddAssign,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use glam::vec3;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{self, ControlFlow, EventLoop},
};

// Folders
mod examples;

// Files
mod input;
pub mod loader;
mod render;
mod window;

// other
mod camera;
mod entity;
mod model;
mod texture;
mod settings;

fn load_model_and_texture(renderer: &mut render::Renderer, model_path: &str, tex_path: &str, translate: glam::Vec3, scale: glam::Vec3) {
    //"models/tests/minecraft-player-rigged.gltf"
    let mut new_model = crate::model::Model::new(model_path,"default-black");
    //new_model.rotate(glam::vec3(1.0,0.0,0.0), 90.0, 0.0, true);

    // Load textures for model
        // Check if texture exists in registry
    let (loaded, index) = renderer.textures.check_registry(tex_path);
    if !loaded {
        let tex_img_samp = renderer.create_imageview_sampler(tex_path);
        renderer.textures.global_textures.push(tex_img_samp);

    }

    new_model.scale(scale);
    new_model.translate(translate, 1.0);
    new_model.texture_index_data.push(index);
    renderer.models.push(new_model);
    println!("textures entries: {:?} | keys entries {:?}", renderer.textures.global_textures.len(), renderer.textures.texture_registry.len()); // 🚫 DEBUG
}


fn main() {

    // for printing messages inside main
    let debug_print = false;

    let event_loop = EventLoop::new();
    let mut renderer = render::Renderer::new(&event_loop);

    let mut winit_input = winit_input_helper::WinitInputHelper::new(); // leave this here? move into main loop?

    // The time between frames
    let mut delta_time = 0.0;

    let mut last_instant = Instant::now();

    // setup geometry beforehand?? 
    // Either use a vec in renderer class or use seperate functions in the renderer (i dont want to change too much)

    
    // stretched debug steve
    load_model_and_texture(&mut renderer, "models/tests/minecraft-player-rigged.gltf", "models/tests/steve-classic.png", glam::vec3(0.0,25.0,25.0), glam::vec3(1.0,1.0,1.0));
    load_model_and_texture(&mut renderer, "models/tests/minecraft-player-rigged.gltf", "models/tests/steve-classic.png", glam::vec3(0.0,25.0,25.0), glam::vec3(40.0,25.0,40.0));

    load_model_and_texture(&mut renderer, "models/tests/minecraft-player-rigged.gltf", "models/tests/alex.png", glam::vec3(3.0,0.0,3.0), glam::vec3(1.0,1.0,1.0));//models/tests/femalehead.png & models\tests\0001990F2.gltf
    
    // plane
    load_model_and_texture(&mut renderer, "models/tests/heightmap-hills-test-modifier.gltf", "models/tests/brick.png", glam::vec3(0.0,5.0,0.0),glam::vec3(1.0,1.0,1.0) * 500.0);
    load_model_and_texture(&mut renderer, 
        "models/tests/cube.gltf",
         "models/tests/brick.png", 
         glam::vec3(0.0,35.0,70.0), 
    glam::vec3(100.0, 100.0, 100.0));

    // entities
    // components
    // systems

    ///////////////////////////////////////////////////////////////////////////////////////////////
    /// Main Loop
    ///////////////////////////////////////////////////////////////////////////////////////////////
    // ❗ Need to call function either here or in renderer to setup textures 

    // See winit github examples, and winit documentation, for templates
        // control_flow defaults to poll (See ControlFlow), also see https://docs.rs/winit/0.25.0/winit/#event-handling
    event_loop.run(move |event, _, control_flow| {
        // current iteration is based on vulkano vsync for loop update timings
        // I have yet to implement something for slower framerates (accumulator?)

        //*control_flow = ControlFlow::Poll;

        // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
        // dispatched any events. This is ideal for games and similar applications.
        // Default is Poll
        //control_flow.set_poll();

        // ControlFlow::Wait pauses the event loop if no events are available to process.
        // This is ideal for non-game applications that only update in response to user
        // input, and uses significantly less power/CPU time than ControlFlow::Poll.
        //control_flow.set_wait();


        // Lockstep. Update then render per frame. (rework if networking later on?)
            // What if I would like to render per frame instead of delta frame? (like old gameboy games)
            // Update everything based on frame by frame (GBA games?) or by using deltas (smooth?)
            // frame by frame calculations or use delta time for calculations?
            // Interpolate with delta time, just calculate frame by frame (like GBA)? for a non networked game i think that would be fine

        // high precision delta time
        let dt = last_instant.elapsed();
            // .00285 = ~2 milliseconds,
            // formula for framerate update is, every 1/60 = .0166
        delta_time = last_instant.elapsed().as_secs_f32(); 
        last_instant = Instant::now();

        if (dt.as_millis() > 1 && debug_print) { println!("Loop Step: New loop {:?}ms | {:?}", dt.as_millis(), dt); }
        

        // does testing need to be reported in milliseconds?
            // save testing into ints, hopefully instants use that( will need to test)

        // get cumulative time for sin and cos (pass shader functions)
/*         let start = SystemTime::now();
        let since_the_epoch = start
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_secs();
        println!("{:?}", since_the_epoch); */

        // This is essentially the same as the below match statement, abstracted into another crate to save time.
        
        if winit_input.update(&event) {
            if (debug_print) { println!("Loop step: Update Input"); }
            

/*             if winit_input.key_held(winit::event::VirtualKeyCode::W) {
                renderer
                    .camera
                    .translate(glam::vec3(0.0, 0.0, 1.0), 50.0 * delta_time);

                println!("{}", renderer.camera.position.to_string());
            } */

            crate::input::update_input(&winit_input, &mut renderer.camera, delta_time);
            //❗renderer.models[0].rotate(glam::Vec3::Y, 1.0, 1.0, true);
        }
       

        // Main loop event //
        // Input
        // Logic
        // Rendering
        // Sleep
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                //renderer.recreate_swapchain = true;
                    // Do i still need this? recreate is set post frame process
                    // This was the cause of the issue earlier, that the image was not presented on start
                    // write better when not sleepy
            }
            Event::MainEventsCleared => {
                // The main game loop
                if (debug_print) { println!("Loop Step: Render"); }
                

                // ! will need to restruture for ECS later on
                // move camera out of render.rs, and move into its own ecs system? will need basic ECS later

                // ECS logic to go here (update all)

                // Do game logic in the event loop / input loop

                // start
                // add geometry to pass
                    // add in instanced things in here somwhere (later)
                // end

                renderer.start(&delta_time);


                // https://github.com/rust-windowing/winit/discussions/3377
            }
            Event::RedrawEventsCleared => {
            }
            _ => (),
        }
    });
}

// let mut player = entity::Player::new();
// player.check();
// player.hit();
// player.check();


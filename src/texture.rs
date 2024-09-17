
// // need to store samplers and imageviews inside same struct

// /* 
// Moved here from main render.rs file.
// See vulkano/imgae/main.rs for the example I followed.
// */


// use vulkano::image::{sampler::Sampler, view::ImageView};

// struct Texture {
//     // For single 
//     texture_imageview: Arc<ImageView>, 
//     texture_sampler: Arc<Sampler>, 

//     // For multiple texture loading
//     texture_imageview_sampler: Vec<(Arc<ImageView>, Arc<Sampler>)>, 
// }

// impl Texture {
//     pub fn new() -> Self {
//         // ❓ Check / keep track if texture is already loaded
//         Self {
//             texture_imageview_sampler: Self::get_texture(),
//         }
//     }

//     fn get_texture() -> (Arc<ImageView>, Arc<Sampler>) {

//     }
// }

//     // First iteration for textures and sampler
// 
use std::sync::Arc;

use cgmath::num_traits::ToPrimitive;
use gltf::json::extensions::texture;
use vulkano::image::{sampler::Sampler, view::ImageView};



pub struct Texture {
    /// Global Vec to hold ImageView & Sampler
    pub global_textures: Vec<(Arc<ImageView>, Arc<Sampler>)>,
    pub texture_registry: Vec<String>,
    /// Path Strings (or UUID) that link model indices to global texture Vec. Used to assign index when loading a new texture.
    pub id: Vec<String>,// Or values. ❓Path string or checksum? Checking against embedded image vs custom path?

}

impl Texture {
    pub fn new() -> Self {
        Self {
            global_textures: Default::default(),
            texture_registry: Vec::new(),
            id: Vec::new(),
        }
    }

    // Try loading new texture, return index(s) for model reference
    pub fn add_texture(&mut self, data: (Arc<ImageView>, Arc<Sampler>)) -> u32 {
        // Check if texture "reference" exists in keys
            // ❗ Determine later if I should use absolute paths or checksums for embedded textures
        // for (index, key) in self.keys.iter().enumerate() {
        //     if key == value {
        //         return index as u32;
        //     }
        // }

        // If not, create and insert into keys and global
        //self.keys.push(value.to_string());
        self.global_textures.push(data);
        //self.create_imageview_sampler();

        // Return index for texture
        0
    }

    /// Check registry if texture is already loaded
    /// 
    /// Return if value is in registry, and the index of the value.
    /// 
    /// Create new entry in keys, to be created by renderer
    pub fn check_registry(&mut self, tex_data: &str) -> (bool, u32) {
        // Check if in registry.
        // Return true and index if found.

        let index = self.texture_registry.iter().position(|r| r == tex_data);
        // return the option? or return index only?

        // need to return index AND if value is present, for creating imageview/samplere

        match index {
            // If found, return the index inside registry for model linking.
            Some(_) => {
                println!("Duplicate texture found: {:?} | index: {:?}", tex_data, index.unwrap());
                return (true, index.unwrap() as u32)
            },
            // If not found, create new entry in registry, and return index
            None => {
                self.texture_registry.push(tex_data.to_string()); // Create entry in registry
                let i = self.texture_registry.len()-1; // Get index of entry in registry
                return (false, i as u32); 
            },
        }

        // If not found, create new entry (keys & texture_registry) and index for model use. 
        // Remember to create imageview & sampler if false.
    }

    /// UNUSED RN
    pub fn assign_registry(&mut self, tex_data: &str) {

    }
}
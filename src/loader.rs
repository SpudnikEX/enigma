use std::mem::size_of_val;
use std::ops::{Deref, DerefMut};
use std::process::abort;

use gltf::accessor::{self, util, Iter};
use gltf::mesh::util::{ReadIndices, ReadTexCoords};
use gltf::Gltf;

use crate::model::Model;


// Struct for object loading
// positions, faces, indices, normals, uvs, joints, etc
// Or should that be in model struct? WIll need to revisit once implementing ECS

// An abstraction for loading glTF models to use with hardware rendering (OpenGL / Vulkan)
// Can either import from gltf or glb.
// glb will be more compressed, but gltf is easier for people to read.

/// Reference https://docs.rs/gltf/latest/gltf/mesh/index.html for loading verticies for meshes
/// Loads / reads vertex colors, vertex indices, joints, morph targets, vertex normals, vertex positions, tangents, texture coordiantes, and vertex weights
// Other references: https://stackoverflow.com/questions/75846989/how-to-load-gltf-files-with-gltf-rs-crate

// For model loading, reference https://taidaesal.github.io/vulkano_tutorial/section_9.html

pub fn load_example() {
    let model_path = "src/models/tests/pyramid.gltf";

    // gltf import vs open? vs from reader?
    let (gltf, buffers, images) = gltf::import(model_path).expect("failed to load glTF model");
    for mesh in gltf.meshes() {
        println!("Mesh #{}", mesh.index());
        for primitive in mesh.primitives() {
            println!("- Primitive #{}", primitive.index());
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            if let Some(iter) = reader.read_positions() {
                // returns an iter containing array of f32
                for (vertex_position) in iter {
                    // iterate over the iterator, working on each element in the iterator
                    println!("vertex_position: {:?}", vertex_position);
                }
            };
            if let Some(gltf::mesh::util::ReadIndices::U16(accessor::Iter::Standard(iter))) =
                reader.read_indices()
            {
                // returns util::ReadIncies
                for (vertex_index) in iter {
                    println!("vertex_index: {:?}", vertex_index);
                }
            };
        }
    }

    return; // TEMPORARY Early return
    let gltf = gltf::Gltf::open(model_path).expect("failed to load glTF model");

    for scene in gltf.scenes() {
        println!("Scene Name: {:?}", scene.name().unwrap());
        for node in scene.nodes() {
            println!(
                "Node {:?} #{} has [{}] children",
                node.name().unwrap(),
                node.index(),
                node.children().count(),
            );
        }
    }
}

/// Get files by extension inside directory
/// - `dir`: The directory to search, relative to the file system
// https://stackoverflow.com/questions/62023605/how-can-i-filter-a-list-of-filenames-by-their-extension
fn get_gltf_paths(dir: &str) -> Result<Vec<std::path::PathBuf>, Box<dyn std::error::Error>> {
    let paths = std::fs::read_dir(dir)?
        // Filter out all those directory entries which couldn't be read
        .filter_map(|res| res.ok())
        // Map the directory entries to paths
        .map(|dir_entry| dir_entry.path())
        // Filter out all paths with extensions other than `csv`
        .filter_map(|path| {
            if path.extension().map_or(false, |ext| ext == "gltf") {
                Some(path)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    Ok(paths)
}

// https://www.reddit.com/r/rust/comments/pqq8p6/comment/ias6dav/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
    // https://github.com/PikminGuts92/grim/blob/master/core/grim/src/model/gltf.rs
        // Web page saved
// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#meshes
    // Gltf data is more than likely encoded as Base64 format
pub fn load_gltf(model_name: &str) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32;2]>, Vec<u16>) {

    let debug_print = false; // Prints debug info below & exits process

    // ❗ temp external model loading for jason
    let result = match model_name.is_empty() {
        true => {
            let binding = get_gltf_paths("./").unwrap();
            let mut result = binding.first().unwrap().to_str().unwrap();
            "./minecraft-player-rigged.gltf"
        },
        false => {
            model_name
            // let directory = "../models/tests/"; // format! begins in current directory, need to travel up
            // //let directory = "models/tests/";
            // let extension = ".gltf";
            // // models/tests/cube.gltf
            // //let mut full_path = format!("{directory}{model_name}{extension}"); // combine multiple variable strings together
            // format!("{directory}{model_name}{extension}") 
        },
    };
    
    println!("LOADED MODEL: {:?}", result);

    let mut full_path = result;//result;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    /// DEBUG
    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Some gltf models need a .bin file, otherwise will produce an error "The system cannot find the path specified."
    // If a bin is provided, import into blender and export both as one .gltf file
    println!("DEBUGGING {:?}", full_path);
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u16> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs = Vec::new();

    ///////////////////////////////////////////////////////////////////////////////////////////////

    let (gltf, buffers, images) = gltf::import(full_path).expect("failed to load glTF model");
    
    // ❗ Unfinished! need to add support for normals and multiple embedded / custom path textures
    // Following vulkano image/main.rs example
        // Load embedded images from GLTF
    for image in images {
        //println!("Image Format: {:?} | Dimensions: {:?}w {:?}h", image.format, image.width, image.height);
        //let pixel_data: Vec<u8> = image.pixels; //Vec<u8>, same as loading a file to a byte array (include_bytes!().as_vec())
        //image.pixels //The image pixel data (8 bits per channel).

        // check runtime memory of values, compare image size vs string size

        // ❗ THIS TANKS PERFORMANCE, ONLY USE FOR DEBUGGING
            // Compare byte size of file vs path vs checksum
        // println!("length: {:?} in bytes", size_of_val(pixel_data.deref()));
        // println!("The string size is {} in bytes", size_of_val("model/tests/minecraft-player-rigged.gltf"));
        // println!("The hash size is {} in bytes", size_of_val("641094577ee247344afbfd7df3bd01e8b291a5a7"));
    }


    /* CANNOT USE THESE (as of 2024-09-13), all values return NONE, even if there is an image attached
    // ❗ This will only work for gltf models, custom textures will need alternate checking / loading
        for tex in gltf.textures() {
        println!("tex: {:?}", tex.name());
    }

    for sampler in gltf.samplers() {
        println!("sampler: {:?}", sampler.name());
    }

    for view in gltf.views() {
        println!("view: {:?}", view.name());
    }
     */

    for mesh in gltf.meshes() {
        println!("- Mesh #{}", mesh.index());
        for primitive in mesh.primitives() {
            println!("|- Primitive #{}", primitive.index());
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            /*
            Template match statement from rust docs https://doc.rust-lang.org/reference/expressions/match-expr.html
            let message = match maybe_digit {
                Some(x) if x < 10 => process_digit(x),
                Some(x) => process_other(x),
                None => panic!(),
            };
             */

            // per vertex data
            positions = reader.read_positions().unwrap().collect(); // outputs xyz for single verts
            normals = match reader.read_normals() {
                Some(x) => x.collect(),
                None => Vec::new(),
            };
            
            uvs = reader.read_tex_coords(0)
            .map(|tc| tc.into_f32()
            .collect::<Vec<_>>())
            .unwrap_or_else(|| (0..positions.len()).map(|_| Default::default()).collect::<Vec<_>>());

            indices = match reader.read_indices().unwrap() {
                ReadIndices::U8(itr) => itr.map(|i| i as u16).collect(),
                ReadIndices::U16(itr) => itr.collect(), // prefer to collect as u16
                ReadIndices::U32(itr) => itr.map(|i| i as u16).collect(),
            };

            let faces_chunked = indices.chunks_exact(3);
            let faces: Vec<[u16; 3]> = faces_chunked
            .map(|f| [
                *f.get(2).unwrap(), // Clockwise -> Anti
                *f.get(1).unwrap(),
                *f.get(0).unwrap(),
            ])
            .collect();


            if debug_print {
                // for pos in &positions {
                //     println!("Vertex local position {:?}", pos);
                // }
                // for index in &indices {
                //     println!("Vertex Index {:?}", index);
                // }
                // for norm in &normals {
                //     println!("Normals: {:?}", norm);
                // }
                // for uv in &uvs {
                //     println!("UVs: {:?}", uv);
                // }
                println!(" |-Vertex Length: {:?}", &positions.len());
                println!(" |-Index Length {:?}", &indices.len());
                println!(" |-Normals Length {:?}", &normals.len());
                println!(" |-UVs Length {:?}", &uvs.len());
            }
            break; // break, only load 1 mesh
        }
        break; // break, only load 1 mesh
    }
    

    // Debug print
    // Researched 2024-07-16, outputs one vertex position for every
        // tested with pyramid, no uv or normal or color
        // tested with pyramid, flat uv flat normals
        // tested with pyramid, smoothed uv smoothed normals
        // The maximum number of vertices that could be required for a particular mesh is: (triangleCount * 3), or 6 * 2 * 3 = 36 for a basic cube



    //exit after loading model  
        // exit 0 = success, exit 1 = error
    if debug_print {std::process::exit(0);} 


    (positions, normals, uvs, indices)
}

    /*
    // The following is an example of gltf loading using the gltf crate.
    // Load model data into collection (vector of structs) "Vertex_3D", for use with vulkano
    // Focus on uploading positions & colors, then work on indices next
    // https://github.com/taidaesal/vulkano_tutorial/blob/gh-pages/lessons/9.%20Model%20Loading/src/obj_loader/loader.rs
    */
// pub fn as_normal_vertices() -> Vec<crate::model::Vertex_3D> {
//     let mut vertex_data: Vec<crate::model::Vertex_3D> = Vec::new();
//     for face in &self.faces {
//         let verts = face.verts;
//         let normals = face.norms.unwrap();
//         // ret.push(crate::model::Vertex_3D {
//         //     position: self.verts.get(verts[0]).unwrap().vals,
//         //     //normal: self.norms.get(normals[0]).unwrap().vals,
//         //     color: self.color,
//         // });
//         // ret.push(crate::model::Vertex_3D {
//         //     position: self.verts.get(verts[1]).unwrap().vals,
//         //     //normal: self.norms.get(normals[1]).unwrap().vals,
//         //     color: self.color,
//         // });
//         // ret.push(crate::model::Vertex_3D {
//         //     position: self.verts.get(verts[2]).unwrap().vals,
//         //     //normal: self.norms.get(normals[2]).unwrap().vals,
//         //     color: self.color,
//         // });
//         vertex_data.push(crate::model::Vertex_3D {
//             position: [0.0, 0.0, 0.0],
//             color: [0.0, 0.0, 0.0, 1.0]
//         });
//     }
//     vertex_data
// }

/*
if let Some(iter) = reader.read_positions() {
    for (vertex_position) in iter {
        println!("vertex_position: {:?}", vertex_position);
    }
};
if let Some(gltf::mesh::util::ReadIndices::U16(accessor::Iter::Standard(iter))) =
    reader.read_indices()
{
    // returns util::ReadIncies
    for (vertex_index) in iter {
        println!("vertex_index: {:?}", vertex_index);
    }
};
*/


/*
// https://www.reddit.com/r/rust/comments/pqq8p6/reading_buffers_for_a_gltf_model_via_crate/
fn read_primitive(&mut self, prim: &Primitive, mesh_name_prefix: &str) -> MeshObject {
    let reader = prim.reader(|buffer| Some(&self.buffers[buffer.index()]));

    let faces: Vec<u16> = match reader.read_indices().unwrap() {
        ReadIndices::U8(itr) => itr.map(|i| i as u16).collect(),
        ReadIndices::U16(itr) => itr.collect(),
        ReadIndices::U32(itr) => itr.map(|i| i as u16).collect(),
    };

    let faces_chunked = faces.chunks_exact(3);

    let faces: Vec<[u16; 3]> = faces_chunked
        .map(|f| [
            *f.get(2).unwrap(), // Clockwise -> Anti
            *f.get(1).unwrap(),
            *f.get(0).unwrap(),
        ])
        .collect();

    let verts_interleaved = izip!(
        reader.read_positions().unwrap(),
        reader.read_normals().unwrap(),
        //reader.read_colors(0).unwrap().into_rgb_f32().into_iter(),
        reader.read_tex_coords(0).unwrap().into_f32(),
    );

    let verts = verts_interleaved
        .map(|(pos, norm, uv)| Vert {
            pos: Vector4 {
                x: match pos.get(0) {
                    Some(p) => *p,
                    _ => 0.0,
                },
                y: match pos.get(1) {
                    Some(p) => *p,
                    _ => 0.0,
                },
                z: match pos.get(2) {
                    Some(p) => *p,
                    _ => 0.0,
                },
                ..Vector4::default()
            },
            normals: Vector4 {
                x: match norm.get(0) {
                    Some(n) => *n,
                    _ => 0.0,
                },
                y: match norm.get(1) {
                    Some(n) => *n,
                    _ => 0.0,
                },
                z: match norm.get(2) {
                    Some(n) => *n,
                    _ => 0.0,
                },
                ..Vector4::default()
            },
            uv: UV {
                u: match uv.get(0) {
                    Some(u) => *u,
                    _ => 0.0,
                },
                v: match uv.get(1) {
                    Some(v) => *v,
                    _ => 0.0,
                },
            },
            ..Vert::default()
        })
        .collect::<Vec<Vert>>();

    let mat_name = match prim.material().index() {
        Some(idx) => self.mats[idx].name.to_owned(),
        None => String::from(""),
    };

    let mesh_name = match prim.index() {
        0 => format!("{}.mesh", mesh_name_prefix),
        _ => format!("{}_{}.mesh", mesh_name_prefix, prim.index()),
    };

    MeshObject {
        name: mesh_name.to_owned(),
        vertices: verts,
        faces,
        mat: mat_name,
        geom_owner: mesh_name,
        parent: String::default(),
        ..MeshObject::default()
    }
}
*/

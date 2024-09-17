// Code holding data for models go here
use glam::{vec3, vec4, Mat4, Vec3, Vec4};
use vulkano::buffer::BufferContents;
use vulkano::pipeline::graphics::vertex_input::Vertex;

/// This is a sample vertex for uploading a 2d position
/// For more examples, check the vulkano repo
#[derive(BufferContents, Vertex, Clone, Copy)] // added copy and clone to trait
#[repr(C)]
pub struct Vertex_2D {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 2],
}

/// The vertex related data passed to shaders.
/// Vectors `vec3` & `vec4` are passed in as arrays of floats `f32`.
///
/// For available formats, see https://docs.rs/vulkano/latest/vulkano/format/enum.Format.html#variant.R32G32B32_UINT
#[derive(BufferContents, Vertex, Clone, Copy)] // added copy and clone to trait
#[repr(C)]
pub struct Vertex_3D {
    // 🟠 Remember to update vertex shaders to accept newly added vertex data
    /// Vertex position within the model's local space
    /// The local position of the vertex in respect to the model matrix
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
    /// The color of the vertex, used in color blending between vertices
    #[format(R32G32B32A32_SFLOAT)]
    color: [f32; 4],
    /// ❓ unneeded later? Currently hard coded normals of model, can be replaced with image
    #[format(R32G32B32_SFLOAT)]
    normal: [f32; 3],
    /// 0.0-1.0 per vertex coordinates, as array of 32-bit floats (GLSL standard)
    #[format(R32G32_SFLOAT)]
    uv: [f32; 2],
    // DATA THAT IS UPLOADED TO SHADER FOR PER VERTEX INFORMATION, DO NOT INCLUDE INDEX
    // Keep index inside the vert, or outside and use with index buffer?
}

pub struct Model {
    /*
    model_matrix: the model matrix, position of the model in camera / world space
    vertex_position: The positions of vertices in local space of the model
    vertex_normal: The baked normals for each vertex??
    vertex_color: The baked color for each vertex, can be used in blending colors
     */
    /// The vertices inside the model.
    ///
    /// Type used when initalizing the `vertex_input_state` to tell the shader the incoming data types, per vertex.
    //pub vertices: [Vertex_3D; 6], // array of unspecified size (procedural?) or loading from the file? (!!! Rename this to something more appropriate like "data")

    /// Collection of Vertex_3D's that hold per vertex information such as
    /// Can also be tought of as `Mesh Data`
    /// The per vertex data sent to vertex and fragment shaders through `vertex_buffer` and `vertex_input_state`
    /// - local position
    /// - index
    /// - UV
    /// - ETC
    pub vertex_data: Vec<Vertex_3D>, // Vec<Vertex_3D> instead of [Vertex_3D] because of unknown size when loading models

    // vertices struct actually needs to be renamed to something like "data", because it holds more than just position
    /// Model to World Matrix.
    /// The M matrix of MVP. Send inside of uniform descriptor.
    /// The local rotation and position(in reference to the camera or world (havent decided)) of the model
    pub model_matrix: Mat4, //column major matrix
    //could also be quaternion & position?

    // Model Data
    // vertices
    // texture coordinates (UV values)
    // vertex normals
    // Face, which lists what vertices make up the face (used in indexed vertices?)

    // Buffers (unsure if they need to be here or in rendering?)
    // Vertex Buffer and Uniform
    angle: f32,            //temp debug angle
    pub indices: Vec<u16>, // in all vulkano repo examples, u16 is the data type that is used in subbuffers

    // Png crate reads image data as as_slice and to_vec
        // as_slice is &[u8], to_vec is Vec<u8>
        // GLTF loads images as Vec<u8> ("The image pixel data (8 bits per channel)."). See loader.rs > image.pixels

    // ❗ Texture Index Data, points to the data in the global texture vec.
    pub texture_index_data: Vec<u32>, 
    //pub texture_name: Vec<String>, // ❗ TEMP!! remove for index data later

    // ❓ how should this be strucutre?
    // string for custom texture to be loaded? or the data that is within the GLTF?
    // better layout for contiguous memory?
    //pub textures: Vec<>
}

impl Model {
    /// Initalize new model, read vertex and textures data from GLTF file format
    pub fn new(model_name: &str, texture_name: &str) -> Self {
        // Load in this order:
            // Textures: Check custom path, then check embedded image, then load default image
            // ❗ Models: Check internal files, then check files in directory (for Jason)?

        // w component set to 1 
            // the fourth component of each axis corresponds to position
        let model_matrix = Mat4::IDENTITY;

        // load from file
        // use data from file to populate vertex struct
        let (vertex_data, indices) = Self::get_data(model_name); //Self::get_debug_tri(false);  //Self::get_debug_cube();//

        // check if string is empty, texture_name.is_empty()
        // let texture_data: Vec<u32> = match texture_name { 
        //     // builtin rust macro for loading file / image data
        //             //You can't use include_bytes!() macro with runtime constructed path.
        //             // use include_bytes! for static / embedded files
        //             // use std::fs::read() for files that are only known at runtime
        //     // ❓ rework texture loading from central source
        //     "" => {
        //         // include_bytes executes from the directory of the current file.
        //         include_bytes!("../models/tests/default-white.png").to_vec() // load default image if empty
        //     }, 
        //     _ => {
        //         // fs::read executes from the root of the project.
        //         let path = format!("./models/tests/{texture_name}.png"); // combine strings together
        //         std::fs::read(path).unwrap()
        //     }
        // };

        Self {
            //vertices: vertices,
            vertex_data: vertex_data,
            indices: indices,
            model_matrix: model_matrix, //model_matrix, // Matrix without rotation
            texture_index_data: Vec::new(),
            angle: 0.0,
        }
    }

    /// Get and set data inside vertex struct, to be sent to the GPU for per vertex calculations.
    /// Returns:
    /// - `Vec<Vertex_3D>`: The per-vertex associated data for the loaded model. 
    /// - `Vec<u16>`: The index list for the loaded model.
    /// - `Vec<u8>`: Single texture data ❗
    fn get_data(model_name: &str) -> (Vec<Vertex_3D>, Vec<u16>) {
        // Load data from GLTF file
        // Put Positions, Normals, and UV in vertex_data
        // Index data does NOT belong in per vertex_data
            // Remove upside-down hack, re-worked into flipping the viewport.

        // ❗ RESTRUCTURE LATER
        // Note to self, on research, imported models will have the same length for normals and positions
            // Vertex, Normals, and UV length will be the same length
            // For clarification, the length for both normals and positions Vec<> will be of the same length
        let (positions, normals, uvs, indices) =
            crate::loader::load_gltf(model_name);

        // The length of positions, uv's, and normals SHOULD be equal. Double check here just in case?
        if (positions.len() != normals.len() && positions.len() != uvs.len()) {
            println!("Position Length: {:?}, UV Length: {:?}, Normal Length: {:?}, indices Length: {:?}", positions.len(), normals.len(), uvs.len(), indices.len());
            std::process::exit(1);
        }

        // Create new Vec array to hold Vertex_3D data.
        let mut vertex_data: Vec<Vertex_3D> = Vec::new(); 

        

        // the length of positions, uv's, and normals SHOULD be equal (print debug data for each to review)
        //https://stackoverflow.com/questions/66288515/how-do-i-get-the-index-of-the-current-element-in-a-for-loop-in-rust
        for (index, element) in positions.iter().enumerate() {
            let color = glam::Vec3::from_array(positions.get(index).unwrap().clone()).normalize(); //❗ not actually the current exported vertex color

            vertex_data.push(Vertex_3D {
                position: positions.get(index).unwrap().clone(),
                color: [color.x, color.y, color.z, 1.0], // ❓ unneeded for models using texture for colors. All other could potentially use this.
                normal: match normals.get(index) {
                    Some(value) => value.clone(),
                    None => [0.0, 0.0, 0.0],
                },
                uv: match uvs.get(index) {
                    Some(value) => value.clone(),
                    None => [0.0, 0.0],
                },
                // ❗ Double Check Later, double check mehes are using indexed and winding order
            });
        }
        // ❓ iterate over entire array once?
        // for now, iterate over entire array twice (contiguous memory)

        (vertex_data, indices)
    }

    ///
    pub fn rotate(&mut self, axis: glam::Vec3, speed: f32, delta: f32, local: bool) {
        // rotate around self (local) or around
        // give option to pass an arbitrary axis

        // angle would be named degrees / radians, needs to be speed because of multiplying by delta time

        // First scale, then rotate, then translate

        // there is probably a better way to do this
        // No need to store a cumulative angle if multiplying the existing matrix. The matrix HAS the angle already
        // Multiplying the two matricies is akin to adding two rotations together

        // Scale then rotate then translate

        if local {
            // Rotate on local axis
            self.model_matrix *= Mat4::from_axis_angle(axis, delta.to_radians());
        //speed * delta
        } else { // Rotate on world axis
        }

        // let temp_mat = Mat4::from_axis_angle(vec3(0.0, 1.0, 0.0), self.angle);
        // self.model_matrix.x_axis = vec4(
        //     temp_mat.x_axis.x,
        //     temp_mat.x_axis.y,
        //     temp_mat.x_axis.z,
        //     temp_mat.x_axis.w,
        // );
        // self.model_matrix.y_axis = vec4(
        //     temp_mat.y_axis.x,
        //     temp_mat.y_axis.y,
        //     temp_mat.y_axis.z,
        //     temp_mat.y_axis.w,
        // );
        // self.model_matrix.z_axis = vec4(
        //     temp_mat.z_axis.x,
        //     temp_mat.z_axis.y,
        //     temp_mat.z_axis.z,
        //     temp_mat.z_axis.w,
        // );
        // self.model_matrix.w_axis = vec4(
        //     self.model_matrix.w_axis.x,
        //     self.model_matrix.w_axis.y,
        //     self.model_matrix.w_axis.z,
        //     self.model_matrix.w_axis.w,
        // ); // Keep W at 1 or 0
        self.angle += delta * 25.0; //0.005;
    }

    /// Move objects in frame of reference to the cemara
    /// without camera translation, movments will be in clip space (test❗?)
    /// 
    /// Move models in relation to camera clip space, as if MVP matrix is (projection * Identity Camera Matrix * Model) (this description needs rework)
    /// 
    /// Model positions are in relation to clip-space coordinates
    pub fn translate(&mut self, amount: glam::Vec3, speed: f32) {
        // Matrix are laid in column major
        // Translation / positions are set in the fourth component of each axis, except the w_axis

        // is there a function that sets all three in one access?
        // self.model_matrix.x_axis.w += amount.x * speed;
        // self.model_matrix.y_axis.w += amount.y * speed;
        // self.model_matrix.z_axis.w += amount.z * speed;

        /*
        glam uses column major axis (hover over Mat4)
            Looking at axis, this is how Mat4s are composed
        x_axis: Vec4::new(m00, m01, m02, m03),
        y_axis: Vec4::new(m10, m11, m12, m13),
        z_axis: Vec4::new(m20, m21, m22, m23),
        w_axis: Vec4::new(m30, m31, m32, m33),

        Like my drawings, the resulting matrix will be (in column order)
         */

        self.model_matrix.w_axis.x += amount.x * speed;
        self.model_matrix.w_axis.y += amount.y * speed;
        self.model_matrix.w_axis.z += amount.z * speed;

        println!(" ");
        println!("x {}", self.model_matrix.x_axis);
        println!("y {}", self.model_matrix.y_axis);
        println!("z {}", self.model_matrix.z_axis);
        println!("w {}", self.model_matrix.w_axis);
    }

    /// First scale, then rotate, then translate
    pub fn scale(&mut self, scale_size: glam::Vec3) {
        self.model_matrix.x_axis.x = scale_size.x;
        self.model_matrix.y_axis.y = scale_size.y;
        self.model_matrix.z_axis.z = scale_size.z;
    }

    /// DEBUG ONLY
    pub fn reset(&mut self) {
        self.model_matrix = Mat4::IDENTITY;
    }

    fn get_debug_tri(fullscreen: bool) -> (Vec<Vertex_3D>, Vec<u16>) {
        // 🔴 Remember to remove full matrix chain combination inside vertex cshader for fullscreen debug triangle to render correctly.
        // Fill indices data...
        let indices: Vec<u16> = Vec::from([1, 0, 2]);
        // counter clockwise, 0 1 2

        // Fill per vertex data...
        let vertex_data = Vec::from([
            Vertex_3D {
                // Top Left
                position: match fullscreen {
                    true => [-1.0, -1.0, 0.0],
                    false => [-0.9, -0.9, 0.0],
                },
                color: [1.0, 0.0, 0.0, 1.0],
                normal: [0.0, 0.0, 0.0],
                uv: [0.0, 0.0], //top left is [0.0,0.0]
            },
            Vertex_3D {
                // Bottom Left
                position: match fullscreen {
                    true => [-1.0, 3.0, 0.0],
                    false => [-0.9, 0.9, 0.0],
                },
                color: [0.0, 1.0, 0.0, 1.0],
                normal: [0.0, 0.0, 0.0],
                uv: [0.0, 1.0], //bottom left is [0.0,1.0]
            },
            Vertex_3D {
                // Top Right
                position: match fullscreen {
                    true => [3.0, -1.0, 0.0],
                    false => [0.9, -0.9, 0.0],
                },
                color: [0.0, 0.0, 1.0, 1.0],
                normal: [0.0, 0.0, 0.0],
                uv: [1.0, 0.0], //top right is [1.0,0.0]
            },
        ]);
        (vertex_data, indices)
    }

    fn get_debug_cube() -> (Vec<Vertex_3D>, Vec<u16>) {
        // counter clockwise winding order
        let mut verts = Vec::new();
        let mut indices = Vec::from([
            0, 1, 3, 3, 1, 2,
            1, 5, 2, 2, 5, 6,
            5, 4, 6, 6, 4, 7,
            4, 0, 7, 7, 0, 3,
            3, 2, 7, 7, 2, 6,
            4, 5, 0, 0, 5, 1
        ]);

        verts.push(Vertex_3D{position: [-10.0, -10.0, -10.0], color: [1.0, 0.0, 0.0, 1.0], normal: Default::default(), uv: [0.0, 0.0]}); //BTL
        verts.push(Vertex_3D{position: [10.0, -10.0, -10.0], color: [0.0, 1.0, 0.0, 1.0], normal: Default::default(), uv: [1.0, 0.0]}); //BTR
        verts.push(Vertex_3D{position: [10.0, 10.0, -10.0], color: [0.0, 0.0, 1.0, 1.0], normal: Default::default(), uv: [1.0, 1.0]}); //BDR
        verts.push(Vertex_3D{position: [-10.0, 10.0, -10.0], color: [0.0, 1.0, 0.0, 1.0], normal: Default::default(), uv: [0.0, 1.0]}); //BDL
        verts.push(Vertex_3D{position: [-10.0, -10.0, 10.0], color: [1.0, 0.0, 0.0, 1.0], normal: Default::default(), uv: [1.0, 0.0]}); //FTL
        verts.push(Vertex_3D{position: [10.0, -10.0, 10.0], color: [0.0, 1.0, 0.0, 1.0], normal: Default::default(), uv: [0.0, 0.0]}); //FTR
        verts.push(Vertex_3D{position: [10.0, 10.0, 10.0], color: [0.0, 0.0, 1.0, 1.0], normal: Default::default(), uv: [0.0, 1.0]}); //FDR
        verts.push(Vertex_3D{position: [-10.0, 10.0, 10.0], color: [0.0, 0.0, 0.0, 1.0], normal: Default::default(), uv: [1.0, 1.0]}); //FDL

        (verts, indices)
        // 8 verts
        // need correct index order however
    }

    /*
    fn get_debug_vector() -> Vec<Vertex_3D> {
        // Placing these coordinates into clip space, matrix needs to be set to identity (no change)
        let mut data: Vec<Vertex_3D> = Vec::new();
        data.push(Vertex_3D {
            position: [-0.75, -0.75, 0.0],
            color: [0.0, 0.0, 0.0, 1.0],
        });
        data.push(Vertex_3D {
            position: [0.0, 1.0, 0.0],
            color: [0.0, 1.0, 0.0, 1.0],
        });
        data.push(Vertex_3D {
            position: [0.75, 0.75, 0.0],
            color: [1.0, 0.0, 0.0, 1.0],
        });

        return data;
        // ret.push(crate::model::Vertex_3D {
        //         //     position: self.verts.get(verts[0]).unwrap().vals,
        //         //     //normal: self.norms.get(normals[0]).unwrap().vals,
        //         //     color: self.color,
        //         // });
    }

    fn get_debug_compass() -> [Vertex_3D; 6] {
        // draw indexed?
        // for visualizing the three world axis orientations in camera space.
        // Will need to exist in camera space, or orient the model to point in world space ONLY
        // draw using lines?
        // Debug lines?
        // That would require a different pipeline i think
        [
            // World +X axis
            Vertex_3D {
                position: [0.0, 0.0, 0.0],
                color: [0.0, 0.0, 0.0, 1.0],
            },
            Vertex_3D {
                position: [0.0, 1.0, 0.0],
                color: [0.0, 1.0, 0.0, 1.0],
            },
            Vertex_3D {
                position: [1.0, 0.0, 0.0],
                color: [1.0, 0.0, 0.0, 1.0],
            },
            // World +Y axis
            Vertex_3D {
                position: [0.0, 0.0, 0.0],
                color: [0.0, 0.0, 0.0, 1.0],
            },
            Vertex_3D {
                position: [0.0, 1.0, 0.0],
                color: [0.0, 1.0, 0.0, 1.0],
            },
            Vertex_3D {
                position: [0.0, 0.0, 1.0],
                color: [0.0, 0.0, 1.0, 1.0],
            },
        ]
    }

    fn get_debug_cube() -> [Vertex_3D; 36] {
        let z_color = [0.0, 0.0, 1.0, 1.0];
        let y_color = [0.0, 1.0, 0.0, 1.0];
        let x_color = [1.0, 0.0, 0.0, 1.0];

        let z_neg_color = [0.0, 0.0, 0.25, 1.0];
        let y_neg_color = [0.0, 0.25, 0.0, 1.0];
        let x_neg_color = [0.25, 0.0, 0.0, 1.0];

        return [
            // Cube
            // 6 verts per face if not doing index drawing
            // clockwise drawing

            // Back Face
            Vertex_3D {
                position: [-1.0, -1.0, -1.0],
                color: z_neg_color,
            }, // top left corner
            Vertex_3D {
                position: [-1.0, 1.0, -1.0],
                color: z_neg_color,
            }, // bottom left corner
            Vertex_3D {
                position: [1.0, -1.0, -1.0],
                color: z_neg_color,
            }, // top right corner
            Vertex_3D {
                position: [1.0, -1.0, -1.0],
                color: z_neg_color,
            }, // top right corner
            Vertex_3D {
                position: [-1.0, 1.0, -1.0],
                color: z_neg_color,
            }, // top right corner
            Vertex_3D {
                position: [1.0, 1.0, -1.0],
                color: z_neg_color,
            }, // top right corner
            // Right Face
            Vertex_3D {
                position: [1.0, -1.0, -1.0],
                color: x_color,
            }, // top right corner
            Vertex_3D {
                position: [1.0, 1.0, -1.0],
                color: x_color,
            }, // top right corner
            Vertex_3D {
                position: [1.0, -1.0, 1.0],
                color: x_color,
            }, // top right corner
            Vertex_3D {
                position: [1.0, -1.0, 1.0],
                color: x_color,
            }, // top right corner
            Vertex_3D {
                position: [1.0, 1.0, -1.0],
                color: x_color,
            }, // top right corner
            Vertex_3D {
                position: [1.0, 1.0, 1.0],
                color: x_color,
            }, // top right corner
            // Front face
            Vertex_3D {
                position: [1.0, -1.0, 1.0],
                color: z_color,
            }, // top right corner
            Vertex_3D {
                position: [-1.0, 1.0, 1.0],
                color: z_color,
            }, // top right corner
            Vertex_3D {
                position: [1.0, 1.0, 1.0],
                color: z_color,
            }, // top right corner
            Vertex_3D {
                position: [1.0, -1.0, 1.0],
                color: z_color,
            }, // top right corner
            Vertex_3D {
                position: [-1.0, -1.0, 1.0],
                color: z_color,
            }, // top right corner
            Vertex_3D {
                position: [-1.0, 1.0, 1.0],
                color: z_color,
            }, // top right corner
            // Left Face
            Vertex_3D {
                position: [-1.0, -1.0, 1.0],
                color: x_neg_color,
            }, // top right corner
            Vertex_3D {
                position: [-1.0, 1.0, -1.0],
                color: x_neg_color,
            }, // top right corner
            Vertex_3D {
                position: [-1.0, 1.0, 1.0],
                color: x_neg_color,
            }, // top right corner
            Vertex_3D {
                position: [-1.0, -1.0, 1.0],
                color: x_neg_color,
            }, // top right corner
            Vertex_3D {
                position: [-1.0, -1.0, -1.0],
                color: x_neg_color,
            }, // top right corner
            Vertex_3D {
                position: [-1.0, 1.0, -1.0],
                color: x_neg_color,
            }, // top right corner
            // Top Face
            Vertex_3D {
                position: [-1.0, -1.0, 1.0],
                color: y_neg_color,
            }, // top right corner
            Vertex_3D {
                position: [-1.0, -1.0, -1.0],
                color: y_neg_color,
            }, // top right corner
            Vertex_3D {
                position: [1.0, -1.0, 1.0],
                color: y_neg_color,
            }, // top right corner
            Vertex_3D {
                position: [1.0, -1.0, 1.0],
                color: y_neg_color,
            }, // top right corner
            Vertex_3D {
                position: [-1.0, -1.0, -1.0],
                color: y_neg_color,
            }, // top right corner
            Vertex_3D {
                position: [1.0, -1.0, -1.0],
                color: y_neg_color,
            }, // top right corner
            // Bottom Face
            Vertex_3D {
                position: [-1.0, 1.0, 1.0],
                color: y_color,
            }, // top right corner
            Vertex_3D {
                position: [-1.0, 1.0, -1.0],
                color: y_color,
            }, // top right corner
            Vertex_3D {
                position: [1.0, 1.0, 1.0],
                color: y_color,
            }, // top right corner
            Vertex_3D {
                position: [1.0, 1.0, 1.0],
                color: y_color,
            }, // top right corner
            Vertex_3D {
                position: [-1.0, 1.0, -1.0],
                color: y_color,
            }, // top right corner
            Vertex_3D {
                position: [1.0, 1.0, -1.0],
                color: y_color,
            }, // top right corner
        ];
    }

    fn get_debug_prism() -> [Vertex_3D; 18] {
        // For the visualization of orientation
        // This prism should be upside down

        let z_color = [0.0, 0.0, 1.0, 1.0];
        let y_color = [0.0, 1.0, 0.0, 1.0];
        let x_color = [1.0, 0.0, 0.0, 1.0];

        let z_neg_color = [0.0, 0.0, 0.25, 1.0];
        let y_neg_color = [0.0, 0.25, 0.0, 1.0];
        let x_neg_color = [0.25, 0.0, 0.0, 1.0];

        return [
            // Back Face
            Vertex_3D {
                position: [0.0, 2.0, 0.0],
                color: z_neg_color,
            },
            Vertex_3D {
                position: [-1.0, 0.0, -1.0],
                color: z_neg_color,
            },
            Vertex_3D {
                position: [1.0, 0.0, -1.0],
                color: z_neg_color,
            },
            // Right Face
            Vertex_3D {
                position: [0.0, 2.0, 0.0],
                color: x_color,
            },
            Vertex_3D {
                position: [1.0, 0.0, -1.0],
                color: x_color,
            }, // top right corner
            Vertex_3D {
                position: [1.0, 0.0, 1.0],
                color: x_color,
            },
            // Front face
            Vertex_3D {
                position: [0.0, 2.0, 0.0],
                color: z_color,
            }, // top right corner
            Vertex_3D {
                position: [1.0, 0.0, 1.0],
                color: z_color,
            }, // top right corner
            Vertex_3D {
                position: [-1.0, 0.0, 1.0],
                color: z_color,
            },
            // Left Face
            Vertex_3D {
                position: [0.0, 2.0, 0.0],
                color: x_neg_color,
            }, // top right corner
            Vertex_3D {
                position: [-1.0, 0.0, 1.0],
                color: x_neg_color,
            }, // top right corner
            Vertex_3D {
                position: [-1.0, 0.0, -1.0],
                color: x_neg_color,
            },
            // Bottom Face
            Vertex_3D {
                position: [-1.0, 0.0, 1.0],
                color: y_color,
            }, // top right corner
            Vertex_3D {
                position: [-1.0, 0.0, -1.0],
                color: y_color,
            }, // top right corner
            Vertex_3D {
                position: [1.0, 0.0, 1.0],
                color: y_color,
            }, // top right corner
            Vertex_3D {
                position: [1.0, 0.0, 1.0],
                color: y_color,
            }, // top right corner
            Vertex_3D {
                position: [-1.0, 0.0, -1.0],
                color: y_color,
            }, // top right corner
            Vertex_3D {
                position: [1.0, 0.0, -1.0],
                color: y_color,
            }, // top right corner
        ];
    }

    fn get_debug_plane(vertex_size: u32) -> [Vertex_3D; 24] {
        // I dont know how to loop over an aray size in Rust
        const SIZE: usize = 24; // 6 verts * 4 squares

        // 6 verts per square if not using indexed renering
        let mut array: [Vertex_3D; 24] = [Vertex_3D {
            position: [0.0, 0.0, 0.0],
            color: [0.0, 0.0, 0.0, 0.0],
        }; 24];
        // Colors https://www.canva.com/colors/color-palettes/dark-road-curve/\
        let ebony = [0.039, 0.027, 0.031, 1.0];
        let dark_gray = [0.267, 0.267, 0.267, 1.0];
        let gray = [0.455, 0.455, 0.455, 1.0];
        let pewter = [0.694, 0.694, 0.694, 1.0];
        let color_ivory = [0.0, 0.0, 0.0, 1.0]; //to do later

        [
            // First Square, top left
            Vertex_3D {
                position: [-1.0, 0.0, 1.0],
                color: dark_gray,
            },
            Vertex_3D {
                position: [-1.0, 0.0, 0.0],
                color: dark_gray,
            },
            Vertex_3D {
                position: [0.0, 0.0, 1.0],
                color: dark_gray,
            },
            Vertex_3D {
                position: [0.0, 0.0, 1.0],
                color: dark_gray,
            },
            Vertex_3D {
                position: [-1.0, 0.0, 0.0],
                color: dark_gray,
            },
            Vertex_3D {
                position: [0.0, 0.0, 0.0],
                color: dark_gray,
            },
            // Second Square, top right
            Vertex_3D {
                position: [0.0, 0.0, 1.0],
                color: ebony,
            },
            Vertex_3D {
                position: [0.0, 0.0, 0.0],
                color: ebony,
            },
            Vertex_3D {
                position: [1.0, 1.0, 1.0],
                color: ebony,
            },
            Vertex_3D {
                position: [1.0, 1.0, 1.0],
                color: ebony,
            },
            Vertex_3D {
                position: [0.0, 0.0, 0.0],
                color: ebony,
            },
            Vertex_3D {
                position: [1.0, 0.0, 0.0],
                color: ebony,
            },
            // Third Square, bottom left
            Vertex_3D {
                position: [-1.0, 0.0, 0.0],
                color: ebony,
            },
            Vertex_3D {
                position: [-1.0, 0.0, -1.0],
                color: ebony,
            },
            Vertex_3D {
                position: [0.0, 0.0, 0.0],
                color: ebony,
            },
            Vertex_3D {
                position: [0.0, 0.0, 0.0],
                color: ebony,
            },
            Vertex_3D {
                position: [-1.0, 0.0, -1.0],
                color: ebony,
            },
            Vertex_3D {
                position: [0.0, 0.0, -1.0],
                color: ebony,
            },
            //Fourth suqare, bottom right
            Vertex_3D {
                position: [0.0, 0.0, 0.0],
                color: dark_gray,
            },
            Vertex_3D {
                position: [0.0, 0.0, -1.0],
                color: dark_gray,
            },
            Vertex_3D {
                position: [1.0, 0.0, 0.0],
                color: dark_gray,
            },
            Vertex_3D {
                position: [1.0, 0.0, 0.0],
                color: dark_gray,
            },
            Vertex_3D {
                position: [0.0, 0.0, -1.0],
                color: dark_gray,
            },
            Vertex_3D {
                position: [1.0, 0.0, -1.0],
                color: dark_gray,
            },
        ]
    }

    fn get_debug_fullscreen_tri() -> [Vertex_3D; 3] {
        // This is the standard fullscreen vertex example from vulkano examples repo.
        [
            /*
            Full Screen Triangle Optimization (could be moved into shader and pipeline removed for 0.1 ms speed increase)
            Vertices are placed in clip space (NDC) (-1 to 1)
                https://www.vincentparizet.com/blog/posts/vulkan_perspective_matrix/
            Each vertex position must be placed in clip / NDC space between:
            x: -1 to 1
            y: -1 to 1
            z: 0 to 1
            */
            // Instead of using a hard-coded array to hold the vertex data,
            // we ask our Model instance to give us the array of the data it’s loaded.
            // Create Vertices from Custom Vertex Struct & Populate fields

            // Fullscreen triangle
            Vertex_3D {
                position: [-0.98, -0.98, 0.0],
                color: [1.0, 0.35, 0.137, 1.0],
            }, // top left corner
            Vertex_3D {
                position: [-0.98, 3.0, 0.0],
                color: [1.0, 0.35, 0.137, 1.0],
            }, // bottom left corner
            Vertex_3D {
                position: [3.0, -0.98, 0.0],
                color: [1.0, 0.35, 0.137, 1.0],
            }, // top right corner
        ]
    }

    */

    // Extrapolations
    // Math extrapolation class to handle all the matrix transformations??
    // simple models to load (triangle, cube, etc)
    // fullscreen triangle?? that would not use the default shader
    // Draw index?
}

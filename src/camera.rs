use cgmath::{num_traits::clamp, Euler, Rotation3};
use glam::{vec3, vec4, Mat4, Quat, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};

// A "camera" is just a matrix, representing a local coordinate space.

/// Brief
/// 
/// Description
/// * `position` - Position in camera's local space.
/// * `direction` - current forwad vector for ray direction, used by lookat.
/// * `rotation` - UNUSED FOR NOW.
/// * `mvp` - camera Model View Projection Matrixes, for use with vertices & transformations.
/// * ``
#[derive(Debug)]
pub struct Camera {
    /// The local position of the camera
    pub position: Vec3,
    pub local_position: Vec3, // The inverse position of the world around the camera
    pub direction: Vec3,
    pub rotation: Quat,
    /// Speed of the freecam
    pub speed: f32,
    /// Model, View, Projection Matrices
    pub mvp: MVP,
    /// Field of View, in RADIANS
    pub fov: f32,
    pub z_near: f32,
    pub z_far: f32,
    pub angle: f32,
}

/// Model View Projection Matrices
/// 
/// Description
/// Note that current implementation for ray generation in fragment shader, rays are already in projection
/// * `model` - Model-to-World Can be used before and after sending to GPU.
/// * `view` - World-to-Camera matrix, CURRENTLY CONSTRUCTED AS Camera-to-World for use in raymarching ray rotations.
/// * `projection` - 
#[derive(Debug)]
pub struct MVP {
    // Model to World
    //pub model: Mat4, // Moved to model.rs, for per model matrix
    /// World to Camera Matrix
    pub view: Mat4,
    /// Camera to Viewport Matrix
    pub projection: Mat4, 
}

impl MVP {
    pub fn new() -> Self {
        Self {
            //model: Mat4::IDENTITY, // Moved to model.rs, for use with per model
            view: Mat4::IDENTITY,
            projection: Mat4::IDENTITY, 
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Functions
///////////////////////////////////////////////////////////////////////////////////////////////////

impl Camera {
    pub fn new() -> Self {

        Self {
            position: vec3(0.0, 0.0, -5.0), // Start a little from camera origin
            local_position: vec3(0.0, 0.0, 5.0), // Move the world slightly in front of the camera
            direction: Vec3::Z, // Point in positive z axis, inline with viewport coordinates
            rotation: Quat::IDENTITY, //The identity quaternion. Corresponds to no rotation.
            speed: 2.0,
            mvp: MVP::new(),
            fov: 1.5, // In radians, ~90 degrees
            z_near: 0.01, // Near clip plane, Unimplemented
            z_far: 100.0, // Far clip plane, Unimplemented
            angle: 0.0,
        }
    }


    /// Creates a rotation matrix and sets it to the view matrix
    pub fn rotate_quat(&mut self, degrees: Vec3, debug: bool) {
        /* https://www.opengl.org/archives/resources/faq/technical/viewing.htm
        As far as OpenGL is concerned, there is no camera. More specifically, the camera is always located at the eye space coordinate (0., 0., 0.). 
        To give the appearance of moving the camera, your OpenGL application must move the scene with the inverse of the camera transformation.
        */


        // Rotate the Quaternion 3 times, one for each axis
        // Rotation order is z, y, x axis

        // To combine two quaternions, much in the same manner as adding two numbers, multiply two quaternions
        // Store rotation as a quaternion
        // Operations are done on a matrix constructed from the quaternion
        
            // learn what each component of the quaternion is?

            // convert quat into matrix for use in shaders
            
            // This will rotate up / down on the x axis
            // How to do a progressive / incremental adjustment?
            // Multiply will represent a combined rotation
            // Need an incremental up / down

            // prevent rotation past certian point??
            // will need to be on the local y axis

        // Angle needs to be converted to radians
        // Current implementation rotates around the LOCAL axis (local coordinate system)

        // frame of refrence?
        // change the axis angle by the absolute value of the input, for rotating on axis
        // abs|input| -> vec3(1,1,0);
        // how can i go about clamping the rotation to not go past certain amount?

        // Add together the rotations

        // Rotate direction vector by quaternion
        //Mat4::from_quat(rotation)

        /* Column Major Matrix
        | x.x | y.x | z.x | pos.x
        | x.y | y.y | y.z | pos.y
        | x.z | y.z | z.z | pos.z
        |  0  |  0  |  0  |   1
        */

        self.rotation = self.rotation.mul_quat(Quat::from_axis_angle(Vec3::Z, f32::to_radians(degrees.z))).normalize(); // Rotate on Z axis (Roll)
        self.rotation = self.rotation.mul_quat(Quat::from_axis_angle(Vec3::Y, f32::to_radians(degrees.y))).normalize(); // Rotate on Y axis (Twist left & right)

        // Simple maximum angle look up / down
        let clamp_angle = 75.0;
        let mut new_angle = self.angle + degrees.x;
        if new_angle < clamp_angle && new_angle > -clamp_angle {
            self.angle = new_angle;
            self.rotation = self.rotation.mul_quat(Quat::from_axis_angle(Vec3::X, f32::to_radians(degrees.x))).normalize(); // Rotate on X axis (Look up & down)
        }

        /*
        Matrices are stored in column order
        Where:
            x = x axis orientation in world space
            y = y axis orientation in world space
            z = z axis orientation in world space
            w = x,y,z position offset in world space

        x y z w
        | | | |
        a a a a
        x x x x
        i i i i
        s s s s
        
         */

        // ❗ Refactor later
        // Convert quaternion back into matrix for use in shaders (GLSL does not support quats).
        self.mvp.view = Mat4::from_rotation_translation(self.rotation, self.mvp.view.w_axis.xyz());
        self.rotation = glam::Quat::from_mat4(&self.mvp.view);
        
        // Compare the axis of rotation (Vec3 direction) with an arbitrary axis of rotation (Vec3 Direction)

        //Console Debug
        if(debug) {
            println!("Updated Camera");
            let UP = self.rotation.angle_between(Quat::from_axis_angle(Vec3::X, f32::to_radians(90.0))); // compare axis of rotation to an axis of rotation pointing UP
            let DOWN = self.rotation.angle_between(Quat::from_axis_angle(Vec3::X, f32::to_radians(-90.0))); // compare axis of rotation to an axis of rotation pointing DOWN
            println!("difference UP {} | DOWN {}", UP.to_string(), DOWN.to_string());
    
            let (axis, angle) = self.rotation.to_axis_angle();
            println!("Angle {} | Angle {}",axis.to_string(), angle.to_string()); 
            println!("X axis direction {}", self.mvp.view.col(0));
            println!("Y axis direction {}", self.mvp.view.col(1));
            println!("Z axis direction {}", self.mvp.view.col(2));
            println!("Scale {}", self.mvp.view.col(3));
            println!(" ");
    
            // Test chagne the camera rotation matrix
            self.mvp.view = self.mvp.view;
        }
        
    }


    /// Translate the camera position for this frame
    /// 
    /// Remember this uses vulkan clip space coordinates, where:
    /// - `x+` is right
    /// - `y+` is down
    /// - `z+` if forward (default in `viewport.depth`)
    pub fn translate(&mut self, amount: glam::Vec3) {
        // translating the whole scene inversely from the matrix position to the origin


        // Position and Rotation are set in "world space" 
            // misunderstanding, all models is relative to clip-space coordinates. 
            // All this matrix will do is translate the model matrix clip-space position when all matrices are IDENTITY



        // 2024-09-08: At the end of the vertex
            // for example, think of placing a vertex in 2D NDC / Clip Space

        // ❗ Objects exist in the camera's local space
            // If moving the camera's translation, need to move in the opposite direction, to line up with clip space coordinates

        // Current implementation, objects exist in the local space of the camera
            // that means the camera doesnt move, the objects move around it in the opposite direction 
            // Ex: input +1 on x (right), then the objects need to move -1 (left)
            // Move all objects in the reverse (the camera doesnt move)


        // "move" (edit) camera matrix or "move" (edit) all other object matrix's
        let new_amount = self.mvp.view.transform_vector3(amount);
        let dir = Vec3::cross(self.mvp.view.z_axis.xyz(), self.mvp.view.y_axis.xyz()).normalize(); // I think this will give me right
        let movement = Vec3::cross(amount, dir).normalize();
        //println!("up {:?} | right: {:?} | new {:?}",self.mvp.view.y_axis, dir, movement);
        self.mvp.view.w_axis += vec4(new_amount.x, new_amount.y, new_amount.z, 0.0); // Keep w.w as 1


        // Move in relation to rotation / orientation options:
        // 1. Use a cross product with the forward direction (learnopengl tutorial)
            // good for constraining movement to a plane.
            // for example, moving along a normal
        // 2. Multiply movement vector by rotation matrix (W component needs to be 0 for directions)
            // good for 360 degree motion.



        // is the camera moving, or the objects moving in the vertex shader (transformed?)
        // Move all objects in the opposite direction of the camera (applied in the vertex shader as of now)
        //println!("CAMERA W: {:?}", self.mvp.view.w_axis);
    }


/// Copied from glam::Mat4::look_to_rh
    /// For a view coordinate system with `+X=right (Vulkan Clip-Space Right)`, `+Y=Up (Vulkan Clip-Space Up)` and `+Z=forward (Vulkan Clip-Space into screen)`.
    pub fn get_view(&mut self) -> glam::Mat4 {
        /* https://www.reddit.com/r/opengl/comments/1bx9uno/view_matrix_is_always_rotating_around_the_origin/
        // From the looks of things this is what you're calculating (i.e. these should be the entries of viewMatrix in your code after all the ma4x4_rotate_* calls):
        [ Xx, Yx, Zx, Px ]
        [ Xy, Yy, Zy, Py ]
        [ Xz, Yz, Zz, Pz ]
        [  0,  0,  0,  1 ]
        // ^Here the X, Y, and Z vectors are the camera's local X, Y, and Z axes in world space, and P is the position of the camera

        // The inverse of that (which is the actual view matrix) looks like this:
        [ Xx, Xy, Xz, -dot(P, X) ]
        [ Yx, Yy, Yz, -dot(P, Y) ]
        [ Zx, Zy, Zz, -dot(P, Z) ]
        [  0,  0,  0,     1      ]
         */

        // Similar to using gluLookAt or glam::look_at, need to use dot product in view matrix calculation
        // I believe that I need to calculate the inverse of the view matrix before using it, like glam::look_at (i think it does that as well)

        // To reverse, can either swap the signs or swap the add/subtract when translating

        let s = self.mvp.view.x_axis; // side (right)
        let u = self.mvp.view.y_axis; //up 
        let f = self.mvp.view.z_axis; // forward
        let eye = self.mvp.view.w_axis; // position

        // crate the inverse of the internal rotation matrix to create the final view matrix
            // ❓ investigate, does creating an inverse matrix put all other matrices inside this coordinate space, that are multiplied?

        // Mat4::from_cols(
        //     Vec4::new(s.x, u.x, f.x, 0.0),
        //     Vec4::new(s.y, u.y, f.y, 0.0),
        //     Vec4::new(s.z, u.z, f.z, 0.0),
        //     Vec4::new(-eye.dot(s), -eye.dot(u), -eye.dot(f), 1.0),
        // )

        // Similar to gluLookAt, but adapted for Vulkan Clip-Space Coordinate System
            // To compare, change the view matrix to IDENTITY for use in shader.
            // This is equal to the inverse of the rotated & translated view matrix

        // Mat4::look_to_lh(
        //     self.mvp.view.w_axis.xyz(), 
        //     self.mvp.view.z_axis.xyz(), 
        //     glam::Vec3::Y)

        // view matrix is in major column order, calculate the inverse
        let mut align = Mat4::IDENTITY;
        align.y_axis.y = -1.0;
        align.z_axis.z = -1.0;
        
        // correct order of multiplication
        align.inverse() * self.mvp.view.inverse()

    }

    /// Sets the projection matrix. Converts vertex positions into clip / nds space
    /// 
    /// Converts all vertices within the near & far clipping range, into clip-space's 0-1 range.
    ///- `fov_y_degrees` = 
    ///- `aspect ratio` = height / width of the screen (or image?)
    ///- `z_near` = 0.01
    ///- `z_far` = 100.0
    pub fn set_projection(&mut self, fov_y_degrees: f32, aspect_ratio: f32, z_near: f32, z_far:f32) {
        // ❗ Should this funtion return a matrix or set directly? 

        // Mat4::perspective_lh uses a depth range of 0-1, the default of vulkan (see viewport.depth)
        // uses a 0 to 1 depth range (see viewport.depth, default is 0-1)

        // Using glam's left hand perspective matrix preserves vulkan's clip space coordinates
        //self.mvp.projection = Mat4::perspective_lh(fov_y_degrees.to_radians(), aspect_ratio, z_near, z_far);

        // infinite vs infinite reverse?

        /*
        x.x = aspect_ratio / (tan (fov/2)
        y.y = 1 / tan(fov/2)
        z.z = far / (far - near)
        w.z = -1 * (near*far) / (far - near)
         */
/* 
        let xX = aspect_ratio / (f32::tan(fov_y_degrees.to_radians() * 0.5)); //degrees or radians?
        let yY = 1.0 / (f32::tan(fov_y_degrees.to_radians() * 0.5));
        let zZ = z_far / (z_far - z_near);
        let wZ = -((z_near * z_far)/(z_far-z_near));

        self.mvp.projection = Mat4::from_cols(
            Vec4::new(xX, 0.0, 0.0, 0.0),
            Vec4::new(0.0, yY, 0.0, 0.0),
            Vec4::new(0.0, 0.0, zZ, 1.0),
            Vec4::new(0.0, 0.0, wZ, 0.0),
        );
 */
        self.mvp.projection = Self::perspective(fov_y_degrees.to_radians(), aspect_ratio, z_near, z_far);

        // self.mvp.projection = glam::Mat4::perspective_lh(fov_y_degrees.to_radians(), aspect_ratio, z_near, z_far);
        //self.mvp.projection.inverse();
    }

    /// Meant to be used for source coordinate systems where +X is right, +Y is up and +Z is forward.
    /// - credit: Marc on vulkano discord
    pub fn perspective(vertical_fov: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Mat4 {

        let t = (vertical_fov / 2.0).tan();
        let sy = 1.0 / t;
        let sx = sy / aspect_ratio;
        let r = z_far / (z_far - z_near);

        Mat4::from_cols(
            Vec4::new(sx, 0.0, 0.0, 0.0),
            Vec4::new(0.0, -sy, 0.0, 0.0),
            Vec4::new(0.0, 0.0, r, 1.0),
            Vec4::new(0.0, 0.0, -z_near * r, 0.0),
        )
    }

    /// Reset position and rotation to 0
    pub fn reset(&mut self) {
        self.mvp.view = Mat4::IDENTITY;
        self.rotation = Quat::IDENTITY; // will remove later during refactor
        self.set_projection(90.0, 2560.0/1440.0, 0.01, 100.0);
        self.angle = 0.0;

        println!("{}", self.position.to_string());
    }
}

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // OLD METHODS 2024-07-17
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /*

    // https://discussions.unity.com/t/how-to-clamp-rotation-following-unitys-quaternion-rules/221052
    fn clamp_rotation(mut q: Quat, minimum: f32, maximum: f32) -> glam::Quat {
        

        q.x /= q.w;
        q.y /= q.w;
        q.z /= q.w;
        q.w = 1.0;
 
        //float angleX = 2.0f * Mathf.Rad2Deg * Mathf.Atan (q.x);
        //angleX = Mathf.Clamp (angleX, MinimumX, MaximumX);
        //q.x = Mathf.Tan (0.5f * Mathf.Deg2Rad * angleX);

        let mut angleX = 2.0 * f32::to_degrees(f32::atan(q.x)); 
        angleX = f32::clamp(angleX, minimum, maximum);
        q.x = f32::tan(0.5 * f32::to_radians(angleX));
    
        return q;
    }

    // https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    fn clamp_rotation_2(mut q: Quat, minimum: f32, maximum: f32) -> glam::Quat {
        // α is a simple rotation angle (the value in radians of the angle of rotation) 
        // cos(βx), cos(βy) and cos(βz) are the "direction cosines" of the angles between the three coordinate axes and the axis of rotation. (Euler's Rotation Theorem).

        // 1x = sin(rotation angle/2)*cos(angle between axis of rotation)
        //qx = sin(a/2) * 1
        let (axis, angle) = q.to_axis_angle();
        let mut clamp = Vec4::ZERO;
        clamp.x = f32::sin(angle/2.0);// * f32::cos(Vec3::angle_between(axis, Vec3::X));
        clamp.y = f32::sin(angle/2.0);// * f32::cos(Vec3::angle_between(axis, Vec3::Y));
        clamp.z = f32::sin(angle/2.0);// * f32::cos(Vec3::angle_between(axis, Vec3::Z));
        clamp.w = f32::cos(angle/2.0);
        println!("clamp value: {}", clamp);
        return q;

    }

    // https://vkguide.dev/docs/new_chapter_5/interactive_camera/
    // fn get_view_matrix(&self) -> Mat4 {
    //     // to create a correct model view, we need to move the world in opposite
    //     // direction to the camera
    //     //  so we will create the camera model matrix and invert

    //     // glam::Mat4::tra
    //     // glm::mat4 cameraTranslation = glm::translate(glm::mat4(1.f), position);
    //     // glm::mat4 cameraRotation = getRotationMatrix();
    //     // return glm::inverse(cameraTranslation * cameraRotation);

    // }

    // https://vkguide.dev/docs/new_chapter_5/interactive_camera/
    fn get_rotation_matrix() -> Mat4 {
        // fairly typical FPS style camera. we join the pitch and yaw rotations into
        // the final rotation matrix

        let x = glam::Mat4::from_axis_angle(vec3(1.0, 0.0, 0.0), 0.0);
        let y = glam::Mat4::from_axis_angle(vec3(0.0, -1.0, 0.0), 0.0);
        let z = glam::Mat4::from_axis_angle(vec3(0.0, 0.0, 1.0), 0.0);
        z * y * x
    }

    pub fn update(&mut self) {
        //update the mvp somehow (with input?)
        // rotation and position
    }

    /// Translate from current position by adding new_position
    pub fn set_translation(&mut self, new_position: Vec3) {
        // Currently fly-cam like behavior
        // move locally in camera space (like in fragment shader)
        let use_camera_space = false;

        if (use_camera_space) {
            self.position += new_position;
        } else {
            self.position += self.mvp.view.transform_vector3(new_position);
        }
    }

    pub fn set_position(&mut self, new_position: Vec3, local_position: bool) {
        if local_position {
            self.position = new_position;
        } else {
            self.position = self.mvp.view.transform_point3(new_position)
        }
    }


    pub fn translate(&mut self, amount: glam::Vec3, speed: f32) {
        // DEBUG!
        // Rememeber than this moves in clip space
        // clip space +x is right, +y is DOWN, +z is forward
        self.mvp.view.w_axis.x += amount.x * speed;
        self.mvp.view.w_axis.y += amount.y * speed;
        self.mvp.view.w_axis.z += amount.z * speed;
    }

    /// Currently Unused.
    /// 
    /// Brief.
    /// Creates a view matrix (camera-to-world space)
    ///
    /// Description. 
    /// Create a view matrix by creating a forward direction that points from the camera's position to the target's position
    /// 
    /// * `target` - The lookat target's position in view space (camera space).
    /// * `eye` - The camera's position in view space (camera space).
    /// * `up` - the "world"'s up vector to convert to (0,1,0)
    pub fn set_view(&mut self, target: Vec3, eye: Vec3, up: Vec3) {
        // https://taidaesal.github.io/vulkano_tutorial/section_3.html > View Matrix
        // https://fby-laboratory.com/articles/article4_en > View Matrix
        println!("Updated View");
        
        self.mvp.view = Self::look_at(
                vec3(0.0, 0.0, 0.0), 
                self.position, 
                vec3(0.0, 1.0, 0.0)
            );//view_matrix(center, eye, up);
        //self.mvp.view.inverse();

        /** https://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/
         * Return a transformation matrix that will transform a ray from view space
         * to world coordinates, given the eye point, the camera target, and an up vector.
         *
         * This assumes that the center of the camera is aligned with the negative z axis in
         * view space when calculating the ray marching direction.
         */
        fn view_matrix(center: Vec3, eye: Vec3, up: Vec3) -> glam::Mat4 {
            // UNUSED!
            let f = Vec3::normalize(center - eye);
            let s = Vec3::normalize(Vec3::cross(f, up));
            let u = Vec3::cross(s, f);
            Mat4 {
                x_axis: vec4(s.x, s.y, s.z, 0.0),
                y_axis: vec4(u.x, u.y, u.z, 0.0),
                z_axis: vec4(-f.x, -f.y, -f.z, 0.0),
                w_axis: vec4(0.0, 0.0, 0.0, 1.0), //TEMPORARY, MOVE LATER
            }
        }
    }

    /// Create a camera-to-world view matrix
    /// Forward direction = target - eye
    /// PEACHY (https://www.shadertoy.com/view/cdGyWW) & PRIMITIVES (https://www.shadertoy.com/view/ltyXD3)
    /// https://learnopengl.com/Getting-started/Camera
    fn look_at(target: Vec3, eye: Vec3, up: Vec3) -> glam::Mat4 {
        // copied from glam::Mat4::look_to() (in fragment shader)

        // This is a camera to world matrix
        //x: (1,0,0)
        //y: (0,1,0)
        //z: (0,0,1)
        let z = Vec3::normalize(target - eye); // +z forward vector, (0,0,1);
        let x = Vec3::normalize(Vec3::cross(up, z));  // +x side vector, (1,0,0)
        let y = Vec3::cross(z, x); // +y up vector (0,1,0)

        // Column order (to work with glsl)
        Mat4::from_cols(
            Vec4::new(x.x, x.y, x.z, 0.0),
            Vec4::new(y.x, y.y, y.z, 0.0),
            Vec4::new(z.x, z.y, z.z, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        )
    }

    /// Brief.
    ///
    /// Description. 
    /// 
    /// * `view_direction` - angle amount to rotate each axis on this frame
    pub fn set_view_direction(&mut self, view_direction: Vec3) {
        /*
        Uses look_at_direction function
        Can adjust forward and up vectors with pitch and yaw
        Immune to gimbal lock, however pitch is locked between -89 & 89 degrees because introduces roll
        
        Will need to use either quaternions or matrix for 6DOF (Pitch, Yaw, and Roll. Equation is too complicated for me without them)
        Accumulates angles, either need to reset after a while or extract from the matrix
         */



        // https://taidaesal.github.io/vulkano_tutorial/section_3.html > View Matrix
        // https://fby-laboratory.com/articles/article4_en > View Matrix

        /* 
        This works
        https://gamedev.stackexchange.com/questions/71320/how-do-i-determine-the-look-at-vector-of-a-free-look-camera#
        Direction.x = (float)( sin(RotateCamera.x) * cos(RotateCamera.y) );
        Direction.y = (float)( sin(RotateCamera.y) );
        Direction.z = (float)( cos(RotateCamera.x) * cos(RotateCamera.y) );

        up.x = (float)( sin(RotateCamera.x) * sin(RotateCamera.y) * -1 );
        up.y = (float)( cos(RotateCamera.y) );
        up.z = (float)( cos(RotateCamera.x) * sin(RotateCamera.y) * -1 );
         */

        /*
        Incorrect?
        https://learnopengl.com/Getting-started/Camera
        front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
        front.y = sin(glm::radians(Pitch));
        front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
         */


        // change view direction based on mouse
        // use existing numbers without caching old ones (buildup potential overflow)
        // 1. rotate around y axis (edit x and z directions)


        


        // take mouse right movement & turn into 1 or -1
        // matrices alread in range of 1 to -1
        // add to them then re-normalize?


        // get unit vector from matrix (first from stored vector)
        // calculate angle from unity vector (atan2(y,x) order is importnant)
        // add result with passed angle
        // use sin + cos?
        // normalize back into matrix
        
        // Get previous normalized vector values
        // Convert to degrees (raidans)
        // (0,0,1) -> 90 degrees

        /*
        takes a vector (unit) and gets the angle of that
        atan2 = (y,x) specify a vetor direction
        (sin(y),cos(x)) | (up & down, left and right)

        atan2 = (1,1) = 45
        atan2 = (0,1) = 90
        atan2 = (1,0) = 0
        atan2 = (-1,-1) = 135
        atan2 = (-1,0) = 180
         */

        let x = self.direction.x;
        let y = self.direction.y;
        let z = self.direction.z;
        

        
        // (vertical axis (y),horizontal axis (x))
        // takes unit vector, converts to degrees / radians
        // for (0,0,1) = 90 degrees
        // so y (sin(90)) = 1, so z goes first
        // atan2 range 0 to 180, -180 to -0

        // let h_angle = f32::atan2(x, z); 
        // let v_angle = f32::atan2(y,z); 
        // let roll_angle = f32::atan2(y,x);
        
        // Additive look direction
        let mut yaw = self.direction.x;
        let mut pitch = self.direction.y;
        let roll = self.direction.z;
        
        println!("yaw {} |, pitch {} |, roll {} ", yaw, pitch, roll);
        println!(" self.direction {}", self.direction);

        self.direction = vec3(
            yaw + view_direction.x.to_radians(),
            f32::clamp(pitch + view_direction.y.to_radians(), -89.0_f32.to_radians(), 89_f32.to_radians()), // clamp degrees to 89 to -89
            roll + view_direction.z.to_radians()
        );

        // // Extract look direction from matrix
        // let matrix_forward = self.mvp.view.col(2);
        // let matrix_up = self.mvp.view.col(1);
        // println!("Martix 3rd column {}", matrix_forward);
        // let yaw = f32::atan2(matrix_forward.x, matrix_forward.z) + view_direction.x.to_radians();
        // let pitch = f32::atan2(matrix_forward.y,matrix_up.y) + view_direction.y.to_radians();
        // //println!("yaw2 {} |, pitch2 {} |", yaw2, pitch2);
        // //let roll = self.direction.z;
        // println!("*****************");


        // println!("Radians: yaw: {} | pitch: {}", (yaw), (pitch));
        // println!("Degrees: yaw: {} | pitch: {}", f32::to_degrees(yaw), f32::to_degrees(pitch));
        // // println!("Radians: yaw: {} | pitch: {} | roll {}", (yaw), (pitch), (roll));
        // // println!("Degrees: yaw: {} | pitch: {} | roll {}", f32::to_degrees(yaw), f32::to_degrees(pitch), f32::to_degrees(roll));
        

        //starting angle is 0
        // sin(0) = 0, cos(0) = 1
        let forward = vec3( 
            yaw.sin() * pitch.cos(),
            pitch.sin(),
            yaw.cos() * pitch.cos()
        ).normalize();

        let up = vec3(
            yaw.sin() * pitch.sin() * -1.0,
            pitch.cos(),
            yaw.cos() * pitch.sin() * -1.0
        ).normalize();
        // should be (0,1,0)

        self.mvp.view = Self::look_at_direction(
            forward,
            up
        );

        // Debug info
        /*
        println!("forward: {} ", forward);
        println!("up: {} ", up);
        println!("matrix x axis {}", self.mvp.view.x_axis);
        println!("matrix y axis {}", self.mvp.view.y_axis);
        println!("matrix z axis {}", self.mvp.view.z_axis);
        println!("matrix w axis {}", self.mvp.view.w_axis);
        println!("********************************************");
         */

    }

    /// Brief.
    ///
    /// Description. 
    /// 
    /// * `view_direction` - pre-computed look direction (ex: target - eye)
    /// * `up` - pre-computed up direction, orthagonal to view_direction
    fn look_at_direction(view_direction: Vec3, up: Vec3) -> glam::Mat4 {
        // https://learnopengl.com/Getting-started/Camera
        // copied from glam::Mat4::look_to()

        let z = Vec3::normalize(view_direction); // +z forward vector, (0,0,1);
        let x = Vec3::normalize(Vec3::cross(up, z));  // +x side vector, (1,0,0)
        let y = Vec3::cross(z, x); // +y up vector (0,1,0)

        // Convert to column-major (to work with glsl)
        Mat4::from_cols(
            Vec4::new(x.x, x.y, x.z, 0.0),
            Vec4::new(y.x, y.y, y.z, 0.0),
            Vec4::new(z.x, z.y, z.z, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        )
    }


    // temporary debug
    pub fn translate_local(&mut self, amount: Vec3, speed: f32) {
        self.mvp.view.x_axis.w += amount.x * speed;
        self.mvp.view.y_axis.w += amount.y * speed;
        self.mvp.view.z_axis.w += amount.z * speed;
    }

        fn round_to_decimal(x: f32) -> f32 {
        // let y = (x * 100.0).round() / 100.0;
        (x * 100.0).round() / 100.0
    }
     */

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // OLD METHODS
    ///////////////////////////////////////////////////////////////////////////////////////////////////

/*
    Unused from fby-labs
    let mut x_thita = 0.0 ;
    let y_thita = 0.0;
    let z_thita = 0.0;

    x_thita += input.mouse_diff().0;

    let nx = Vector3::new(1.0, 0.0, 0.0);
    let rotation_x = Matrix3::from_axis_angle(nx, Rad(x_thita));

    let ny = Vector3::new(0.0, 1.0, 0.0);
    let rotation_y = Matrix3::from_axis_angle(ny, Rad(y_thita));

    let nz = Vector3::new(0.0, 0.0, 1.0);
    let rotation_z = Matrix3::from_axis_angle(nz, Rad(z_thita));

    let rotation = Matrix4::from(rotation_x * rotation_y * rotation_z);

    let x_translation = 0.0;
    let y_translation = 0.0;
    let z_translation = 0.0;

    let translation = Matrix4::from_translation(Vector3::new(
        x_translation,
        y_translation,
        z_translation,
    ));

    let x_scale = 1.0;
    let y_scale = 1.0;
    let z_scale = 1.0;

    let scale = Matrix4::from_nonuniform_scale(x_scale, y_scale, z_scale);

    let model = translation * rotation * scale;

    let model_array = [
        [model.x.x, model.x.y, model.x.z, model.x.w],
        [model.y.x, model.y.y, model.y.z, model.y.w],
        [model.z.x, model.z.y, model.z.z, model.z.w],
        [model.w.x, model.w.y, model.w.z, model.w.w],
    ];

    let eye_position = Point3::new(0.0, 1.0, 1.0);
    let looking_point = Point3::new(0.0, 0.0, 0.0);

    let looking_dir = looking_point - eye_position;
    let unit_z = Vector3::new(0.0, 0.0, 1.0);
    let e = unit_z.cross(-looking_dir);
    let up_direction = e.cross(looking_dir).normalize();

    let view = Matrix4::look_at_rh(eye_position, looking_point, up_direction);

    let view_array = [
        [view.x.x, view.x.y, view.x.z, view.x.w],
        [view.y.x, view.y.y, view.y.z, view.y.w],
        [view.z.x, view.z.y, view.z.z, view.z.w],
        [view.w.x, view.w.y, view.w.z, view.w.w],
    ];

    let aspect_ratio =
        swapchain.image_extent()[0] as f32 / swapchain.image_extent()[1] as f32;

    let mut proj = cgmath::perspective(
        Rad(std::f32::consts::FRAC_PI_2),
        aspect_ratio,
        0.01,
        100.0,
    );

    proj.x *= -1.0;
    proj.y *= -1.0;

    let proj_array = [
        [proj.x.x, proj.x.y, proj.x.z, proj.x.w],
        [proj.y.x, proj.y.y, proj.y.z, proj.y.w],
        [proj.z.x, proj.z.y, proj.z.z, proj.z.w],
        [proj.w.x, proj.w.y, proj.w.z, proj.w.w],
    ];
*/

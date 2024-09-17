/*
Structs have an extra level of visibility with their fields.
The visibility defaults to private, and can be overridden with the pub modifier.
This visibility only matters when a struct is accessed from outside the module where it is defined, and has the goal of hiding information (encapsulation).
*/

struct Entity {


    // ID: u32, // internal memory usually converts to at least u32 (even bools)
    // bit_mask: u32,
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Ideas
///////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
struct Attributes {
    health: u8,
    stamina: u8,
}

#[derive(Debug)]
pub struct Player {
    name: String,
    attributes: Attributes,
    speed: u8,
    weight: u8,
    race: Race,
}

pub struct NPC {
    name: String,
    attributes: Attributes,
    valor: u8,
}

#[derive(Debug)]
enum Race {
    Human,
    Elf,
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Functions
///////////////////////////////////////////////////////////////////////////////////////////////////

impl Attributes {
    fn new() -> Attributes {
        Attributes {
            health: 100,
            stamina: 100,
        }
    }
}

impl Player {
    //https://doc.rust-lang.org/book/ch05-01-defining-structs.html
    // Return a new instance of Player
    // Can return either Self or the type (Player)
    pub fn new() -> Self {
        // Self {
        //     //         attributes: Attributes::new()
        //     //
        // }
        Player {
            // Can return either Self or the type (Player)
            name: String::from("player"),
            attributes: Attributes {
                health: 100,
                stamina: 100,
            },
            speed: 100,
            weight: 0,
            race: Race::Human,
        }
    }

    pub fn check(&self) {
        //println!("{}, {}", x.value(), y.value());
        //println!("{}", value);
        println!("{}, {}", self.attributes.health, self.attributes.stamina);
    }

    pub fn hit(&mut self) {
        self.attributes.health -= 10;
    }
}

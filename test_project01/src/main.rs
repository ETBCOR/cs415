mod libs;
use libs::{Simulation};



fn main() {
    let mut sim1 = Simulation::new(
        20,
        8,
        1,
        0.0,
        0.4
    );
    sim1.init();
    sim1.run();
}

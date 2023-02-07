
use rand::{
    prelude::{random, Distribution},
    distributions::Standard, thread_rng, seq::SliceRandom, Rng,
};

#[derive(Debug, Clone, Copy)]
enum Nucleotide {
    A,
    C,
    G,
    T
}

impl Distribution<Nucleotide> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Nucleotide {
        match rng.gen_range(0..=3) {
            0 => Nucleotide::A,
            1 => Nucleotide::C,
            2 => Nucleotide::G,
            _ => Nucleotide::T,
        }
    }
}

type Genome = Vec<Nucleotide>;

#[derive(Debug, Clone)]
pub struct Individual {
    genome: Genome,
    fitness: u32,
}

impl Individual {
    pub fn new() -> Self {
        Self {
            genome: Vec::new(),
            fitness: 0,
        }
    }

    pub fn init(&mut self, genome_size: u32) {
        for _ in 0..genome_size {
            self.genome.push(random());
        }
    }

    pub fn print(&self) {
        print!("Member {{ genome: [");
        for nuc in self.genome.iter() {
            print!("{}", match nuc {
                Nucleotide::A => "A",
                Nucleotide::C => "C",
                Nucleotide::T => "T",
                Nucleotide::G => "G",
            });
        }
        println!("], fitness: {} }}", self.fitness);

    }

    pub fn update_fitness(&mut self) {
        self.fitness = 0;
        for nuc in self.genome.iter() {
            self.fitness += match nuc {
                Nucleotide::T => 1,
                _ => 0
            }
        }
    }
}

type Population = Vec<Individual>;

#[derive(Debug)]
pub struct Simulation {
    population_size: u32,
    genome_size: u32,
    generations: u32,
    mutation_rate: f32,
    selection_pressure: f32,
    population: Population,
}

impl Simulation {
    pub fn new(
        population_size: u32,
        genome_size: u32,
        generations: u32,
        mutation_rate: f32,
        selection_pressure: f32,

    ) -> Self { Self {
        population_size,
        genome_size,
        generations,
        mutation_rate,
        selection_pressure,
        population: Vec::new(),
    }}

    pub fn init(&mut self) {
        for _ in 0..self.population_size {
            let mut mem = Individual::new();
            mem.init(self.genome_size);
            self.population.push(mem);
        }
        self.fit();
    }

    pub fn print(&self) {
        println!("Population {{\n\tmembers: [");
        for mem in self.population.iter() {
            print!("\t\t");
            mem.print();
        }
        println!("\t]\n}}");
    }

    fn fit(&mut self) {
        for mem in self.population.iter_mut() {
            mem.update_fitness();
        }
        self.population.sort_by(|a, b| a.fitness.cmp(&b.fitness));
    }

    pub fn run(&mut self) {
        println!(
            "--------------------------------------------------------\n| Running simulation with the following parameters:\n| population_size: {}\n| genome_size: {}\n| generations: {}\n| mutation_rate: {}\n| selection_pressure: {}\n--------------------------------------------------------",
            self.population_size, self.genome_size, self.generations, self.mutation_rate, self.selection_pressure
        );

        for gen in 1..=self.generations {
            println!("Generation #{}:", gen);
            
            let parents = select(&self.population, self.selection_pressure);
            let children = breed(parents, self.genome_size);

            println!("Parents: {:?}\nChildren: {:?}", parents, children);

            self.fit();
            //self.print();
        }
    }

}

fn select(population: &Population, _selection_pressure: f32) -> (&Vec<Nucleotide>, &Vec<Nucleotide>) {
    let mut rng = thread_rng();
    (&population.choose(&mut rng).unwrap().genome, &population.choose(&mut rng).unwrap().genome)
}

fn breed(parents: (&Vec<Nucleotide>, &Vec<Nucleotide>), genome_size: u32) -> (Vec<Nucleotide>, Vec<Nucleotide>) {
    let mut rng = thread_rng();
    let split = rng.gen_range(1..(genome_size - 1));
    println!("split: {}", split);

    let parents = (parents.0.iter().enumerate(), parents.1.iter().enumerate());
    let mut children = (Vec::<Nucleotide>::new(), Vec::<Nucleotide>::new());
    
    //for i in 0..split {
        //children.0.push(parents.1.next());
    //}
    
    children
}
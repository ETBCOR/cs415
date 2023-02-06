
use rand::{
    prelude::{random, Distribution},
    distributions::Standard, thread_rng, seq::SliceRandom, Rng,
};

#[derive(Debug, Clone)]
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
    pub fn new(genome_size: u32) -> Self {
        let mut g: Vec<Nucleotide> = Vec::new();
        for _ in 0..genome_size {
            g.push(random());
        }
        Self {
            genome: g,
            fitness: 0,
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
            let mem = Individual::new(self.genome_size);
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

    fn select(&mut self) -> (&Individual, &Individual) {
        let mut rng = thread_rng();
        (self.population.choose(&mut rng).unwrap(), self.population.choose(&mut rng).unwrap())
    }

    fn breed(&mut self, parents: (&Individual, &Individual)) -> (Individual, Individual) {
        let mut rng = thread_rng();
        let mut children = (parents.0.to_owned(), parents.1.to_owned());
        let split = rng.gen_range(1..(self.genome_size - 1));
        println!("split: {}", split);
        children
    }

    pub fn run(&mut self) {
        println!(
            "--------------------------------------------------------\n| Running simulation with the following parameters:\n| population_size: {}\n| genome_size: {}\n| generations: {}\n| mutation_rate: {}\n| selection_pressure: {}\n--------------------------------------------------------",
            self.population_size, self.genome_size, self.generations, self.mutation_rate, self.selection_pressure
        );

        for gen in 0..self.generations {
            println!("Generation #{}:", gen);
            
            let parents = self.select();
            println!("Parents: {:?}", parents);
            let children = self.breed(parents);

            println!("Children: {:?}", children);

            self.fit();
            self.print();
        }
    }

}


use core::num;

use genevo::{
    self,
    prelude::*,
    population::ValueEncodedGenomeBuilder
};

const STRAND_SIZE: usize = 20;
const POPULATION_SIZE: usize = 20;
const GENERATION_LIMIT: u64 = 200;
const NUM_INDIVIDUALS_PER_PARENTS: usize = 2;
const SELECTION_RATION: f64 = 0.5;
const MUTATION_RATE: f64 = 0.05;
const REINSERTION_RATION: f64 = 0.5;

/// The phenotype
type Phenome = String;

/// The genotype
type Genome = Vec<u8>;

/// How do the genes of the genotype show up in the phenotype
trait AsPhenotype {
    fn as_phenome(&self) -> Phenome;
}

impl AsPhenotype for Genome {
    fn as_phenome(&self) -> Phenome {
        String::from_utf8(self.to_vec()).unwrap()
    }
}

/// The fitnes function for `Genome`s.
#[derive(Clone, Debug)]
struct FitnessCalc;

impl FitnessFunction<Genome, usize> for FitnessCalc {
    fn fitness_of(&self, genome: &Genome) -> usize {
        let mut num_Ts = 0;
        for g in genome.into_iter() {
            if g.to_string() == "T" {
                num_Ts += 1;
            }
        }
        num_Ts
    }

    fn average(&self, values: &[usize]) -> usize {
        values.iter().sum::<usize>() / values.len()
    }

    fn highest_possible_fitness(&self) -> usize {
        STRAND_SIZE
    }
    
    fn lowest_possible_fitness(&self) -> usize {
        0
    }
}



fn main() {
    let initial_population: Population<Genome> = build_population()
        .with_genome_builder(ValueEncodedGenomeBuilder::new(STRAND_SIZE, 0, 3))
        .of_size(POPULATION_SIZE)
        .uniform_at_random();

    let mut sim = simulate(
        genetic_algorithm()
        .with_evaluation(FitnessCalc)
        .with_selection(MaximizeSelector::new())
        .with_crossover(crossover_op)
        .with_mutation(mutation_op)
        .with_reinsertion(reinsertion_op)
        .with_initial_population(initial_population)
        
    );
}
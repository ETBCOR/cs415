use genevo::{
    self,
    prelude::*,
    selection::truncation::*,
    recombination::discrete::SinglePointCrossBreeder,
    reinsertion::elitist::ElitistReinserter, operator::prelude::{RandomValueMutation, RandomValueMutator}, types::fmt::Display,
};
use rand::{
    distributions::{Distribution, Standard},
    Rng
};

use plotters::prelude::*;


// Invarient simulation parameters
const STRAND_SIZE: usize = 128;
const GENERATION_LIMIT: u64 = 8192; // 2^13

// The Parameter struct defines the parameters need to run a simulation
// (along with the above constants, which will not be varied)
#[derive(Debug)]
struct Parameters {
    population_size: usize,
    num_individuals_per_parents: usize,
    selection_ratio: f64,
    mutation_rate: f64,
    reinsertion_ratio: f64,
}

impl Parameters {
    fn new(
        population_size: usize,
        num_individuals_per_parents: usize,
        selection_ratio: f64,
        mutation_rate: f64,
        reinsertion_ratio: f64
    ) -> Self {
        Self {
            population_size,
            num_individuals_per_parents,
            selection_ratio,
            mutation_rate,
            reinsertion_ratio
        }
    }
}

impl Default for Parameters {
    fn default() -> Self {
        Self::new(
            64,
            2,
            0.5,
            0.05,
            0.5
        )
    }
}


// The phenotype
type Phenome = String;


// The genotype
#[derive(Clone, Debug, PartialEq, PartialOrd)]
enum Nucleotide { A, C, T, G }
type Genome = Vec<Nucleotide>;


// How do the genes of the genotype show up in the phenotype
trait AsPhenotype {
    fn as_phenome(&self) -> Phenome;
}

impl AsPhenotype for Genome {
    fn as_phenome(&self) -> Phenome {
        self.into_iter().map(|x| {
            match x {
                Nucleotide::A => 'A',
                Nucleotide::C => 'C',
                Nucleotide::T => 'T',
                Nucleotide::G => 'G',
            }
        }).collect::<String>()
    }
}


// Enable random Nucleotide generation
impl Distribution<Nucleotide> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Nucleotide {
        match rng.gen_range(0..4) {
            0 => Nucleotide::A,
            1 => Nucleotide::C,
            2 => Nucleotide::G,
            _ => Nucleotide::T
        }
    }
}

impl RandomValueMutation for Nucleotide {
    fn random_mutated<R>(_: Self, _: &Self, _: &Self, _: &mut R) -> Self
    where
        R: Rng + Sized
    {
        rand::random()
    }
}


// The "T" counting fitness function for `Genome`s.
#[derive(Clone, Debug)]
struct NumTsFitnessCalculator;

impl FitnessFunction<Genome, usize> for NumTsFitnessCalculator {
    fn fitness_of(&self, genome: &Genome) -> usize {
        let mut t_count = 0;
        for n in genome.into_iter() {
            if *n == Nucleotide::T {
                t_count += 1;
            }
        }
        t_count
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


// The clusters-of-4 counting fitness function for `Genome`s.
#[derive(Clone, Debug)]
struct ClustersOf4FitnessCalculator;

impl FitnessFunction<Genome, usize> for ClustersOf4FitnessCalculator {
    fn fitness_of(&self, genome: &Genome) -> usize {
        // let dst = genome.chunks(4).into_iter().filter(|&n| n);

        let mut cluster_count = 0;
        let chunks = genome.chunks(4);
        for n in chunks.into_iter() {
            if n.get(0)
                .map(|first| n.iter().all(|x| x == first))
                .unwrap_or(true)
            {
                cluster_count += 1;
            }
        }
        cluster_count
    }

    fn average(&self, values: &[usize]) -> usize {
        values.iter().sum::<usize>() / values.len()
    }

    fn highest_possible_fitness(&self) -> usize {
        STRAND_SIZE / 4
    }

    fn lowest_possible_fitness(&self) -> usize {
        0
    }
}


// Build some random DNA strands.
struct RandomStrandBuilder;

impl GenomeBuilder<Genome> for RandomStrandBuilder {
    fn build_genome<R>(&self, _: usize, _: &mut R) -> Genome
    where
        R: Rng + Sized
    {
        (0..STRAND_SIZE)
            .map(|_| rand::random())
            .collect()
    }
}


// Runs a simulation based on a set of give parameters
fn run_sim_from_parms(parms: Option<Parameters>) {
    let parms = parms.unwrap_or_default();

    let initial_population: Population<Genome> = build_population()
        .with_genome_builder(RandomStrandBuilder)
        .of_size(parms.population_size)
        .uniform_at_random();

    let alg = genetic_algorithm()
        .with_evaluation(ClustersOf4FitnessCalculator)
        .with_selection(MaximizeSelector::new(
            parms.selection_ratio,
            parms.num_individuals_per_parents,
        ))
        .with_crossover(SinglePointCrossBreeder::new())
        .with_mutation(RandomValueMutator::new(
            parms.mutation_rate,
            Nucleotide::A,
            Nucleotide::A,
        ))
        .with_reinsertion(ElitistReinserter::new(
            ClustersOf4FitnessCalculator,
            true,
            parms.reinsertion_ratio,
        ))
        .with_initial_population(initial_population)
        .build();

    let mut simulator = simulate(alg)
        .until(or(
            FitnessLimit::new(ClustersOf4FitnessCalculator.highest_possible_fitness()),
            GenerationLimit::new(GENERATION_LIMIT)
        ))
        .build();

    println!("Starting a simulation with the following parameters: {:#?}", parms);

    loop {
        let result = simulator.step();
        match result {
            Ok(SimResult::Intermediate(step)) => {
                let _evaluated_population = step.result.evaluated_population;
                let _best_solution = step.result.best_solution;
                /*println!(
                    "Step #{}: average_fitness: {}, best fitness: {}, duration: {}, processing_time: {}",//\n\tpopulation: {:?}",
                    step.iteration,
                    evaluated_population.average_fitness(),
                    best_solution.solution.fitness,
                    step.duration.fmt(),
                    step.processing_time.fmt(),
                    //evaluated_population.individuals().iter().map(|x| x.as_phenome()).collect::<Vec<String>>()
                );*/
            }
            Ok(SimResult::Final(step, processing_time, duration, stop_reason)) => {
                let best_solution = step.result.best_solution;
                println!("{}", stop_reason);
                println!(
                    "Final Result after {}: generation: {}, best solution with fitness {} found in generation {}, processing_time: {}",//\n\tpopulation: {:?}",
                    duration.fmt(),
                    step.iteration,
                    best_solution.solution.fitness,
                    best_solution.generation,
                    processing_time.fmt(),
                    //step.result.evaluated_population.individuals().iter().map(|x| x.as_phenome()).collect::<Vec<String>>()
                );
                println!("\t{}", best_solution.solution.genome.as_phenome());
                break;
            }
            Err(error) => {
                println!("{}", error);
                break;
            }
        }
    }
}

fn main() {
    assert_eq!(STRAND_SIZE % 4, 0);

    run_sim_from_parms(None);
}
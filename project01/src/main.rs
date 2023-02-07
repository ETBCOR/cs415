use genevo::{
    self,
    prelude::*,
    selection::truncation::*,
    recombination::discrete::SinglePointCrossBreeder,
    reinsertion::elitist::ElitistReinserter, operator::prelude::{RandomValueMutation, RandomValueMutator}, types::fmt::Display
};
use rand::{
    distributions::{Distribution, Standard},
    Rng
};

const STRAND_SIZE: usize = 8;
const POPULATION_SIZE: usize = 10;
const GENERATION_LIMIT: u64 = 1000;
const NUM_INDIVIDUALS_PER_PARENTS: usize = 2;
const SELECTION_RATIO: f64 = 0.5;
const MUTATION_RATE: f64 = 0.05;
const REINSERTION_RATION: f64 = 0.5;

/// The phenotype
type Phenome = String;

/// The genotype
#[derive(Clone, Debug, PartialEq, PartialOrd)]
enum Nucleotide { A, C, T, G }
type Genome = Vec<Nucleotide>;

/// How do the genes of the genotype show up in the phenotype
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

/// The fitnes function for `Genome`s.
#[derive(Clone, Debug)]
struct FitnessCalc;

impl FitnessFunction<Genome, usize> for FitnessCalc {
    fn fitness_of(&self, genome: &Genome) -> usize {
        let mut t_count = 0;
        for g in genome.into_iter() {
            if *g == Nucleotide::T {
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

impl RandomValueMutation for Nucleotide {
    fn random_mutated<R>(_: Self, _: &Self, _: &Self, _: &mut R) -> Self
    where
        R: Rng + Sized
    {
        rand::random()
    }
}

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

fn main() {
    let initial_population: Population<Genome> = build_population()
        .with_genome_builder(RandomStrandBuilder)
        .of_size(POPULATION_SIZE)
        .uniform_at_random();

    let alg = genetic_algorithm()
        .with_evaluation(FitnessCalc)
        .with_selection(MaximizeSelector::new(
            SELECTION_RATIO,
            NUM_INDIVIDUALS_PER_PARENTS,
        ))
        .with_crossover(SinglePointCrossBreeder::new())
        .with_mutation(RandomValueMutator::new(
            MUTATION_RATE,
            Nucleotide::A,
            Nucleotide::A,
        ))
        .with_reinsertion(ElitistReinserter::new(
            FitnessCalc,
            true,
            REINSERTION_RATION,
        ))
        .with_initial_population(initial_population)
        .build();

    let mut simulator = simulate(alg)
        .until(or(
            FitnessLimit::new(FitnessCalc.highest_possible_fitness()),
            GenerationLimit::new(GENERATION_LIMIT)
        ))
        .build();

    println!("Starting simulation.");

    loop {
        let result = simulator.step();
        match result {
            Ok(SimResult::Intermediate(step)) => {
                let evaluated_population = step.result.evaluated_population;
                let best_solution = step.result.best_solution;
                println!(
                    "Step #{}: average_fitness: {}, best fitness: {}, duration: {}, processing_time: {}",//\n\tpopulation: {:?}",
                    step.iteration,
                    evaluated_population.average_fitness(),
                    best_solution.solution.fitness,
                    step.duration.fmt(),
                    step.processing_time.fmt(),
                    //evaluated_population.individuals().iter().map(|x| x.as_phenome()).collect::<Vec<String>>()
                );
            }
            Ok(SimResult::Final(step, processing_time, duration, stop_reason)) => {
                let best_solution = step.result.best_solution;
                println!("{}", stop_reason);
                println!(
                    "Final Result after {}: generation: {}, best solution with fitness {} found in generation {}, processing_time: {}\n\tpopulation: {:?}",
                    duration.fmt(),
                    step.iteration,
                    best_solution.solution.fitness,
                    best_solution.generation,
                    processing_time.fmt(),
                    step.result.evaluated_population.individuals().iter().map(|x| x.as_phenome()).collect::<Vec<String>>()
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
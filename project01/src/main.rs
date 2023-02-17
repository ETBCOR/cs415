use genevo::{
    self,
    operator::prelude::{RandomValueMutation, RandomValueMutator},
    prelude::*,
    recombination::discrete::SinglePointCrossBreeder,
    reinsertion::elitist::ElitistReinserter,
    selection::truncation::*,
};
use plotters::prelude::*;
use rand::{
    distributions::{Distribution, Standard},
    Rng,
};
use std::{
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

// Output file paths
const OUT_FILE: &'static str = "output/0.png";

// Unchangable simulation parameters
const STRAND_SIZE: usize = 100;
const POPULATION_SIZE: usize = 500;
const GENERATION_LIMIT: u64 = 16384; // 2^14
const BATCH_SIZE: u64 = 32;
// The Parameter struct defines the changable parameters need to run a simulation
#[derive(Debug, Clone, Copy)]
struct Parameters<'a> {
    parms_name: &'a str,
    num_individuals_per_parents: usize,
    selection_ratio: f64,
    mutation_rate: f64,
    reinsertion_ratio: f64,
}

/* impl<'a> Parameters<'a> {
    fn new(
        parms_name: &'a str,
        population_size: usize,
        num_individuals_per_parents: usize,
        selection_ratio: f64,
        mutation_rate: f64,
        reinsertion_ratio: f64,
    ) -> Self {
        Self {
            parms_name,
            population_size,
            num_individuals_per_parents,
            selection_ratio,
            mutation_rate,
            reinsertion_ratio,
        }
    }
} */

impl<'a> Default for Parameters<'a> {
    fn default() -> Self {
        Self {
            parms_name: "default",
            num_individuals_per_parents: 2,
            selection_ratio: 0.5,
            mutation_rate: 0.05,
            reinsertion_ratio: 0.5,
        }
    }
}

// The phenotype
type Phenome = String;

// The genotype
#[derive(Clone, Debug, PartialEq, PartialOrd)]
enum Nucleotide {
    A,
    C,
    T,
    G,
}
type Genome = Vec<Nucleotide>;

// How do the genes of the genotype show up in the phenotype
trait AsPhenotype {
    fn as_phenome(&self) -> Phenome;
}

impl AsPhenotype for Genome {
    fn as_phenome(&self) -> Phenome {
        self.into_iter()
            .map(|x| match x {
                Nucleotide::A => 'A',
                Nucleotide::C => 'C',
                Nucleotide::T => 'T',
                Nucleotide::G => 'G',
            })
            .collect::<String>()
    }
}

// Enable random Nucleotide generation
impl Distribution<Nucleotide> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Nucleotide {
        match rng.gen_range(0..4) {
            0 => Nucleotide::A,
            1 => Nucleotide::C,
            2 => Nucleotide::G,
            _ => Nucleotide::T,
        }
    }
}

impl RandomValueMutation for Nucleotide {
    fn random_mutated<R>(_: Self, _: &Self, _: &Self, _: &mut R) -> Self
    where
        R: Rng + Sized,
    {
        rand::random()
    }
}

// The "T" counting fitness function for `Genome`s.
/* #[derive(Clone, Debug)]
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
} */

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
        R: Rng + Sized,
    {
        (0..STRAND_SIZE).map(|_| rand::random()).collect()
    }
}

type Data = Vec<(u32, u32)>;

// Runs a simulation based on a set of give parameters
fn run_sim_from_parms(parms: &Parameters, thread_number: u64) -> Option<Data> {
    let initial_population: Population<Genome> = build_population()
        .with_genome_builder(RandomStrandBuilder)
        .of_size(POPULATION_SIZE)
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

    let mut sim = simulate(alg)
        .until(or(
            FitnessLimit::new(ClustersOf4FitnessCalculator.highest_possible_fitness()),
            GenerationLimit::new(GENERATION_LIMIT),
        ))
        .build();

    println!(
        "[thread #{}]: Starting a simulation with {} parameters.",
        thread_number, parms.parms_name
    );

    let mut data = vec![];

    loop {
        let result = sim.step();
        match result {
            Ok(SimResult::Intermediate(step)) => {
                let best_fitness = step.result.best_solution.solution.fitness;
                data.push((step.iteration as u32, best_fitness as u32));
            }
            Ok(SimResult::Final(step, _, _, _)) => {
                let best_fitness = step.result.best_solution.solution.fitness;
                if best_fitness == ClustersOf4FitnessCalculator.highest_possible_fitness() {
                    println!(
                        "[thread #{}]: Optimal solution was found after {} generations.",
                        thread_number, step.iteration
                    )
                } else {
                    println!(
                        "[thread #{}]: Optimal solution was not found after the max of {} generations.",
                        thread_number, step.iteration
                    );
                }
                data.push((step.iteration as u32, best_fitness as u32));
                return Some(data);
            }
            Err(error) => {
                println!("[thread #{}]: {}", thread_number, error);
                return None;
            }
        }
    }
}

fn run_sim_batch_from_parms(parms: &Parameters, runs: u64) {
    // Create a thread scope for parms
    thread::scope(|scope| {
        let start_time = Instant::now();
        let parms = Arc::new(parms);
        let sum = Arc::new(Mutex::new(0));
        let mut handles = vec![];

        // Create a pool of threads
        println!(
            "[thread pool]: Creating threadpool with {} parameters of size {}.",
            parms.parms_name, runs
        );
        for i in 1..=runs {
            let parms = Arc::downgrade(&parms);
            let sum = Arc::clone(&sum);
            let handle = scope.spawn(move || -> Option<_> {
                thread::sleep(Duration::from_millis(i * 20));
                let parms = parms.upgrade()?;
                let data = run_sim_from_parms(&parms, i)?;
                if (*data.last()?).1 as usize
                    == ClustersOf4FitnessCalculator.highest_possible_fitness()
                {
                    let mut sum = sum.lock().unwrap();
                    *sum += data.len();
                    Some(())
                } else {
                    None
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            if handle.join().unwrap() == None {
                println!(
                    "[thread pool]: With {} parameters, optimal solution was not always found within the generation limit!",
                    parms.parms_name
                );
                return;
            }
        }

        let avg = (*sum.lock().unwrap() as f64 / runs as f64).round();
        // Some(avg)
        println!(
            "[thread pool]: With {} paremeters, a perfect solution was found in all {} simulations, at generation {} on average. Batch took {} seconds.",
            parms.parms_name, runs, avg, start_time.elapsed().as_secs()
        );
    }) // thread::scope
}

fn generate_graph(data: Data, out_file: &'static str) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(out_file, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("caption", ("sans-serif", 32).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            1 as u32..data.len() as u32,
            ClustersOf4FitnessCalculator.lowest_possible_fitness() as u32
                ..ClustersOf4FitnessCalculator.highest_possible_fitness() as u32,
        )?;

    chart.configure_mesh().disable_x_mesh().x_labels(8).draw()?;

    chart
        .draw_series(LineSeries::new(data.iter().map(|(x, y)| (*x, *y)), &RED))?
        .label("v1")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}

fn main() {
    assert_eq!(STRAND_SIZE % 4, 0);

    let parms_default = Parameters::default();
    let parms_16_indiv_per_parent = Parameters {
        parms_name: "16-individuals-per-parent",
        num_individuals_per_parents: 16,
        selection_ratio: parms_default.selection_ratio,
        mutation_rate: parms_default.mutation_rate,
        reinsertion_ratio: parms_default.reinsertion_ratio,
    };

    //run_sim_batch_from_parms(&parms_default, BATCH_SIZE);
    //run_sim_batch_from_parms(&parms_16_indiv_per_parent, BATCH_SIZE);
    //generate_graph(OUT_FILE).unwrap();

    generate_graph(run_sim_from_parms(&parms_default, 0).unwrap(), OUT_FILE).unwrap();
}

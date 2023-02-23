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
    time::Instant,
};

// Output file paths
const OUT_DEFAULT: &'static str = "output/default_parameters.png";
const OUT_VAR_NUM_INDIV: &'static str = "output/variation_over_num_indiv.png";
const OUT_VAR_SELECTION: &'static str = "output/variation_over_selection_ratio.png";
const OUT_VAR_MUTATION: &'static str = "output/variation_over_mutation_rate.png";
const OUT_VAR_REINSERTION: &'static str = "output/variation_over_reinsertion_ratio.png";
const OUT_BEST_OF_EACH: &'static str = "output/best_value_of_each_varied_parameter.png";

// Unchanging simulation parameters
const STRAND_SIZE: usize = 100;
const POPULATION_SIZE: usize = 256;
const GENERATION_LIMIT: u64 = 16384; // 2^14
const BATCH_SIZE: u64 = 8;

// The Parameter struct defines the changing parameters need to run a simulation
#[derive(Debug, Clone)]
struct Parameters {
    parms_name: String,
    num_individuals_per_parents: usize,
    selection_ratio: f64,
    mutation_rate: f64,
    reinsertion_ratio: f64,
}

enum Variation {
    None,
    NumIdiv,
    Selection,
    Mutation,
    Reinsertion,
    BestOfEach,
}

impl<'a> Parameters {
    /*fn new(
        parms_name: String,
        num_individuals_per_parents: usize,
        selection_ratio: f64,
        mutation_rate: f64,
        reinsertion_ratio: f64,
    ) -> Self {
        Self {
            parms_name,
            num_individuals_per_parents,
            selection_ratio,
            mutation_rate,
            reinsertion_ratio,
        }
    }*/

    fn new_with_variation(var: Variation) -> Vec<Self> {
        let mut n = vec![Self::default(); 5];

        match var {
            Variation::None => {
                n.truncate(1);
            }
            Variation::NumIdiv => {
                n[0].num_individuals_per_parents = 2; // default
                n[0].parms_name = "num_indiv_per_parent = 2 (default)".to_string();
                n[1].num_individuals_per_parents = 4;
                n[1].parms_name = "num_indiv_per_parent = 4".to_string();
                n[2].num_individuals_per_parents = 16;
                n[2].parms_name = "num_indiv_per_parent = 16".to_string();
                n[3].num_individuals_per_parents = 32;
                n[3].parms_name = "num_indiv_per_parent = 32".to_string();
                n[4].num_individuals_per_parents = 64;
                n[4].parms_name = "num_indiv_per_parent = 64".to_string();
            }
            Variation::Selection => {
                n[0].selection_ratio = 0.1;
                n[0].parms_name = "selection_ratio = 0.10".to_string();
                n[1].selection_ratio = 0.25;
                n[1].parms_name = "selection_ratio = 0.25".to_string();
                n[2].selection_ratio = 0.5; // default
                n[2].parms_name = "selection_ratio = 0.50 (default)".to_string();
                n[3].selection_ratio = 0.75;
                n[3].parms_name = "selection_ratio = 0.75".to_string();
                n[4].selection_ratio = 1.0;
                n[4].parms_name = "selection_ratio = 1.00".to_string();
            }
            Variation::Mutation => {
                n[0].mutation_rate = 0.001;
                n[0].parms_name = "mutation_rate = 0.001".to_string();
                n[1].mutation_rate = 0.005;
                n[1].parms_name = "mutation_rate = 0.005".to_string();
                n[2].mutation_rate = 0.01;
                n[2].parms_name = "mutation_rate = 0.010".to_string();
                n[3].mutation_rate = 0.05; // default
                n[3].parms_name = "mutation_rate = 0.050 (default)".to_string();
                n[4].mutation_rate = 0.06;
                n[4].parms_name = "mutation_rate = 0.060".to_string();
            }
            Variation::Reinsertion => {
                n[0].reinsertion_ratio = 0.1;
                n[0].parms_name = "reinsertion_ratio = 0.10".to_string();
                n[1].reinsertion_ratio = 0.25;
                n[1].parms_name = "reinsertion_ratio = 0.25".to_string();
                n[2].reinsertion_ratio = 0.5; // default
                n[2].parms_name = "reinsertion_ratio = 0.50 (default)".to_string();
                n[3].reinsertion_ratio = 0.75;
                n[3].parms_name = "reinsertion_ratio = 0.75".to_string();
                n[4].reinsertion_ratio = 0.9;
                n[4].parms_name = "reinsertion_ratio = 0.90".to_string();
            }
            Variation::BestOfEach => {
                n.truncate(1);
                n[0].parms_name = "best of each varied parameter".to_string();
                n[0].num_individuals_per_parents = 64;
                n[0].selection_ratio = 1.0;
                n[0].mutation_rate = 0.01;
                n[0].reinsertion_ratio = 0.1;
            }
        }
        n
    }
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            parms_name: "default".to_string(),
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

type Data = Vec<u32>;
type DataSetWithLables = Vec<(String, Data)>;

// Runs a simulation based on a set of give parameters
fn run_sim_from_parms(parms: &Parameters, thread_number: Option<u64>) -> Option<DataSetWithLables> {
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

    match thread_number {
        Some(n) => println!(
            "[thread #{}]: Starting a simulation with {} parameters.",
            n, parms.parms_name
        ),
        None => println!(
            "Starting a simulation with {} parameters.",
            parms.parms_name
        ),
    }

    // Stores the best fitness value at each iteration of the simulation
    let mut data = vec![];

    // Iterate the simulation
    loop {
        let result = sim.step();
        match result {
            Ok(SimResult::Intermediate(step)) => {
                let best_fitness = step.result.best_solution.solution.fitness;
                // println!("parms: {} best_fitness: {}", parms.parms_name, best_fitness);

                // Push this intermediate result's best fitness to the vector
                data.push(best_fitness as u32);
            }
            Ok(SimResult::Final(step, _, _, _)) => {
                let best_fitness = step.result.best_solution.solution.fitness;

                // Push the final result's best fitness to the vector
                data.push(best_fitness as u32);

                // Print information about the final result
                if best_fitness == ClustersOf4FitnessCalculator.highest_possible_fitness() {
                    match thread_number {
                        Some(n) => println!(
                            "[thread #{}]: Optimal solution was found after {} generations with {} paremeters.",
                            n, step.iteration, parms.parms_name
                        ),
                        None => println!(
                            "Optimal solution was found after {} generations with {} paremeters.",
                            step.iteration, parms.parms_name
                        ),
                    }
                } else {
                    match thread_number {
                        Some(n) => println!(
                            "[thread #{}]: Optimal solution was not found after the max of {} generations with {} parameters.",
                            n, step.iteration, parms.parms_name
                        ),
                        None => println!(
                            "Optimal solution was not found after the max of {} generations with {} parameters.",
                            step.iteration, parms.parms_name
                        ),
                    }
                }

                // Because this result was final, return the data
                return Some(vec![(parms.parms_name.clone(), data)]);
            }
            Err(error) => {
                match thread_number {
                    Some(n) => println!("[thread #{}]: {}", n, error),
                    None => println!("{}", error),
                }

                // Return the none varient if we encouter an error
                return None;
            }
        }
    }
}

#[allow(dead_code)]
fn run_sim_batch_from_parms(parms: &Parameters) -> Option<DataSetWithLables> {
    // Create a thread scope for parms
    thread::scope(|scope| {
        // let start_time = Instant::now();

        let parms = Arc::new(parms);
        let sum = Arc::new(Mutex::new(0));
        let mut data: Vec<Data> = vec![];
        let mut handles = vec![];

        // Create a pool of threads
        println!(
            "[thread pool]: Creating a threadpool (batch size: {}) with parameters: {}",
            BATCH_SIZE, parms.parms_name
        );
        for i in 1..=BATCH_SIZE {
            let parms = Arc::downgrade(&parms);
            let sum = Arc::clone(&sum);

            // Spawn a new thread
            let handle = scope.spawn(move || -> Option<Data> {
                let parms = parms.upgrade()?;
                let data = run_sim_from_parms(&parms, Some(i))?.first()?.1.clone();
                if (*data.last()?) as usize
                    == ClustersOf4FitnessCalculator.highest_possible_fitness()
                {
                    let mut sum = sum.lock().unwrap();
                    *sum += data.len();
                    Some(data)
                } else {
                    None
                }
            });
            handles.push(handle);
        }

        // Wait for all the threads to finish
        for handle in handles {
            match handle.join().unwrap() {
                Some(d) => data.push(d),
                None => {
                    println!(
                        "[thread pool]: With {} parameters, optimal solution was not always found within the generation limit!",
                        parms.parms_name
                    );
                    return None;
                }
            }
        }

        let avg = (*sum.lock().unwrap() as f64 / BATCH_SIZE as f64).round();

        let max_size = data.iter().map(|d| d.len()).max().unwrap();
        let mut combined_data = vec![0; max_size];
        for (i, d) in combined_data.iter_mut().enumerate() {
            for s in data.iter() {
                *d += if i < s.len() {
                    s[i]
                } else {
                    ClustersOf4FitnessCalculator.highest_possible_fitness() as u32
                };
            }
            *d = (*d as f64 / BATCH_SIZE as f64) as u32;
        }

        println!(
            "[thread pool]: With {} paremeters, a perfect solution was found in all {} simulations, at generation {} on average",//. Batch took {} seconds.",
            parms.parms_name, BATCH_SIZE, avg//, start_time.elapsed().as_secs()
        );
        Some(vec![(parms.parms_name.clone(), combined_data)])
    }) // thread::scope
}

fn run_sim_batch_from_parms_list(parms_list: &Vec<Parameters>) -> Option<DataSetWithLables> {
    println!("num of parms in the parm list: {}", parms_list.len());
    // Create a thread scope for parms
    thread::scope(|scope| {
        // let start_time = Instant::now();

        let parms_list = parms_list
            .iter()
            .map(|p| Arc::new(p))
            .collect::<Vec<Arc<&Parameters>>>();
        let sums_list = vec![Arc::new(Mutex::new(0)); parms_list.len()];
        let mut data_list: Vec<Vec<Data>> = vec![vec![]; parms_list.len()];
        let mut handles = vec![];

        // Create a pool of threads
        println!(
            "[thread pool]: Creating a threadpool (batch size: {}) with a parameters list: {:?}",
            BATCH_SIZE, parms_list
        );
        for i in 0..BATCH_SIZE {
            for (j, parms) in parms_list.iter().enumerate() {
                let parms = Arc::downgrade(&parms);
                let sum = Arc::clone(&sums_list[j]);

                // Spawn a new thread
                let handle = scope.spawn(move || -> (usize, Option<Data>) {
                    let parms = match parms.upgrade() {
                        Some(parms) => parms,
                        None => return (j, None),
                    };

                    let data =
                        match run_sim_from_parms(&parms, Some((j as u64 * BATCH_SIZE + i) + 1)) {
                            Some(data) => data,
                            None => return (j, None),
                        };
                    let data = match data.first() {
                        Some(data) => data,
                        None => return (j, None),
                    }
                    .1
                    .clone();

                    if (*match data.last() {
                        Some(l) => l,
                        None => return (j, None),
                    }) as usize
                        == ClustersOf4FitnessCalculator.highest_possible_fitness()
                    {
                        let mut sum = sum.lock().unwrap();
                        *sum += data.len();
                        (j, Some(data))
                    } else {
                        (j, None)
                    }
                });
                handles.push(handle);
            }
        }

        // Wait for all the threads to finish
        for handle in handles {
            match handle.join().unwrap() {
                (i, Some(d)) => {
                    println!("[thread pool]: Thread #{} finished.", i);
                    data_list[i].push(d);
                }
                (i, None) => {
                    println!(
                        "[thread pool]: With {} parameters, optimal solution was not always found within the generation limit!",
                        parms_list[i].parms_name
                    );
                    return None;
                }
            }
        }

        let mut combined_data_list = vec![];

        for (i, data) in data_list.iter().enumerate() {
            let _avg = (*sums_list[i].lock().unwrap() as f64 / BATCH_SIZE as f64).round();

            let max_size = data.iter().map(|d| d.len()).max().unwrap();
            let mut combined_data = vec![0; max_size];
            for (i, d) in combined_data.iter_mut().enumerate() {
                for s in data.iter() {
                    *d += if i < s.len() {
                        s[i]
                    } else {
                        ClustersOf4FitnessCalculator.highest_possible_fitness() as u32
                    };
                }
                *d = (*d as f64 / BATCH_SIZE as f64) as u32;
            }
            combined_data_list.push((parms_list[i].parms_name.clone(), combined_data));
        }

        let max_size = combined_data_list
            .iter()
            .map(|(_, d)| d.len())
            .max()
            .unwrap();
        for (_, combined_data) in combined_data_list.iter_mut() {
            while combined_data.len() < max_size {
                combined_data.push(ClustersOf4FitnessCalculator.highest_possible_fitness() as u32);
            }
        }

        Some(combined_data_list)
    }) // thread::scope
}

fn generate_graph(
    graph_name: &str,
    dataset: DataSetWithLables,
    out_file: &'static str,
) -> Result<(), Box<dyn std::error::Error>> {
    let gens = dataset.first().unwrap().1.len() as u32;

    let root = BitMapBackend::new(out_file, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(graph_name, ("Consolas", 40).into_font())
        .margin(10)
        .x_label_area_size(20)
        .y_label_area_size(20)
        .build_cartesian_2d(
            1 as u32..gens,
            ClustersOf4FitnessCalculator.lowest_possible_fitness() as u32
                ..ClustersOf4FitnessCalculator.highest_possible_fitness() as u32,
        )?;

    chart.configure_mesh().disable_x_mesh().x_labels(5).draw()?;

    // Draw each line in the dataset
    for (i, (label, data)) in dataset.iter().enumerate() {
        let data = data.iter().enumerate();
        let color = Palette99::pick(i).mix(0.6);
        chart
            .draw_series(LineSeries::new(
                data.map(|(x, y)| (x as u32 + 1, *y)),
                color.stroke_width(2),
            ))?
            .label(label.clone())
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.filled()));
    }

    chart
        .configure_series_labels()
        .label_font(("Consolas", 18).into_font())
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}

fn generate_graph_with_variation(
    graph_name: &str,
    variation: Variation,
    out_file: &'static str,
) -> Result<(), Box<dyn std::error::Error>> {
    let parms_list = Parameters::new_with_variation(variation);
    let data = run_sim_batch_from_parms_list(&parms_list).unwrap();
    generate_graph(graph_name, data, out_file)?;
    Ok(())
}

fn main() {
    assert_eq!(STRAND_SIZE % 4, 0);

    let start_time = Instant::now();
/*
    generate_graph_with_variation("Default Parameters", Variation::None, OUT_DEFAULT).unwrap();

    generate_graph_with_variation(
        "Various Numbers of Individuals Per Parent",
        Variation::NumIdiv,
        OUT_VAR_NUM_INDIV,
    )
    .unwrap();

    generate_graph_with_variation(
        "Various Selection Ratios",
        Variation::Selection,
        OUT_VAR_SELECTION,
    )
    .unwrap();

    generate_graph_with_variation(
        "Various Mutation Rates",
        Variation::Mutation,
        OUT_VAR_MUTATION,
    )
    .unwrap();

    generate_graph_with_variation(
        "Various Reinsertion Ratios",
        Variation::Reinsertion,
        OUT_VAR_REINSERTION,
    )
    .unwrap();
*/

    generate_graph_with_variation("Using Best Value of Each Varied Parameter", Variation::BestOfEach, OUT_BEST_OF_EACH).unwrap();

    println!(
        "Finished execution in {} seconds!",
        start_time.elapsed().as_secs()
    );
}

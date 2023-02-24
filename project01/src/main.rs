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
    fs::remove_file,
    io::ErrorKind,
    sync::{Arc, Mutex},
    thread,
    time::Instant,
};

// Output file paths and flags for whether or not to generate the file
const OUT_DEFAULT: (&'static str, bool) = ("output/default_parameters.png", true);
const OUT_VAR_NUM_INDIV: (&'static str, bool) = ("output/various_num_indivs.png", true);
const OUT_VAR_SELECTION: (&'static str, bool) = ("output/various_selection_ratios.png", true);
const OUT_VAR_MUTATION: (&'static str, bool) = ("output/various_mutation_rates.png", true);
const OUT_VAR_REINSERTION: (&'static str, bool) = ("output/various_reinsertion_ratios.png", true);
const OUT_BEST_OF_EACH: (&'static str, bool) = ("output/best_of_each_varied_parm.png", true);

// Unchanging simulation parameters
const STRAND_SIZE: usize = 100;
const POPULATION_SIZE: usize = 256;
const GENERATION_LIMIT: u64 = 16384; // 2^14
const BATCH_SIZE: u64 = 16;

// The Parameter struct defines the changing parameters need to run a simulation
#[derive(Debug, Clone)]
struct Parameters {
    parms_name: String,
    num_individuals_per_parents: usize,
    selection_ratio: f64,
    mutation_rate: f64,
    reinsertion_ratio: f64,
}

#[derive(Debug, Default)]
enum Variation {
    #[default]
    Default,
    NumIdiv(Vec<usize>),
    Selection(Vec<f64>),
    Mutation(Vec<f64>),
    Reinsertion(Vec<f64>),
    BestOfEach,
}

impl<'a> Parameters {
    fn new(var: &Variation) -> Vec<Self> {
        let mut parms_list: Vec<Parameters> = vec![];

        match var {
            Variation::Default => {
                parms_list.push(Parameters::default());
            }
            Variation::NumIdiv(v) => {
                for x in v {
                    let mut p = Parameters::default();
                    p.parms_name = format!(
                        "num_indiv_per_parent = {}{}",
                        *x,
                        if *x == p.num_individuals_per_parents {
                            " (default)"
                        } else {
                            ""
                        }
                    );
                    p.num_individuals_per_parents = *x;
                    parms_list.push(p);
                }
            }
            Variation::Selection(v) => {
                for x in v {
                    let mut p = Parameters::default();
                    p.parms_name = format!(
                        "selection_ratio = {}{}",
                        *x,
                        if *x == p.selection_ratio {
                            " (default)"
                        } else {
                            ""
                        }
                    );
                    p.selection_ratio = *x;
                    parms_list.push(p);
                }
            }
            Variation::Mutation(v) => {
                for x in v {
                    let mut p = Parameters::default();
                    p.parms_name = format!(
                        "mutation_rate = {}{}",
                        *x,
                        if *x == p.mutation_rate {
                            " (default)"
                        } else {
                            ""
                        }
                    );
                    p.mutation_rate = *x;
                    parms_list.push(p);
                }
            }
            Variation::Reinsertion(v) => {
                for x in v {
                    let mut p = Parameters::default();
                    p.parms_name = format!(
                        "reinsertion_ratio = {}{}",
                        *x,
                        if *x == p.reinsertion_ratio {
                            " (default)"
                        } else {
                            ""
                        }
                    );
                    p.reinsertion_ratio = *x;
                    parms_list.push(p);
                }
            }
            Variation::BestOfEach => {
                parms_list.push(Parameters {
                    parms_name: "best of each varied parm".to_string(),
                    num_individuals_per_parents: 128,
                    selection_ratio: 1.0,
                    mutation_rate: 0.01,
                    reinsertion_ratio: 0.1,
                });
            }
        }
        parms_list
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
            "\t[thread #{}]: Starting a simulation with {} parms.",
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
                // println!("parms: {} best_fitness: {}", parms.parms_name, best_fitness); // spam std out with best fitness

                // Push this intermediate result's best fitness to the vector
                data.push(best_fitness as u32);
            }
            Ok(SimResult::Final(step, _, _, _)) => {
                let best_fitness = step.result.best_solution.solution.fitness;

                // Push the final result's best fitness to the vector
                data.push(best_fitness as u32);

                // Print information about the final result
                println!(
                    "{}Optimal solution was {}found after {} generationns with {} parms.",
                    if let Some(n) = thread_number {
                        format!("\t[thread #{}]: ", n)
                    } else {
                        "".to_string()
                    },
                    if best_fitness == ClustersOf4FitnessCalculator.highest_possible_fitness() {
                        ""
                    } else {
                        "not "
                    },
                    step.iteration,
                    parms.parms_name
                );

                // Because this result was final, return the data
                return Some(vec![(parms.parms_name.clone(), data)]);
            }
            Err(error) => {
                match thread_number {
                    Some(n) => println!("\t[thread #{}]: {}", n, error),
                    None => println!("{}", error),
                }

                // Return the none varient if we encouter an error
                return None;
            }
        }
    }
}

// Runs a simulation batch from a given parameters list. Returns an option of a labled dataset
fn run_sim_batch(
    parms_list: &Vec<Parameters>,
    variation: Option<Variation>,
) -> Option<DataSetWithLables> {
    // Create a thread scope for parms
    thread::scope(|scope| {
        let parms_list = parms_list
            .iter()
            .map(|p| Arc::new(p))
            .collect::<Vec<Arc<&Parameters>>>();
        let variation = variation.unwrap_or_default();
        let sums_list = vec![Arc::new(Mutex::new(0)); parms_list.len()];
        let mut data_list: Vec<Vec<Data>> = vec![vec![]; parms_list.len()];
        let mut handles = vec![];

        // Create a pool of threads
        let start_time = Instant::now();
        println!(
            "[thread pool]: Creating a threadpool (batch size: {}) with {:?} variation.",
            BATCH_SIZE, variation
        );
        for thread_idx in 0..BATCH_SIZE {
            for (parm_idx, parms) in parms_list.iter().enumerate() {
                let parms = Arc::downgrade(&parms);
                let sum = Arc::clone(&sums_list[parm_idx]);

                // Spawn a new thread
                let handle = scope.spawn(move || -> (u64, usize, Option<Data>) {
                    let parms = match parms.upgrade() {
                        Some(parms) => parms,
                        None => return (thread_idx + 1, parm_idx, None),
                    };

                    let data = match run_sim_from_parms(
                        &parms,
                        Some((parm_idx as u64 * BATCH_SIZE + thread_idx) + 1),
                    ) {
                        Some(data) => data,
                        None => return (thread_idx + 1, parm_idx, None),
                    };
                    let data = match data.first() {
                        Some(data) => data,
                        None => return (thread_idx + 1, parm_idx, None),
                    }
                    .1
                    .clone();

                    if (*match data.last() {
                        Some(l) => l,
                        None => return (thread_idx + 1, parm_idx, None),
                    }) as usize
                        == ClustersOf4FitnessCalculator.highest_possible_fitness()
                    {
                        let mut sum = sum.lock().unwrap();
                        *sum += data.len();
                        (thread_idx + 1, parm_idx, Some(data))
                    } else {
                        (thread_idx + 1, parm_idx, None)
                    }
                });
                handles.push(handle);
            }
        }

        // Wait for all the threads to finish
        for handle in handles {
            match handle.join().unwrap() {
                (thread_idx, parm_idx, Some(d)) => {
                    println!(
                        "[thread pool]: Joined thread #{}.",
                        (parm_idx as u64 * BATCH_SIZE + thread_idx) + 1
                    );
                    data_list[parm_idx].push(d);
                }
                (thread_idx, parm_idx, None) => {
                    println!(
                        "[thread pool]: With {} parameters, optimal solution was not always found within the generation limit! Failed in thread #{}.",
                        parms_list[parm_idx].parms_name, (parm_idx as u64 * BATCH_SIZE + thread_idx) + 1
                    );
                    return None;
                }
            }
        }

        // Combine the data into a labeled dataset
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

        println!(
            "[thread pool]: Finished threadpool with {:?} variation after {} seconds.\n",
            variation,
            start_time.elapsed().as_secs()
        );

        Some(combined_data_list)
    }) // thread::scope
}

fn generate_graph(
    graph_name: &str,
    mut dataset: DataSetWithLables,
    out_file: &'static str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Store the gen at which each simulation finished
    let gens_list = dataset
        .iter()
        .map(|d| d.1.len() as u32)
        .collect::<Vec<u32>>();
    // And the max gens any simulation took (width of graph)
    let gens_max = *gens_list.iter().max().unwrap();

    // Normalize the length of each of the lines in the dataset
    for (_, d) in dataset.iter_mut() {
        while (d.len() as u32) < gens_max {
            d.push(ClustersOf4FitnessCalculator.highest_possible_fitness() as u32);
        }
    }

    // Drawing root
    let root = BitMapBackend::new(out_file, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    // Chart
    let mut chart = ChartBuilder::on(&root)
        .caption(graph_name, ("Consolas", 50).into_font())
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .build_cartesian_2d(
            1 as u32..gens_max,
            ClustersOf4FitnessCalculator.lowest_possible_fitness() as u32
                ..ClustersOf4FitnessCalculator.highest_possible_fitness() as u32,
        )?;

    // Mesh configuration
    chart
        .configure_mesh()
        .y_labels(6)
        .x_labels(16)
        .y_desc("fitness")
        .x_desc("gens")
        .label_style(("Consolas", 25).into_font())
        .draw()?;

    // Draw each line in the dataset
    for (idx, (label, data)) in dataset.iter().enumerate() {
        let data = data.iter().enumerate();
        let color = Palette99::pick(idx).mix(0.6);

        chart
            .draw_series(LineSeries::new(
                data.map(|(x, y)| (x as u32 + 1, *y)),
                color.stroke_width(3),
            ))?
            .label(format!("{} (gens: {})", label.clone(), gens_list[idx]))
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(3))
            });
    }

    chart
        .configure_series_labels()
        .label_font(("Consolas", 25).into_font())
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}

fn generate_graph_from_variation(
    graph_name: &str,
    variation: Variation,
    out_file: (&'static str, bool),
) -> Result<(), Box<dyn std::error::Error>> {
    if out_file.1 {
        let parms_list = Parameters::new(&variation);
        let data = run_sim_batch(&parms_list, Some(variation)).unwrap();
        generate_graph(graph_name, data, out_file.0)?;
    }
    Ok(())
}

fn delete_file(file: (&'static str, bool)) {
    if file.1 {
        match remove_file(file.0) {
            Ok(_) => (),
            Err(error) if error.kind() == ErrorKind::NotFound => { /* do nothing */ }
            Err(error) => panic!("Problem deleting file: {:?}", error),
        }
    }
}

fn main() {
    assert_eq!(STRAND_SIZE % 4, 0);

    delete_file(OUT_DEFAULT);
    delete_file(OUT_VAR_NUM_INDIV);
    delete_file(OUT_VAR_SELECTION);
    delete_file(OUT_VAR_MUTATION);
    delete_file(OUT_VAR_REINSERTION);
    delete_file(OUT_BEST_OF_EACH);

    let start_time = Instant::now();

    generate_graph_from_variation("Default Parameters", Variation::Default, OUT_DEFAULT).unwrap();

    generate_graph_from_variation(
        "Various Numbers of Individuals Per Parent",
        Variation::NumIdiv(vec![2, 4, 8, 16, 32, 64, 128]),
        OUT_VAR_NUM_INDIV,
    )
    .unwrap();

    generate_graph_from_variation(
        "Various Selection Ratios",
        Variation::Selection(vec![0.25, 0.5, 1.0, 2.0, 4.0, 8.0]),
        OUT_VAR_SELECTION,
    )
    .unwrap();

    generate_graph_from_variation(
        "Various Mutation Rates",
        Variation::Mutation(vec![0.001, 0.005, 0.01, 0.025, 0.05]),
        OUT_VAR_MUTATION,
    )
    .unwrap();

    generate_graph_from_variation(
        "Various Reinsertion Ratios",
        Variation::Reinsertion(vec![0.01, 0.1, 0.25, 0.5, 0.75, 0.9]),
        OUT_VAR_REINSERTION,
    )
    .unwrap();

    generate_graph_from_variation(
        "Using Best Value of Each Varied Parameter",
        Variation::BestOfEach,
        OUT_BEST_OF_EACH,
    )
    .unwrap();

    println!(
        "Finished execution in {} seconds!",
        start_time.elapsed().as_secs()
    );
}

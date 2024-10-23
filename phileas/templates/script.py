import argparse
import datetime
import logging
import pathlib
import shutil
import sys

import rich.logging
from tqdm.rich import tqdm

import phileas

{% if experiment_config is defined %}
import {{ experiment_config }}

    {% for dependency in dependencies %}
import {{ dependency }}

    {% endfor %}
{% else %}
# Import here the module defining all the loaders that you want to use.
import <path_of_the_experiment_config_module>

{% endif %}

# Configure logging
console_handler = rich.logging.RichHandler(rich_tracebacks=True)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter("%(name)s:: %(message)s"))

{% if not no_explanation %}
# Use `logger` to log any information about your experiment.
{% endif %}
logger = logging.getLogger("experiment")
logger.setLevel(logging.DEBUG)
logger.propagate = False
logger.addHandler(console_handler)

logging.getLogger("phileas").addHandler(console_handler)
logging.getLogger("phileas").setLevel(logging.INFO)
logging.getLogger("phileas").propagate = False

{% if not no_cli %}
def get_cli_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("<Short description of the script>")
    parser.add_argument(
        "--bench",
        "-b",
        help="Bench configuration file.",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "--experiment",
        "-e",
        help="Experiment configuration file.",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "--results",
        "-r",
        help="Directory to save results to. If not set, results are not stored",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--dry-run",
        "-d",
        help=(
            "When set, just prepare the instruments required by the experiment "
            "file, apply their default configuration, and exit."
        ),
        action="store_true",
    )

    return parser.parse_args()


def prepare_save_directory(
    args: argparse.Namespace, exp_description: dict
) -> pathlib.Path | None:
    if args.results is None:
        logger.warning("The campaign results will not be saved.")
        return None

    start_time = datetime.datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    chip = exp_description["chip"]
    target = exp_description["target"]
    exp_name = exp_description["name"]
    name = f"./{chip}_{target}_{exp_name}_{timestamp}/"
    save_directory = args.results / pathlib.Path(name)

    # You can specify here which files you want to save
    # For a project managed with pyproject and Poetry, it is advised to copy
    # `pyproject.toml` and `poetry.lock`.
    logger.info(f"Prepare save directory {save_directory}.")
    save_directory.mkdir(parents=True)
    shutil.copy(args.bench, save_directory / pathlib.Path("bench.yaml"))
    shutil.copy(args.experiment, save_directory / pathlib.Path("experiment.yaml"))
    {% if experiment_config is none %}
    shutil.copy(<experiment_config_path>, save_directory)
    {% else %}
    shutil.copy({{ experiment_config }}, save_directory)
    {% endif %}
    shutil.copy(__file__, save_directory / pathlib.Path(__file__).name)
    shutil.copy("pyproject.toml", save_directory)
    shutil.copy("poetry.lock", save_directory)

    file_logs_handler = logging.FileHandler(save_directory / pathlib.Path("logs.txt"))
    file_logs_handler.setFormatter(
        logging.Formatter("%(asctime)s - (%(levelname)s) %(name)s:: %(message)s")
    )
    file_logs_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_logs_handler)

    return save_directory


{% endif %}
if __name__ == "__main__":
    {% if not no_cli %}
    ### CLI ###

    args = get_cli_arguments()
    if args.results is None:
        logger.warning(
            "This run won't save anything to the filesystem. Specify the "
            "--results option if you want to save the results."
        )

    {% endif %}
    ### Load configuration files ###

    {% if not no_cli %}
    factory = phileas.ExperimentFactory(args.bench, args.experiment)
    {% else %}
    factory = phileas.ExperimentFactory(<bench_file_path>, <experiment_file_path>)
    {% endif %}
    p_experiment_config = factory.experiment_config.to_pseudo_data_tree()
    assert isinstance(p_experiment_config, dict)

    ### Prepare iteration tree ###

    {% if not no_explanation %}
    # The experiment configuration does not allow to represent any iteration
    # tree. In particular, all the internal nodes of the tree are parsed as
    # cartesian products. So, if you want to use other iteration methods - or
    # modify the iteration tree in any other way -, you can do it here.
    {% endif %}
    factory.experiment_config = factory.experiment_config.replace_node(
        [], phileas.iteration.CartesianProduct, lazy=True, snake=True
    ).replace_node(
        ["instrument1"], phileas.iteration.CartesianProduct, lazy=True, snake=False
    )

    {% if not no_cli %}
    ### Prepare save directory ###

    description = p_experiment_config["description"]
    assert isinstance(description, dict)
    save_directory = prepare_save_directory(args, description)

    {% endif %}
    #### Instruments setup ####

    logger.info("Start instruments setup.")

    {% if not no_explanation %}
    # Prepare the drivers of all the bench instruments, using the
    # `initiate_connection` method of their loader.
    {% endif %}
    factory.initiate_connections()
    {% if not no_explanation %}
    # Configure all the instruments to their default configuration.
    {% endif %}
    factory.configure_experiment()

    {% for instrument, instrument_type in instruments_and_type %}
        {% if instrument_type is not none %}
    {{ instrument }}: {{ instrument_type }} = factory.experiment_instruments["{{ instrument }}"]
        {% else %}
    {{ instrument }} = factory.experiment_instruments["{{ instrument }}"]
        {% endif %}
    {% endfor %}

    logger.info("Instruments setup is done.")

    if args.dry_run:
        logger.info("Dry run terminated")
        sys.exit(0)

    ### Run the experiment ###

    start_date = datetime.datetime.now()
    try:
        logger.info("Start the experiment.")

        {% if not no_explanation %}
        # This iterates over all the configurations of the iteration tree, while
        # displaying a progress bar.
        {% endif %}
        progress_bar = tqdm.tqdm(factory.experiment_config, smoothing=0.01)
        global_config = dict()
        for config in progress_bar:
            assert isinstance(config, dict)
            global_config |= config
            factory.configure_experiment(config)

            {% if not no_explanation %}
            # Carry out measurements or use the instruments here.
            {% endif %}

            {% if not no_cli %}
            # Intermediate save
            if save_directory is not None:
                logger.info("Partial results saved.")
            {% endif %}

            {% if not no_explanation %}
            # Display information alongside the progress bar here.
            {% endif %}
            progress_bar.set_postfix(
                {
                    "param1": f"{global_config['param1']:.3f}",
                    "param2": f"{global_config['param2']:.3f}",
                    "param3": f"{global_config['param3']:.3f}",
                }
            )
        logger.info("The experiment has been successfully carried out.")
    except:
        logger.warning(
            "Interrupting the experiment early because an exception was "
            "raised, see the log file for more details."
        )
        logger.debug(
            "Interrupting the experiment because of the following exception.",
            exc_info=True,
            stack_info=True,
        )
    finally:
        stop_date = datetime.datetime.now()
        logger.info(f"The experiment lasted for {stop_date - start_date}.")

        {% if not no_cli %}
        if save_directory is not None:
            logger.info(f"Results saved to {save_directory}.")

        {% endif %}
        try:
            factory.configure_experiment()
        except:
            logging.warning(
                "The instruments could not be configured to their default state."
            )
            logger.debug(
                "The following exception was raised while configuring the "
                "instruments back to their default settings.",
                exc_info=True,
                stack_info=True,
            )

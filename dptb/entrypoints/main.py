import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dptb.entrypoints.train import train
from dptb.entrypoints.postrun import postrun
from dptb.utils.loggers import set_log_handles

def get_ll(log_level: str) -> int:
    """Convert string to python logging level.

    Parameters
    ----------
    log_level : str
        allowed input values are: DEBUG, INFO, WARNING, ERROR, 3, 2, 1, 0

    Returns
    -------
    int
        one of python logging module log levels - 10, 20, 30 or 40
    """
    if log_level.isdigit():
        int_level = (4 - int(log_level)) * 10
    else:
        int_level = getattr(logging, log_level)

    return int_level

def main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepTB: A deep learning package for Tight-Binding Model"
                    " with first-principle accuracy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(title="Valid subcommands", dest="command")

    # log parser
    parser_log = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_log.add_argument(
        "-v",
        "--log-level",
        choices=["DEBUG", "3", "INFO", "2", "WARNING", "1", "ERROR", "0"],
        default="INFO",
        help="set verbosity level by string or number, 0=ERROR, 1=WARNING, 2=INFO "
             "and 3=DEBUG",
    )

    parser_log.add_argument(
        "-l",
        "--log-path",
        type=str,
        default=None,
        help="set log file to log messages to disk, if not specified, the logs will "
             "only be output to console",
    )

    # train parser
    parser_train = subparsers.add_parser(
        "train",
        parents=[parser_log],
        help="train a model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser_train.add_argument(
        "INPUT", help="the input parameter file in json or yaml format",
        type=str,
        default=None
    )
    parser_train.add_argument(
        "-i",
        "--init-model",
        type=str,
        default=None,
        help="Initialize the model by the provided checkpoint.",
    )
    
    parser_train.add_argument(
        "-r",
        "--restart",
        type=str,
        default=None,
        help="Restart the training from the provided checkpoint.",
    )

    parser_train.add_argument(
        "-sk",
        "--train-sk",
        action="store_true",
        help="Trainging NNSKTB parameters.",
    )

    parser_train.add_argument(
        "-crt",
        "--use-correction",
        type=str,
        default=None,
        help="Use nnsktb correction when training dptb",
    )

    parser_train.add_argument(
        "-f",
        "--freeze",
        action="store_true",
        help="Initialize the training from the frozen model.",
    )

    parser_train.add_argument(
        "-o",
        "--output",
        type=str,
        default="./",
        help="The output files in training.",
    )
    
    parser_run = subparsers.add_parser(
        "run",
        parents=[parser_log],
        help="run the TB with a model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser_run.add_argument(
        "INPUT", help="the input parameter file for postprocess run in json format",
        type=str,
        default=None
    )

    parser_run.add_argument(
        "-ckpt", 
        "--model_ckpt",
        help="the checkpointfile for postprocess run in json format, prior to the model_ckpt tags in the input json. ",
        type=str,
        default=None
    )
    
    parser_run.add_argument(
        "-str",
        "--structure",
        type=str,
        default=None,
        help="the structure file name wiht its suffix of format, such as, .vasp, .cif etc., prior to the model_ckpt tags in the input json. "
    )

    parser_run.add_argument(
        "-o",
        "--output",
        type=str,
        default="./",
        help="The output files in postprocess run."
    )

    parser_run.add_argument(
        "-sk",
        "--run_sk",
        action="store_true",
        help="using NNSKTB parameters TB models for post-run."
    )

    parser_run.add_argument(
        "-crt",
        "--use-correction",
        type=str,
        default=None,
        help="Use nnsktb correction when training dptb",
    )



    return parser

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse arguments and convert argument strings to objects.

    Parameters
    ----------
    args: List[str]
        list of command line arguments, main purpose is testing default option None
        takes arguments from sys.argv

    Returns
    -------
    argparse.Namespace
        the populated namespace
    """
    parser = main_parser()
    parsed_args = parser.parse_args(args=args)
    if parsed_args.command is None:
        parser.print_help()
    else:
        parsed_args.log_level = get_ll(parsed_args.log_level)

    return parsed_args

def main():
    args = parse_args()

    if args.command not in (None, "train", "run"):
        set_log_handles(args.log_level, Path(args.log_path) if args.log_path else None)

    dict_args = vars(args)

    if args.command == 'train':
        train(**dict_args)

    if args.command == 'run':
        postrun(**dict_args)

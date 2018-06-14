#! /usr/bin/env python
import argparse as ap
import logging
import os
import re
import sys

import fimpg
import pandas as pd


def parse_pos(pos, option):
    """
    Parse a specified genomic position.
    Should be digits followed optionally by case-insensitive Mb or Kb modifiers.
    """
    match = re.match("^(\d+)(mb|kb)?$", pos, flags=re.IGNORECASE)
    position = None
    if match:
        pos_tmp = int(match.group(1)) # position
        pos_mod = match.group(2) # modifier
        if pos_mod:
            pos_mod = pos_mod.upper()
            if pos_mod == "MB":
                pos_tmp *= 1000000
            elif pos_mod == "KB":
                pos_tmp *= 1000

        position = pos_tmp
    else:
        raise ValueError("Option {} {} is an invalid genomic position".format(option, pos))

    return position


def get_command_string(args):
    """
    Format fimpg call and options into a string for logging/printing
    """
    base = "fimpg.py " + " ".join(args[:2]) + os.linesep
    rest = args[2:]
    rest_strs = []
    for idx, cmd in enumerate(rest[::2]):
        rest_strs.append("\t{}".format(cmd) + " " + rest[idx + 1] + os.linesep)

    return base + "".join(rest_strs) + os.linesep


def main(argsv):
    argp = ap.ArgumentParser(description="GWAS summary statistics imputation with functional data.")

    # main arguments
    argp.add_argument("gwas", type=ap.FileType("r"), help="GWAS summary data.")
    argp.add_argument("ref", help="Path to reference panel PLINK data.")

    # imputation location options
    argp.add_argument("--chr",
        help="Perform imputation for specific chromosome.")
    argp.add_argument("--start",
        help="Perform imputation starting at specific location (in base pairs). Requires --chr to be specified.")
    argp.add_argument("--stop",
        help="Perform imputation until at specific location (in base pairs). Requires --chr to be specified.")
    argp.add_argument("--locations", type=ap.FileType("r"),
        help="Perform imputation at locations listed in bed-formatted file.")

    # imputation options
    argp.add_argument("--window-size", type=int,
        help="Size of imputation window (in base pairs).")

    # misc options
    argp.add_argument("--quiet", default=False, action="store_true",
        help="Do not print anything to stdout")
    argp.add_argument("-o", "--output", default="FIMPG",
        help="Prefix for output data")

    args = argp.parse_args(argsv)

    try:
        # basic arg check passed, reprint to log before formatted logging begins
        cmd_str = get_command_string(argsv)

        masthead =  "====================================" + os.linesep
        masthead += "              FIMPG v{}             ".format(fimpg.VERSION) + os.linesep
        masthead += "====================================" + os.linesep

        # setup logging
        FORMAT = "[%(asctime)s - %(levelname)s] %(message)s"
        DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
        log = logging.getLogger("fimpg")
        fmt = logging.Formatter(fmt=FORMAT, datefmt=DATE_FORMAT)

        # setup log file, but write PLINK-style command first
        disk_log_stream = open("{}.log".format(args.output), "w")
        disk_log_stream.write(masthead)
        disk_log_stream.write(cmd_str)
        disk_log_stream.write("Starting log..." + os.linesep)

        disk_handler = logging.StreamHandler(disk_log_stream)
        disk_handler.setFormatter(fmt)
        disk_handler.setLevel(logging.INFO)
        log.addHandler(disk_handler)

        # write to stdout unless quiet is set
        if not args.quiet:
            # setup log file, but write PLINK-style command first
            sys.stdout.write(masthead)
            sys.stdout.write(cmd_str)
            sys.stdout.write("Starting log..." + os.linesep)
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(fmt)
            stdout_handler.setLevel(logging.INFO)
            log.addHandler(stdout_handler)

        locations = None
        # perform sanity arguments checking before continuing
        if any(x is not None for x in [args.chr, args.start, args.stop]):
            if args.start is not None and args.chr is None:
                raise ValueError("Option --start cannot be set unless --chr is specified")
            if args.stop is not None and args.chr is None:
                raise ValueError("Option --stop cannot be set unless --chr is specified")

            # parse start/stop positions and make sure they're ordered (if exist)
            if args.start is not None:
                start_bp = parse_pos(args.start, "--start")
            else:
                start_bp = fimpg.IGNORE_POS

            if args.stop is not None:
                stop_bp = parse_pos(args.stop, "--stop")
            else:
                stop_bp = fimpg.IGNORE_POS

            if args.start is not None and args.stop is not None:
                if start_bp >= stop_bp:
                    raise ValueError("Specified --start position must be before --stop position")
            locations = pd.DataFrame({"CHR": [args.chr], "START": [start_bp], "STOP": [stop_bp]})

        # this will override specific location setting
        if args.locations is not None:
            if locations is not None:
                log.warning("Option --locations overrides previous --chr, --start, --stop options")
            locations = pd.read_csv(args.locations, delim_whitespace=True)

        # load GWAS summary data
        log.info("Preparing GWAS summary file")
        gwas = fimpg.GWAS.parse_gwas(args.gwas)

        # load reference genotype data
        log.info("Preparing reference SNP data")
        ref = fimpg.parse_plink(args.ref)

        # load functional annotations
        # TBD

        log.info("Starting summary statistics imputation")
        with open("{}.sumstat".format(args.output), "w") as output:
            # for each partition in reference genotype data
            for idx, partition in enumerate(fimpg.partition_data(gwas, ref, args.window_size, loc=locations)):
                gwasp, obsV, unobsV = partition

                # check if we should impute or skip based on some QC...
                # TBD

                # impute GWAS data for this partition
                imputed_gwas = fimpg.impute_gwas(gwasp, obsV, unobsV)

                # write to disk
                fimpg.write_output(imputed_gwas, output, append=bool(idx))

    except Exception as err:
        log.error(err.message)
    finally:
        log.info("Finished summary statistic imputation")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

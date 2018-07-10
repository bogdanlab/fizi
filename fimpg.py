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
    match = re.match("^(([0-9]*[.])?[0-9]+)(mb|kb)?$", pos, flags=re.IGNORECASE)
    position = None
    if match:
        pos_tmp = float(match.group(1)) # position
        pos_mod = match.group(3) # modifier
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
    for cmd in rest:
        if "--" in cmd:
            if cmd == "--quiet":
                rest_strs.append("\t{}".format(cmd) + os.linesep)
            else:
                rest_strs.append("\t{}".format(cmd))
        else:
            rest_strs.append(" " + cmd + os.linesep)

    return base + "".join(rest_strs) + os.linesep


def main(argsv):
    argp = ap.ArgumentParser(description="GWAS summary statistics imputation with functional data.")

    # main arguments
    argp.add_argument("gwas", type=ap.FileType("r"),
        help="GWAS summary data.")
    argp.add_argument("ref",
        help="Path to reference panel PLINK data.")

    # imputation location options
    argp.add_argument("--chr",
        help="Perform imputation for specific chromosome.")
    argp.add_argument("--start",
        help="Perform imputation starting at specific location (in base pairs). Accepts kb/mb modifiers. Requires --chr to be specified.")
    argp.add_argument("--stop",
        help="Perform imputation until at specific location (in base pairs). Accepts kb/mb modifiers. Requires --chr to be specified.")

    # imputation options
    argp.add_argument("--window-size", default="250kb",
        help="Size of imputation window (in base pairs). Accepts kb/mb modifiers.")

    # misc options
    argp.add_argument("--quiet", default=False, action="store_true",
        help="Do not print anything to stdout.")
    argp.add_argument("-o", "--output", default="FIMPG",
        help="Prefix for output data.")

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
        log = logging.getLogger(fimpg.LOG)
        log.setLevel(logging.INFO)
        fmt = logging.Formatter(fmt=FORMAT, datefmt=DATE_FORMAT)

        # setup log file, but write PLINK-style command first
        disk_log_stream = open("{}.log".format(args.output), "w")
        disk_log_stream.write(masthead)
        disk_log_stream.write(cmd_str)
        disk_log_stream.write("Starting log..." + os.linesep)

        disk_handler = logging.StreamHandler(disk_log_stream)
        disk_handler.setFormatter(fmt)
        log.addHandler(disk_handler)

        # write to stdout unless quiet is set
        if not args.quiet:
            # setup log file, but write PLINK-style command first
            sys.stdout.write(masthead)
            sys.stdout.write(cmd_str)
            sys.stdout.write("Starting log..." + os.linesep)
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(fmt)
            log.addHandler(stdout_handler)

        # perform sanity arguments checking before continuing
        chrom = None
        start_bp = None
        stop_bp = None
        if any(x is not None for x in [args.chr, args.start, args.stop]):
            if args.start is not None and args.chr is None:
                raise ValueError("Option --start cannot be set unless --chr is specified")
            if args.stop is not None and args.chr is None:
                raise ValueError("Option --stop cannot be set unless --chr is specified")

            chrom = args.chr

            # parse start/stop positions and make sure they're ordered (if exist)
            if args.start is not None:
                start_bp = parse_pos(args.start, "--start")
            else:
                start_bp = None

            if args.stop is not None:
                stop_bp = parse_pos(args.stop, "--stop")
            else:
                stop_bp = None

            if args.start is not None and args.stop is not None:
                if start_bp >= stop_bp:
                    raise ValueError("Specified --start position must be before --stop position")

        window_size = parse_pos(args.window_size, "--window-size")

        # load GWAS summary data
        log.info("Preparing GWAS summary file")
        gwas = fimpg.GWAS.parse_gwas(args.gwas)

        # load reference genotype data
        log.info("Preparing reference SNP data")
        ref = fimpg.RefPanel.parse_plink(args.ref)

        # load functional annotations
        # TBD

        log.info("Starting summary statistics imputation")
        with open("{}.sumstat".format(args.output), "w") as output:

            import pdb; pdb.set_trace()
            partitions = ref.get_partitions(window_size, chrom, start_bp, stop_bp)
            for idx, partition in enumerate(partitions):
                chrom, start, stop = partition
                part_ref = ref.subset_by_pos(chrom, start, stop)

                if len(part_ref) == 0:
                    log.warning("No reference SNPs found at {}:{} - {}. Skipping".format(chrom, int(start), int(stop)))
                    continue

                part_gwas = gwas.subset_by_pos(chrom, start, stop)
                if len(part_gwas) == 0:
                    log.warning("No GWAS SNPs found at {}:{} - {}. Skipping".format(chrom, int(start), int(stop)))
                    continue

                # impute GWAS data for this partition
                imputed_gwas = fimpg.impute_gwas(part_gwas, part_ref)
                if imputed_gwas is not None:
                    fimpg.write_output(imputed_gwas, output, append=bool(idx))

    except Exception as err:
        log.error(err.message)
    finally:
        log.info("Finished summary statistic imputation")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

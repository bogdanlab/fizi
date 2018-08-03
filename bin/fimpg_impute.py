#! /usr/bin/env python

#import sys
#idx = sys.path.index('/u/local/apps/python/2.7.3/lib/python2.7/site-packages/python_dateutil-2.2-py2.7.egg')
#del sys.path[idx]

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
        pos_tmp = float(match.group(1))  # position
        pos_mod = match.group(3)  # modifier
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


def parse_locations(locations, chrom=None, start_bp=None, stop_bp=None):
    """
    Parse user-specified BED file with [CHR, START, STOP] windows defining where to perform
    imputation.

    If user also specified chr, start-bp, or stop-bp arguments filter on those as well.
    """
    for idx, line in enumerate(locations):
        # skip comments
        if "#" in line:
            continue

        row = line.split()

        if len(row) < 3:
            raise ValueError("Line {} in locations file does not contain [CHR, START, STOP]".format(idx))

        chrom_arg = row[0]
        start_arg = parse_pos(row[1], "start argument in locations file")
        stop_arg = parse_pos(row[2], "stop argument in locations file")

        if chrom is not None and chrom_arg != chrom:
            continue
        elif start_bp is not None and start_arg < start_bp:
            continue
        elif stop_bp is not None and stop_arg > stop_bp:
            continue

        yield [chrom_arg, start_arg, stop_arg]

    return


def get_command_string(args):
    """
    Format fimpg call and options into a string for logging/printing
    """
    base = "fimpg_impute.py " + " ".join(args[:2]) + os.linesep
    rest = args[2:]
    rest_strs = []
    for cmd in rest:
        if "--" in cmd:
            if cmd in ["--quiet", "--verbose", "--force-non-negative"]:
                rest_strs.append("\t{}".format(cmd) + os.linesep)
            else:
                rest_strs.append("\t{}".format(cmd))
        else:
            rest_strs.append(" " + cmd + os.linesep)

    return base + "".join(rest_strs) + os.linesep


def main(argsv):
    argp = ap.ArgumentParser(description="GWAS summary statistics imputation with functional data.",
        formatter_class=ap.ArgumentDefaultsHelpFormatter)

    # main arguments
    argp.add_argument("gwas", type=ap.FileType("r"),
        help="GWAS summary data. Supports gzip and bz2 compression.")
    argp.add_argument("ref",
        help="Path to reference panel PLINK data.")

    # functional arguments
    argp.add_argument("--annot", default=None, type=ap.FileType("r"),
        help="Path to SNP functional annotation data. Should be in LDScore regression-style format. Supports gzip and bz2 compression.")
    argp.add_argument("--sigmas", default=None, type=ap.FileType("r"),
        help="Path to LDScore regression output. Must contain coefficient estimates. Supports gzip and bz2 compression.")
    argp.add_argument("--alpha", default=1.00, type=float,
        help="Significance threshold to determine which functional categories to keep.")
    argp.add_argument("--force-non-negative", default=False, action="store_true",
        help="Set negative variance parameters to zero.")

    # GWAS options
    argp.add_argument("--gwas-n", default=None, type=int,
        help="GWAS sample size.")

    # imputation location options
    argp.add_argument("--chr", default=None,
        help="Perform imputation for specific chromosome.")
    argp.add_argument("--start", default=None,
        help="Perform imputation starting at specific location (in base pairs). Accepts kb/mb modifiers. Requires --chr to be specified.")
    argp.add_argument("--stop", default=None,
        help="Perform imputation until at specific location (in base pairs). Accepts kb/mb modifiers. Requires --chr to be specified.")
    argp.add_argument("--locations", default=None, type=ap.FileType("r"),
        help="Path to a BED file containing windows (e.g., CHR START STOP) to impute. Start and stop values may contain kb/mb modifiers.")

    # imputation options
    argp.add_argument("--window-size", default="250kb",
        help="Size of imputation window (in base pairs). Accepts kb/mb modifiers.")
    argp.add_argument("--buffer-size", default="250kb",
        help="Size of buffer window (in base pairs). Accepts kb/mb modifiers.")
    argp.add_argument("--min-prop", default=0.5, type=float,
        help="Minimum required proportion of gwas/reference panel overlap to perform imputation.")
    argp.add_argument("--ridge-term", default=0.1, type=float,
        help="Diagonal adjustment for linkage-disequilibrium (LD) estimate.")

    # misc options
    argp.add_argument("-q", "--quiet", default=False, action="store_true",
        help="Do not print anything to stdout.")
    argp.add_argument("--verbose", default=False, action="store_true",
        help="Verbose logging. Includes debug info.")
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
        if args.verbose:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)
        fmt = logging.Formatter(fmt=FORMAT, datefmt=DATE_FORMAT)

        # write to stdout unless quiet is set
        if not args.quiet:
            # setup log file, but write PLINK-style command first
            sys.stdout.write(masthead)
            sys.stdout.write(cmd_str)
            sys.stdout.write("Starting log..." + os.linesep)
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(fmt)
            log.addHandler(stdout_handler)

        # setup log file, but write PLINK-style command first
        disk_log_stream = open("{}.log".format(args.output), "w")
        disk_log_stream.write(masthead)
        disk_log_stream.write(cmd_str)
        disk_log_stream.write("Starting log..." + os.linesep)

        disk_handler = logging.StreamHandler(disk_log_stream)
        disk_handler.setFormatter(fmt)
        log.addHandler(disk_handler)

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
        buffer_size = parse_pos(args.buffer_size, "--buffer-size")

        min_prop = args.min_prop
        if min_prop <= 0 or min_prop >= 1:
            raise ValueError("--min-prop must be strictly bounded in (0, 1)")

        # load GWAS summary data
        log.info("Preparing GWAS summary file")
        gwas = fimpg.GWAS.parse_gwas(args.gwas)

        # load reference genotype data
        log.info("Preparing reference SNP data")
        ref = fimpg.RefPanel.parse_plink(args.ref)

        # load functional annotations and sigmas
        if args.annot is not None and args.sigmas is not None:
            log.info("Preparing annotation file")
            annot = fimpg.Annot.parse_annot(args.annot)
            log.info("Preparing SNP effect-size variance file")
            sigmas = fimpg.Sigmas.parse_sigmas(args.sigmas)
            sigmas = sigmas.subset_by_enrich_pvalue(args.alpha)
            if args.force_non_negative:
                sigmas.set_nonnegative()

            # we should check columns in sigmas are contained in annot here...
            sig_cnames = sigmas[fimpg.Sigmas.NAMECOL]
            annot_cnames = annot.columns.values
            for cn in sig_cnames:
                if cn not in annot_cnames:
                    raise KeyError("Prior variance for {} not found in annotation file".format(cn))

        # parity check for functional data
        if args.annot is not None and args.sigmas is None:
            raise ValueError("Annotation requires corresponding LDSC escimates (--sigmas) file")
        if args.annot is None and args.sigmas is not None:
            raise ValueError("LDSC estimates requires corresponding annotation (--annot) file")

        log.info("Starting summary statistics imputation with window size {} and buffer size {}".format(window_size, buffer_size))
        with open("{}.sumstat".format(args.output), "w") as output:

            if args.locations is not None:
                log.info("Preparing user-defined locations")
                partitions = parse_locations(args.locations, chrom, start_bp, stop_bp)
            else:
                partitions = ref.get_partitions(window_size, chrom, start_bp, stop_bp)

            for idx, partition in enumerate(partitions):
                chrom, start, stop = partition
                pstart = max(1, start - buffer_size)
                pstop = stop + buffer_size

                log.debug("Subsetting GWAS data by {}:{} - {}:{}".format(chrom, int(pstart), chrom, int(pstop)))
                part_gwas = gwas.subset_by_pos(chrom, pstart, pstop)
                if len(part_gwas) == 0:
                    log.warning("No GWAS SNPs found at {}:{} - {}:{}. Skipping".format(chrom, int(pstart), chrom, int(pstop)))
                    continue

                log.debug("Subsetting reference SNP data by {}:{} - {}:{}".format(chrom, int(pstart), chrom, int(pstop)))
                part_ref = ref.subset_by_pos(chrom, pstart, pstop)
                if len(part_ref) == 0:
                    log.warning("No reference SNPs found at {}:{} - {}:{}. Skipping".format(chrom, int(pstart), chrom, int(pstop)))
                    imputed_gwas = fimpg.create_output(part_gwas, start=start, stop=stop)
                    fimpg.write_output(imputed_gwas, output, append=bool(idx))
                    continue

                # should we just fall back to IMPG when no annotations overlap?
                if args.annot is not None and args.sigmas is not None:
                    log.debug("Subsetting annotation data by {}:{} - {}:{}".format(chrom, int(pstart), chrom, int(pstop)))
                    part_annot = annot.subset_by_pos(chrom, pstart, pstop)
                    if len(part_annot) == 0:
                        log.warning("No annotations found at {}:{} - {}:{}. Skipping".format(chrom, int(pstart), chrom, int(pstop)))
                        imputed_gwas = fimpg.create_output(part_gwas, start=start, stop=stop)
                        fimpg.write_output(imputed_gwas, output, append=bool(idx))
                        continue

                # impute GWAS data for this partition
                if args.annot is not None and args.sigmas is not None:
                    imputed_gwas = fimpg.impute_gwas(part_gwas, part_ref, annot=part_annot, sigmas=sigmas,
                                                    prop=min_prop, start=start, stop=stop, ridge=args.ridge_term)
                else:
                    imputed_gwas = fimpg.impute_gwas(part_gwas, part_ref,
                                                    prop=min_prop, start=start, stop=stop, ridge=args.ridge_term)

                fimpg.write_output(imputed_gwas, output, append=bool(idx))

    except Exception as err:
        log.error(err.message)
    finally:
        log.info("Finished summary statistic imputation")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

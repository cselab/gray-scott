#!/usr/bin/env python3
# File       : run_simulation.py
# Created    : Sat Jan 30 2021 09:13:51 PM (+0100)
# Description: Gray-Scott driver.  Use the --help argument for all options
# Copyright 2021 ETH Zurich. All Rights Reserved.
import argparse

# the file gray_scott.py must be in the PYTHONPATH or in the current directory
from gray_scott import GrayScott


def parse_args():
    """
    Driver arguments.  These are passed to the GrayScott class
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-F', '--feed_rate', default=0.04, type=float, help='Feed rate F')
    parser.add_argument('-k', '--death_rate', default=0.06, type=float, help='Death rate kappa')
    parser.add_argument('-T', '--end_time', default=3500, type=float, help='Final time')
    parser.add_argument('-d', '--dump_freq', default=100, type=int, help='Dump frequency (integration steps)')
    parser.add_argument('--demo', action='store_true', help='Run demo (https://www.chebfun.org/examples/pde/GrayScott.html)')
    parser.add_argument('--movie', action='store_true', help='Create a movie')
    parser.add_argument('--outdir', default='.', type=str, help='Output directory')
    return parser.parse_known_args()


def demo(args):
    """
    Reproduces the example at https://www.chebfun.org/examples/pde/GrayScott.html
    Pass the --demo option to the driver to run this demo.
    """

    # 1. Rolls
    rolls = GrayScott(F=0.04, kappa=0.06, movie=True, outdir="demo_rolls")
    rolls.integrate(0, 3500, dump_freq=args.dump_freq)

    # 2. Spots
    spots = GrayScott(F=0.025, kappa=0.06, movie=True, outdir="demo_spots")
    spots.integrate(0, 3500, dump_freq=args.dump_freq)


def main():
    args, _ = parse_args()

    if args.demo:
        demo(args)
        return

    gs = GrayScott(F=args.feed_rate, kappa=args.death_rate, movie=args.movie, outdir=args.outdir)
    gs.integrate(0, args.end_time, dump_freq=args.dump_freq)


if __name__ == "__main__":
    main()

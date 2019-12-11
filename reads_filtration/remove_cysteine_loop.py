def remove_cysteine(fasta_file, out_fasta_file):
    f_in = open(fasta_file)
    f_out = open(out_fasta_file, 'w')
    for header in f_in:
        seq = f_in.readline().rstrip()
        if seq.startswith('C') and seq.endswith('C'):
            seq = f'{seq[1:-1]}'  # remove Cys loop
        f_out.write(f'{header}{seq}\n')


if __name__ == '__main__':
    from sys import argv
    print(f'Starting {argv[0]}. Executed command is:\n{" ".join(argv)}')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta_file', help='A fasta file with aa sequences')
    parser.add_argument('out_fasta_file', help='A fasta file to write the input sequences without the flanking Cysteine')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase output verbosity')
    args = parser.parse_args()

    import logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('main')

    remove_cysteine(args.fasta_file, args.out_fasta_file)
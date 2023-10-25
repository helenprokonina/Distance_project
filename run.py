import argparse
import json
import sys

from source import app  # pylint: disable=E0401


def parse_args():
    # define application parameters
    parser = argparse.ArgumentParser(
        description="ML app: proximity alert system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # configuration file
    parser.add_argument('-c', '--config', type=str, help='Path to configuration file.')

    # model
    parser.add_argument('--model', type=str)

    # input video
    parser.add_argument('--video_in', type=str)
    
    # output video
    parser.add_argument('--video_out', type=str)
    
    
    

    # read command line args
    command_line_args = parser.parse_args()

    # read config args
    with open(command_line_args.config, 'r') as config_file:
        config_file_args = json.load(config_file)

    # override config file args with command line args
    parser.set_defaults(**config_file_args)
    args = parser.parse_args()

    print(args)

    return args


def run():
    args = parse_args()

    app.main(args)


if __name__ == '__main__':
    sys.exit(run())

import argparse


def main():
    parser = argparse.ArgumentParser(
        prog='Object measure',
        description='Measure objects in the image provided\nmeasure is calibrated by the leftmost object in the image'
    )
    parser.add_argument('-i', '--image', required=True)  # Image path
    parser.add_argument('-s', '--size', required=True)  # Size of the reference 'leftmost' object

    args = parser.parse_args()


if __name__ == '__main__':
    main()
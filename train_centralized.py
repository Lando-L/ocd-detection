from ocddetection.centralized import learning


def main() -> None:
    learning.run(
        'Human Activity Recognition',
        'Centralized'
    )


if __name__ == "__main__":
    main()

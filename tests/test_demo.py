from hgan.demo import main


def test_demo():
    # The demo should run in all cases - with or without GPU, and by only accessing data already available
    # to the package.
    main()

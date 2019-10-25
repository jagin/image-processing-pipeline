# Image processing pipeline

Modular image processing pipeline using OpenCV and Python generators.  

## Setup environment

This project is using [Conda](https://conda.io) for project environment management.

Setup the project environment:

    $ conda env create -f environment.yml
    $ conda activate pipeline
    
or update the environment if you `git pull` the repo:

    $ conda env update -f environment.yml

## Getting started

For detailed description read the Medium stories in order:
* [Modular image processing pipeline using OpenCV and Python generators](https://medium.com/deepvisionguru/modular-image-processing-pipeline-using-opencv-and-python-generators-9edca3ccb696)
* [Video processing pipeline with OpenCV](https://medium.com/deepvisionguru/video-processing-pipeline-with-opencv-ac10187d75b)

Don't forget to clap a bit if you like it. If you like it very much, you can clap a few times :)  
Thank you!

## Tests

`pytest` is used as a test framework. All tests are stored in `tests` folder. Run the tests:

```bash
$ pytest
```

## Resources and Credits

* For Unix like pipeline idea credits goes to this [Gist](https://gist.github.com/alexmacedo/1552724)
* Source of the example images and videos is [pixbay](https://pixabay.com)
* Some ideas and code snippets are borrowed from [pyimagesearch](https://www.pyimagesearch.com/)
* Color constants from [Python Color Constants Module](https://www.webucator.com/blog/2015/03/python-color-constants-module/)

## License

[MIT License](LICENSE)
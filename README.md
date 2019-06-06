# MahlerNet 
### Unbounded Orchestral Music with Neural Networks
### Website with samples: [MahlerNet.se](http://www.mahlernet.se)
MahlerNet is a neural network resulting from a master thesis that models music in MIDI format inspired by MusicVAE, PerformanceRNN, BachProp and BALSTM. Simply put, MahlerNet is a sequence to sequence model that uses a conditional variational autoencoder (C-VAE) for its latent space, conditioned on a previous unit of music to create long-term structure. MahlerNet uses a custom MIDI preprocessor that incorporates heuristics from music theory and works by modelling the properties of offset to previous event, duration, pitch and instrument for an event, resulting in a representation with no immediate restriction on the number of instruments used. In practice, the current version of MahlerNet has restricted the number of instruments to 23 so that all the 128 GM MIDI instruments are mapped to one of these. Furthermore, 60 classes are used to represent time for both offset and duration and finally 96 pitches out of the 128 MIDI pitches are used. MahlerNet was written from scratch in the spring of 2019 in Tensorflow and at [http://www.mahlernet.se](http://www.mahlernet.se) there is a plethora of listening samples as well as a link to the thesis itself.

MahlerNet has been trained on both the PIANOMIDI and MUSEDATA data sets as well as a new custom MAHLER data set consisting of most of the movements from Mahler's 10 symphonies with good results. Weights from these training sessions are available, stored at the website due to size, and can be fetched [here](http://www.mahlernet.se/files/weights.rar). For someone who knows music, it is fairly easy to hear that MahlerNet is able to learn style in terms of harmonies, instrumentation and the way voices interact with each other, and long-term structure is present in these domains. As often with neural networks modelling music however, long-term structure in terms of reoccuring themes and motives is yet to come.

### Dependencies and versions
MahlerNet was implemented using the following two setups where it is thus known to work:

```Anaconda 4.6.8, Python 3.7.2, Numpy 1.15.4, Tensorflow 1.13.1, Mido 1.2.9, Matplotlib 3.0.3```

and

```Python 3.6.8, Numpy 1.16.2, Tensorflow 1.13.1, Mido 1.2.8, Matplotlib 2.1.0```

The ```Mido``` package was used as a low-level handler of MIDI files.

### Run MahlerNet
MahlerNet comes with a variety of settings, some set via command line others set in the config file, but here is outlined how to get the most straightforward setup up and running with a folder of input MIDI. The notion of  ```ROOT``` folder refers to a folder associated with a specific training set and in this folder, MahlerNet expects to find all input data and it will place all output and trained models in this very same directory.

First, place all your MIDIs, in a flat or non-flat hierarchy, in a folder named ```input``` residing in the ```ROOT``` folder.

```python ProcessFolder.py ROOT -ds```

Now, the input MIDI files have been processed by the preprocessor which have yielded two new folders in the ```ROOT``` folder. In the ```data``` folder lie the complete data representation of every song that was processed, and in the ```seq``` folder lie training patterns from the input files spread into different files based on the length of each training sequence. Alas, we want to make the training batches as uniform as possible.

```python RunMahlerNet.py train NAME --root ROOT --config CONFIG```

Uses the name ```NAME``` for the current training session and will subsequently result in MahlerNet placing all the output from this training session under ```ROOT/runs/NAME```. The ```CONFIG``` configuration file may be the ```example_config.txt``` that is available in this repository or some other configuration file with the same structure and keys as this one. The configuration file will be saved in the training session folder created by MahlerNet. During training, hitting ```ctrl-c``` will signal to MahlerNet to end training after the current epoch. At this point, if the user hits ```ctrl-c``` again, MahlerNet will stop training abruptly. The former is a controlled premature termination where models and training statistics are saved after the current epoch whereas the latter results in an uncontrolled immediate termination where no data is saved (except for already saved data). To erase a training session, simply erase the corresponding folder in ```ROOT/runs```.

```python RunMahlerNet.py generate NAME --root ROOT --type pred --length 10 --samples 2 --units - --use_start_ctx --model MODEL```

Uses the model ```MODEL```, expected to be found in the folder ```ROOT/runs/NAME/trained_models``` (omit any suffixes of the three training files named ```MODEL``` with different suffixes), to generate 2 10-unit (bars in this case) long random samples (not seeded) and that uses a special start token only in the context conditioning. Resulting MIDI output will be placed in the folder ```ROOT/runs/NAME/generated```. To seed the generation, use the argument ```--file FILE``` to refer to a file ```ROOT/data/FILE.pickle``` and supply what unit to use for seeding in the ```--units``` argument. Units are 0-indexed and refer to the unit to reconstruct; for example ```10cz``` means that we use both the context (10th unit) and the input (11th unit) to seed the model. If no context is used, then a 1-step all 0's context is used unless the argument ```--use_start_ctx```is used, which signals that the 1-step context should contain the special start token instead, used as initial context for all songs during training.

During inference, parameters for the model is fetched from the file ```ROOT/runs/NAME/params.txt``` which is saved before training. To alter softmax temperatures, edit this file in a text editor appropriately. Notice well that it is in Json form, NOT literal Python form, as is the case with the input file given when the training starts. The impact this has is that while writing ```False``` in the training configuration file, the saved file would instead have ```"false"``` since this is the Json equivalent.

```python RunMahlerNet.py generate NAME --root ROOT --type recon --file FILE --units 0cz --model MODEL```

Uses the same setting as above but instead tries to reconstruct the first bar of the file ```FILE```, expected to be found under the name ```FILE.pickle``` in the directory ```ROOT/data```. Output is saved in ```ROOT/runs/NAME/generated``` and will contain both the original and the reconstructed unit.

```python RunMahlerNet.py generate NAME --root ROOT --type n_recon --model MODEL```

Tries to reconstruct all the files in the training set with interactive precision feedback. Use the additional flag ```--use_teacher_forcing``` to do the same reconstruction but with teacher forcing, mimicking the behaviour at training time exactly. Produces no output.

```python RunMahlerNet.py generate NAME --root ROOT --type intpol --file FILE --units 9z 19z --steps 5 --model MODEL```

Performs a 5-step interpolation between the 10th and the 20th unit (here bar) of input file ```FILE``` fetched from the file ```ROOT/data/FILE.pickle``` and places the output in ```ROOT/runs/NAME/generated```. Currently only works for models that does not use context conditioning. The endpoints are copied into the result.

Finally, to visualize training from one or several runs, run

```python display_stats.py l "NAME1" RUN_FOLDER1 "NAME2" RUN_FOLDER2 ...```

The ```l``` compares loss and arguments ```p``` and ```d``` can be used instead to compare accuracy (precision) or average distance to correct class instead. Takes an arbitrary number of pairs of arguments where the first argument in the pair is a name to use for reference in the output graph and the second is the path to a run folder, typically on the form ```ROOT/runs/NAME``` which is expected to hold a folder named ```records``` with a records file from a training session.

### Contact
Please send a mail to music@mahlernet.se with comments or suggestions. To cite this work use the following BibTex entry (or convert it accordingly):

```
@mastersthesis{mahlernet,
    author       = {Elias Lousseief}, 
    title        = {MahlerNet - Unbounded Orchestral Music with Neural Networks},
    school       = {Royal College of Technology},
    year         = 2019,
    address      = {Stockholm, Sweden},
    month        = 6
}
```

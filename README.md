# Lunar Lander Model

This repo contains a machine learning model trained on the Lunar Lander environment from OpenAI's Gym. The model is contained in the `model` file, and can be tested with the `test.py` script. Additionally, the model can be further trained using the `train.py` script.

## Requirements

The dependencies for this project are listed in the `requirements.txt` file, which was generated by Conda. To install the necessary dependencies, run the following command:

```
conda create --name lunar_lander --file requirements.txt

```

## Usage

To test the model, run the following command:

```
python test.py


```

This will load the model from the `model` file, and run it on a sample episode of the Lunar Lander environment.

To train the model further, run the following command:

```
python train.py

```

This will load the model from the `model` file (if it exists), and train it on the Lunar Lander environment using a reinforcement learning algorithm.
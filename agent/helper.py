import os
import matplotlib.pyplot as plt

MODEL_DIR = 'models'
SCORE_FILENAME = 'scores.png'

def get_model_dir(modelname):
    """Convenience function to get the directory where to save the model."""
    return os.path.join(os.getcwd(), MODEL_DIR, modelname)

def create_model_dir(modelname):
    """Convenience function to create a directory where to save the model."""
    os.mkdir(os.path.join(os.getcwd(), MODEL_DIR, modelname))

def save_scores(modelname, scores):
    """Convenience function to save scores as a graph."""

    plt.plot(list(range(1, len(scores) + 1)), scores)
    plt.ylabel('score')
    plt.xlabel('episode')
    save_path = os.path.join(get_model_dir(modelname), SCORE_FILENAME)
    plt.savefig(save_path)

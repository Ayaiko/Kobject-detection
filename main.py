# main.py is now a launcher for training and export
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    import train
    import export
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator
from pathlib import Path


def plot_learning_curve(train_loss, val_loss, save_path):
    epochs = np.arange(1, len(train_loss) + 1)
    best_idx = np.argmin(val_loss)
    best_val_loss = val_loss[best_idx]
    overfit_detect = False

    counter = 0
    for val in val_loss[best_idx+1:]:
        if val > best_val_loss:
            counter += 1
        else:
            counter = 0
        
        if counter == 10:
            overfit_detect = True
            break

    plt.figure(figsize=(10,6))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(epochs, train_loss, 'o-', color='r', label='Training loss')
    plt.plot(epochs, val_loss, 'o-', color='b', label='Validation loss')
    plt.axvline(x=best_idx+1, color='g', label=f'Best model at Epoch {best_idx+1}')
    if overfit_detect:
        # plt.axvspan(xmin=best_idx+1, xmax=epochs[-1], color='gray', alpha=0.2, label='Overfitting Zone')
        plt.fill_between(x=epochs, y1=train_loss, y2=val_loss, where=(epochs > best_idx), color='gray', alpha=0.8, label='Overfitting Zone')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__=='__main__':
    LOG_FOLDER = Path(__file__).parent.parent / 'logs'
    file_log = list(LOG_FOLDER.glob('*.csv'))[0]
    df = pd.read_csv(file_log)
    train_loss, val_loss = np.array(df['train_total_loss']), np.array(df['val_total_loss'])
    save_path = LOG_FOLDER / 'learning_curve.png'
    plot_learning_curve(train_loss, val_loss, save_path)
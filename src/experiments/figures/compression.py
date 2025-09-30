from ...constants import RESULTS_DIR, FIGURES_DIR
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from pathlib import Path

RESULT_PATH = RESULTS_DIR / 'compression.csv'
FIGURE_PATH = FIGURES_DIR / 'compression.pdf'

OURS = "ScribeTokens"

def load_data(file_path: Path = RESULT_PATH) -> pd.DataFrame:
    return pd.read_csv(file_path)

def _setup_subplot_grid(n_tokenizers: int, figsize: tuple[int, int]) -> tuple[Figure, list]:
    """Set up the subplot grid based on number of tokenizers."""
    cols = 2
    rows = (n_tokenizers + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=True)
    
    if n_tokenizers == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    return fig, axes

def _plot_tokenizer_data(ax: Axes, tokenizer_data: pd.DataFrame, tokenizer_type: str, colors: list) -> None:
    """Plot data for a single tokenizer type on the given axis."""
    deltas = sorted(tokenizer_data['delta'].unique(), reverse=True)
    
    for j, delta in enumerate(deltas):
        delta_data = tokenizer_data[tokenizer_data['delta'] == delta].sort_values('vocab_size')
        
        ax.plot(
            delta_data['vocab_size'] / 1000,
            delta_data['compression_rate'],
            color=colors[j % len(colors)],
            label=f"δ={delta}",
            marker='o',
            markersize=6,
            linewidth=2,
            alpha=0.7
        )
    title = f'$\\mathbf{{{tokenizer_type}}}_{{\\boldsymbol{{\\delta}}}}$ (ours)' if tokenizer_type == OURS \
        else f'$\\mathbf{{{tokenizer_type}}}_{{\\boldsymbol{{\\delta}}}}$'
    ax.set_title(title, fontsize=24)
    ax.legend(loc='upper left', fontsize=20)
    ax.grid(True, alpha=0.1)
    
    ax.tick_params(axis='both', which='major', labelsize=20)

def _finalize_plot_layout(fig: Figure, axes: list, n_tokenizers: int) -> None:
    """Hide unused subplots and add axis labels."""
    # Hide unused subplots
    for i in range(n_tokenizers, len(axes)):
        axes[i].set_visible(False)
    
    # Add axis labels with more spacing to avoid overlap
    fig.text(0.5, -0.03, 'Vocabulary Size (k)', ha='center', fontsize=24)
    fig.text(0.01, 0.5, 'Compression Rate', va='center', rotation='vertical', fontsize=24)
    
    # Use subplots_adjust instead of tight_layout for better control
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.92, wspace=0.2, hspace=0.2)

def create_compression_plot(df: pd.DataFrame, figsize: tuple[int, int] = (12, 8)) -> Figure:
    tokenizer_types = df['tokeniser_type'].unique()
    n_tokenizers = len(tokenizer_types)
    
    fig, axes = _setup_subplot_grid(n_tokenizers, figsize)
    colors = plt.colormaps['viridis'](np.linspace(0, 1.0, 6))
    
    for i, tokenizer_type in enumerate(tokenizer_types):
        tokenizer_data = df[df['tokeniser_type'] == tokenizer_type]
        _plot_tokenizer_data(axes[i], tokenizer_data, tokenizer_type, list(colors))
    
    _finalize_plot_layout(fig, axes, n_tokenizers)
    return fig

def save_figure(fig: Figure, output_path: Path = FIGURE_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', dpi=300)

def plot(input_path: Path = RESULT_PATH, output_path: Path = FIGURE_PATH, 
         figsize: tuple[int, int] = (12, 8), show: bool = True) -> None:
    df = load_data(input_path)
    fig = create_compression_plot(df, figsize)
    save_figure(fig, output_path)
    
    if show:
        plt.show()

if __name__ == "__main__":
    plot()
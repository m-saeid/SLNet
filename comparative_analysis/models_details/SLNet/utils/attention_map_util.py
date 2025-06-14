import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import io
import torch
import plotly.graph_objects as go
import plotly.io as pio
import kaleido  # ensure Kaleido is importable for static exports
from PIL import Image

import torch.nn as nn
from encoder.Encoder import Encoder

# Optional: configure default Kaleido image settings
pio.kaleido.scope.default_format = "png"
pio.kaleido.scope.default_width = 800
pio.kaleido.scope.default_height = 600


def normalize_features_minmax(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Compute per-dimension min and max over batch and points
    # result shapes: (1, 1, D)
    f_min = features.amin(dim=(0,1), keepdim=True)   # global min per channel :contentReference[oaicite:0]{index=0}
    f_max = features.amax(dim=(0,1), keepdim=True)   # global max per channel :contentReference[oaicite:1]{index=1}

    # Min–max scaling: (x − min)/(max − min)
    features_norm = (features - f_min) / (f_max - f_min + eps)
    return features_norm


def plot_attention_map_points_only(
    xyz: torch.Tensor,
    features: torch.Tensor,
    batch_idx: int = 0,
    dir: str = '',
    out_html: str = None,          # path for HTML (unchanged)
    out_image: str = None,         # path for static image via to_image()
    marker_size: int = 6,
    opacity: float = 0.8,
    cmap: str = "Viridis",
):
    """
    Interactive 3D scatter of attention-normalized pointcloud,
    with axes and background hidden. Saves HTML (via write_html)
    and static image (via to_image + manual write).
    """
    features = normalize_features_minmax(features)-0.1
    out_image = f"{dir}/{batch_idx}.png"

    # 1) Make sure output dirs exist
    for path in (out_html, out_image):
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # 2) Compute per-point attention normalization
    coords = xyz[batch_idx].cpu().numpy()    # (N,3)
    feats = features[batch_idx]              # (N,D)
    attn_logits = feats @ feats.T            # (N,N)
    attn = torch.softmax(attn_logits, dim=1) # (N,N)
    attn_score = attn.sum(dim=0).cpu().numpy()
    attn_norm = (attn_score - attn_score.min()) / (attn_score.ptp() + 1e-8)

    

    # 3) Build Plotly figure (no axes, no colorbar)
    scatter = go.Scatter3d(
        x=coords[:,0], y=coords[:,1], z=coords[:,2],
        mode='markers',
        marker=dict(
            size=marker_size,
            opacity=opacity,
            color=attn_norm,
            colorscale=cmap,
            cmin=0,
            cmax=1,
            showscale=False,
        ),
        hoverinfo='none'
    )
    layout = go.Layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='rgba(0,0,0,0)',
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    fig = go.Figure(data=[scatter], layout=layout)
    
    # 4) Save interactive HTML if requested
    if out_html:
        fig.write_html(out_html)
        print(f"[+] Saved interactive HTML → {out_html}")

    # 5) Save static image via to_image()
    if out_image:
        # Determine format from file extension (default to png)
        ext = os.path.splitext(out_image)[1].lstrip(".").lower() or "png"
        img_bytes = fig.to_image(format=ext)
        with open(out_image, "wb") as f:
            f.write(img_bytes)
        print(f"[+] Saved static image → {out_image}")
    
    # 6) Display in notebook or browser
    #fig.show()
    #return fig





def plot_attention_map_interactive(
    xyz: torch.Tensor,
    features: torch.Tensor,
    batch_idx: int = 0,
    out_html: str = "myplot.html",            # If given, will write a standalone HTML file
    marker_size: int = 4,            # Point size
    opacity: float = 0.8,            # Point opacity
    cmap: str = "Viridis",           # Any Plotly colorscale
):
    """
    Creates an interactive 3D scatter of attention-normalized pointcloud.

    Args:
        xyz (B x N x 3 Tensor):  point coordinates
        features (B x N x D Tensor): per-point features (for attention)
        batch_idx (int):             which batch entry to plot
        out_html (str, optional):     path to write standalone HTML
        marker_size (int):           scatter marker size
        opacity (float):             scatter opacity
        cmap (str):                  Plotly colorscale name
    Returns:
        fig (plotly.graph_objs.Figure): interactive figure
    """
    # 1) Extract coords and compute attention
    coords = xyz[batch_idx].cpu().numpy()               # (N,3)
    feats = features[batch_idx]                         # (N,D)
    attn_logits = feats @ feats.T                       # (N,N)
    attn = torch.softmax(attn_logits, dim=1)            # along source points
    attn_score = attn.sum(dim=0).cpu().detach().numpy() # (N,)
    # normalize to [0,1]
    attn_norm = (attn_score - attn_score.min()) / (attn_score.ptp() + 1e-8)

    # 2) Build Plotly scatter3d
    scatter = go.Scatter3d(
        x=coords[:,0], y=coords[:,1], z=coords[:,2],
        mode='markers',
        marker=dict(
            size=marker_size,
            opacity=opacity,
            color=attn_norm,
            colorscale=cmap,
            colorbar=dict(title="Attention"),
            showscale=True,
        ),
        hovertemplate=
            "Point %{pointNumber}<br>" +
            "X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>" +
            "Attn: %{marker.color:.3f}<extra></extra>"
    )

    # 3) Layout: set axis labels
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title=f"Interactive Attention Map (batch {batch_idx})"
    )

    fig = go.Figure(data=[scatter], layout=layout)

    # 4) Optionally export
    if out_html:
        fig.write_html(out_html)
        print(f"Saved interactive plot to {out_html}")

    fig.show()
    return fig


if __name__ == "__main__":
    # demo
    B, N, D = 2, 2048, 64
    xyz = torch.randn(B, N, 3)
    feats = torch.randn(B, N, D)
    fig = plot_attention_map_interactive(xyz, feats, batch_idx=0, out_html="attn3d.html")
    fig.show()



def plot_attention_map_to_file(
    xyz: torch.Tensor,
    features: torch.Tensor,
    batch_idx: int = 0,
    out_path: str = "attention_map.png",
    figsize: tuple = (8, 6),
    dpi: int = 200,
    cmap: str = "viridis",
    elev: float = 30,
    azim: float = 45,
    point_size: int = 40,
):
    # Select batch, compute attention as before…
    coords = xyz[batch_idx].cpu().numpy()
    feats = features[batch_idx]
    attn_logits = feats @ feats.t()
    attn = torch.softmax(attn_logits, dim=1)
    attn_score = attn.sum(dim=0).cpu().detach().numpy()
    attn_norm = (attn_score - attn_score.min()) / (attn_score.max() - attn_score.min() + 1e-8)

    # Plot
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=attn_norm, cmap=cmap, s=point_size, depthshade=True
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"Attention Map (batch {batch_idx})")
    fig.colorbar(sc, ax=ax, pad=0.1, label="Normalized Attention")
    plt.tight_layout()

    # Save instead of show
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved attention map to {out_path}")


if __name__ == "__main__":
    B, N, D = 2, 1024, 64
    xyz = torch.randn(B, N, 3)
    feats = torch.randn(B, N, D)
    plot_attention_map_to_file(xyz, feats, batch_idx=0, out_path="attn.png")



# SLNet

from encoder.Encoder import Encoder
from utils.util import Classifier
import torch.nn.functional as F

class Classification(nn.Module):
    def __init__(self,
                 n=1024,
                 embed=[3, 32, 'no', 'yes', 0.4],
                 res_dim_ratio=0.25,
                 bias=False,
                 use_xyz=True,
                 norm_mode="anchor",
                 std_mode="BN11",
                 dim_ratio=[2, 2, 2, 1],
                 num_blocks1=[1, 1, 2, 1],
                 transfer_mode=['mlp', 'mlp', 'mlp', 'mlp'],
                 block1_mode=['mlp', 'mlp', 'mlp', 'mlp'],
                 num_blocks2=[1, 1, 2, 1],
                 block2_mode=['mlp', 'mlp', 'mlp', 'mlp'],
                 k_neighbors=[32, 32, 32, 32],
                 sampling_mode=['fps', 'fps', 'fps', 'fps'],
                 sampling_ratio=[2, 2, 2, 2],
                 classifier_mode='mlp_very_large'):
        super(Classification, self).__init__()
        self.encoder = Encoder(
            n=n,
            embed=embed,
            res_dim_ratio=res_dim_ratio,
            bias=bias,
            use_xyz=use_xyz,
            norm_mode=norm_mode,
            std_mode=std_mode,
            dim_ratio=dim_ratio,
            num_blocks1=num_blocks1,
            transfer_mode=transfer_mode,
            block1_mode=block1_mode,
            num_blocks2=num_blocks2,
            block2_mode=block2_mode,
            k_neighbors=k_neighbors,
            sampling_mode=sampling_mode,
            sampling_ratio=sampling_ratio,
        )
        
        last_dim = embed[1]
        for d in dim_ratio:
            last_dim *= d
        
        self.classifier = Classifier(last_dim, classifier_mode, 40)
    
    def forward(self, xyz, feature, embed_dim):
        xyz_list, f_list = self.encoder(xyz, feature)   # 1024, 512, 256, 128

        xyz = xyz_list[0]                      # 16,64,3
        f = f_list[0].permute(0,2,1)           # 16,64,256

        # Attention Map
        dir = f"attention/SLNet_embedDim{embed_dim}"
        for i in [1,2,5,6,7,10,11,13,19,21,24,47,59]:
            plot_attention_map_points_only(xyz, f, i, dir)
        x = F.adaptive_max_pool1d(f_list[-1], 1).squeeze(dim=-1)
        return self.classifier(x)
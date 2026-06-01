import logging
import numpy as np
from stl import mesh as stl_mesh

logger = logging.getLogger(__name__)

def create_pixel_mesh(mask, thickness, dpmm):
    """
    Cria uma malha 3D estilo voxel de forma ultra-otimizada e vetorizada,
    garantindo que todas as normais fiquem corretas (manifold).
    """
    logger.info(f"Iniciando geração de malha Pixel vetorizada (DPMM={dpmm})...")
    
    scale = 1.0 / dpmm
    facets = []

    # --- 1. TAMPAS HORIZONTAIS (Top e Bottom) ---
    active_pixels = np.argwhere(mask)
    N = len(active_pixels)
    
    if N > 0:
        logger.debug(f"Processando {N} pixels ativos para as tampas...")
        r, c = active_pixels[:, 0], active_pixels[:, 1]
        
        x1, x2 = c * scale, (c + 1) * scale
        y1, y2 = r * scale, (r + 1) * scale
        z0 = np.zeros(N)
        zt = np.full(N, thickness)
        
        cap_facets = np.zeros((N * 4, 3, 3))
        
        # Bottom Cap (Z=0) - Normal para baixo (-Z)
        cap_facets[0::4, 0, :] = np.stack([x1, y1, z0], axis=-1)
        cap_facets[0::4, 1, :] = np.stack([x1, y2, z0], axis=-1)
        cap_facets[0::4, 2, :] = np.stack([x2, y1, z0], axis=-1)
        
        cap_facets[1::4, 0, :] = np.stack([x2, y1, z0], axis=-1)
        cap_facets[1::4, 1, :] = np.stack([x1, y2, z0], axis=-1)
        cap_facets[1::4, 2, :] = np.stack([x2, y2, z0], axis=-1)
        
        # Top Cap (Z=thickness) - Normal para cima (+Z)
        cap_facets[2::4, 0, :] = np.stack([x1, y1, zt], axis=-1)
        cap_facets[2::4, 1, :] = np.stack([x2, y1, zt], axis=-1)
        cap_facets[2::4, 2, :] = np.stack([x1, y2, zt], axis=-1)
        
        cap_facets[3::4, 0, :] = np.stack([x2, y1, zt], axis=-1)
        cap_facets[3::4, 1, :] = np.stack([x2, y2, zt], axis=-1)
        cap_facets[3::4, 2, :] = np.stack([x1, y2, zt], axis=-1)
        
        facets.append(cap_facets)

    # --- 2. PAREDES VERTICAIS ALINHADAS EM Y (Gradiente Horizontal) ---
    diff_h = np.diff(mask.astype(int), axis=1)
    
    # Transição Cobre -> Vazio (Parede Direita, normal aponta +X)
    edges_r = np.argwhere(diff_h == -1)
    Nr = len(edges_r)
    if Nr > 0:
        r, c = edges_r[:, 0], edges_r[:, 1]
        x = (c + 1) * scale
        y1, y2 = r * scale, (r + 1) * scale
        wall = np.zeros((Nr * 2, 3, 3))
        wall[0::2, 0, :] = np.stack([x, y1, np.zeros(Nr)], axis=-1)
        wall[0::2, 1, :] = np.stack([x, y2, np.zeros(Nr)], axis=-1)
        wall[0::2, 2, :] = np.stack([x, y2, np.full(Nr, thickness)], axis=-1)
        wall[1::2, 0, :] = np.stack([x, y1, np.zeros(Nr)], axis=-1)
        wall[1::2, 1, :] = np.stack([x, y2, np.full(Nr, thickness)], axis=-1)
        wall[1::2, 2, :] = np.stack([x, y1, np.full(Nr, thickness)], axis=-1)
        facets.append(wall)

    # Transição Vazio -> Cobre (Parede Esquerda, normal aponta -X)
    edges_l = np.argwhere(diff_h == 1)
    Nl = len(edges_l)
    if Nl > 0:
        r, c = edges_l[:, 0], edges_l[:, 1]
        x = (c + 1) * scale
        y1, y2 = r * scale, (r + 1) * scale
        wall = np.zeros((Nl * 2, 3, 3))
        wall[0::2, 0, :] = np.stack([x, y2, np.zeros(Nl)], axis=-1)
        wall[0::2, 1, :] = np.stack([x, y1, np.zeros(Nl)], axis=-1)
        wall[0::2, 2, :] = np.stack([x, y2, np.full(Nl, thickness)], axis=-1)
        wall[1::2, 0, :] = np.stack([x, y2, np.full(Nl, thickness)], axis=-1)
        wall[1::2, 1, :] = np.stack([x, y1, np.zeros(Nl)], axis=-1)
        wall[1::2, 2, :] = np.stack([x, y1, np.full(Nl, thickness)], axis=-1)
        facets.append(wall)

    # --- 3. PAREDES VERTICAIS ALINHADAS EM X (Gradiente Vertical) ---
    diff_v = np.diff(mask.astype(int), axis=0)
    
    # Transição Cobre -> Vazio (Parede Superior, normal aponta +Y)
    edges_t = np.argwhere(diff_v == -1)
    Nt = len(edges_t)
    if Nt > 0:
        r, c = edges_t[:, 0], edges_t[:, 1]
        y = (r + 1) * scale
        x1, x2 = c * scale, (c + 1) * scale
        wall = np.zeros((Nt * 2, 3, 3))
        wall[0::2, 0, :] = np.stack([x2, y, np.zeros(Nt)], axis=-1)
        wall[0::2, 1, :] = np.stack([x1, y, np.zeros(Nt)], axis=-1)
        wall[0::2, 2, :] = np.stack([x2, y, np.full(Nt, thickness)], axis=-1)
        wall[1::2, 0, :] = np.stack([x1, y, np.zeros(Nt)], axis=-1)
        wall[1::2, 1, :] = np.stack([x1, y, np.full(Nt, thickness)], axis=-1)
        wall[1::2, 2, :] = np.stack([x2, y, np.full(Nt, thickness)], axis=-1)
        facets.append(wall)

    # Transição Vazio -> Cobre (Parede Inferior, normal aponta -Y)
    edges_b = np.argwhere(diff_v == 1)
    Nb = len(edges_b)
    if Nb > 0:
        r, c = edges_b[:, 0], edges_b[:, 1]
        y = (r + 1) * scale
        x1, x2 = c * scale, (c + 1) * scale
        wall = np.zeros((Nb * 2, 3, 3))
        wall[0::2, 0, :] = np.stack([x1, y, np.zeros(Nb)], axis=-1)
        wall[0::2, 1, :] = np.stack([x2, y, np.zeros(Nb)], axis=-1)
        wall[0::2, 2, :] = np.stack([x2, y, np.full(Nb, thickness)], axis=-1)
        wall[1::2, 0, :] = np.stack([x1, y, np.zeros(Nb)], axis=-1)
        wall[1::2, 1, :] = np.stack([x2, y, np.full(Nb, thickness)], axis=-1)
        wall[1::2, 2, :] = np.stack([x1, y, np.full(Nb, thickness)], axis=-1)
        facets.append(wall)

    # --- 4. CONSTRUÇÃO DA MALHA FINAL ---
    if facets:
        all_facets = np.vstack(facets)
    else:
        all_facets = np.zeros((0, 3, 3))

    logger.info(f"Malha Pixel concluída. Total de facetas: {len(all_facets)}")

    data = np.zeros(len(all_facets), dtype=stl_mesh.Mesh.dtype)
    data['vectors'] = all_facets
    return stl_mesh.Mesh(data)
from typing import Union, Tuple

import torch
import nvdiffrast.torch as dr

def render_texture(
    uvs: torch.Tensor, # [V1, 2]
    uv_faces: torch.Tensor, # [F, 3]
    attrs: torch.Tensor, # [V2, C]
    attr_faces: torch.Tensor, # [F, 3]
    size: Tuple[int, int], # (H, W)
    glctx: Union[dr.RasterizeGLContext, dr.RasterizeCudaContext]
) -> torch.Tensor:
    hack_verts_clip = uvs * 2.0 - 1.0 # (0, 1) => (-1, 1)
    hack_verts_clip = torch.nn.functional.pad(hack_verts_clip, [0, 1], value=0.0)
    hack_verts_clip = torch.nn.functional.pad(hack_verts_clip, [0, 1], value=1.0)

    uv_faces = uv_faces.to(torch.int32)
    rast_out, _ = dr.rasterize(glctx, hack_verts_clip.unsqueeze(0), uv_faces, resolution=(size[0], size[1]))

    attr_faces = attr_faces.to(torch.int32) # caution: attr_faces is different from uv_faces
    interp_out, _ = dr.interpolate(attrs, rast_out, attr_faces)
    return interp_out.squeeze(0) # [H, W, C]


def compute_rast_info(
    uvs: torch.Tensor, # [V1, 2]
    uv_faces: torch.Tensor, # [F, 3]
    size: Tuple[int, int], # (H, W)
    glctx: Union[dr.RasterizeGLContext, dr.RasterizeCudaContext]
) -> tuple[torch.Tensor, torch.Tensor]:
    hack_verts_clip = uvs * 2.0 - 1.0 # (0, 1) => (-1, 1)
    hack_verts_clip = torch.nn.functional.pad(hack_verts_clip, [0, 1], value=0.0)
    hack_verts_clip = torch.nn.functional.pad(hack_verts_clip, [0, 1], value=1.0)

    uv_faces = uv_faces.to(torch.int32)
    rast_out, _ = dr.rasterize(glctx, hack_verts_clip.unsqueeze(0), uv_faces, resolution=(size[0], size[1]))

    rast_out = rast_out.squeeze(0)
    face_uv = rast_out[..., :2]
    face_id = rast_out[..., 3:] # face_id == 0 when no triangle
    return face_uv, face_id.to(torch.int) # [H, W, 2], [H, W, 1]

# ========================================== Refactoring
# Mac CPU Rasterizer
import torch
def compute_rast_info(uvs, uv_faces, size, glctx=None):
    print("🚀 [CPU Rasterizer] Mapping Gaussians to FLAME Mesh (Mac Mode)...")
    H, W = size
    face_id = torch.zeros((H, W), dtype=torch.int32)
    face_uv = torch.zeros((H, W, 2), dtype=torch.float32)
    
    uvs_cpu = uvs.cpu()
    uv_faces_cpu = uv_faces.cpu()
    
    uvs_v0 = uvs_cpu[uv_faces_cpu[:, 0]]
    uvs_v1 = uvs_cpu[uv_faces_cpu[:, 1]]
    uvs_v2 = uvs_cpu[uv_faces_cpu[:, 2]]
    
    for f in range(len(uv_faces_cpu)):
        v0, v1, v2 = uvs_v0[f], uvs_v1[f], uvs_v2[f]
        min_u = min(v0[0].item(), v1[0].item(), v2[0].item())
        max_u = max(v0[0].item(), v1[0].item(), v2[0].item())
        min_v = min(v0[1].item(), v1[1].item(), v2[1].item())
        max_v = max(v0[1].item(), v1[1].item(), v2[1].item())
        
        min_x = max(0, int(min_u * W))
        max_x = min(W-1, int(max_u * W))
        min_y = max(0, int(min_v * H))
        max_y = min(H-1, int(max_v * H))
        if min_x > max_x or min_y > max_y: continue
        
        gy, gx = torch.meshgrid(torch.arange(min_y, max_y+1), torch.arange(min_x, max_x+1), indexing='ij')
        pts = torch.stack([(gx.float() + 0.5)/W, (gy.float() + 0.5)/H], dim=-1)
        
        v0_v1, v0_v2, v0_pts = v1 - v0, v2 - v0, pts - v0
        d00 = torch.dot(v0_v1, v0_v1)
        d01 = torch.dot(v0_v1, v0_v2)
        d11 = torch.dot(v0_v2, v0_v2)
        d20 = (v0_pts * v0_v1).sum(-1)
        d21 = (v0_pts * v0_v2).sum(-1)
        
        denom = d00 * d11 - d01 * d01
        if denom == 0: continue
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        
        inside = (u >= 0) & (v >= 0) & (w >= 0)
        if inside.any():
            face_id[gy[inside], gx[inside]] = f + 1
            face_uv[gy[inside], gx[inside], 0] = v[inside]
            face_uv[gy[inside], gx[inside], 1] = w[inside]

    return face_uv.to(uvs.device), face_id.to(uvs.device)
# ==========================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointcept.models import build_model
from pointcept.models.losses import build_criteria
from collections import OrderedDict
from torch_cluster import knn
from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point

class MSEFeatureLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.mse = nn.MSELoss()

    def forward(self, student_feat, teacher_feat):
        return self.mse(student_feat, teacher_feat) * self.loss_weight


# def match_teacher_logits_to_student(student_coords, teacher_coords,
#     teacher_logits, k=1):
#     row, col = knn(student_coords, teacher_coords, k=k)
#     matched = teacher_logits[col]
#     print("row.shape", row.shape)
#     print("col.shape",col.shape)
#     print("matched.shape",matched.shape)
#     print("student_coords.shape",student_coords.shape)
#     print("teacher_coords.shape",teacher_coords.shape)
#     print("teacher_logits.shape",teacher_logits.shape)
#     if k > 1:
#         matched = matched.view(-1, k, teacher_logits.size(-1)).mean(1)
#     return matched


def _pairwise_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a, b – одинаковой длины (N, 3)
    return (a - b).pow(2).sum(-1).sqrt()        # -> (N,)

# ────────────────────────────────────────────────────────────────────────────────
# ЛОГИТЫ (порог 0.2)
# ────────────────────────────────────────────────────────────────────────────────
def match_teacher_logits_to_student_thr(s_coord, t_coord, t_logits,
                                        k: int = 1, thr: float = 0.2):
    """
    Возвращает:
        matched_t_logits – (N_s_valid, C)
        s_idx_valid      – (N_s_valid,) индексы точек студента,
        mask             – (N_s,)       True, если пара удовлетворяет порогу.
    """
    # x = t_coord, y = s_coord
    row, col = knn(t_coord.contiguous(), s_coord.contiguous(), k=k)  # [2, N_s*k]
    # row – индексы точек STUDENT, col – индексы TEACHER

    # берём ближайшего (k==1 => len == N_s)
    if k > 1:
        row, col = row.view(-1, k)[:, 0], col.view(-1, k)[:, 0]

    dist = _pairwise_distance(s_coord[row], t_coord[col])            # (N_s,)
    mask = dist <= thr                                               # порог 0.2

    matched_t_logits = t_logits[col]                                 # (N_s, C)

    return matched_t_logits[mask], row[mask], mask                   # (N_valid, …)
    

# ────────────────────────────────────────────────────────────────────────────────
# ПРИЗНАКИ (порог 0.1)
# ────────────────────────────────────────────────────────────────────────────────
def match_teacher_feats_by_knn_thr(s_coord, t_coord,
                                   s_feat, t_feat,
                                   k: int = 1, thr: float = 0.1):
    """
    s_feat – (N_s, D_s)     t_feat – (N_t, D_t)
    Возвращает:
        s_feat_valid  – (N_s_valid, D_s)
        t_feat_valid  – (N_s_valid, D_t)
        mask          – (N_s,)
    """
    row, col = knn(t_coord.contiguous(), s_coord.contiguous(), k=k)

    if k > 1:
        row, col = row.view(-1, k)[:, 0], col.view(-1, k)[:, 0]

    dist = _pairwise_distance(s_coord[row], t_coord[col])
    mask = dist <= thr                                               # порог 0.1

    return s_feat[row][mask], t_feat[col][mask], mask

def load_weight(path, model, seg_head=None):
    keywords = ''
    replacement = None
    replacement = replacement if replacement is not None else keywords
    strict = False
    print("=> Loading checkpoint & weight ...")
    
    print(f"Loading weight at: {path}")
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage.cuda())
    print(
        f"Loading layer weights with keyword: {keywords}, "
        f"replace keyword with: {replacement}"
    )
    weight = OrderedDict()

    if seg_head is not None:
        if 'module.seg_head.weight' in checkpoint["state_dict"]:
            seg_head.weight.data.copy_(checkpoint["state_dict"]['module.seg_head.weight'])
        if 'module.seg_head.bias' in checkpoint["state_dict"]:
            seg_head.bias.data.copy_(checkpoint["state_dict"]['module.seg_head.bias'])

    checkpoint["state_dict"].pop('module.seg_head.weight', None)
    checkpoint["state_dict"].pop('module.seg_head.bias', None)

    for key, value in checkpoint["state_dict"].items():
        key = key[16:] 
        weight[key] = value

    model.load_state_dict(weight, strict=strict)
    return model

@MODELS.register_module()
class MatchesLayerDistillationSegmentorV6(nn.Module):
    def __init__(self, teacher_cfg, student_cfg, criteria, teacher_pretrained=None, student_pretrained=None):
        super().__init__()
        num_classes = 22
        backbone_out_channels = 64
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.seg_head_teacher = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.teacher = build_model(teacher_cfg)
        if teacher_pretrained is not None:
            self.teacher = load_weight(teacher_pretrained, self.teacher, seg_head=self.seg_head_teacher)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        for param in self.seg_head_teacher.parameters():
            param.requires_grad = False
        self.seg_head_teacher.eval()
        
        self.student = build_model(student_cfg)
        if student_pretrained is not None:
            self.student = load_weight(student_pretrained, self.student, seg_head=self.seg_head)

        self.criteria = build_criteria(criteria)
        
        self.teacher_features = {}
        self.student_features = {}
        self.teacher_coord = {}
        self.student_coord = {}
        
        self.register_teacher_hooks()
        self.register_student_hooks()
        self.proj_layers = nn.ModuleDict()
        self.feature_losses = nn.ModuleDict()

        student_dims = self.get_student_dims()
        teacher_dims = self.get_teacher_dims()

        self.layer_weights = [0.01, 0.02, 0.03, 0.05, 0.01]
        # for i in range(3,5):
        s_dim = student_dims[0]
        t_dim = teacher_dims[0]
        self.proj_layers[f'enc{4}'] =nn.Sequential(
                                        nn.Linear(s_dim, s_dim//2),
                                        nn.BatchNorm1d(s_dim//2, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                                        nn.ReLU(inplace=True),
                                        
                                        nn.Linear(s_dim//2, 128),
                                        nn.BatchNorm1d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                                        nn.ReLU(inplace=True),
                                        
                                        nn.Linear(128, 128),
                                        nn.BatchNorm1d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                                        nn.ReLU(inplace=True),
                                        
                                        nn.Linear(128, s_dim//2),
                                        nn.BatchNorm1d(s_dim//2, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                                        nn.ReLU(inplace=True),
                                        
                                        nn.Linear(s_dim//2, t_dim),
                                        nn.BatchNorm1d(t_dim, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                                        nn.ReLU(inplace=True))

        self.feature_losses[f'enc{4}'] = MSEFeatureLoss(loss_weight=1.0)
        
        

    def get_teacher_dims(self):
        return [
            # self.teacher.enc.enc0[-1].mlp[0].fc2.out_features,
            # self.teacher.enc.enc1[-1].mlp[0].fc2.out_features,
            # self.teacher.enc.enc2[-1].mlp[0].fc2.out_features,
            # self.teacher.enc.enc3[-1].mlp[0].fc2.out_features,
            self.teacher.enc.enc4[-1].mlp[0].fc2.out_features
        ]

    def get_student_dims(self):
        return [
            # self.student.enc.enc0[-1].mlp[0].fc2.out_features,
            # self.student.enc.enc1[-1].mlp[0].fc2.out_features,
            # self.student.enc.enc2[-1].mlp[0].fc2.out_features,
            # self.student.enc.enc3[-1].mlp[0].fc2.out_features,
            self.student.enc.enc4[-1].mlp[0].fc2.out_features
        ]

    def register_teacher_hooks(self):
        def get_teacher_hook(layer_name):
            def hook(module, input, output):
                self.teacher_features[layer_name] = output.feat if hasattr(output, 'feat') else output.features
                self.teacher_coord[layer_name] = output.coord if hasattr(output, 'coord') else output.coord
                # for k,v in output.items():
                #     try:
                #         if isinstance(v,int):
                #             print(f"teacher {k} - shape {v}")
                #         elif isinstance(v,list):
                #             print(f"teacher {k} - shape {len(v)}")
                #         else:
                #             print(f"teacher {k} - shape {v.shape}")
                #     except:
                #         print(f"teacher {k} value {v}")
                #         continue
            return hook
        
        # self.teacher.enc.enc0.register_forward_hook(get_teacher_hook('enc0'))
        # self.teacher.enc.enc1.register_forward_hook(get_teacher_hook('enc1'))
        # self.teacher.enc.enc2.register_forward_hook(get_teacher_hook('enc2'))
        # self.teacher.enc.enc3.register_forward_hook(get_teacher_hook('enc3'))
        self.teacher.enc.enc4.block1.register_forward_hook(get_teacher_hook('enc4'))

    def register_student_hooks(self):
        def get_student_hook(layer_name):
            def hook(module, input, output):
                self.student_features[layer_name] = output.feat if hasattr(output, 'feat') else output.features
                self.student_coord[layer_name] = output.coord if hasattr(output, 'coord') else output.coord

                # print(output)
                # for k,v in output.items():
                #     try:
                #         if isinstance(v,int):
                #             print(f"student {k} - shape {v}")
                #         elif isinstance(v,list):
                #             print(f"student {k} - shape {len(v)}")
                #         else:
                #             print(f"student {k} - shape {v.shape}")
                #     except:
                #         print(f"student {k} value {v}")
                #         continue
            return hook
        
        # self.student.enc.enc0.register_forward_hook(get_student_hook('enc0'))
        # self.student.enc.enc1.register_forward_hook(get_student_hook('enc1'))
        # self.student.enc.enc2.register_forward_hook(get_student_hook('enc2'))
        # self.student.enc.enc3.register_forward_hook(get_student_hook('enc3'))
        self.student.enc.enc4.block1.register_forward_hook(get_student_hook('enc4'))

    def calculate_layer_distillation_loss(self):
        total_loss, layer_losses = 0.0, {}

        ln = 'enc4'
        if ln in self.student_features and ln in self.teacher_features:
            sf = self.student_features[ln]      # (N_s, D_s)
            tf = self.teacher_features[ln]      # (N_t, D_t)
            sc = self.student_coord[ln]         # (N_s, 3)
            tc = self.teacher_coord[ln]         # (N_t, 3)

            sf_valid, tf_valid, mask = match_teacher_feats_by_knn_thr(
                sc, tc, sf, tf, k=1, thr=0.05
            )
            if mask.any():                      # могут быть пустые
                proj_sf = self.proj_layers[ln](sf_valid)
                feat_loss = self.feature_losses[ln](proj_sf, tf_valid)
                total_loss += feat_loss * self.layer_weights[-1]
                layer_losses[f'{ln}_mse'] = feat_loss.detach()
            else:
                layer_losses[f'{ln}_mse'] = torch.tensor(0.0,
                                                         device=sf.device)

        return total_loss, layer_losses

    def forward(self, input_dict):
        if self.training:
            teacher_input = input_dict['teacher']
            student_input = input_dict['student']
        else:
            student_input = input_dict

        # ── прямой проход студента ────────────────────────────────
        student_out = self.student(Point(student_input))
        s_feat = student_out.feat if isinstance(student_out, Point) \
                                else student_out
        s_logits = self.seg_head(s_feat)                            # (N_s, C)

        if not self.training:
            # … inference как раньше …
            if 'segment' in student_input:
                loss = self.criteria(s_logits, student_input['segment'])
                return {'loss': loss, 'seg_logits': s_logits}
            return {'seg_logits': s_logits}

        # ── прямой проход учителя (no-grad) ───────────────────────
        with torch.no_grad():
            t_out  = self.teacher(Point(teacher_input))
            t_feat = t_out.feat if isinstance(t_out, Point) else t_out
            t_logits = self.seg_head_teacher(t_feat)               # (N_t, C)

        # ── сопоставляем логиты по порогу 0.2 ─────────────────────
        mt_logits, s_idx_valid, mask_logits = match_teacher_logits_to_student_thr(
            s_coord = student_input['coord'],
            t_coord = teacher_input['coord'],
            t_logits = t_logits,
            k = 1,
            thr = 0.05
        )

        # Возможно, пар нет (mask_logits.sum()==0)
        if mask_logits.any():
            s_logp = F.log_softmax(s_logits[s_idx_valid] / 2.0, dim=1)
            t_prob = F.softmax   (mt_logits          / 2.0, dim=1)
            kl_loss = F.kl_div(s_logp, t_prob,
                               reduction='batchmean') * (2.0 ** 2) * 0.2
        else:
            kl_loss = torch.tensor(0.0, device=s_logits.device)

        # ── обычный сегментационный loss ──────────────────────────
        seg_loss = self.criteria(s_logits, student_input['segment'])

        # ── feature-distillation loss (enc4, порог 0.1) ───────────
        dist_loss, layer_losses = self.calculate_layer_distillation_loss()

        total_loss = seg_loss + kl_loss + dist_loss

        return {
            'loss'      : total_loss,
            'seg_loss'  : seg_loss,
            'kl_loss'   : kl_loss,
            'dist_loss' : dist_loss,
            **layer_losses
        }
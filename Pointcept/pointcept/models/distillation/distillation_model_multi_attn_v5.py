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


def match_teacher_logits_to_student(student_coords, teacher_coords, teacher_logits, k=1):
    """
    student_coords: [N_student, 3]
    teacher_coords: [N_teacher, 3]
    teacher_logits: [N_teacher, num_classes]
    """
    # Найдём ближайшие точки учителя для каждой точки студента
    # knn возвращает индексы ближайших соседей
    student_coords = student_coords.contiguous()
    teacher_coords = teacher_coords.contiguous()
    _, idx = knn(teacher_coords, student_coords, k=k)  # idx: [N_student * k]

    # Если k=1, просто выбираем логиты ближайших точек
    matched_teacher_logits = teacher_logits[idx.view(-1)]

    if k > 1:
        # Если k>1, усредняем логиты по k ближайшим соседям
        matched_teacher_logits = matched_teacher_logits.view(-1, k, teacher_logits.size(-1)).mean(dim=1)

    save_dir =  '/home/myugolyadkin/storonkin_3d/save_knn'
    import os
    import numpy as np
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        # Сохраняем данные как numpy массивы
        np.save(os.path.join(save_dir, 'student_coords.npy'), student_coords.cpu().numpy())
        np.save(os.path.join(save_dir, 'teacher_coords.npy'), teacher_coords.cpu().numpy())
        np.save(os.path.join(save_dir, 'teacher_logits.npy'), teacher_logits.cpu().numpy())
        np.save(os.path.join(save_dir, 'knn_indices.npy'), idx.cpu().numpy())
    # print(3423/0)
    return matched_teacher_logits

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
class MatchesLayerDistillationSegmentorV5(nn.Module):
    def __init__(self, teacher_cfg, student_cfg, criteria, teacher_pretrained=None, student_pretrained=None):
        super().__init__()
        self.logit_distance_threshold = 0.2
        self.feat_distance_threshold  = 0.2
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

        self.layer_weights = [0.01, 0.02, 0.03, 0.05, 0.01]



    def forward(self, input_dict):
        if self.training:
            teacher_input = input_dict['teacher']
            student_input = input_dict['student']
        else:
            student_input = input_dict

        # student -> seg_logits
        student_out = self.student(Point(student_input))
        s_feat = student_out.feat if isinstance(student_out, Point) else student_out
        seg_logits = self.seg_head(s_feat)

        if self.training:
            with torch.no_grad():
                t_out = self.teacher(Point(teacher_input))
                t_feat = t_out.feat if isinstance(t_out, Point) else t_out
                t_seg_logits = self.seg_head_teacher(t_feat)

            # 1) получаем логи и маску
            matched_teacher_logits = match_teacher_logits_to_student(
                student_coords=student_input['coord'],
                teacher_coords=teacher_input['coord'],
                teacher_logits=t_seg_logits,
                k=1
            )

            student_logp = F.log_softmax(seg_logits / 2.0, dim=1)  # [N_s, C]
            teacher_p    = F.softmax(matched_teacher_logits / 2.0, dim=1)  # [N_s, C]


            kl1 = F.kl_div(student_logp, teacher_p, reduction='batchmean') * (2.0**2)

            kl_loss = 0.2 * kl1
            seg_loss = self.criteria(seg_logits, student_input['segment'])

            total_loss = seg_loss + kl_loss #+ dist_loss
            return {
                'loss': total_loss,
                'seg_loss': seg_loss,
                'kl_loss': kl_loss,
            }
        else:
            if 'segment' in student_input:
                loss = self.criteria(seg_logits, student_input['segment'])
                return {'loss': loss, 'seg_logits': seg_logits}
            else:
                return {'seg_logits': seg_logits}
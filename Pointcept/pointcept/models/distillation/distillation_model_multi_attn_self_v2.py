import torch
import torch.nn as nn
import torch.nn.functional as F
from pointcept.models import build_model
from pointcept.models.losses import build_criteria
from collections import OrderedDict
from torch_cluster import knn
from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point
from copy import deepcopy
import random

def _pairwise_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).pow(2).sum(-1).sqrt()

def match_teacher_logits_to_student_thr(s_coord, t_coord, t_logits, k=1, thr=0.01):
    row, col = knn(t_coord.contiguous(), s_coord.contiguous(), k=k)
    if k > 1:
        row, col = row.view(-1, k)[:, 0], col.view(-1, k)[:, 0]
    dist = _pairwise_distance(s_coord[row], t_coord[col])
    mask = dist <= thr
    return t_logits[col][mask], row[mask], mask


def load_weight(path, model, seg_head=None):
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage.cuda())
    if seg_head is not None:
        if 'module.seg_head.weight' in checkpoint["state_dict"]:
            seg_head.weight.data.copy_(checkpoint["state_dict"]['module.seg_head.weight'])
        if 'module.seg_head.bias' in checkpoint["state_dict"]:
            seg_head.bias.data.copy_(checkpoint["state_dict"]['module.seg_head.bias'])
    checkpoint["state_dict"].pop('module.seg_head.weight', None)
    checkpoint["state_dict"].pop('module.seg_head.bias', None)
    weight = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        key = key[16:] 
        weight[key] = value
    model.load_state_dict(weight, strict=False)
    return model

@MODELS.register_module()
class MatchesLayerDistillationSegmentorSelfV2(nn.Module):
    def __init__(self, ema_update_interval=5,ema_momentum=0.996, student_pretrained=None, student_cfg=None, criteria=None):
        super().__init__()
        self.ema_update_interval = ema_update_interval
        self.ema_update_counter = 0
        num_classes = 22
        backbone_out_channels = 64
        
        # Студент
        self.student = build_model(student_cfg)
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)
        if student_pretrained:
            self.student = load_weight(student_pretrained, self.student, self.seg_head)
        
        # Учитель (EMA копия студента)
        self.teacher = deepcopy(self.student)
        self.seg_head_teacher = deepcopy(self.seg_head)
        self._freeze_teacher()
        
        self.ema_momentum = ema_momentum
        self.criteria = build_criteria(criteria)
        

    
    def _freeze_teacher(self):
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        for param in self.seg_head_teacher.parameters():
            param.requires_grad_(False)
        self.teacher.eval()
        self.seg_head_teacher.eval()

    def update_teacher(self):
        self.ema_update_counter += 1
        if self.ema_update_counter % self.ema_update_interval != 0:
            return
        
        with torch.no_grad():
            for param_s, param_t in zip(self.student.parameters(), 
                                      self.teacher.parameters()):
                param_t.data = param_t.data * self.ema_momentum + \
                             param_s.data * (1 - self.ema_momentum)
            for param_s, param_t in zip(self.seg_head.parameters(), self.seg_head_teacher.parameters()):
                param_t.data = param_t.data * self.ema_momentum + param_s.data * (1 - self.ema_momentum)
    
    def forward(self, input_dict):
        if self.training:
            use_teacher_data = random.random() < 0.5
            teacher_input = input_dict['teacher']
            if use_teacher_data:
                student_input = input_dict['teacher']
            else:
                student_input = input_dict['student']

        else:
            student_input = input_dict
        
        # Forward студента
        student_out = self.student(Point(student_input))
        s_feat = student_out.feat
        s_logits = self.seg_head(s_feat)
        
        if not self.training:
            if 'segment' in student_input:
                loss = self.criteria(s_logits, student_input['segment'])
                return {'loss': loss, 'seg_logits': s_logits}
            return {'seg_logits': s_logits}
        
        # Forward учителя
        with torch.no_grad():
            teacher_out = self.teacher(Point(teacher_input))
            t_feat = teacher_out.feat
            t_logits = self.seg_head_teacher(t_feat)
        
        # Расчет лоссов
        seg_loss = self.criteria(s_logits, student_input['segment'])
        
        # Logits distillation
        mt_logits, s_idx, mask = match_teacher_logits_to_student_thr(
            student_input['coord'], teacher_input['coord'], t_logits, thr=0.05
        )
        kl_loss = 0.0
        if mask.sum() > 0:
            s_logp = F.log_softmax(s_logits[s_idx]/2.0, dim=1)
            t_prob = F.softmax(mt_logits/2.0, dim=1)
            kl_loss = F.kl_div(s_logp, t_prob, reduction='batchmean') * (2.0**2) * 0.2
        
        total_loss = seg_loss + kl_loss
        
        return {
            'loss': total_loss,
            'seg_loss': seg_loss.detach(),
            'kl_loss': kl_loss.detach()
        }
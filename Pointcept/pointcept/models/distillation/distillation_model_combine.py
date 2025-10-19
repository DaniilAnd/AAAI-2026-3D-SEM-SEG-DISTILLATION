import torch
import torch.nn as nn
import torch.nn.functional as F
from pointcept.models import build_model
from pointcept.models.losses import build_criteria
from collections import OrderedDict
from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point

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

    # Extract seg_head weights if provided
    if seg_head is not None:
        if 'module.seg_head.weight' in checkpoint["state_dict"]:
            seg_head.weight.data.copy_(checkpoint["state_dict"]['module.seg_head.weight'])
        if 'module.seg_head.bias' in checkpoint["state_dict"]:
            seg_head.bias.data.copy_(checkpoint["state_dict"]['module.seg_head.bias'])

    # Remove seg_head weights from the checkpoint to avoid loading them into the model
    checkpoint["state_dict"].pop('module.seg_head.weight', None)
    checkpoint["state_dict"].pop('module.seg_head.bias', None)

    for key, value in checkpoint["state_dict"].items():
        key = key[16:] 
        weight[key] = value

    model.load_state_dict(weight, strict=strict)
    return model

@MODELS.register_module()
class CombineDistillationSegmentor(nn.Module):
    def __init__(self, teacher_cfg, student_cfg, criteria, teacher_pretrained=None, student_pretrained=None):
        super().__init__()
        num_classes = 22
        backbone_out_channels = 64

        self.teacher = build_model(teacher_cfg)
        if teacher_pretrained is not None:
            self.teacher = load_weight(teacher_pretrained, self.teacher)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        self.student = build_model(student_cfg)
        self.seg_head = nn.Linear(backbone_out_channels, num_classes) if num_classes > 0 else nn.Identity()
        if student_pretrained is not None:
            self.student = load_weight(student_pretrained, self.student, seg_head=self.seg_head)

        self.criteria = build_criteria(criteria)

        # Hooks for feature capture
        self.teacher_features = None
        self.student_features = None
        self.teacher.enc.enc3.register_forward_hook(self.get_teacher_hook)
        self.student.enc.enc3.register_forward_hook(self.get_student_hook)
        student_dim = self.student.enc.enc3[-1].mlp[0].fc2.out_features
        self.extra_conv = nn.Sequential(
            nn.Linear(student_dim, 128),
            nn.BatchNorm1d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, student_dim),
            nn.BatchNorm1d(student_dim, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        # Instance-aware affinity distillation
        self.affinity_loss = nn.MSELoss()

    def get_teacher_hook(self, module, input, output):
        self.teacher_features = output.feat if hasattr(output, 'feat') else output.features

    def get_student_hook(self, module, input, output):
        self.student_features = output.feat if hasattr(output, 'feat') else output.features

    def interpolate_features(self, source, target_size):
        return F.interpolate(
            source, 
            size=target_size, 
            mode='linear', 
            align_corners=False
        )
    def compute_affinity_matrix(self, features):
        if features.dim() == 3:
            features = features.permute(0, 2, 1)
        norm_features = F.normalize(features, p=2, dim=-1)
        return torch.bmm(norm_features, norm_features.transpose(1, 2))

    def forward(self, input_dict):
        if self.training:
            teacher_input = input_dict['teacher']
            student_input = input_dict['student']
        else:
            student_input = input_dict

        self.student.train()
        student_output = self.student(Point(student_input))
        

        if self.training:
            with torch.no_grad():
                teacher_output = self.teacher(Point(teacher_input))
        
        if isinstance(student_output, Point):
            feat = student_output.feat
        else:
            feat = student_output
        seg_logits = self.seg_head(feat)
        
        if self.training:
            # global dist
            loss = self.criteria(seg_logits, student_input['segment'])
            student_feat = self.extra_conv(self.student_features)
            teacher_feat = self.teacher_features
            global_dist = F.mse_loss(student_feat.mean(dim=0), teacher_feat.mean(dim=0))

            loss_student = loss
            loss += 0.3 * global_dist

            return {'loss': loss, 'global_dist': 0.3 * global_dist, 'loss_student':loss_student}
        elif 'segment' in student_input:
            loss = self.criteria(seg_logits, student_input['segment'])
            return {'loss': loss, 'seg_logits': seg_logits}
        else:
            return {'seg_logits': seg_logits}
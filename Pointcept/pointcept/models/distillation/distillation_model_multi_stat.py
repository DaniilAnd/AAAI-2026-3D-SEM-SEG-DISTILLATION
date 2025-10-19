import torch
import torch.nn as nn
import torch.nn.functional as F
from pointcept.models import build_model
from pointcept.models.losses import build_criteria
from collections import OrderedDict
import pointcept.utils.comm as comm

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point



class StatAlignLoss(nn.Module):
    """
    L = ‖µ_s - µ_t‖² + ‖σ²_s - σ²_t‖² + λ * ‖G_s - G_t‖²
    Gram-часть по-умолчанию весит 0.01, чтобы не доминировать.
    """
    def __init__(self, gram_weight: float = 1e-2):
        super().__init__()
        self.gram_weight = gram_weight

    def forward(self, feat_s: torch.Tensor, feat_t: torch.Tensor):
        # feat_* : [N, C]
        if feat_s.numel() == 0 or feat_t.numel() == 0:
            return feat_s.new_tensor(0.)

        # ---------- 1-е моменты ---------------------------------------
        mu_s, mu_t = feat_s.mean(0), feat_t.mean(0)
        var_s, var_t = feat_s.var(0, unbiased=False), feat_t.var(0, unbiased=False)
        loss = F.mse_loss(mu_s, mu_t) + F.mse_loss(var_s, var_t)

        # ---------- 2-е моменты (Gram) --------------------------------
        # feat_s_c = feat_s - mu_s              # [N,C]
        # feat_t_c = feat_t - mu_t
        # gram_s = feat_s_c.T @ feat_s_c / feat_s_c.size(0)   # [C,C]
        # gram_t = feat_t_c.T @ feat_t_c / feat_t_c.size(0)
        # loss = loss + self.gram_weight * F.mse_loss(gram_s, gram_t)
        return loss

class KLDivLoss(nn.Module):
    def __init__(self, loss_weight=1.0, temperature=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits):
        student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        loss = self.kl_div(student_probs, teacher_probs.detach()) * (self.temperature ** 2)
        return loss * self.loss_weight

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
class StatDistillationSegmentor(nn.Module):
    def __init__(self, teacher_cfg, student_cfg, criteria, teacher_pretrained=None, student_pretrained=None):
        super().__init__()
        num_classes = 22
        backbone_out_channels = 64
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.teacher = build_model(teacher_cfg)
        if teacher_pretrained is not None:
            self.teacher = load_weight(teacher_pretrained, self.teacher)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        self.student = build_model(student_cfg)

        
        if student_pretrained is not None:
            self.student = load_weight(student_pretrained, self.student, seg_head=self.seg_head)

        self.criteria = build_criteria(criteria)
        self.kv_loss = KLDivLoss()

        self.teacher_features = {}
        self.student_features = {}
        
        self.register_teacher_hooks()
        self.register_student_hooks()
        self.proj_layers = nn.ModuleDict()
        student_dims = self.get_student_dims()
        teacher_dims = self.get_teacher_dims()
        self.stat_loss = StatAlignLoss()
        for i in range(5):  # enc0 to enc4
            self.proj_layers[f'enc{i}'] = nn.Sequential(
            nn.Linear(student_dims[i], 128),
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
            
            nn.Linear(128, student_dims[i]),
            nn.BatchNorm1d(student_dims[i], eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
            

        self.layer_weights = [0.01, 0.01, 0.05, 0.1, 0.1]
#         self.layer_weights = [0.1, 0.1, 0.1, 0.4, 0.4]
        # [0.1, 0.1, 0.1, 0.4, 0.4]

    def get_teacher_dims(self):
        """Get output dimensions of teacher encoder layers"""
        return [
            self.teacher.enc.enc0[-1].mlp[0].fc2.out_features,
            self.teacher.enc.enc1[-1].mlp[0].fc2.out_features,
            self.teacher.enc.enc2[-1].mlp[0].fc2.out_features,
            self.teacher.enc.enc3[-1].mlp[0].fc2.out_features,
            self.teacher.enc.enc4[-1].mlp[0].fc2.out_features
        ]

    def get_student_dims(self):
        """Get output dimensions of student encoder layers"""
        return [
            self.student.enc.enc0[-1].mlp[0].fc2.out_features,
            self.student.enc.enc1[-1].mlp[0].fc2.out_features,
            self.student.enc.enc2[-1].mlp[0].fc2.out_features,
            self.student.enc.enc3[-1].mlp[0].fc2.out_features,
            self.student.enc.enc4[-1].mlp[0].fc2.out_features
        ]

    def register_teacher_hooks(self):
        """Register hooks for all teacher encoder layers"""
        def get_teacher_hook(layer_name):
            def hook(module, input, output):
                self.teacher_features[layer_name] = output.feat if hasattr(output, 'feat') else output.features
            return hook
        
        self.teacher.enc.enc0.register_forward_hook(get_teacher_hook('enc0'))
        self.teacher.enc.enc1.register_forward_hook(get_teacher_hook('enc1'))
        self.teacher.enc.enc2.register_forward_hook(get_teacher_hook('enc2'))
        self.teacher.enc.enc3.register_forward_hook(get_teacher_hook('enc3'))
        self.teacher.enc.enc4.register_forward_hook(get_teacher_hook('enc4'))

    def register_student_hooks(self):
        """Register hooks for all student encoder layers"""
        def get_student_hook(layer_name):
            def hook(module, input, output):
                self.student_features[layer_name] = output.feat if hasattr(output, 'feat') else output.features
            return hook
        
        self.student.enc.enc0.register_forward_hook(get_student_hook('enc0'))
        self.student.enc.enc1.register_forward_hook(get_student_hook('enc1'))
        self.student.enc.enc2.register_forward_hook(get_student_hook('enc2'))
        self.student.enc.enc3.register_forward_hook(get_student_hook('enc3'))
        self.student.enc.enc4.register_forward_hook(get_student_hook('enc4'))

    def calculate_layer_distillation_loss(self):
        """Calculate distillation loss for all encoder layers"""
        total_loss = 0.0
        layer_losses = {}
        
        for i in range(5):
            layer_name = f'enc{i}'
            if layer_name in self.student_features and layer_name in self.teacher_features:
                projected_student = self.proj_layers[layer_name](self.student_features[layer_name])
                teacher_feat = self.teacher_features[layer_name]
                
                # layer_loss = F.mse_loss(
                #     projected_student.mean(dim=0),
                #     teacher_feat.mean(dim=0),
                #     reduction='mean'
                # )
                layer_loss = self.stat_loss(projected_student, teacher_feat)
                
                weighted_loss = layer_loss * self.layer_weights[i]
                # print("test loss",layer_loss)
                total_loss += weighted_loss
                layer_losses[f'{layer_name}_loss'] = weighted_loss.detach()
        
        return total_loss, layer_losses

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
            seg_loss = self.criteria(seg_logits, student_input['segment'])
            dist_loss, layer_losses = self.calculate_layer_distillation_loss()
            total_loss = seg_loss + dist_loss
            output_dict = {
                'loss': total_loss,
                'seg_loss': seg_loss,
                'dist_loss': dist_loss,
                **layer_losses
            }
            
            return output_dict
        elif 'segment' in student_input:
            loss = self.criteria(seg_logits, student_input['segment'])
            return {'loss': loss, 'seg_logits': seg_logits}
        else:
            return {'seg_logits': seg_logits}
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


def knn_distillation_loss(teacher_coords, teacher_feats, student_coords, student_feats, reduction='mean'):


    # 1. Compute pairwise distances between student_coords and teacher_coords.
    #    Note: This is O(N_t * N_s) in memory. For very large point clouds, you
    #          should use more efficient spatial lookup methods in practice.
    dist_mat = torch.cdist(student_coords, teacher_coords)  # (N_s, N_t)

    # 2. For each student point, find the closest teacher point index.
    nn_teacher_idx = dist_mat.argmin(dim=1)  # (N_s,)

    # 3. Gather teacher features for each student point's nearest neighbor.
    matched_teacher_feats = teacher_feats[nn_teacher_idx]  # (N_s, C)

    # 4. Compute an MSE between matched features. 
    loss = F.mse_loss(student_feats, matched_teacher_feats, reduction=reduction)
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
class LogitsDistillationSegmentor(nn.Module):
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
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        
        if student_pretrained is not None:
            self.student = load_weight(student_pretrained, self.student, seg_head=self.seg_head)

        self.criteria = build_criteria(criteria)
        self.kv_loss = KLDivLoss()

        # Feature storage for all encoder layers
        self.teacher_features = {}
        self.student_features = {}
        self.teacher_coords = {}
        self.student_coords = {}

        
        # Register hooks for all encoder layers (enc0 to enc4)
        self.register_teacher_hooks()
        self.register_student_hooks()
        
        # Projection layers for each encoder level
        self.proj_layers = nn.ModuleDict()
        student_dims = self.get_student_dims()
        teacher_dims = self.get_teacher_dims()
        
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
            
        # Weighting factors for each layer's distillation loss
        self.layer_weights = [0.1, 0.2, 0.3, 0.2, 0.2]  # weights for enc0 to enc4

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
                self.teacher_coords[layer_name] = output.coord
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
                self.student_coords[layer_name] = output.coord
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
                # Project student features to teacher's dimension space
                # projected_student = self.proj_layers[layer_name](self.student_features[layer_name])
                teacher_feat = self.teacher_features[layer_name]

                t_coords = self.teacher_coords[layer_name]
                s_coords = self.student_coords[layer_name]

                if (t_coords is not None and s_coords is not None and
                    teacher_feat is not None and self.student_features[layer_name] is not None):
                    dist_loss = knn_distillation_loss(
                        teacher_coords=t_coords,
                        teacher_feats=teacher_feat,
                        student_coords=s_coords,
                        student_feats=self.student_features[layer_name],
                        reduction="mean"  # "sum" or "mean"
                    ) * self.layer_weights[i]
                weighted_loss = dist_loss * self.layer_weights[i]
                # Calculate MSE loss between projected student and teacher features
                # layer_loss = F.mse_loss(
                #     projected_student.mean(dim=0),
                #     teacher_feat.mean(dim=0),
                #     reduction='mean'
                # )
                
                # weighted_loss = layer_loss * self.layer_weights[i]
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
            # Calculate segmentation loss
            seg_loss = self.criteria(seg_logits, student_input['segment'])
            
            # Calculate distillation losses for all layers
            dist_loss, layer_losses = self.calculate_layer_distillation_loss()
            
            # Combine losses
            total_loss = seg_loss + dist_loss
            
            # Prepare output dictionary
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
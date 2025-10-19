import torch
import torch.nn as nn
import torch.nn.functional as F
from pointcept.models import build_model
from pointcept.models.losses import build_criteria
from collections import OrderedDict
import pointcept.utils.comm as comm
from torch_cluster import knn

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

    return matched_teacher_logits


try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point



class ProjFeatureMatching(nn.Module):
    def __init__(self, student_dim, teacher_dim, hidden_dim=128):
        super().__init__()
        self.student_proj = nn.Sequential(
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


    def forward(self, student_feat, teacher_feat):
        student_proj = self.student_proj(student_feat).unsqueeze(0)  # [1, N, hidden_dim]
        return student_proj

class CosineSimilarityLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def forward(self, student_feat, teacher_feat):
        # Flatten the features to 2D (batch*num_points, feature_dim)
        student_feat = student_feat.view(-1, student_feat.size(-1))
        teacher_feat = teacher_feat.view(-1, teacher_feat.size(-1))
        # Create target with ones of size (batch*num_points)
        target = torch.ones(student_feat.size(0)).to(student_feat.device)
        loss = self.cosine_loss(student_feat, teacher_feat, target)
        return loss * self.loss_weight

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
class MatchesLayerDistillationSegmentor(nn.Module):
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
        self.seg_kldiv_loss = KLDivLoss(loss_weight=0.5, temperature=2.0)
        self.matchers = nn.ModuleDict()
        self.cosine_losses = nn.ModuleDict()


        
        # Feature storage for all encoder layers
        self.teacher_features = {}
        self.student_features = {}
        
        # Register hooks for all encoder layers (enc0 to enc4)
        self.register_teacher_hooks()
        self.register_student_hooks()
        
        # Projection layers for each encoder level
        self.proj_layers = nn.ModuleDict()
        student_dims = self.get_student_dims()
        teacher_dims = self.get_teacher_dims()

        self.layer_weights = [0.1, 0.1, 0.1, 0.2, 0.4]  # weights for enc0 to enc4
        for i in range(5):
            self.matchers[f'enc{i}'] = ProjFeatureMatching(student_dims[i], teacher_dims[i])
            self.cosine_losses[f'enc{i}'] = CosineSimilarityLoss(loss_weight=self.layer_weights[i])
            
        # Weighting factors for each layer's distillation loss
        

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
        total_loss = 0.0
        layer_losses = {}

        for i in range(5):
            layer_name = f'enc{i}'
            if layer_name in self.student_features and layer_name in self.teacher_features:
                student_feat = self.student_features[layer_name]
                teacher_feat = self.teacher_features[layer_name]

                # matching
                matched_student_feat = self.matchers[layer_name](student_feat, teacher_feat)

                # Cosine similarity loss
                layer_loss = self.cosine_losses[layer_name](matched_student_feat, teacher_feat.mean(dim=0, keepdim=True).expand_as(matched_student_feat))

                total_loss += layer_loss
                layer_losses[f'{layer_name}_loss'] = layer_loss.detach()

        return total_loss, layer_losses

    def forward(self, input_dict):
        if self.training:
            teacher_input = input_dict['teacher']
            student_input = input_dict['student']
        else:
            student_input = input_dict

        self.student.train()
        student_output = self.student(Point(student_input))
        student_feat = student_output.feat if isinstance(student_output, Point) else student_output
        seg_logits = self.seg_head(student_feat)

        if self.training:
            with torch.no_grad():
                teacher_output = self.teacher(Point(teacher_input))
                teacher_feat = teacher_output.feat if isinstance(teacher_output, Point) else teacher_output
                teacher_logits = self.seg_head(teacher_feat)

            # Согласуем логиты учителя с логитами студента
            matched_teacher_logits = match_teacher_logits_to_student(
                student_coords=student_input['coord'],  # координаты студента
                teacher_coords=teacher_input['coord'],  # координаты учителя
                teacher_logits=teacher_logits,
                k=1  # можно использовать k>1 для более гладких логитов
            )

            # Теперь размеры совпадают, можно считать KLDivLoss
            kl_loss = self.seg_kldiv_loss(seg_logits, matched_teacher_logits)

            # Остальные потери (сегментация, дистилляция признаков)
            seg_loss = self.criteria(seg_logits, student_input['segment'])
            dist_loss, layer_losses = self.calculate_layer_distillation_loss()

            total_loss = seg_loss + dist_loss + kl_loss

            output_dict = {
                'loss': total_loss,
                'seg_loss': seg_loss,
                'dist_loss': dist_loss,
                'kl_loss': kl_loss,
                **layer_losses
            }

            return output_dict
        elif 'segment' in student_input:
            loss = self.criteria(seg_logits, student_input['segment'])
            return {'loss': loss, 'seg_logits': seg_logits}
        else:
            return {'seg_logits': seg_logits}
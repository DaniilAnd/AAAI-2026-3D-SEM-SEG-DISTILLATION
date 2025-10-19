import torch
import torch.nn as nn
from pointcept.models import build_model
from pointcept.models.losses import build_criteria
from collections import OrderedDict
from pointcept.models.builder import MODELS

class MSEFeatureLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.mse = nn.MSELoss()

    def forward(self, student_feat, teacher_feat):
        return self.mse(student_feat, teacher_feat) * self.loss_weight


def load_weight(path, model):
    print(f"=> Loading weight from: {path}")
    checkpoint = torch.load(path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    # Remove 'module.' prefix if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # remove 'module.' prefix
        new_state_dict[k] = v
        
    # Load with strict=False to ignore mismatched keys
    model.load_state_dict(new_state_dict, strict=False)
    return model

@MODELS.register_module()
class SPUnetDistillationBase(nn.Module):
    def __init__(self, teacher_cfg, student_cfg, criteria, 
                 teacher_pretrained=None, student_pretrained=None,
                 layer_weights=[0.1, 0.1],  # Now only 2 weights: [enc3, dec0]
                 teacher_dims=None,
                 student_dims=None):
        super().__init__()
        # Initialize teacher and student models
        self.teacher = build_model(teacher_cfg)
        if teacher_pretrained:
            self.teacher = load_weight(teacher_pretrained, self.teacher)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        self.student = build_model(student_cfg)
        if student_pretrained:
            self.student = load_weight(student_pretrained, self.student)
        
        # Loss and distillation setup
        self.criteria = build_criteria(criteria)
        self.layer_weights = layer_weights
        
        # Get dimensions if not provided
        if student_dims is None:
            student_dims = self.get_student_dims()
        if teacher_dims is None:
            teacher_dims = self.get_teacher_dims()
        
        # Initialize projectors and losses
        self.proj_layers = nn.ModuleDict()
        self.feature_losses = nn.ModuleDict()
        self._create_projectors(student_dims, teacher_dims)
        
        # Feature storage dictionaries
        self.teacher_features = {}
        self.teacher_indices = {}
        self.student_features = {}
        self.student_indices = {}
        
        # Hook references
        self.teacher_hooks = []
        self.student_hooks = []

    def _recursive_to_device(self, data, device):
        """Recursively move data to the specified device"""
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {k: self._recursive_to_device(v, device) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._recursive_to_device(v, device) for v in data]
        else:
            return data
            
    def get_student_dims(self):
        """Get output dimensions of student encoder layers"""
        return {
            'enc3': 256,
            'dec0': 96
        }

    def get_teacher_dims(self):
        """Get output dimensions of teacher encoder layers"""
        return {
            'enc3': 256,
            'dec0': 96
        }

    def _create_projectors(self, student_dims, teacher_dims):
        """Initialize projectors for enc3 and dec0 only"""
        # layer_keys = ['enc3', 'dec0']
        layer_keys = ['dec0']
        for i, key in enumerate(layer_keys):
            s_dim = student_dims[key]
            t_dim = teacher_dims[key]

            # Create projection head
            proj = nn.Sequential(
                nn.Linear(s_dim, s_dim // 4),
                nn.BatchNorm1d(s_dim // 4, eps=0.001, momentum=0.01),
                nn.ReLU(inplace=True),
                nn.Linear(s_dim // 4, t_dim),
                nn.BatchNorm1d(t_dim, eps=0.001, momentum=0.01),
                nn.ReLU(inplace=True)
            )
            self.proj_layers[key] = proj
            self.feature_losses[key] = MSEFeatureLoss(loss_weight=self.layer_weights[i])

    def _register_teacher_hooks(self):
        """Register hooks for teacher model using closure style"""
        def get_teacher_hook(layer_name):
            def hook(module, input, output):
                self.teacher_features[layer_name] = output.features
                self.teacher_indices[layer_name] = output.indices
            return hook
        
        # Target layers: last conv block in encoder3 and first decoder block
        self.teacher_hooks = [
            # self.teacher.enc[3][-1].conv2.register_forward_hook(get_teacher_hook('enc3')),
            self.teacher.dec[0][-1].conv2.register_forward_hook(get_teacher_hook('dec0'))
        ]
        
    def _register_student_hooks(self):
        """Register hooks for student model using closure style"""
        def get_student_hook(layer_name):
            def hook(module, input, output):
                self.student_features[layer_name] = output.features
                self.student_indices[layer_name] = output.indices
            return hook
        
        # Target layers: last conv block in encoder3 and first decoder block
        self.student_hooks = [
            # self.student.enc[3][-1].conv2.register_forward_hook(get_student_hook('enc3')),
            self.student.dec[0][-1].conv2.register_forward_hook(get_student_hook('dec0'))
        ]

    def _remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.teacher_hooks + self.student_hooks:
            hook.remove()
        self.teacher_hooks = []
        self.student_hooks = []

    def _distill_features(self):
        """Compute distillation loss for enc3 and dec0 only using efficient hashing."""
        total_distill_loss = 0
        layer_keys = ['dec0']  # Only using dec0 layer per current config
        
        for key in layer_keys:
            # Skip if layer not captured
            if key not in self.teacher_features or key not in self.student_features:
                continue
                
            t_features = self.teacher_features[key]
            t_indices = self.teacher_indices[key]
            s_features = self.student_features[key]
            s_indices = self.student_indices[key]

            # Convert student indices to hashable tuples for efficient lookup
            s_indices_tuples = set()
            for idx in s_indices:
                # Convert tensor to tuple for hashing
                s_indices_tuples.add(tuple(idx.cpu().numpy()))
            
            # Identify common indices using set intersection
            common_indices_set = set()
            t_mask = []
            
            # First pass: find common indices and create teacher mask
            for idx in t_indices:
                idx_tuple = tuple(idx.cpu().numpy())
                if idx_tuple in s_indices_tuples:
                    common_indices_set.add(idx_tuple)
                    t_mask.append(True)
                else:
                    t_mask.append(False)
            
            t_mask = torch.tensor(t_mask, device=t_features.device)
            
            # Skip if no common indices
            if len(common_indices_set) == 0:
                continue
                
            # Create dictionaries for fast feature lookup
            t_feat_dict = {}
            for i, idx in enumerate(t_indices):
                idx_tuple = tuple(idx.cpu().numpy())
                if idx_tuple in common_indices_set:
                    t_feat_dict[idx_tuple] = t_features[i]
            
            s_feat_dict = {}
            for i, idx in enumerate(s_indices):
                idx_tuple = tuple(idx.cpu().numpy())
                if idx_tuple in common_indices_set:
                    s_feat_dict[idx_tuple] = s_features[i]
            
            # Sort common indices to ensure aligned ordering
            common_indices_sorted = sorted(common_indices_set)
            
            # Extract features in consistent order
            t_common_feat = torch.stack(
                [t_feat_dict[idx] for idx in common_indices_sorted]
            )
            s_common_feat = torch.stack(
                [s_feat_dict[idx] for idx in common_indices_sorted]
            )

            # Project student features and compute loss
            proj_feat = self.proj_layers[key](s_common_feat)
            total_distill_loss += self.feature_losses[key](proj_feat, t_common_feat)
        
        return total_distill_loss
    # def _distill_features(self):
    #     """Compute distillation loss for enc3 and dec0 only"""
    #     total_distill_loss = 0
    #     # layer_keys = ['enc3', 'dec0']
    #     layer_keys = ['dec0']
    #     for key in layer_keys:
    #         # Skip if layer not captured
    #         if key not in self.teacher_features or key not in self.student_features:
    #             continue
                
    #         t_features = self.teacher_features[key]
    #         t_indices = self.teacher_indices[key]
    #         s_features = self.student_features[key]
    #         s_indices = self.student_indices[key]

    #         # print(t_features.shape)
    #         # print(s_features.shape)

    #         # print(s_indices,s_indices)
            
    #         # Skip if no features
    #         if t_features.shape[0] == 0 or s_features.shape[0] == 0:
    #             continue
                
    #         # Find common indices between teacher and student
    #         common_indices = []
    #         for idx in t_indices:
    #             if (s_indices == idx).all(dim=1).any():
    #                 common_indices.append(idx)
            
    #         if not common_indices:
    #             continue
                
    #         common_indices = torch.stack(common_indices)
            
    #         # Get features at common indices
    #         t_mask = (t_indices.unsqueeze(1) == common_indices.unsqueeze(0))
    #         t_mask = t_mask.all(dim=2).any(dim=1)
    #         t_common_feat = t_features[t_mask]
            
    #         s_mask = (s_indices.unsqueeze(1) == common_indices.unsqueeze(0))
    #         s_mask = s_mask.all(dim=2).any(dim=1)
    #         s_common_feat = s_features[s_mask]
            
    #         # Project student features
    #         proj_feat = self.proj_layers[key](s_common_feat)
            
    #         # Calculate layer loss
    #         total_distill_loss += self.feature_losses[key](proj_feat, t_common_feat)
        
    #     return total_distill_loss

    def forward(self, input_dict):
        if self.training:
            # Clear previous features
            self.teacher_features = {}
            self.teacher_indices = {}
            self.student_features = {}
            self.student_indices = {}

            device = next(self.parameters()).device
            input_dict['teacher'] = self._recursive_to_device(input_dict['teacher'], device)
            input_dict['student'] = self._recursive_to_device(input_dict['student'], device)
            
            # Register hooks
            self._register_teacher_hooks()
            self._register_student_hooks()
            
            # Run teacher and student
            with torch.no_grad():
                self.teacher(input_dict['teacher'])
            student_out = self.student(input_dict['student'])
            
            # Remove hooks immediately after use
            self._remove_hooks()
            
            # Calculate losses
            seg_loss = self.criteria(student_out, input_dict['student']['segment'])
            distill_loss = self._distill_features()
            total_loss = seg_loss + distill_loss
            
            return {'loss': total_loss}
        
        else:
            # Evaluation mode
            student_out = self.student(input_dict)
            if 'segment' in input_dict:
                loss = self.criteria(student_out, input_dict['segment'])
                return {'loss': loss, 'seg_logits': student_out}
            return {'seg_logits': student_out}
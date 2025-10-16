import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import cv2
from utils.metrics import loss_fn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from transformers import ViTConfig, ViTModel
from einops import rearrange

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})
    return None

def enable_gradients(model):
    for param in model.parameters():
        param.requires_grad = True

def check_gradients(model):
    for param in model.parameters():
        if not param.requires_grad:
            return False
    return True

####################### Curriculum 1 #############################

def extract_roi(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    s2 = cv2.blur(s, (5, 5))
    _, saturation_thresholded_mask = cv2.threshold(s2, 15, 255, cv2.THRESH_BINARY)
    
    roi = saturation_thresholded_mask / 255
    return roi

def reshape_transform(tensor):
    b, n, l = tensor.shape
    n = np.sqrt(n-1).astype(int)
    result = tensor[:, 1:, :].reshape(b, n, n, l)

    # Bring the channels to the first dimension,
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class Atten_aggregation(nn.Module):
    def __init__(self, dim, inner_dim):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.Tanh(),
            nn.Linear(inner_dim, 1)
        )
        self.attend = nn.Softmax(dim = -1)

    def forward(self, x, return_attention=False):
        b = x.shape[0]
        x1 = rearrange(x, 'b n d -> (b n) d')
        a = self.attention(x1)
        a = rearrange(a, '(b n) d -> b d n', b=b)
        a = self.attend(a)
        x = torch.bmm(a, x).squeeze(1)
        if return_attention: return x, a
        return x

class Vit(nn.Module):
    def __init__(self, i):
        super(Vit, self).__init__()   
        out_dim = 768
        mid_dim = 3072

        configuration = ViTConfig( hidden_size = out_dim,num_hidden_layers = 8 + 2 * i,num_attention_heads = 12,
                                   intermediate_size = mid_dim,hidden_act = 'gelu',hidden_dropout_prob = 0.5,
                                   attention_probs_dropout_prob = 0.5,initializer_range = 0.02,layer_norm_eps = 1e-12,
                                   image_size = 128 * (2**i), patch_size = 16, num_channels = 3,qkv_bias = True,encoder_stride = 16) 
        self.vit = ViTModel(configuration, add_pooling_layer=False, use_mask_token=True)
        
        self.attention = Atten_aggregation(out_dim, mid_dim)
        
        self.to_latent = nn.Sequential(nn.LayerNorm(out_dim), nn.Identity())

        self.mlp_head = nn.Sequential(
            nn.Linear(out_dim, 2),
        )     
        self.fea, self.last_fea = None, None
    def forward(self, input):
        x, mask = input

        x = self.vit(x, output_attentions = False, bool_masked_pos=mask)
        
        x = self.to_latent(self.attention(x.last_hidden_state))
        
        fea = x if self.last_fea == None else self.to_latent(self.last_fea + x)

        self.save_fea(fea)

        y = self.mlp_head(fea)

        return y

    def save_last_fea(self, last_fea):
        self.last_fea = last_fea

    def save_fea(self, fea):
        self.fea = fea.clone().detach()

    def get_fea(self):
        return self.fea
        
class VitBranch(nn.Module):
    def __init__(self):
        super(VitBranch, self).__init__()                            

        self.vit_1 = Vit(0)
        self.vit_2 = Vit(1)
        self.vit_3 = Vit(2)
        
        self.modules = [self.vit_1, self.vit_2, self.vit_3]

        # blocks = []
        # for i in range(3):
        #     blocks.append(Vit(i))
        # self.blocks = nn.ModuleList(blocks)     

        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)

    def forward(self, input, y=None, pretraining=False, **kwargs):
        res_pro = []
        mask = None
        for i, (x, module) in enumerate(zip(input, self.modules)):
            if i > 0:
                mask, _ = self.cammap(self.modules[i-1], [input[i-1],mask], y)
                mask = mask.to(kwargs['device'])
                module.save_last_fea(self.modules[i-1].get_fea())
                    
            pro = module([x, mask])
            res_pro.append(pro)
            if pretraining: break

        res_pro = torch.stack(res_pro, dim=1)
        res_hat = torch.argmax(res_pro, -1).detach().cpu().numpy()

        return res_pro, res_hat
    
    @torch.no_grad()
    def extract_fea(self, input, y=None, **kwargs):
        res_pro = []
        mask = None
        for i, (x, module) in enumerate(zip(input, self.modules)):
            if i > 0:
                mask, _ = self.cammap(self.modules[i-1], [input[i-1],mask], y)
                mask = mask.to(kwargs['device'])
                module.save_last_fea(self.modules[i-1].get_fea())
            pro = module([x, mask])
            res_pro.append(pro)

        fea = self.modules[-1].get_fea()
        res_pro = torch.stack(res_pro, dim=1)
        res_hat = torch.argmax(res_pro, -1).detach()
        return fea, res_pro, res_hat
    
    @torch.no_grad()
    def visual(self, input, y=None, smooth=False, raw_img=None, **kwargs):
        atten, maps, pros = [], [], []
        mask = None

        k=0
        for i, (x, module) in enumerate(zip(input, self.modules)):
            if i > 0:
                module.save_last_fea(self.modules[i-1].get_fea())       
            
            pro = module([x,mask]).detach()
            pro_softmax = torch.nn.functional.softmax(pro, dim=-1).detach().cpu()
            pros.append(pro_softmax[:,1])
            
            a = loss_fn(pro, (1-y).to(kwargs['device'])).detach().cpu()
            a[a>1] = 1
            atten.append(a)

            mask, heatmap = self.cammap(module, [x,mask], y, smooth)
            maps.append(heatmap)
            
            mask = mask.to(kwargs['device'])

            # if raw_img != None:
            #     img = raw_img[i][k]
            #     img = cv2.cvtColor((np.transpose(img.numpy(),(1,2,0))*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            #     cv2.imwrite(f'./plot/img.jpg', img)
                
            #     roi = extract_roi(img)
            #     one_map = heatmap[k] * roi
            #     cv2.imwrite(f'./plot/mask_{i}.jpg', (one_map*255).astype(np.uint8))
                
            #     one_map = cv2.GaussianBlur(one_map, tuple((np.array([32,32]) * (1-0.)).astype(int) * 2 +1),0)
            #     self.cam_show(img, one_map, i) 

        return maps, atten, pros

    def cammap(self, model, x, y=None, smooth=False):
        cam_model = copy.deepcopy(model)

        target_layers = [cam_model.vit.encoder.layer[-1].layernorm_before] 
        cam = GradCAM(model=cam_model, target_layers=target_layers, use_cuda=False, reshape_transform=reshape_transform)
        cam.batch_size = x[0].shape[0]

        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested category.
        if y == None:
            targets = None
        else:
            targets = [ClassifierOutputTarget(category) for category in y]            
            
        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        with torch.enable_grad():
            grayscale_cam = cam(input_tensor=x,
                                targets=targets,
                                eigen_smooth=smooth,
                                aug_smooth=smooth)
        
        mask = self.upsample(torch.FloatTensor(grayscale_cam).unsqueeze(1))
        mask = rearrange(mask, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 16, p2 = 16)
        mask = torch.max(mask,-1).values
        mask = torch.where(mask <= 0.4, torch.ones_like(mask),torch.zeros_like(mask))

        return mask, grayscale_cam

    def cam_show(self, img, cam, name):
        cam_image = show_cam_on_image(img/255, cam)
        cv2.imwrite(f'./plot/cam_{name}.jpg', cam_image)
######################### Curriculum 2 ###########################

def get_hard_attention(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    y_soft = logits.softmax(dim)
    y_hard = torch.ones_like(logits, memory_format=torch.legacy_contiguous_format)
    ret = y_hard - y_soft.detach() + y_soft

    return ret

def get_gumbel_attention(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> torch.Tensor:
    
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        y_hard = torch.ones_like(logits, memory_format=torch.legacy_contiguous_format)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class Self_Attention(nn.Module):
    def __init__(self, dim, inner_dim = 32, dropout = 0.):
        super().__init__()
        self.scale = inner_dim ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.to_latent = nn.Identity()

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = self.to_latent(out.mean(dim = 1))

        return self.to_out(out)
    
class SurvModel(nn.Module):
    def __init__(self, dim = 128, input_dim = 768, inner_dim = 32, k = 10, dropout = 0.):
        super(SurvModel, self).__init__()
        """
        Arguments:
            dims: List(raw_dim, hidden1_dim, hidden2_dim)
        """        
        self.k = k

        self.linear = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
        )
        self.integrate = Integrate(dim, inner_dim)

        self.attention = Attention(dim = dim, inner_dim = inner_dim)

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(p=dropout),
            nn.Linear(dim, 1),
        )

        self.self_attention = Self_Attention(dim = dim, inner_dim = inner_dim, dropout = dropout)

        self.dropout = nn.Dropout(dropout)
        self.to_latent = nn.Identity()

    def forward(self, x, x_normal=None, training=False):
        x = self.linear(x)

        a = self.attention(x)
        x = x * torch.transpose(get_gumbel_attention(a, hard=True), 1, 0)  
        ind = torch.sort(a, descending=True)[1].squeeze(0)

        z = x[ind[:self.k]].unsqueeze(0)
        z = self.dropout(z)
        z = self.self_attention(z)

        y = self.mlp(z)

        if training:  
            z_bar = x[ind[self.k:]]
            z_bar = self.integrate(z_bar)

            z_normal = self.linear(x_normal)
            z_normal = z_normal * torch.transpose(get_gumbel_attention(self.attention(z_normal), hard=True), 1, 0)

            z_normal = self.dropout(z_normal)
            z_normal = self.self_attention(z_normal.unsqueeze(0))

            return y, z, z_bar, z_normal
        
        return y
    
    @torch.no_grad()
    def get_atten(self, x):
        x = self.linear(x)
        a = self.attention(x)
        
        return x, a

    @torch.no_grad()
    def get_feature(self, x, is_normal=False, is_integration=True):
        x = self.linear(x)
        a = self.attention(x)
        ind = torch.sort(a, descending=True)[1].squeeze(0)
        if is_integration:
            if not is_normal:  
                z = self.self_attention(x[ind[:self.k]].unsqueeze(0))
                z_bar = self.integrate(x[ind[self.k:]])
                return z.cpu(), z_bar.cpu()  
            else:
                z_normal = self.self_attention(x.unsqueeze(0))
                return z_normal.cpu()
        else:
            return x[ind].cpu(), ind.cpu()
   
class Attention(nn.Module):
    def __init__(self, dim, inner_dim):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.Tanh(),
            nn.Linear(inner_dim, 1)
        )

    def forward(self, x):
        a = self.attention(x)
        a = torch.transpose(a, 1, 0)  

        return a

class Integrate(nn.Module):
    def __init__(self, dim, inner_dim):
        super().__init__()
        
        self.attention = Attention(dim = dim, inner_dim = inner_dim)
        self.attend = nn.Softmax(dim = -1)

    def forward(self, x):
        a = self.attention(x)
        a = self.attend(a)
        x = torch.matmul(a, x)
        return x

class TCLModel(nn.Module):
    def __init__(self, dim = 128, inner_dim = 32):
        super(TCLModel, self).__init__()
        self.dim = dim

        self.integrate = Integrate(dim, inner_dim)

        self.weight_0 = nn.Linear(dim, dim)
        self.weight_1 = nn.Linear(dim, dim)

        self.lsoftmax = nn.LogSoftmax(dim=1)

        self.cosine_loss = torch.nn.CosineEmbeddingLoss()
        
        self.hinge_loss = torch.nn.HingeEmbeddingLoss()

    def forward(self, **kwargs):
        z_low, z_high, z_bar, z_normal = kwargs['z_low'], kwargs['z_high'], kwargs['z_bar'], kwargs['z_normal']
        z_all = torch.cat([z_low, z_high]) # b * dim

        batch_low = z_low.shape[0]
        batch_high = z_high.shape[0]
        batch = batch_low + batch_high

        z_tumor = torch.empty((batch, self.dim)).to(kwargs['device'])
        z_hat = torch.empty((batch, self.dim)).to(kwargs['device'])

        for i in range(batch_low):
            z_hat[i,] = self.integrate(z_low[torch.arange(batch_low) != i])
            z_tumor[i,] = self.integrate(z_all[torch.arange(batch) != i])
        for i in range(batch_high):
            z_hat[batch_low + i,] = self.integrate(z_high[torch.arange(batch_high) != i])
            z_tumor[batch_low + i,] = self.integrate(z_all[torch.arange(batch) != batch_low + i])

        pred_tumor = self.weight_0(z_hat) + self.weight_1(z_bar) + self.weight_0(z_tumor)
        pred_normal = self.weight_0(z_hat) + self.weight_1(z_bar) + self.weight_0(self.integrate(z_normal))

        cpc_tumor = torch.matmul(z_all, torch.transpose(pred_tumor, 0, 1)) # b * b
        cpc_normal = torch.matmul(z_all, torch.transpose(pred_normal, 0, 1)) # b * b

        eye_tensor = torch.eye(batch).to(kwargs['device'])
        total = eye_tensor * cpc_tumor + (1 - eye_tensor) * cpc_normal

        mask = torch.ones([batch,batch]).to(kwargs['device'])

        ones_tensor = torch.ones(batch).to(kwargs['device'])
        loss_adc = (self.cosine_loss(z_all, z_hat, ones_tensor)
                    + self.cosine_loss(z_all, z_bar, ones_tensor)
                    + self.hinge_loss(F.cosine_similarity(z_all, pred_tumor)
                                 - F.cosine_similarity(z_all, pred_normal), -1 * ones_tensor)
                    )
        
        loss_tcl = torch.sum(torch.diag(self.lsoftmax(total * mask)))
        loss_tcl /= -1.*batch
        # correct = torch.sum(torch.eq(torch.argmax(self.softmax(total).detach().cpu(), dim=0), torch.arange(0, batch)))
        # accuracy_w = 1.*correct.item()/batch
        # print(f"accuracy_s/w: {accuracy_s} and {accuracy_w}")
        return loss_tcl, loss_adc


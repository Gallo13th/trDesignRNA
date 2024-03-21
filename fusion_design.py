import tempfile

import glob

import string
import time
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn

from collections import defaultdict
from argparse import ArgumentParser
from pathlib import Path
from torch.optim import Adam
pkg_dir = os.path.abspath(str(Path(__file__).parent))
sys.path.insert(0, pkg_dir)
sys.path.insert(1, f'{pkg_dir}/network')
from utils import *
from network.RNAformer import DistPredictorforDesign
from network.config import n_bins, obj
import torch

from torch.autograd import Variable
import random
import tqdm
import fm
import torch.nn.functional as F
import RNA
import design_utils.pointidx as pointidx
import design_utils.designloss as designloss
import design_utils.pdb2cst as pdb2cst


parser = ArgumentParser()
parser.add_argument('-o', '--npz', help='output NPZ file')
parser.add_argument('-mdir', '--model_pth', default=f'params/model_1', help='pretrained params directory')
parser.add_argument('-nrows', '--nrows', default=1000, type=int, help='maximum number of rows in the MSA repr.')
parser.add_argument('-gpu', '--gpu', type=str, default='-1', help='use which gpu')
parser.add_argument('-cpu', '--cpu', type=int, default=2, help='number of CPUs to use')
parser.add_argument('-method', '--method', type=str, default='mcmc', help='mcmc or gd')
parser.add_argument('-template', '--template', type=str, default=None, help='template msa file')
parser.add_argument('-seqmask', '--seqmask', type=str, default=None, help='seqmask string')
parser.add_argument('-ssmask', '--ssmask', type=str, default=None, help='seqmask string')
parser.add_argument('-length', '--length', type=int, default=20, help='denovo length')
parser.add_argument('-ss', '--ss', type=str, default=None, help='ss mask')
parser.add_argument('-pdb', '--pdb', type=str, default=None, help='pdb filepath')
parser.add_argument('-pdbmask', '--pdbmask', type=str, default=None, help='pdbmask filepath, use | to separate the two masks')
parser.add_argument('-impaintlr', '--impaintlr', type=float, default=1)
parser.add_argument('-structlr', '--structlr', type=float, default=1)
parser.add_argument('-lr', '--lr', type=float, default=1)
parser.add_argument('-task','--task',default='inversebyss')
# parser.add_argument('-dt')
args = parser.parse_args()

############################################################################################################
# RNA-FM
############################################################################################################

pad = nn.ZeroPad2d((4,16,0,0))
pad_start = nn.ZeroPad2d((0,0,1,1))

def vienna_ss_pred(seq):
    fc = RNA.fold_compound(seq)
    (mfe_struct, mfe) = fc.mfe()
    return mfe_struct,mfe

def get_nc_matrix(seq):
    seq_len = len(seq)
    nc_matrix = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        for j in range(i, seq_len):
            if seq[i] + seq[j] in ['AU', 'UA', 'GC', 'CG', 'GU', 'UG']:
                nc_matrix[i, j] = 1
                nc_matrix[j, i] = 1
    return nc_matrix

def dot2ssmat(ss):
    mat = np.zeros((len(ss),len(ss)))
    stk = []
    for i in range(len(ss)):
        if ss[i] == '(':
            stk.append(i)
        elif ss[i] == ')':
            j = stk.pop()
            mat[i,j] = 1
            mat[j,i] = 1
    return mat

def postprocess(prob_map, seq, threshold=0.5, allow_nc=True):
    # we suppose that probmay cantains values range from [0,1], so is the threshold
    canonical_pairs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG']

    # candidate 1: threshold  (we obatin the full contact matrix)
    prob_map = prob_map * (1 - torch.eye(prob_map.shape[0],device=prob_map.device))  # no  care about the diagonal
    pred_map = (prob_map > threshold)

    # candidate 2: split the multiplets, resulting in cm without multiplets. Also filter the non-canonical pairs
    # when several pairs are conflict by presenting in the same row or column, we choose the one with highest score.
    seq_len = len(seq)
    x_array, y_array = torch.nonzero(pred_map,as_tuple=True)
    prob_array = []
    for i in range(x_array.shape[0]):
        prob_array.append(prob_map[x_array[i], y_array[i]])
    prob_array = torch.tensor(prob_array)

    sort_index = torch.argsort(-prob_array)

    mask_map = torch.zeros_like(pred_map)
    already_x = set()
    already_y = set()
    multiplet_list = []
    for index in sort_index:
        x = x_array[index]
        y = y_array[index]

        # # no sharp stem-loop
        if abs(x - y) <= 1:    # when <=1, allow 1 element loop
            continue

        seq_pair = seq[x] + seq[y]
        if seq_pair not in canonical_pairs and allow_nc == False:
            # print(seq_pair)
            continue
            pass

        if x in already_x or y in already_y:  # this is conflict
            # multiplet_list.append([x+1,y+1])
            continue
        else:
            mask_map[x, y] = 1
            # already_x.add(x)
            # already_y.add(y)

    pred_map_without_multiplets = pred_map * mask_map

    return pred_map, pred_map_without_multiplets, multiplet_list, prob_map, mask_map, mask_map

class PSSMRNABertModel(fm.model.RNABertModel):
    
    def forward(self, pssm, repr_layers=..., need_head_weights=False, return_contacts=False, masked_tokens=None):
        x = pssm
        x = self.embed_scale * (x.permute(0, 1, 2)
                                     @ self.embed_tokens.weight.clone())
        x_tokens = torch.argmax(x, dim=-1)
        x_1 = x + self.embed_positions(x_tokens)
        padding_mask = x_tokens.eq(self.padding_idx)
        
        if self.model_version == 'ESM-1b':
            x_1 = self.emb_layer_norm_before(x_1)
            if padding_mask is not None:
                x_1 = x_1 * (1 - padding_mask.unsqueeze(-1).type_as(x_1))

    
        x_result = self.forwardhead(x_1, padding_mask, x_tokens, repr_layers,
                                    need_head_weights, return_contacts,
                                    masked_tokens)
        return x_result

    def forwardhead(self,
                    x,
                    padding_mask,
                    tokens,
                    repr_layers=[],
                    need_head_weights=False,
                    return_contacts=False,
                    masked_tokens=None):
        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(x,
                            self_attn_padding_mask=padding_mask,
                            need_head_weights=need_head_weights)
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        if self.model_version == 'ESM-1b':
            x = self.emb_layer_norm_after(x)
            x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

            # last hidden representation should have layer norm applied
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x
            # CJY at 2021.10.20 add masked_tokens for lm_head
            x = self.lm_head(x, masked_tokens)
        else:
            x = F.linear(x, self.embed_out, bias=self.embed_out_bias)
            x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if self.model_version == "ESM-1":
                # ESM-1 models have an additional null-token for attention, which we remove
                attentions = attentions[..., :-1]
            if padding_mask is not None:
                attention_mask = (1 - padding_mask.type_as(attentions))
                attention_mask = attention_mask.unsqueeze(
                    1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts
        return result

class PSSMBaseline(fm.downstream.baseline.Baseline):
    
    def __init__(self,
                 backbone_name,
                 seqwise_predictor_name="none",
                 elewise_predictor_name="none",
                 pairwise_predictor_name="none",
                 backbone_frozen=0,
                 backbone_random_initialization=False,):
        super().__init__(
                 backbone_name,
                 seqwise_predictor_name,
                 elewise_predictor_name,
                 pairwise_predictor_name,
                 backbone_frozen,
                 backbone_random_initialization,)
        self.backbone, self.backbone_alphabet = fm.pretrained.rna_fm_t12(model_location="/WORK/luzlab/.cache/torch/hub/checkpoints/RNA-FM_pretrained.pth",sp_model=PSSMRNABertModel)
        
    def forward(self, pssm):
        
        pssm = pssm[0].to(device)
        pssm = pad_start(pad(pssm)).unsqueeze(0)
        pssm[:, 0] = torch.tensor([1.0] + [0] * 24)
        pssm[:, -1] = torch.tensor([0] * 2 + [1.0] + [0] * 22)
        data = {'token':torch.argmax(pssm, dim=-1)}
        need_head_weights = self.backbone_control_info["need_head_weights"]
        repr_layers = self.backbone_control_info["repr_layers"]
        return_contacts = self.backbone_control_info["return_contacts"]

        if self.backbone_frozen != -1:
            if self.backbone_frozen == 1:
                self.backbone.eval()
                with torch.no_grad():
                    results = self.backbone(pssm, need_head_weights=need_head_weights,
                                            repr_layers=repr_layers, return_contacts=return_contacts)
            elif self.backbone_frozen == 2:
                need_forward = False
                for des in data["description"]:
                    if des not in self.backbone_frozen_output_cache:
                        need_forward = True

                if need_forward == True:
                    self.backbone.eval()
                    with torch.no_grad():
                        results = self.backbone(pssm, need_head_weights=need_head_weights,
                                                repr_layers=repr_layers, return_contacts=return_contacts)
                    self.save_backbone_output_cache(data, results)
                else:
                    results = self.load_backbone_output_cache(data)
            else:  # 0
                results = self.backbone(pssm, need_head_weights=need_head_weights,
                                        repr_layers=repr_layers, return_contacts=return_contacts)
        else:
            results = {}

        for key in self.downstream_modules.keys():
            key_info = key.split(":")
            predictor_type, target_name = key_info[0], key_info[1]
            ds_module_input = self.fetch_ds_module_input(data, results, self.downstream_modules[key].input_type)
            results[target_name] = self.downstream_modules[key](data['token'], ds_module_input).float()    # .float() for fp16 to fp32

        return results

fm_model, alphabet = fm.pretrained.rna_fm_t12(model_location="/WORK/luzlab/.cache/torch/hub/checkpoints/RNA-FM_pretrained.pth",sp_model=None)
batch_converter = alphabet.get_batch_converter()
fm_model = PSSMBaseline(backbone_name='rna-fm',pairwise_predictor_name='pc-resnet_1_sym_first:r-ss')
fm_model.load_param('overall',model_path='./RNA-FM-ResNet_PDB-All.pth')
fm_model.eval()

def predict(model, pssm, ss_, window=150, shift=50):
    feat = pssm
    if isinstance(ss_, np.ndarray):
        ss_ = torch.from_numpy(ss_).to(device)
    L = feat.shape[1]
    res_id = torch.arange(L, device=device).view(1, L)
    if L > 300:  # predict by crops for long RNA
        pred_dict = {
            'contact': torch.zeros((L, L), device=device),
            'distance': {k: torch.zeros((L, L, n_bins['2D']['distance']), device=device) for k in
                            obj['2D']['distance']},
        }

        count_1d = torch.zeros((L)).to(device)
        count_2d = torch.zeros((L, L)).to(device)
        #
        grids = np.arange(0, L - window + shift, shift)
        ngrids = grids.shape[0]
        print("ngrid:     ", ngrids)
        print("grids:     ", grids)
        print("windows:   ", window)

        idx_pdb = torch.arange(L).long().view(1, L)
        for i in range(ngrids):
            for j in range(i, ngrids):
                start_1 = grids[i]
                end_1 = min(grids[i] + window, L)
                start_2 = grids[j]
                end_2 = min(grids[j] + window, L)
                sel = np.zeros((L)).astype(np.bool_)
                sel[start_1:end_1] = True
                sel[start_2:end_2] = True

                input_msa = feat[:, sel]
                input_ss = ss_[sel][:, sel]
                mask = torch.sum(input_msa == 4, dim=-1) < .7 * sel.sum()  # remove too gappy sequences

                input_msa = input_msa[mask]
                input_idx = idx_pdb[:, sel]
                input_res_id = res_id[:, sel]

                print("running crop: %d-%d/%d-%d" % (start_1, end_1, start_2, end_2), input_msa.shape)
                pred_gemos = model(input_msa, input_ss, res_id=input_res_id.to(device),
                                    msa_cutoff=args.nrows)['geoms']
                weight = 1
                sub_idx = input_idx[0].cpu()
                sub_idx_2d = np.ix_(sub_idx, sub_idx)
                count_2d[sub_idx_2d] += weight
                count_1d[sub_idx] += weight

                for k in obj['2D']:
                    if k == 'contact':
                        pred_dict['contact'][sub_idx_2d] += weight * pred_gemos['contact']
                    else:
                        for a in obj['2D'][k]:
                            pred_dict[k][a][sub_idx_2d] += weight * pred_gemos[k][a]
        for k in obj['2D']:
            if k == 'contact':
                pred_dict['contact'] /= count_2d
            else:
                for a in obj['2D'][k]:
                    if pred_dict[k][a].size().__len__() == 3:
                        pred_dict[k][a] /= count_2d[:, :, None]
                    else:
                        pred_dict[k][a] /= count_2d
    else:
        pred_dict_ = model(feat, ss_, res_id=res_id.to(device), msa_cutoff=args.nrows, is_training=True)
        pred_dict = pred_dict_['geoms']
        err = pred_dict_['estogram']
    for l in pred_dict:
        if isinstance(pred_dict[l], dict):
            for k in pred_dict[l]:
                pred_dict[l][k] = pred_dict[l][k]#.cpu().detach().numpy()
        else:
            pred_dict[l] = pred_dict[l]#.cpu().detach().numpy()
    return pred_dict,err

def score_with_pdb(pssm, template, noise_pred_dict, pdb, pdbmask, mask_):
    loss = 0
    total_kl_impainting = 0
    total_structure_loss = 0
    pssm_ = pssm * 1
    pssm_ = torch.softmax(pssm_,dim=-1)
    pssm_one_hot = torch.nn.functional.one_hot(torch.argmax(pssm_,dim=-1),4).float()
    pssm_ = pssm_ + (pssm_one_hot - pssm_).detach()
    pssm_ = nn.functional.pad(pssm_,(0,1),value=0)
    pssm_ = pssm_ * ( 1 - mask_.unsqueeze(-1)) + template.detach() * mask_.unsqueeze(-1)
    
    seq = pssm_.argmax(-1)[0]
    seq = ''.join([str(i) for i in seq.cpu().detach().numpy() if i != 5])
    seq = seq.replace('0','A').replace('1','U').replace('2','C').replace('3','G').replace('4','-')
    with torch.cuda.amp.autocast():
        pssm_for_fm = pssm_ * 1
        pssm_for_fm[:,:,[1,2,3]] = pssm_for_fm[:,:,[3,1,2]]
        p_map = fm_model(pssm_for_fm)
        mask_map = get_nc_matrix(seq).to(device)
        prob_map = torch.sigmoid(p_map['r-ss'])[0]
        prob_map = prob_map * mask_map
        pred_dict,err = predict(model, pssm_, prob_map, window=150, shift=50)
    for pdbmask_ in pdbmask:
        structure_loss = designloss.structure_loss_with_pdb_template(pred_dict,pdb,pdbmask_,)
        for a in structure_loss.keys():
            total_structure_loss += structure_loss[a]
    geom_loss = designloss.hallucinate_loss(pred_dict,noise_pred_dict,mask_)
    for a in geom_loss['impainting']['distance'].keys():
        total_kl_impainting += geom_loss['impainting']['distance'][a]
    with torch.cuda.amp.autocast():
        loss = - args.impaintlr * total_kl_impainting + args.structlr * total_structure_loss
    return loss, total_kl_impainting, total_structure_loss,pred_dict

def score_without_pdb(pssm, template, template_pred_dict, noise_pred_dict, mask_):
    loss = 0
    total_kl_impainting = 0
    total_structure_loss = 0

    pssm_ = pssm * 1
    pssm_ = torch.softmax(pssm_,dim=-1)
    pssm_one_hot = torch.nn.functional.one_hot(torch.argmax(pssm_,dim=-1),4).float()
    pssm_ = pssm_ + (pssm_one_hot - pssm_).detach()
    pssm_ = nn.functional.pad(pssm_,(0,1),value=0)
    pssm_ = pssm_ * ( 1 - mask_.unsqueeze(-1)) + template.detach() * mask_.unsqueeze(-1)
    
    seq = pssm_.argmax(-1)[0]
    seq = ''.join([str(i) for i in seq.cpu().detach().numpy() if i != 5])
    seq = seq.replace('0','A').replace('1','U').replace('2','C').replace('3','G').replace('4','-')
    with torch.cuda.amp.autocast():
        pssm_for_fm = pssm_ * 1
        pssm_for_fm[:,:,[1,2,3]] = pssm_for_fm[:,:,[3,1,2]]
        p_map = fm_model(pssm_for_fm)
        mask_map = get_nc_matrix(seq).to(device)
        prob_map = torch.sigmoid(p_map['r-ss'])[0]
        prob_map = prob_map * mask_map
        pred_dict,err = predict(model, pssm_, prob_map, window=150, shift=50)
    structure_loss = designloss.hallucinate_loss(pred_dict,template_pred_dict,mask_,valid=True)['template']['distance']
    for a in structure_loss.keys():
        total_structure_loss += structure_loss[a]
    geom_loss = designloss.hallucinate_loss(pred_dict,noise_pred_dict,mask_,valid=True)
    for a in geom_loss['impainting']['distance'].keys():
        total_kl_impainting += geom_loss['impainting']['distance'][a]
    with torch.cuda.amp.autocast():
        loss = - args.impaintlr * total_kl_impainting + args.structlr * total_structure_loss
    return loss, total_kl_impainting, total_structure_loss, pred_dict

def inverse_3dscore(pssm, template, pdb, mask_, pdbmask):
    loss = 0
    pssm_ = pssm * 1
    pssm_ = torch.softmax(pssm_,dim=-1)
    pssm_one_hot = torch.nn.functional.one_hot(torch.argmax(pssm_,dim=-1),4).float()
    pssm_ = pssm_ + (pssm_one_hot - pssm_).detach()
    pssm_ = nn.functional.pad(pssm_,(0,1),value=0)
    pssm_ = pssm_ * ( 1 - mask_[None,:,None]) + template.detach() * mask_[None,:,None]
    
    seq = pssm_.argmax(-1)[0]
    seq = ''.join([str(i) for i in seq.cpu().detach().numpy() if i != 5])
    seq = seq.replace('0','A').replace('1','U').replace('2','C').replace('3','G').replace('4','-')
    with torch.cuda.amp.autocast():
        pssm_for_fm = pssm_ * 1
        pssm_for_fm[:,:,[1,2,3]] = pssm_for_fm[:,:,[3,1,2]]
        p_map = fm_model(pssm_for_fm)
        mask_map = get_nc_matrix(seq).to(device)
        prob_map = torch.sigmoid(p_map['r-ss'])[0]
        prob_map = prob_map * mask_map
        pred_dict,err = predict(model, pssm_, prob_map, window=150, shift=50)
    for pdbmask_ in pdbmask:
        structure_loss = designloss.structure_loss_with_pdb_template(pred_dict,pdb,pdbmask_,)
        for a in structure_loss.keys():
            loss += structure_loss[a]
    return loss, pred_dict

def inverse_2dscore(pssm, template, mask_, valid=False):
    ss_matrix = dot2ssmat(args.ss)
    target = torch.from_numpy(ss_matrix).to(device)
    pssm_ = pssm * 1
    pssm_ = torch.softmax(pssm_,dim=-1)
    pssm_one_hot = torch.nn.functional.one_hot(torch.argmax(pssm_,dim=-1),4).float()
    pssm_ = pssm_ + (pssm_one_hot - pssm_).detach()
    pssm_ = nn.functional.pad(pssm_,(0,1),value=0)
    pssm_ = pssm_ * ( 1 - mask_.unsqueeze(-1)) + template.detach() * mask_.unsqueeze(-1)
    
    seq = pssm_.argmax(-1)[0]
    seq = ''.join([str(i) for i in seq.cpu().detach().numpy() if i != 5])
    seq = seq.replace('0','A').replace('1','U').replace('2','C').replace('3','G').replace('4','-')
    with torch.cuda.amp.autocast():
        pssm_for_fm = pssm_ * 1
        pssm_for_fm[:,:,[1,2,3]] = pssm_for_fm[:,:,[3,1,2]]
        p_map = fm_model(pssm_for_fm)
        mask_map = get_nc_matrix(seq).to(device)
        prob_map = p_map['r-ss'][0]
        prob_map = prob_map * mask_map
        if valid:
            pred_dict,err = predict(model, pssm_, prob_map, window=150, shift=50)
        # print('prob_map:',prob_map)
        # print('target:',target)
        loss = nn.functional.binary_cross_entropy_with_logits(prob_map, target)
    if valid:
        return loss, pred_dict
    else:
        return loss

def mcmcdesign(model,bkg_model,template, ss_, mask, window=150, shift=50, pdb=None, pdbmask=None):
    mask_ = torch.zeros(ss_.shape[0]).to(device)
    mask_[mask] = 1
    length = ss_.shape[0]
    pssm = torch.randn_like(template[:,:,:4])
    pssm = pssm.to(device)
    raw_msa = torch.argmax(template,dim=-1)
    raw_ss_ = ss_
    noise = torch.randn_like(template)
    with torch.no_grad():
        noise = torch.softmax(noise,dim=-1)
        noise_pred_dict,_ = predict(bkg_model, noise, ss_, window=window, shift=shift)
    for l in noise_pred_dict:
        if isinstance(noise_pred_dict[l], dict):
            for k in noise_pred_dict[l]:
                noise_pred_dict[l][k] = noise_pred_dict[l][k].detach()
        else:
            noise_pred_dict[l] = noise_pred_dict[l].detach()
    if ss_.shape[0] != raw_msa.shape[-1]:
        raise ValueError(f'ss length {ss_.shape[0]}, msa length {raw_msa.shape[1]}!')

    nomutsite = mask
    site = list(range(length))
    mutsite = list(set(site) - set(nomutsite))
    print(mutsite)
    import math
    total_stpes = int(10000*((length/128))) + 1
    nt = np.array([0,1,2,3])
    M = np.linspace(int(6), int(2), total_stpes)
    pbar = tqdm.tqdm(range(total_stpes))
    
    seq = np.random.choice(nt,(1,length))
    seq = torch.from_numpy(seq).to(device)
    seq[:,mask] = raw_msa[:,mask]

    best_score = -100000
    best_seq = seq.cpu().detach().numpy()

    bsl_seq = template[:,:,:4]
    factor = 0.043 * math.exp(-0.045*ss_.shape[0]) + 1.015
    factor = 1 / factor
    with torch.no_grad():
        template_for_fm = template.detach().clone()
        template_for_fm[:,:,[1,2,3]] = template_for_fm[:,:,[3,1,2]]
        p_map = fm_model(template_for_fm)
        seq_ = torch.argmax(seq,dim=-1).cpu().detach().numpy()
        seq_ = ''.join([str(i) for i in seq_ if i != 5])
        seq_ = seq_.replace('0','A').replace('1','U').replace('2','C').replace('3','G').replace('4','-')
        ss,_ = vienna_ss_pred(seq_)
        ss_ = dot2ssmat(ss)
        ss_ = torch.from_numpy(ss_).to(device)
        prob_map = torch.sigmoid(p_map['r-ss'])[0] # * 0.2 + 0.2 * ss_ + 0.6 * raw_ss_
        bsl_seq_ = template
        template_pred_dict,_ = predict(model, bsl_seq_, prob_map, window=window, shift=shift)
        if args.task == 'inversebyss':
            bsl_score = inverse_2dscore(bsl_seq,template,mask_)
            print('baseline score:',- bsl_score.item())
        elif args.task == 'inverse3d':
            bsl_score,_ = inverse_3dscore(bsl_seq,template,pdb,mask_,pdbmask)
            print('baseline score:',- bsl_score.item())
        elif args.task == 'designwithpdb':
            bsl_score, total_kl_impainting, struct_loss,_ = score_with_pdb(bsl_seq,noise_pred_dict,pdb,pdbmask,mask_)
            print('baseline score:',- bsl_score.item())
            print('baseline structure loss:',struct_loss.item())
            print('baseline kl loss:',total_kl_impainting.item())
        elif args.task == 'designwithoutpdb':
            bsl_score, total_kl_impainting, struct_loss,_ = score_without_pdb(bsl_seq,template_pred_dict,noise_pred_dict,mask_)
            print('baseline score:',- bsl_score.item())
            print('baseline structure loss:',struct_loss.item())
            print('baseline kl loss:',total_kl_impainting.item())
        else:
            raise ValueError('task not supported')
    print('target score:',- bsl_score.item() * factor,flush=True)
    for step in pbar:
        T = 0.05 * (np.exp(np.log(0.5) / (total_stpes/5)) ** step)
        n_mutations = round(M[step])
        seq = best_seq.copy()
        seq[:,np.random.choice(mutsite,n_mutations)] = np.random.choice(nt,n_mutations)
        seq = torch.from_numpy(seq).to(device)
        seq = torch.nn.functional.one_hot(seq,4).float()
        # seq = nn.functional.pad(seq,(0,1),value=0)
        with torch.no_grad():
            if args.task == 'inversebyss':
                loss = inverse_2dscore(seq,template,mask_)
            elif args.task == 'inverse3d':
                loss, _ = inverse_3dscore(seq,template,pdb,mask_,pdbmask)
            elif args.task == 'designwithpdb':
                loss, total_kl_impainting, struct_loss, _ = score_with_pdb(seq,template,noise_pred_dict,pdb,pdbmask,mask_)
            elif args.task == 'designwithoutpdb':
                loss, total_kl_impainting, struct_loss, _ = score_without_pdb(seq,template,template_pred_dict,noise_pred_dict,mask_)
            else:
                raise ValueError('task not supported')
            score = - loss
            delta = - score + best_score
            delta = delta.cpu().numpy()
            if delta < 0 or np.random.uniform(0,1) < np.exp(-delta/T):
                best_score = score
                best_seq = torch.argmax(seq,dim=-1).cpu().detach().numpy()
                seq = ''.join([str(i) for i in best_seq[0] if i != 5])
                seq = seq.replace('0','A').replace('1','U').replace('2','C').replace('3','G').replace('4','-')
                if args.task == 'inversebyss' or args.task == 'inverse3d':
                    print('ACCEPT MUTATION',step,score.item(),seq,flush=True)
                else:
                    print('ACCEPT MUTATION',step,score.item(),total_kl_impainting.item(),struct_loss.item(),seq,flush=True)
        if best_score > - factor * bsl_score:
            print('EARLY STOP',step,best_score.item(),bsl_score.item())
            break
    # last validation
    seq = ''.join([str(i) for i in best_seq[0] if i != 5])
    seq = seq.replace('0','A').replace('1','U').replace('2','C').replace('3','G').replace('4','-')
    print('best seq:',seq)
    seq = best_seq
    seq = torch.from_numpy(seq).to(device)
    seq = torch.nn.functional.one_hot(seq,4).float()
    with torch.no_grad():
        if args.task == 'inversebyss':
            loss, pred_dict = inverse_2dscore(seq,template,mask_,valid=True)
        elif args.task == 'inverse3d':
            loss, pred_dict = inverse_3dscore(seq,template,pdb,mask_,pdbmask)
        elif args.task == 'designwithpdb':
            loss, struct_loss, total_kl_impainting, pred_dict = score_with_pdb(seq,template,noise_pred_dict,pdb,pdbmask,mask_)
        elif args.task == 'designwithoutpdb':
            loss, struct_loss, total_kl_impainting, pred_dict = score_without_pdb(seq,template,template_pred_dict,noise_pred_dict,mask_)
        else:
            raise ValueError('task not supported')
        score = - loss
        best_score = score
        best_seq = torch.argmax(seq,dim=-1).cpu().detach().numpy()
        seq = ''.join([str(i) for i in best_seq[0] if i != 5])
        seq = seq.replace('0','A').replace('1','U').replace('2','C').replace('3','G').replace('4','-')
        if args.task == 'inversebyss' or args.task == 'inverse3d':
            print('validation',score.item(),seq,flush=True)
        else:
            print('validation',score.item(),total_kl_impainting.item(),struct_loss.item(),seq,flush=True)
    
    for l in pred_dict:
        if isinstance(pred_dict[l], dict):
            for k in pred_dict[l]:
                pred_dict[l][k] = pred_dict[l][k].cpu().detach().numpy()
        else:
            pred_dict[l] = pred_dict[l].cpu().detach().numpy()
    
    raw_seq = template[0,:,:4]
    raw_seq = torch.argmax(raw_seq,dim=-1)
    raw_seq = ''.join([str(i) for i in raw_seq.cpu().detach().numpy() if i != 5])
    raw_seq = raw_seq.replace('0','A').replace('1','U').replace('2','C').replace('3','G').replace('4','-')
    # calculate difference between raw_seq and best_seq
    diff = 0
    same = 0
    for i in range(len(raw_seq)):
        if raw_seq[i] != seq[i]:
            diff += 1
        else:
            same += 1
    print('diff:',diff)
    print('same:',same-2)
    print('recovery:',same/len(raw_seq))
    return seq, pred_dict

def gddesign(model, bkg_model, template, ss_, mask, window=150, shift=50, pdb=None, pdbmask=None):
    mask_ = torch.zeros(ss_.shape[0]).to(device)
    mask_[mask] = 1
    pssm = template.detach().clone()[:,:,:4] * 0.01
    pssm = torch.randn_like(pssm) * 0.01 # + pssm
    pssm = pssm.to(device)
    pssm = pssm * ( 1 - mask_[None,:,None]) + template[:,:,:4].detach() * mask_[None,:,None]
    pssm.requires_grad = True
    noise = torch.randn_like(template)
    with torch.no_grad():
        noise = torch.softmax(noise,dim=-1)
        noise_pred_dict,_ = predict(bkg_model, noise, ss_, window=window, shift=shift)
    for l in noise_pred_dict:
        if isinstance(noise_pred_dict[l], dict):
            for k in noise_pred_dict[l]:
                noise_pred_dict[l][k] = noise_pred_dict[l][k].detach()
        else:
            noise_pred_dict[l] = noise_pred_dict[l].detach()
    raw_ss_ = ss_
    bsl_seq = template[:,:,:4]
    if args.pdb is None:
        template_for_fm = template.detach().clone()
        template_for_fm[:,:,[1,2,3]] = template_for_fm[:,:,[3,1,2]]
        p_map = fm_model(template_for_fm)
        mask_map = get_nc_matrix(template).to(device)
        prob_map = torch.sigmoid(p_map['r-ss'])[0] * 0.2 + 0.6 * ss_ + 0.2 * raw_ss_
        prob_map = prob_map * mask_map
        template_pred_dict,err = predict(model, template, prob_map, window=window, shift=shift)
        if args.task == 'inversebyss':
            bsl_score = inverse_2dscore(bsl_seq,template,mask_)
            print('baseline score:',- bsl_score.item())
        elif args.task == 'inverse3d':
            bsl_score,_ = inverse_3dscore(bsl_seq,template,pdb,mask_,pdbmask)
            print('baseline score:',- bsl_score.item())
        elif args.task == 'designwithpdb':
            bsl_score, total_kl_impainting, struct_loss,_ = score_with_pdb(bsl_seq,template,noise_pred_dict,pdb,pdbmask,mask_)
            print('baseline score:',- bsl_score.item())
            print('baseline structure loss:',struct_loss.item())
            print('baseline kl loss:',total_kl_impainting.item())
        elif args.task == 'designwithoutpdb':
            bsl_score, total_kl_impainting, struct_loss,_ = score_without_pdb(bsl_seq,template,template_pred_dict,noise_pred_dict,mask_)
            print('baseline score:',- bsl_score.item())
            print('baseline structure loss:',struct_loss.item())
            print('baseline kl loss:',total_kl_impainting.item())
        else:
            raise ValueError('task not supported')
    best_results = {
        'pssm':pssm,
        'loss':10000,
    }
    if args.task == 'designwithpdb' or args.task == 'designwithoutpdb':
        best_results['struct_loss'] = struct_loss.item()
        best_results['kl_loss'] = total_kl_impainting.item()
    pbar = tqdm.tqdm(range(1000))
    for step in pbar:
        ss_ = ss_.detach()
        if args.task == 'inversebyss':
            loss = inverse_2dscore(pssm,template,mask_)
        elif args.task == 'inverse3d':
            loss, _ = inverse_3dscore(pssm,template,pdb,mask_,pdbmask)
        elif args.task == 'designwithpdb':
            loss, total_kl_impainting, struct_loss, _ = score_with_pdb(pssm,template,noise_pred_dict,pdb,pdbmask,mask_)
        elif args.task == 'designwithoutpdb':
            loss, total_kl_impainting, struct_loss, _ = score_without_pdb(pssm,template,template_pred_dict,noise_pred_dict,mask_)
        else:
            raise ValueError('task not supported')
        loss.backward()
        pssm = (pssm - args.lr * pssm.grad).detach()
        pssm.requires_grad = True
        if loss < best_results['loss']:
            best_results = {
                'pssm':pssm,
                'loss':loss,
            }
            seq = pssm.argmax(-1)[0]
            seq = ''.join([str(i) for i in seq.cpu().detach().numpy() if i != 5])
            seq = seq.replace('0','A').replace('1','U').replace('2','C').replace('3','G').replace('4','-')
            mfess,mfeen = vienna_ss_pred(seq)
            print(mfess,mfeen,best_results['loss'].item(),seq,flush=True)
            if args.task == 'designwithpdb' or args.task == 'designwithoutpdb':
                best_results['struct_loss'] = struct_loss.item()
                best_results['kl_loss'] = total_kl_impainting.item()
        if loss < bsl_score:
            break
    # final validation
    with torch.no_grad():
        if args.task == 'inversebyss':
            loss, pred_dict = inverse_2dscore(best_results['pssm'],template,mask_,valid=True)
        elif args.task == 'inverse3d':
            loss, pred_dict = inverse_3dscore(best_results['pssm'],template,pdb,mask_,pdbmask,valid=True)
        elif args.task == 'designwithpdb':
            loss, struct_loss, total_kl_impainting, pred_dict = score_with_pdb(best_results['pssm'],template,noise_pred_dict,pdb,pdbmask,mask_)
        elif args.task == 'designwithoutpdb':
            loss, struct_loss, total_kl_impainting, pred_dict = score_without_pdb(best_results['pssm'],template,template_pred_dict,noise_pred_dict,mask_)
        else:
            raise ValueError('task not supported')
        best_results['loss'] = loss
        if args.task == 'designwithpdb' or args.task == 'designwithoutpdb':
            best_results['struct_loss'] = struct_loss
            best_results['kl_loss'] = total_kl_impainting
    print('final validation')
    mfess,mfeen = vienna_ss_pred(seq)
    print(mfess,mfeen,seq)
    if args.task == 'designwithpdb' or args.task == 'designwithoutpdb':
        print('final validation',best_results['loss'].item(),best_results['kl_loss'].item(),best_results['struct_loss'].item())
    for l in pred_dict:
        if isinstance(pred_dict[l], dict):
            for k in pred_dict[l]:
                pred_dict[l][k] = pred_dict[l][k].cpu().detach().numpy()
        else:
            pred_dict[l] = pred_dict[l].cpu().detach().numpy()
    return seq, pred_dict

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.set_num_threads(args.cpu)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    py = sys.executable
    fm_model.to(device)
    out_dir = os.path.dirname(os.path.abspath(args.npz))
    os.makedirs(out_dir, exist_ok=True)

    cwd = os.getcwd()
    print('inpainting lr:',args.impaintlr)
    print('structure lr:',args.structlr)
    
    print('predict geometries')
    config = json.load(open(f'{args.model_pth}/config/model_1.json', 'r'))

    model = DistPredictorforDesign(dim_2d=config['channels'], layers_2d=config['n_blocks'])

    model_ckpt = torch.load(f'{args.model_pth}/models/model_1.pth.tar', map_location=device)
    model.load_state_dict(model_ckpt,strict=False)
    model.eval()
    model.to(device)
    
    bkg_config = json.load(open('./params/background/config/model_1.json', 'r'))
    bkg_model = DistPredictorforDesign(dim_2d=bkg_config['channels'], layers_2d=bkg_config['n_blocks'])
    bkg_model_ckpt = torch.load(f'./params/background/models/model_1.pth.tar', map_location=device)
    bkg_model.load_state_dict(bkg_model_ckpt,strict=False)
    bkg_model.eval()
    bkg_model.to(device)
    if args.ss:
        if os.path.exists(args.ss):
            dot_ss_ = open(args.ss).readlines()[0].split('\n')
        else:
            dot_ss_ = args.ss
        ss_ = dot2ssmat(dot_ss_)
        ss_ = torch.from_numpy(ss_).to(device)
    pdb = None
    pdbmask = None
    if args.pdb:
        pdb = pdb2cst.pdb2cst(args.pdb)
    if args.pdbmask:
        rsts1,rsts2 = args.pdbmask.split('|')
        pdbmask = pointidx.rst_struc_trans(rsts1,rsts2)
        
    print('loading template......')
    template = args.template
    seq = template
    # predict SS by SPOT-RNA
    print('predict SS by RNA-FM')
    
    seqmask = pointidx.rst_seq_trans(args.seqmask)

    print('done!')
    print('predicting......')
    
    
    template = ['AUCG-'.index(i) for i in template]
    template = torch.tensor(template).unsqueeze(0).to(device)
    template = torch.nn.functional.one_hot(template,5).float()
    template_for_fm = template.detach().clone()
    template_for_fm[:,:,[1,2,3]] = template_for_fm[:,:,[3,1,2]]
    p_map = fm_model(template_for_fm)
    prob_map = torch.sigmoid(p_map['r-ss'])[0]
    
    if args.ss is None:
        ss_ = prob_map
    if args.method == 'mcmc':
        print('mcmc design......')
        seq,pred = mcmcdesign(model, bkg_model, template, ss_, seqmask, window=150, shift=50, pdb=pdb, pdbmask=pdbmask)
        print('done!')
        print('saving......')
        np.savez_compressed(args.npz, **pred)
        # seq = pred['seq']
        # seq = ''.join([str(i) for i in seq if i != 5])
        # seq = seq.replace('0','A').replace('1','U').replace('2','C').replace('3','G').replace('4','-')
        with open(args.npz.replace('.npz',f'.fa'),'w') as f:
            print(seq)
            f.write('>design\n'+seq)
    elif args.method == 'gd':
        print('gradient descent design......')
        seq,pred = gddesign(model, bkg_model, template, ss_, seqmask, window=150, shift=50, pdb=pdb, pdbmask=pdbmask)
        print('done!')
        print('saving......')
        np.savez_compressed(args.npz, **pred)
        # seq = pssm[0,:]
        # print(seq)
        # seq = ''.join([str(i) for i in seq if i != 5])
        # seq = seq.replace('0','A').replace('1','U').replace('2','C').replace('3','G').replace('4','-')
        with open(args.npz.replace('.npz',f'.fa'),'w') as f:
            f.write('>design\n'+seq)
    else:
        raise ValueError('method not supported')